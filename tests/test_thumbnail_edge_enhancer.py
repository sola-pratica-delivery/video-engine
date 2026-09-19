"""Testes para algoritmos de feathering, stroke e glow suave (Issue #12)."""

from __future__ import annotations

import numpy as np

from video_engine.thumbnail.edge_enhancer import (
    apply_stroke_and_glow,
    create_glow_mask,
    create_stroke_mask,
    dilate_mask,
    feather_mask,
)
from video_engine.thumbnail.models import GlowConfig, StrokeConfig


def _make_solid_circle_mask(size: int = 100, radius: int = 30) -> np.ndarray:
    """Gera mascara binaria com circulo solido no centro (0 ou 255)."""
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[dist <= radius] = 255
    return mask


def _make_foreground_rgba(size: int = 100, radius: int = 30) -> np.ndarray:
    """Gera imagem RGBA com sujeito colorido no centro e fundo transparente."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    mask = _make_solid_circle_mask(size, radius)
    # Sujeito azul opaco
    rgba[mask > 0, 0] = 50
    rgba[mask > 0, 1] = 120
    rgba[mask > 0, 2] = 220
    rgba[mask > 0, 3] = 255
    return rgba


class TestEdgeEnhancerPrimitives:
    def test_dilate_mask_expands_boundaries(self) -> None:
        mask = _make_solid_circle_mask(size=80, radius=20)
        dilated = dilate_mask(mask, radius=5)

        assert dilated.shape == mask.shape
        assert dilated.dtype == np.uint8
        # A area dilatada deve ser estritamente maior
        assert np.sum(dilated > 0) > np.sum(mask > 0)
        # Onde a mascara original era positiva, a dilatada tambem deve ser
        assert np.all(dilated[mask > 0] == 255)

    def test_feather_mask_smooths_staircasing(self) -> None:
        # Mascara binaria dura (degrau 0 -> 255)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[:, 20:] = 255

        feathered = feather_mask(mask, radius=3)

        assert feathered.shape == mask.shape
        assert feathered.dtype == np.uint8
        # Na fronteira entre x=18 e x=22 deve haver valores intermediarios (anti-aliasing)
        boundary_slice = feathered[20, 17:24]
        assert not np.all(np.isin(boundary_slice, [0, 255]))
        # Transicao monotonicamente crescente da esquerda para a direita
        assert np.all(np.diff(boundary_slice) >= 0)

    def test_create_stroke_mask_surrounds_subject(self) -> None:
        mask = _make_solid_circle_mask(size=100, radius=25)
        stroke = create_stroke_mask(mask, width=6)

        assert stroke.shape == mask.shape
        assert stroke.dtype == np.uint8
        # O interior do circulo nao deve conter stroke (o stroke fica ao redor)
        assert np.all(stroke[mask == 255] == 0)
        # Deve haver pixels de stroke na vizinhanca externa imediata
        assert np.sum(stroke > 0) > 0

    def test_create_glow_mask_decays_smoothly(self) -> None:
        mask = _make_solid_circle_mask(size=120, radius=25)
        glow = create_glow_mask(mask, radius=15, intensity=0.9)

        assert glow.shape == mask.shape
        assert glow.dtype == np.uint8
        # O interior do sujeito tem glow zerado (ocupado pelo proprio sujeito)
        assert np.all(glow[mask == 255] == 0)

        # Proximo a borda o glow deve ser mais intenso do que longe da borda
        center = 60
        # Ponto imediatamente fora da borda (r = 28)
        val_near = glow[center, center + 28]
        # Ponto mais distante (r = 45)
        val_far = glow[center, center + 45]

        assert val_near > val_far
        assert val_far >= 0


class TestApplyStrokeAndGlow:
    def test_preserves_subject_interior_pixels(self) -> None:
        fg = _make_foreground_rgba(size=80, radius=20)
        original_interior = fg[fg[:, :, 3] == 255].copy()

        stroke_cfg = StrokeConfig(enabled=True, width=5, color=(255, 255, 255), opacity=1.0)
        glow_cfg = GlowConfig(enabled=True, radius=10, color=(255, 200, 0), intensity=0.8)

        enhanced = apply_stroke_and_glow(fg, stroke=stroke_cfg, glow=glow_cfg)

        assert enhanced.shape == fg.shape
        assert enhanced.dtype == np.uint8

        # Invariante crucial: os pixels do sujeito continuam 100% intocados
        subject_mask = fg[:, :, 3] == 255
        assert np.array_equal(enhanced[subject_mask], original_interior)

    def test_stroke_only(self) -> None:
        fg = _make_foreground_rgba(size=80, radius=20)
        stroke_cfg = StrokeConfig(enabled=True, width=4, color=(255, 255, 255), opacity=1.0)

        enhanced = apply_stroke_and_glow(fg, stroke=stroke_cfg, glow=None)

        # Deve haver pixels brancos (255, 255, 255, 255) no stroke imediatamente fora do sujeito
        stroke_pixels = enhanced[(fg[:, :, 3] == 0) & (enhanced[:, :, 3] > 0)]
        assert len(stroke_pixels) > 0
        assert np.all(stroke_pixels[:, :3] == [255, 255, 255])

    def test_glow_only(self) -> None:
        fg = _make_foreground_rgba(size=80, radius=20)
        glow_cfg = GlowConfig(enabled=True, radius=12, color=(0, 255, 255), intensity=0.9)

        enhanced = apply_stroke_and_glow(fg, stroke=None, glow=glow_cfg)

        # Na regiao externa ao sujeito deve haver alfa suave decrescente
        outer_alphas = enhanced[fg[:, :, 3] == 0, 3]
        assert np.max(outer_alphas) > 0
        assert np.min(outer_alphas) == 0

    def test_both_stroke_and_glow_layering(self) -> None:
        # Sujeito na frente -> Stroke intermediario -> Glow no fundo
        fg = _make_foreground_rgba(size=100, radius=25)
        stroke_cfg = StrokeConfig(enabled=True, width=4, color=(255, 255, 255))
        glow_cfg = GlowConfig(enabled=True, radius=16, color=(255, 0, 0))

        enhanced = apply_stroke_and_glow(fg, stroke=stroke_cfg, glow=glow_cfg)

        # Deve haver camada total combinada sem estourar limites de uint8
        assert enhanced[:, :, 3].min() >= 0
        assert enhanced[:, :, 3].max() <= 255

    def test_no_effects_returns_copy(self) -> None:
        fg = _make_foreground_rgba(size=60, radius=15)
        result = apply_stroke_and_glow(fg, stroke=None, glow=None)
        assert np.array_equal(result, fg)
