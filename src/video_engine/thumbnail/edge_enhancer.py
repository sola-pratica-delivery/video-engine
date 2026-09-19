"""Algoritmos de feathering, contorno (stroke) e glow suave (Issue #12).

Implementa tecnicas de anti-aliasing em mascaras alfa e geracao de camadas
de destaque (stroke e glow suave) sem halos ou serrilhados, otimizando o
recorte do locutor para thumbnails de alto CTR.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from video_engine.thumbnail.models import GlowConfig, StrokeConfig


def _gaussian_kernel_1d(radius: int, sigma: Optional[float] = None) -> np.ndarray:
    """Gera kernel gaussiano 1D normalizado de tamanho 2*radius + 1."""
    if radius <= 0:
        return np.array([1.0], dtype=np.float64)
    sig = sigma if sigma is not None and sigma > 0 else max(radius / 2.5, 0.8)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sig) ** 2)
    return kernel / np.sum(kernel)


def _convolve_separable_2d(array: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Aplica convolucao 2D separavel com kernel 1D em array float64."""
    radius = (len(kernel_1d) - 1) // 2
    if radius <= 0:
        return array.copy()

    # Passagem horizontal
    pad_h = np.pad(array, ((0, 0), (radius, radius)), mode="edge")
    h_out = np.zeros_like(array, dtype=np.float64)
    for i, w in enumerate(kernel_1d):
        h_out += w * pad_h[:, i : i + array.shape[1]]

    # Passagem vertical
    pad_v = np.pad(h_out, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(array, dtype=np.float64)
    for i, w in enumerate(kernel_1d):
        out += w * pad_v[i : i + array.shape[0], :]

    return out


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilatacao morfologica com elemento estruturante circular.

    Expande as bordas da mascara em ``radius`` pixels.
    """
    if radius <= 0:
        return mask.copy()

    src = np.asarray(mask, dtype=np.uint8)
    h, w = src.shape
    dilated = src.copy()

    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= r2:
                y_src_start = max(0, -dy)
                y_src_end = min(h, h - dy)
                x_src_start = max(0, -dx)
                x_src_end = min(w, w - dx)

                y_dst_start = max(0, dy)
                y_dst_end = min(h, h + dy)
                x_dst_start = max(0, dx)
                x_dst_end = min(w, w + dx)

                np.maximum(
                    dilated[y_dst_start:y_dst_end, x_dst_start:x_dst_end],
                    src[y_src_start:y_src_end, x_src_start:x_src_end],
                    out=dilated[y_dst_start:y_dst_end, x_dst_start:x_dst_end],
                )

    return dilated


def feather_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Suavizacao (anti-aliasing) de bordas duras da mascara alfa.

    Aplica filtro gaussiano suave na fronteira da mascara eliminando
    serrilhados e criando transicoes graduais.
    """
    if radius <= 0:
        return mask.copy()

    kernel = _gaussian_kernel_1d(radius)
    blurred = _convolve_separable_2d(mask.astype(np.float64), kernel)
    return np.clip(np.round(blurred), 0, 255).astype(np.uint8)


def create_stroke_mask(mask: np.ndarray, width: int = 8) -> np.ndarray:
    """Gera mascara alfa isolada do contorno (stroke) ao redor do sujeito.

    O stroke e obtido dilatando a mascara original e subtraindo a silhueta
    do sujeito para que nao oclua pixels interiores.
    """
    if width <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    dilated = dilate_mask(mask, radius=width)
    stroke = np.clip(dilated.astype(np.int32) - mask.astype(np.int32), 0, 255)
    return stroke.astype(np.uint8)


def create_glow_mask(
    mask: np.ndarray,
    radius: int = 16,
    intensity: float = 0.8,
) -> np.ndarray:
    """Gera mascara alfa de brilho suave (glow) difuso ao redor do sujeito.

    Aplica difusao gaussiana na silhueta e mascara o interior do sujeito.
    """
    if radius <= 0 or intensity <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    kernel = _gaussian_kernel_1d(radius, sigma=radius / 2.0)
    blurred = _convolve_separable_2d(mask.astype(np.float64), kernel)

    # Mascara o interior do sujeito para manter o glow externamente
    outer_factor = 1.0 - (mask.astype(np.float64) / 255.0)
    glow = blurred * outer_factor * intensity
    return np.clip(np.round(glow), 0, 255).astype(np.uint8)


def _blend_layer_over(
    bottom_rgb: np.ndarray,
    bottom_alpha: np.ndarray,
    top_rgb: np.ndarray,
    top_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Composicao 'A over B' (Porter-Duff) em formato float normalizado."""
    a_top = top_alpha[:, :, np.newaxis]
    a_bot = bottom_alpha[:, :, np.newaxis]

    out_alpha = top_alpha + bottom_alpha * (1.0 - top_alpha)
    out_alpha_safe = np.maximum(out_alpha[:, :, np.newaxis], 1e-9)

    out_rgb = (top_rgb * a_top + bottom_rgb * a_bot * (1.0 - a_top)) / out_alpha_safe
    return np.clip(out_rgb, 0.0, 1.0), np.clip(out_alpha, 0.0, 1.0)


def apply_stroke_and_glow(
    foreground_rgba: np.ndarray,
    stroke: Optional[StrokeConfig] = None,
    glow: Optional[GlowConfig] = None,
) -> np.ndarray:
    """Aplica contorno (stroke) e brilho suave (glow) em um sujeito RGBA.

    A ordem de renderizacao e:
    1. Base: Glow suave (fundo mais distante)
    2. Meio: Stroke solido/semitransparente (contorno)
    3. Topo: Sujeito original intacto (preservando 100% dos pixels interiores)
    """
    array = np.asarray(foreground_rgba)
    if array.ndim != 3 or array.shape[2] != 4 or array.size == 0:
        raise ValueError("Esperado array RGBA 3D HxWx4")

    has_stroke = stroke is not None and stroke.enabled and stroke.width > 0
    has_glow = glow is not None and glow.enabled and glow.radius > 0

    if not has_stroke and not has_glow:
        return array.copy()

    h, w = array.shape[:2]
    subject_rgb = array[:, :, :3].astype(np.float64) / 255.0
    subject_alpha = array[:, :, 3].astype(np.float64) / 255.0
    binary_mask = (array[:, :, 3] > 0).astype(np.uint8) * 255

    # 1. Camada de Glow (fundo)
    curr_rgb = np.zeros((h, w, 3), dtype=np.float64)
    curr_alpha = np.zeros((h, w), dtype=np.float64)

    if has_glow and glow is not None:
        glow_mask = create_glow_mask(binary_mask, radius=glow.radius, intensity=glow.intensity)
        curr_alpha = glow_mask.astype(np.float64) / 255.0
        glow_color = np.array(glow.color, dtype=np.float64) / 255.0
        curr_rgb = np.tile(glow_color, (h, w, 1))

    # 2. Camada de Stroke (sobre o glow)
    if has_stroke and stroke is not None:
        stroke_mask = create_stroke_mask(binary_mask, width=stroke.width)
        stroke_alpha = (stroke_mask.astype(np.float64) / 255.0) * stroke.opacity
        stroke_color = np.array(stroke.color, dtype=np.float64) / 255.0
        stroke_rgb = np.tile(stroke_color, (h, w, 1))

        curr_rgb, curr_alpha = _blend_layer_over(
            bottom_rgb=curr_rgb,
            bottom_alpha=curr_alpha,
            top_rgb=stroke_rgb,
            top_alpha=stroke_alpha,
        )

    # 3. Camada do Sujeito (no topo de tudo)
    final_rgb, final_alpha = _blend_layer_over(
        bottom_rgb=curr_rgb,
        bottom_alpha=curr_alpha,
        top_rgb=subject_rgb,
        top_alpha=subject_alpha,
    )

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = np.clip(np.round(final_rgb * 255.0), 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(np.round(final_alpha * 255.0), 0, 255).astype(np.uint8)

    # Garante preservacao exata dos pixels onde o sujeito e 100% opaco
    opaque_subject = array[:, :, 3] == 255
    out[opaque_subject] = array[opaque_subject]

    return out


__all__ = [
    "apply_stroke_and_glow",
    "create_glow_mask",
    "create_stroke_mask",
    "dilate_mask",
    "feather_mask",
]
