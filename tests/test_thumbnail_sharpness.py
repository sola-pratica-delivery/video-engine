"""Testes de avaliacao de nitidez e variancia do Laplaciano (Issue #11)."""

from __future__ import annotations

import numpy as np
import pytest

from video_engine.thumbnail.sharpness import (
    evaluate_sharpness,
    is_blurry,
    laplacian_variance,
)


def _make_checkerboard(size: int = 100, block: int = 10) -> np.ndarray:
    """Gera padrao xadrez de alto contraste e bordas abruptas (alta nitidez)."""
    grid = (np.indices((size, size)) // block).sum(axis=0) % 2
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[grid == 1] = 255
    return rgb


def _make_uniform(size: int = 100, value: int = 128) -> np.ndarray:
    """Gera frame liso uniforme sem bordas (variancia zero)."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def _box_blur(image: np.ndarray, radius: int = 5) -> np.ndarray:
    """Aplica blur uniforme simples reduzindo as frequencias espaciais."""
    blurred = image.astype(np.float64)
    h, w, _ = image.shape
    out = np.zeros_like(blurred)
    for y in range(h):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        for x in range(w):
            x0, x1 = max(0, x - radius), min(w, x + radius + 1)
            out[y, x] = blurred[y0:y1, x0:x1].mean(axis=(0, 1))
    return np.clip(out, 0, 255).astype(np.uint8)


class TestLaplacianVariance:
    def test_sharp_vs_uniform(self) -> None:
        sharp = _make_checkerboard(size=120, block=10)
        uniform = _make_uniform(size=120, value=128)

        var_sharp = laplacian_variance(sharp)
        var_uniform = laplacian_variance(uniform)

        assert var_sharp > 500.0
        assert var_uniform == pytest.approx(0.0, abs=1e-6)

    def test_blurring_reduces_variance_significantly(self) -> None:
        sharp = _make_checkerboard(size=80, block=8)
        blurred = _box_blur(sharp, radius=3)

        var_sharp = laplacian_variance(sharp)
        var_blurred = laplacian_variance(blurred)

        assert var_sharp > var_blurred * 5.0
        assert is_blurry(var_blurred, min_sharpness_threshold=500.0)
        assert not is_blurry(var_sharp, min_sharpness_threshold=500.0)

    def test_small_images_return_zero(self) -> None:
        tiny = np.ones((2, 2, 3), dtype=np.uint8)
        assert laplacian_variance(tiny) == 0.0

    def test_invalid_dimensions_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="esperado HxWx3"):
            laplacian_variance(np.ones((50, 50), dtype=np.uint8))  # 2D sem canais

        with pytest.raises(ValueError, match="esperado HxWx3"):
            laplacian_variance(np.ones((50, 50, 4), dtype=np.uint8))  # 4 canais RGBA

        with pytest.raises(ValueError, match="esperado HxWx3"):
            laplacian_variance(np.empty((0, 0, 3), dtype=np.uint8))  # vazio

    def test_evaluate_sharpness_tuple(self) -> None:
        sharp = _make_checkerboard(size=80, block=8)
        variance, blurry = evaluate_sharpness(sharp, min_sharpness_threshold=50.0)

        assert variance > 50.0
        assert blurry is False

        _, blurry_high_threshold = evaluate_sharpness(sharp, min_sharpness_threshold=variance + 1000.0)
        assert blurry_high_threshold is True
