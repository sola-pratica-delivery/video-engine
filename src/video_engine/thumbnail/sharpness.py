"""Avaliacao de nitidez via variancia do Laplaciano (Issue #11).

Implementa a convolucao 2D com o kernel Laplaciano discretizado
(``nabla^2 I``) sobre a luma BT.601 e quantifica o foco do frame pela
variancia :math:`\\sigma^2(\\nabla^2 I)`. Frames com motion blur ou desfoque
(abaixo de ``min_sharpness_threshold``) sao descartados pelo seletor.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from video_engine.thumbnail.lighting import luminance_channel

# Kernel Laplaciano 3x3 discretizado
_LAPLACIAN_KERNEL = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ],
    dtype=np.float64,
)


def laplacian_variance(frame_rgb: np.ndarray) -> float:
    """Variancia da resposta Laplaciana do frame (medida de foco).

    Converte o frame para luma BT.601, aplica a convolucao 2D com o kernel
    Laplaciano descretizado e retorna ``sigma^2(nabla^2 I)``. Imagens menores
    que 3x3 (sem bordas mensuraveis) retornam ``0.0``.

    Raises:
        ValueError: se ``frame_rgb`` nao for um array HxWx3 RGB valido.
    """
    luma = luminance_channel(frame_rgb)
    return _variance_from_luma(luma)


def _variance_from_luma(luma: np.ndarray) -> float:
    array = np.asarray(luma, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Array de luma invalido: esperado 2D nao vazio")
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0
    padded = np.pad(array, 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    response = np.einsum("ijkl,kl->ij", windows, _LAPLACIAN_KERNEL)
    return float(np.var(response))


def is_blurry(variance: float, min_sharpness_threshold: float = 80.0) -> bool:
    """True se a variancia do Laplaciano ficar abaixo do limiar de foco."""
    return variance < min_sharpness_threshold


def evaluate_sharpness(
    frame_rgb: np.ndarray,
    min_sharpness_threshold: float = 80.0,
) -> Tuple[float, bool]:
    """Atalho que retorna ``(variancia, is_blurry)`` para um frame RGB."""
    variance = laplacian_variance(frame_rgb)
    return variance, is_blurry(variance, min_sharpness_threshold)


__all__ = [
    "evaluate_sharpness",
    "is_blurry",
    "laplacian_variance",
]
