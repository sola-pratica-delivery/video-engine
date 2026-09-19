"""Avaliacao de iluminacao e exposicao de frames (Issue #11).

Calcula a luminancia media (ITU-R BT.601) e o desvio padrao (contraste) de
um frame RGB e produz um ``lighting_score`` normalizado (0.0 a 1.0) que
penaliza frames subexpostos (escuros demais) e superexpostos (estourados),
ambos configuravelmente descartaveis pelo ``KeyframeSelector``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Coeficientes de luminancia ITU-R BT.601
_Y_COEFFS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def luminance_channel(frame_rgb: np.ndarray) -> np.ndarray:
    """Converte um frame RGB (HxWx3, uint8/float) em luma Y (2D float64).

    Raises:
        ValueError: se ``frame_rgb`` nao for um array 3D com 3 canais.
    """
    array = np.asarray(frame_rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise ValueError("Array de frame invalido: esperado HxWx3 com 3 canais RGB")
    return (array.astype(np.float64) * _Y_COEFFS).sum(axis=2)


def luminance_stats(frame_rgb: np.ndarray) -> Tuple[float, float]:
    """Retorna ``(mean, std)`` da luminancia BT.601 do frame (0-255)."""
    luma = luminance_channel(frame_rgb)
    mean = float(np.mean(luma))
    std = float(np.std(luma))
    return mean, std


def lighting_score(
    frame_rgb: np.ndarray,
    min_luminance: float = 35.0,
    max_luminance: float = 225.0,
) -> float:
    """Pontua a qualidade de iluminacao do frame em escala 0.0-1.0.

    A pontuacao favorece o centro da janela aceitavel ``[min, max]`` e
    recompensa contraste moderado. Frames fora da janela (sub/superexpostos)
    recebem score ``0.0``.
    """
    mean, std = luminance_stats(frame_rgb)
    return _score_from_stats(mean, std, min_luminance, max_luminance)


def _score_from_stats(
    mean: float,
    std: float,
    min_luminance: float,
    max_luminance: float,
) -> float:
    if mean < min_luminance or mean > max_luminance:
        return 0.0
    interval = max(max_luminance - min_luminance, 1e-9)
    illumination = (mean - min_luminance) / interval  # 0 no minimo, 1 no maximo
    center_score = 1.0 - abs(illumination - 0.5) * 2.0  # 1 no centro, 0 nas bordas
    contrast_score = min(std / 80.0, 1.0) * 0.3
    return float(min(max(0.0, 0.7 * center_score + contrast_score), 1.0))


def is_poor_lighting(
    frame_rgb: np.ndarray,
    min_luminance: float = 35.0,
    max_luminance: float = 225.0,
) -> bool:
    """True se o frame estiver subexposto ou superexposto (fora da janela)."""
    mean, _ = luminance_stats(frame_rgb)
    return mean < min_luminance or mean > max_luminance


__all__ = [
    "is_poor_lighting",
    "lighting_score",
    "luminance_channel",
    "luminance_stats",
]
