"""Camada de sinal pura NumPy: emendas (splice) com micro-crossfade.

Implementa a "juncao cirurgica" da Issue #3 na camada de sinal: os intervalos
selecionados sao recortados e concatenados em ordem cronologica aplicando um
micro-crossfade na regiao de cada emenda (fade-out no final do segmento ``i`` e
fade-in no inicio do segmento ``i+1``) para eliminar estalos/ruidos.

Nao ha sobreposicao (overlap-add): o tempo total e preservado (a soma das
duracoes dos segmentos), garantindo o alinhamento estrito frame-a-frame com o
stream de video (CA2 da spec).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from video_engine.audio.models import TimeInterval
from video_engine.audio.vad_utils import merge_intervals as merge_intervals
from video_engine.editing.models import FadeCurve, SplicerConfig


def _segments_to_spans(
    segments: Sequence[TimeInterval],
    sample_rate: int,
    n_samples: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for seg in sorted(segments, key=lambda s: (s.start_ms, s.end_ms)):
        start = int(round(seg.start_ms * sample_rate / 1000.0))
        end = int(round(seg.end_ms * sample_rate / 1000.0))
        start = min(max(start, 0), n_samples)
        end = min(max(end, 0), n_samples)
        if end > start:
            spans.append((start, end))
    return spans


def _fade_gain(fade_len: int, curve: FadeCurve, fade_out: bool) -> np.ndarray:
    """Ganhos do janelamento de um fade de ``fade_len`` amostras.

    Equal power: ``sin(pi/2*t)`` no fade-in e ``cos(pi/2*t)`` no fade-out.
    Linear: rampa ``t`` / ``1-t``. Em ambos, as extremidades da emenda tendem a
    zero, garantindo continuidade no ponto de juncao.
    """
    if fade_len == 1:
        return np.array([0.0], dtype=np.float32)
    t = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
    if curve is FadeCurve.EQUAL_POWER:
        gain = np.cos(np.pi / 2.0 * t) if fade_out else np.sin(np.pi / 2.0 * t)
    elif curve is FadeCurve.LINEAR:
        gain = 1.0 - t if fade_out else t
    else:  # pragma: no cover - enum restrito
        raise ValueError(f"Curva de fade desconhecida: {curve!r}")
    return gain.astype(np.float32, copy=False)


def _apply_fade(part: np.ndarray, gain: np.ndarray, head: bool) -> None:
    """Aplica ``gain`` no inicio (fade-in) ou fim (fade-out) de ``part`` in-place."""
    fade_len = gain.shape[0]
    window = part[:fade_len] if head else part[-fade_len:]
    if window.ndim == 1:
        window *= gain
    else:
        window *= gain.reshape(-1, 1)


def splice_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    segments: Sequence[TimeInterval],
    config: Optional[SplicerConfig] = None,
) -> np.ndarray:
    """Recorta ``segments`` de ``audio`` e concatena com fades nas emendas.

    Args:
        audio: Array float mono (``(N,)``) ou estereo (``(N, C)``) em [-1, 1].
        sample_rate: Taxa de amostragem em Hz (deve ser > 0).
        segments: Intervalos (ms) a manter, em ordem cronologica.
        config: Opcoes do splicer (default: 15ms equal power).

    Returns:
        Audio concatenado, com o mesmo ``dtype`` e numero de canais da entrada,
        cujo numero total de amostras e a soma das duracoes (ms->amostras
        arredondadas) dos segmentos fundidos.

    Raises:
        ValueError: se ``segments`` estiver vazio, ``sample_rate <= 0``, o array
            estiver vazio ou nenhum intervalo valido restar apos o
            arredondamento ms->amostras.
    """
    if config is None:
        config = SplicerConfig()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if audio is None or audio.size == 0:
        raise ValueError("audio array cannot be empty")
    if not segments:
        raise ValueError("Segments list cannot be empty")

    audio = np.asarray(audio)
    merged = merge_intervals(segments)
    spans = _segments_to_spans(merged, sample_rate, audio.shape[0])
    if not spans:
        raise ValueError("No valid segments remaining after rounding to samples")

    parts: List[np.ndarray] = [audio[s:e].copy() for s, e in spans]
    n = len(spans)

    fade_len_by_segment: List[int] = []
    for start, end in spans:
        seg_len = end - start
        dur_ms = 1000.0 * seg_len / sample_rate
        fade_ms = min(config.crossfade_ms, dur_ms / 2.0)
        fade_len = min(int(round(fade_ms * sample_rate / 1000.0)), seg_len // 2)
        fade_len_by_segment.append(fade_len)

    for i in range(1, n):
        fade_len = fade_len_by_segment[i]
        if fade_len <= 0:
            continue
        _apply_fade(parts[i], _fade_gain(fade_len, config.curve, fade_out=False), head=True)

    for i in range(n - 1):
        fade_len = fade_len_by_segment[i]
        if fade_len <= 0:
            continue
        _apply_fade(parts[i], _fade_gain(fade_len, config.curve, fade_out=True), head=False)

    return np.concatenate(parts, axis=0)


__all__ = ["merge_intervals", "splice_audio_array"]
