"""Utilitarios puros de pos-processamento dos intervalos de fala/silencio.

Funcoes deterministas e sem dependencia de rede neural, usadas para
aplicar a margem de seguranca (``apply_padding_and_merge``) e calcular as
pausas elegiveis para corte (``extract_silence_intervals``).
"""

from __future__ import annotations

from typing import List, Sequence

from .models import SilenceSegment, SpeechSegment, TimeInterval


def apply_padding_and_merge(
    raw_segments: Sequence[TimeInterval],
    padding_ms: int,
    total_duration_ms: int,
) -> List[SpeechSegment]:
    """Aplica margem de seguranca (padding) e funde segmentos sobrepostos.

    Expande cada segmento em ``padding_ms`` para antes e depois, limitando
    os limites ao intervalo ``[0, total_duration_ms]``. Segmentos expandidos
    que se sobreponham ou se toquem (``proximo.start_ms <= atual.end_ms``)
    sao fundidos em um unico segmento continuo.

    Args:
        raw_segments: Segmentos de fala crus, sem padding e sem fusao,
            tipicamente em ordem cronologica.
        padding_ms: Margem de seguranca em milissegundos aplicada em cada lado.
        total_duration_ms: Duracao total do audio, usada para limitar o
            padding posterior.

    Returns:
        Lista de ``SpeechSegment`` com padding aplicado e fusoes resolvidas.
    """
    padded: List[SpeechSegment] = [
        SpeechSegment(
            start_ms=max(0, segment.start_ms - padding_ms),
            end_ms=min(total_duration_ms, segment.end_ms + padding_ms),
        )
        for segment in raw_segments
    ]

    merged: List[SpeechSegment] = []
    for segment in padded:
        if merged and segment.start_ms <= merged[-1].end_ms:
            previous = merged[-1]
            merged[-1] = _merge(previous, segment)
        else:
            merged.append(segment)
    return merged


def _merge(left: SpeechSegment, right: SpeechSegment) -> SpeechSegment:
    """Funde dois segmentos adjacentes/sobrepostos em um unico."""
    return SpeechSegment(start_ms=left.start_ms, end_ms=max(left.end_ms, right.end_ms))


def extract_silence_intervals(
    speech_segments: Sequence[SpeechSegment],
    total_duration_ms: int,
    min_silence_duration_ms: int,
) -> List[SilenceSegment]:
    """Calcula os intervalos de silencio entre as falas e nas extremidades.

    Os intervalos com duracao menor que ``min_silence_duration_ms`` sao
    ignorados. O caso em que não ha fala produz um unico intervalo cobrindo
    todo o audio (desde que atenda ao threshold minimo).

    Args:
        speech_segments: Segmentos de fala ja com padding e fusoes aplicadas.
        total_duration_ms: Duracao total do audio.
        min_silence_duration_ms: Threshold minimo de silencio (em ms) para
            registrar o intervalo como pausa.

    Returns:
        Lista de ``SilenceSegment`` ordenada cronologicamente.
    """
    silences: List[SilenceSegment] = []
    cursor = 0
    for segment in speech_segments:
        gap = segment.start_ms - cursor
        if gap >= min_silence_duration_ms:
            silences.append(SilenceSegment(start_ms=cursor, end_ms=segment.start_ms))
        cursor = max(cursor, segment.end_ms)

    trailing = total_duration_ms - cursor
    if trailing >= min_silence_duration_ms:
        silences.append(SilenceSegment(start_ms=cursor, end_ms=total_duration_ms))
    return silences
