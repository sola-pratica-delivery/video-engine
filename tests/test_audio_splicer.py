"""Testes da camada de sinal NumPy: ``audio.splicer.splice_audio_array``.

Cobre o contrato da Issue #3 (CA1 micro-crossfade e CA2 sincronizacao):
recorte/concatenacao de intervalos, fades de emenda (equal power e linear),
adaptacao de fades para segmentos curtos, ordenacao/fusao automatica e casos
de borda (lista vazia, segmento unico, arredondamento ms->amostras).
"""

from __future__ import annotations

import numpy as np
import pytest

from video_engine.audio.models import TimeInterval
from video_engine.audio.splicer import merge_intervals, splice_audio_array
from video_engine.editing.models import FadeCurve, SplicerConfig

SR = 16000


def time_ms(*pairs):
    return [TimeInterval(start_ms=s, end_ms=e) for s, e in pairs]


def fade_gains(length: int, curve: FadeCurve, fade_out: bool) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length)
    if curve is FadeCurve.EQUAL_POWER:
        arr = np.cos(np.pi / 2.0 * t) if fade_out else np.sin(np.pi / 2.0 * t)
    else:
        arr = 1.0 - t if fade_out else t
    return arr.astype(np.float32)


def test_empty_segments_raises_value_error():
    audio = np.ones(SR, dtype=np.float32)
    with pytest.raises(ValueError, match="Segments list cannot be empty"):
        splice_audio_array(audio, SR, [])


def test_sample_rate_zero_raises_value_error():
    audio = np.ones(SR, dtype=np.float32)
    with pytest.raises(ValueError, match="sample_rate"):
        splice_audio_array(audio, 0, time_ms((0, 100)))


def test_empty_audio_raises_value_error():
    with pytest.raises(ValueError, match="audio array cannot be empty"):
        splice_audio_array(np.empty((0,), dtype=np.float32), SR, time_ms((0, 100)))


def test_single_segment_is_plain_slice():
    audio = np.random.RandomState(0).rand(SR).astype(np.float32)
    result = splice_audio_array(audio, SR, time_ms((500, 1000)))
    np.testing.assert_array_equal(result, audio[8000:16000])


def test_multiple_segments_total_length_is_sum_of_durations():
    audio = np.ones(2 * SR, dtype=np.float32)
    result = splice_audio_array(audio, SR, time_ms((0, 500), (1000, 1500)))
    assert result.shape == (SR,)
    assert result.shape[0] == (500 + 500) * SR // 1000


def test_interior_audio_within_segments_is_preserved():
    audio = np.arange(2 * SR, dtype=np.float32)
    result = splice_audio_array(audio, SR, time_ms((0, 500), (1000, 1500)))
    np.testing.assert_array_equal(result[100:200], audio[100:200])


def test_equal_power_fade_out_matches_cos():
    audio = np.ones(2 * SR, dtype=np.float32)
    config = SplicerConfig(crossfade_ms=15.0, curve=FadeCurve.EQUAL_POWER)
    result = splice_audio_array(audio, SR, time_ms((0, 500), (1000, 1500)), config)
    fade_len = int(round(15.0 * SR / 1000))
    junction = 500 * SR // 1000
    np.testing.assert_allclose(
        result[junction - fade_len:junction],
        fade_gains(fade_len, FadeCurve.EQUAL_POWER, True),
        atol=1e-6,
    )


def test_equal_power_fade_in_matches_sin():
    audio = np.ones(2 * SR, dtype=np.float32)
    config = SplicerConfig(crossfade_ms=15.0, curve=FadeCurve.EQUAL_POWER)
    result = splice_audio_array(audio, SR, time_ms((0, 500), (1000, 1500)), config)
    fade_len = int(round(15.0 * SR / 1000))
    junction = 500 * SR // 1000
    np.testing.assert_allclose(
        result[junction:junction + fade_len],
        fade_gains(fade_len, FadeCurve.EQUAL_POWER, False),
        atol=1e-6,
    )


def test_linear_fades_match_ramps():
    audio = np.ones(2 * SR, dtype=np.float32)
    config = SplicerConfig(crossfade_ms=15.0, curve=FadeCurve.LINEAR)
    result = splice_audio_array(audio, SR, time_ms((0, 500), (1000, 1500)), config)
    fade_len = int(round(15.0 * SR / 1000))
    junction = 500 * SR // 1000
    np.testing.assert_allclose(
        result[junction - fade_len:junction],
        fade_gains(fade_len, FadeCurve.LINEAR, True),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result[junction:junction + fade_len],
        fade_gains(fade_len, FadeCurve.LINEAR, False),
        atol=1e-6,
    )


def test_junction_is_continuous():
    audio = (0.5 + 0.5 * np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR)).astype(np.float32)
    result = splice_audio_array(audio, SR, time_ms((0, 700), (1000, 1700)))
    junction = 700 * SR // 1000
    assert abs(float(result[junction - 1]) - float(result[junction])) < 1e-3


def test_short_segment_uses_adapted_fade_len():
    audio = np.ones(SR, dtype=np.float32)
    # Segmento de 12ms (< crossfade 15ms) => fade de 6ms (dur/2).
    config = SplicerConfig(crossfade_ms=15.0, curve=FadeCurve.EQUAL_POWER)
    result = splice_audio_array(audio, SR, time_ms((0, 12), (20, 60)), config)
    seg0_len = 12 * SR // 1000
    assert seg0_len == 192
    expected_len = 6 * SR // 1000
    np.testing.assert_allclose(
        result[seg0_len - expected_len:seg0_len],
        fade_gains(expected_len, FadeCurve.EQUAL_POWER, True),
        atol=1e-6,
    )


def test_unsorted_overlapping_segments_are_merged():
    audio = np.ones(3 * SR, dtype=np.float32)
    config = SplicerConfig(crossfade_ms=15.0, curve=FadeCurve.EQUAL_POWER)
    result = splice_audio_array(
        audio, SR, time_ms((1000, 2000), (0, 500), (1200, 3000)), config
    )
    # [1000,2000] u [1200,3000] = [1000,3000]; total = 500ms + 2000ms = 2.5s.
    assert result.shape[0] == (500 + 2000) * SR // 1000


def test_merge_intervals_sorts_and_unions():
    merged = merge_intervals(time_ms((1500, 2200), (0, 500), (800, 1000), (900, 1200)))
    assert [(s.start_ms, s.end_ms) for s in merged] == [(0, 500), (800, 1200), (1500, 2200)]


def test_merge_intervals_accepts_speech_segments():
    from video_engine.audio.models import SpeechSegment

    segments = [SpeechSegment(start_ms=900, end_ms=1200), SpeechSegment(start_ms=0, end_ms=500)]
    merged = merge_intervals(segments)
    assert [(s.start_ms, s.end_ms) for s in merged] == [(0, 500), (900, 1200)]
    assert all(isinstance(s, SpeechSegment) for s in merged)


def test_merge_intervals_keeps_touching_segments_separate():
    # Segmentos que apenas se tocam nao sao fundidos (cada um e uma juncao).
    merged = merge_intervals(time_ms((0, 1000), (1000, 2000)))
    assert [(s.start_ms, s.end_ms) for s in merged] == [(0, 1000), (1000, 2000)]


def test_stereo_is_supported_and_both_channels_are_faded():
    audio = np.ones((2 * SR, 2), dtype=np.float32)
    config = SplicerConfig(crossfade_ms=15.0, curve=FadeCurve.EQUAL_POWER)
    result = splice_audio_array(audio, SR, time_ms((0, 500), (1000, 1500)), config)
    assert result.shape == (SR, 2)
    np.testing.assert_array_equal(result[:, 0], result[:, 1])
    junction = 500 * SR // 1000
    np.testing.assert_allclose(result[junction, :], 0.0, atol=1e-6)


def test_dtype_is_preserved():
    audio = np.ones(2 * SR, dtype=np.float64)
    result = splice_audio_array(audio, SR, time_ms((0, 500), (1000, 1500)))
    assert result.dtype == np.float64


def test_segments_rounding_to_zero_length_raises():
    audio = np.ones(SR, dtype=np.float32)
    with pytest.raises(ValueError, match="No valid segments"):
        splice_audio_array(audio, SR, time_ms((0, 0)))


def test_fade_len_respects_config_crossfade():
    audio = np.ones(SR, dtype=np.float32)
    config = SplicerConfig(crossfade_ms=30.0, curve=FadeCurve.LINEAR)
    result = splice_audio_array(audio, SR, time_ms((0, 300), (400, 700)), config)
    fade_len = int(round(30.0 * SR / 1000))
    np.testing.assert_allclose(
        result[300 * SR // 1000 - fade_len:300 * SR // 1000],
        fade_gains(fade_len, FadeCurve.LINEAR, True),
        atol=1e-6,
    )
