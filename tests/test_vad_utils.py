from __future__ import annotations

from video_engine.audio.models import (
    SilenceSegment,
    SpeechSegment,
    TimeInterval,
    VadConfig,
)
from video_engine.audio.vad_utils import (
    apply_padding_and_merge,
    extract_silence_intervals,
)


class TestApplyPaddingAndMerge:
    def test_basic_padding(self) -> None:
        raw = [TimeInterval(start_ms=1000, end_ms=2000)]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=5000)
        assert len(result) == 1
        assert result[0].start_ms == 920
        assert result[0].end_ms == 2080

    def test_returns_speech_segments(self) -> None:
        raw = [TimeInterval(start_ms=1000, end_ms=2000)]
        result = apply_padding_and_merge(raw, padding_ms=0, total_duration_ms=5000)
        assert isinstance(result[0], SpeechSegment)
        assert result[0].confidence is None

    def test_padding_clamped_at_audio_start(self) -> None:
        raw = [TimeInterval(start_ms=0, end_ms=500)]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=5000)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 580

    def test_padding_clamped_at_audio_end(self) -> None:
        raw = [TimeInterval(start_ms=1000, end_ms=4400)]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=4400)
        assert result[0].start_ms == 920
        assert result[0].end_ms == 4400

    def test_merge_on_padding_overlap(self) -> None:
        raw = [
            TimeInterval(start_ms=1000, end_ms=2000),
            TimeInterval(start_ms=2080, end_ms=3000),
        ]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=5000)
        assert len(result) == 1
        assert result[0].start_ms == 920
        assert result[0].end_ms == 3080

    def test_merge_when_gap_equals_two_paddings(self) -> None:
        raw = [
            TimeInterval(start_ms=1000, end_ms=2000),
            TimeInterval(start_ms=2160, end_ms=3000),
        ]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=5000)
        assert len(result) == 1
        assert result[0].end_ms == 3080

    def test_no_merge_when_gap_above_two_paddings(self) -> None:
        raw = [
            TimeInterval(start_ms=1000, end_ms=2000),
            TimeInterval(start_ms=2200, end_ms=3000),
        ]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=5000)
        assert len(result) == 2
        assert result[0].end_ms == 2080
        assert result[1].start_ms == 2120

    def test_chained_merge(self) -> None:
        raw = [
            TimeInterval(start_ms=100, end_ms=500),
            TimeInterval(start_ms=540, end_ms=900),
            TimeInterval(start_ms=940, end_ms=1300),
        ]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=5000)
        assert len(result) == 1
        assert result[0].start_ms == 20
        assert result[0].end_ms == 1380

    def test_merge_on_touch_with_zero_padding(self) -> None:
        raw = [
            TimeInterval(start_ms=0, end_ms=100),
            TimeInterval(start_ms=100, end_ms=200),
        ]
        result = apply_padding_and_merge(raw, padding_ms=0, total_duration_ms=5000)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 200

    def test_empty_input(self) -> None:
        assert apply_padding_and_merge([], padding_ms=80, total_duration_ms=5000) == []

    def test_invariants_start_leq_end_leq_total(self) -> None:
        raw = [
            TimeInterval(start_ms=100, end_ms=300),
            TimeInterval(start_ms=0, end_ms=20),
            TimeInterval(start_ms=4900, end_ms=4999),
        ]
        result = apply_padding_and_merge(raw, padding_ms=80, total_duration_ms=5000)
        for seg in result:
            assert 0 <= seg.start_ms <= seg.end_ms <= 5000


class TestExtractSilenceIntervals:
    def test_gap_above_threshold(self) -> None:
        speech = [
            SpeechSegment(start_ms=0, end_ms=1000),
            SpeechSegment(start_ms=1800, end_ms=3000),
        ]
        result = extract_silence_intervals(speech, total_duration_ms=5000, min_silence_duration_ms=500)
        assert [(s.start_ms, s.end_ms) for s in result] == [(1000, 1800), (3000, 5000)]

    def test_gap_below_threshold_excluded(self) -> None:
        speech = [
            SpeechSegment(start_ms=0, end_ms=1000),
            SpeechSegment(start_ms=1350, end_ms=3000),
        ]
        result = extract_silence_intervals(speech, total_duration_ms=4000, min_silence_duration_ms=500)
        assert [(s.start_ms, s.end_ms) for s in result] == [(3000, 4000)]

    def test_leading_and_trailing_silence(self) -> None:
        speech = [SpeechSegment(start_ms=800, end_ms=1200)]
        result = extract_silence_intervals(speech, total_duration_ms=3000, min_silence_duration_ms=500)
        assert [(s.start_ms, s.end_ms) for s in result] == [(0, 800), (1200, 3000)]

    def test_leading_silence_below_threshold_ignored(self) -> None:
        speech = [SpeechSegment(start_ms=400, end_ms=1200)]
        result = extract_silence_intervals(speech, total_duration_ms=2000, min_silence_duration_ms=500)
        assert [(s.start_ms, s.end_ms) for s in result] == [(1200, 2000)]

    def test_no_leading_silence_when_speech_at_start(self) -> None:
        speech = [SpeechSegment(start_ms=0, end_ms=1200)]
        result = extract_silence_intervals(speech, total_duration_ms=3000, min_silence_duration_ms=500)
        assert [(s.start_ms, s.end_ms) for s in result] == [(1200, 3000)]

    def test_exact_threshold_included(self) -> None:
        speech = [
            SpeechSegment(start_ms=0, end_ms=1000),
            SpeechSegment(start_ms=1500, end_ms=2500),
        ]
        result = extract_silence_intervals(speech, total_duration_ms=3000, min_silence_duration_ms=500)
        assert [(s.start_ms, s.end_ms) for s in result] == [(1000, 1500), (2500, 3000)]

    def test_total_silence_single_segment(self) -> None:
        result = extract_silence_intervals([], total_duration_ms=2000, min_silence_duration_ms=500)
        assert [(s.start_ms, s.end_ms) for s in result] == [(0, 2000)]

    def test_total_silence_below_threshold_empty(self) -> None:
        result = extract_silence_intervals([], total_duration_ms=300, min_silence_duration_ms=500)
        assert result == []

    def test_continuous_speech_no_silence(self) -> None:
        speech = [SpeechSegment(start_ms=0, end_ms=1000)]
        result = extract_silence_intervals(speech, total_duration_ms=1000, min_silence_duration_ms=500)
        assert result == []

    def test_from_utils_returns_silence_segments(self) -> None:
        speech = [SpeechSegment(start_ms=0, end_ms=1000)]
        result = extract_silence_intervals(speech, total_duration_ms=2000, min_silence_duration_ms=500)
        assert isinstance(result[0], SilenceSegment)

    def test_zero_duration_speech_invalid_gap(self) -> None:
        speech = [
            SpeechSegment(start_ms=0, end_ms=1000),
            SpeechSegment(start_ms=1000, end_ms=2000),
        ]
        result = extract_silence_intervals(speech, total_duration_ms=2000, min_silence_duration_ms=500)
        assert result == []


def test_vad_config_defaults() -> None:
    config = VadConfig()
    assert config.threshold == 0.5
    assert config.min_speech_duration_ms == 100
    assert config.min_silence_duration_ms == 500
    assert config.padding_ms == 80
    assert config.sample_rate == 16000
