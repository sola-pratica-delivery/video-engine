from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from video_engine.audio.models import SilenceSegment, SpeechSegment, VadConfig
from video_engine.audio.silero_vad import SileroVadDetector

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "tests" / "data"
CONTROLLED_WAV = TEST_DATA / "controlled.wav"
MP3_SAMPLE = TEST_DATA / "sample.mp3"
VENDORED_MODEL = ROOT / "models" / "silero_vad.onnx"


@pytest.fixture(scope="session")
def detector() -> SileroVadDetector:
    return SileroVadDetector(onnx_model_path=VENDORED_MODEL)


def _assert_invariants(result) -> None:
    assert result.total_duration_ms >= 0
    for segment in result.speech_segments:
        assert 0 <= segment.start_ms <= segment.end_ms <= result.total_duration_ms
    for segment in result.silence_segments:
        assert 0 <= segment.start_ms <= segment.end_ms <= result.total_duration_ms


class TestSileroVadDetector:
    def test_detect_controlled_clip(self, detector: SileroVadDetector) -> None:
        data, _ = sf.read(CONTROLLED_WAV, dtype="float32")
        result = detector.detect(data, sample_rate=16000)

        assert result.total_duration_ms == 7000
        assert [(s.start_ms, s.end_ms) for s in result.speech_segments] == [
            (0, 2576),
            (3440, 7000),
        ]
        assert [(s.start_ms, s.end_ms) for s in result.silence_segments] == [
            (2576, 3440),
        ]
        for segment in result.speech_segments:
            assert segment.confidence is not None
            assert 0.0 <= segment.confidence <= 1.0
        _assert_invariants(result)

    def test_detect_file_wav_matches_detect(self, detector: SileroVadDetector) -> None:
        file_result = detector.detect_file(CONTROLLED_WAV)
        data, _ = sf.read(CONTROLLED_WAV, dtype="float32")
        array_result = detector.detect(data, sample_rate=16000)
        assert file_result == array_result

    def test_detect_file_missing_raises(self, detector: SileroVadDetector) -> None:
        with pytest.raises(FileNotFoundError):
            detector.detect_file(ROOT / "does_not_exist.wav")

    def test_detect_file_unreadable_raises(
        self, detector: SileroVadDetector, tmp_path: Path
    ) -> None:
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"this is not audio data at all")
        with pytest.raises(RuntimeError):
            detector.detect_file(bogus)

    def test_detect_all_silence(self, detector: SileroVadDetector) -> None:
        result = detector.detect(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        assert result.speech_segments == []
        assert [(s.start_ms, s.end_ms) for s in result.silence_segments] == [
            (0, 1000)
        ]

    def test_detect_empty_array_raises(self, detector: SileroVadDetector) -> None:
        with pytest.raises(ValueError, match="empty or invalid"):
            detector.detect(np.empty(0, dtype=np.float32))

    def test_detect_nan_array_raises(self, detector: SileroVadDetector) -> None:
        audio = np.zeros(3200, dtype=np.float32)
        audio[10] = np.nan
        with pytest.raises(ValueError, match="empty or invalid"):
            detector.detect(audio)

    def test_detect_inf_array_raises(self, detector: SileroVadDetector) -> None:
        audio = np.zeros(3200, dtype=np.float32)
        audio[10] = np.inf
        with pytest.raises(ValueError, match="empty or invalid"):
            detector.detect(audio)

    def test_detect_invalid_sample_rate_raises(
        self, detector: SileroVadDetector
    ) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            detector.detect(np.zeros(1600, dtype=np.float32), sample_rate=0)

    def test_detect_stereo_array_downmix(self, detector: SileroVadDetector) -> None:
        data, _ = sf.read(CONTROLLED_WAV, dtype="float32")
        stereo = np.column_stack([data, np.zeros_like(data)])
        result = detector.detect(stereo, sample_rate=16000)
        assert result.total_duration_ms == 7000
        assert len(result.speech_segments) == 2
        _assert_invariants(result)

    def test_detect_resamples_8k(self, detector: SileroVadDetector) -> None:
        data, _ = sf.read(CONTROLLED_WAV, dtype="float32")
        downsampled = data[::2]
        result = detector.detect(downsampled, sample_rate=8000)
        assert result.total_duration_ms == 7000
        assert len(result.speech_segments) == 2
        _assert_invariants(result)

    def test_detect_file_mp3(self, detector: SileroVadDetector) -> None:
        result = detector.detect_file(MP3_SAMPLE)
        assert result.total_duration_ms == 2534
        _assert_invariants(result)

    def test_detect_file_mp4_via_ffmpeg(
        self, detector: SileroVadDetector, tmp_path: Path
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("ffmpeg nao disponivel no PATH")
        mp4 = tmp_path / "sample.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(CONTROLLED_WAV),
                "-c:v",
                "mpeg4",
                "-b:v",
                "64k",
                "-c:a",
                "aac",
                str(mp4),
            ],
            check=True,
            capture_output=True,
        )
        result = detector.detect_file(mp4)
        assert abs(result.total_duration_ms - 7000) <= 50
        assert result.speech_segments
        assert result.silence_segments
        _assert_invariants(result)

    def test_model_path_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            SileroVadDetector(onnx_model_path=ROOT / "missing.onnx")

    def test_explicit_model_path_works(self) -> None:
        other = SileroVadDetector(onnx_model_path=VENDORED_MODEL)
        result = other.detect(np.zeros(16000, dtype=np.float32))
        assert result.speech_segments == []


class TestDeterministicPipeline:
    """Pipeline prob->segmentos->padding->pausas com probabilities controladas."""

    TOTAL_MS = 3000

    @staticmethod
    def probs() -> np.ndarray:
        speech = [0.9] * 20
        silence = [0.1] * 30
        speech_b = [0.9] * 20
        tail = [0.1] * 24
        return np.asarray(speech + silence + speech_b + tail, dtype=np.float32)

    def make_detector(self, config: VadConfig) -> SileroVadDetector:
        detector = SileroVadDetector(onnx_model_path=VENDORED_MODEL, config=config)
        detector._infer_probs = lambda audio: self.probs()  # type: ignore[method-assign]
        return detector

    def test_default_config(self) -> None:
        result = self.make_detector(VadConfig()).detect(
            np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000
        )
        assert [(s.start_ms, s.end_ms) for s in result.speech_segments] == [
            (0, 720),
            (1520, 2320),
        ]
        assert [(s.start_ms, s.end_ms) for s in result.silence_segments] == [
            (720, 1520),
            (2320, 3000),
        ]

    def test_confidence_populated(self) -> None:
        result = self.make_detector(VadConfig()).detect(
            np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000
        )
        for segment in result.speech_segments:
            assert segment.confidence is not None
            assert 0.0 <= segment.confidence <= 1.0

    def test_min_silence_threshold_high_excludes_gap(self) -> None:
        result = self.make_detector(
            VadConfig(min_silence_duration_ms=900)
        ).detect(np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000)
        # a pausa de 800ms no meio fica abaixo do novo limiar de 900ms
        assert [(s.start_ms, s.end_ms) for s in result.speech_segments] == [
            (0, 720),
            (1520, 3000),
        ]
        assert result.silence_segments == []

    def test_threshold_high_suppresses_speech(self) -> None:
        result = self.make_detector(
            VadConfig(threshold=0.95)
        ).detect(np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000)
        assert result.speech_segments == []
        assert [(s.start_ms, s.end_ms) for s in result.silence_segments] == [
            (0, 3000)
        ]

    def test_threshold_low_still_detects(self) -> None:
        result = self.make_detector(
            VadConfig(threshold=0.85)
        ).detect(np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000)
        assert len(result.speech_segments) == 2

    def test_min_speech_filters_brief_segments(self) -> None:
        result = self.make_detector(
            VadConfig(min_speech_duration_ms=2000)
        ).detect(np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000)
        assert result.speech_segments == []
        assert [(s.start_ms, s.end_ms) for s in result.silence_segments] == [
            (0, 3000)
        ]

    def test_all_silence_probs(self) -> None:
        detector = self.make_detector(VadConfig())
        detector._infer_probs = lambda audio: np.full(94, 0.1, dtype=np.float32)  # type: ignore[method-assign]
        result = detector.detect(
            np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000
        )
        assert result.speech_segments == []
        assert [(s.start_ms, s.end_ms) for s in result.silence_segments] == [
            (0, 3000)
        ]

    def test_segments_are_typed_models(self) -> None:
        result = self.make_detector(VadConfig()).detect(
            np.zeros(self.TOTAL_MS * 16, dtype=np.float32), sample_rate=16000
        )
        assert all(isinstance(s, SpeechSegment) for s in result.speech_segments)
        assert all(isinstance(s, SilenceSegment) for s in result.silence_segments)
