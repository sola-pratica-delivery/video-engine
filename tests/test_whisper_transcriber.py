"""Testes do motor de transcricao fonetica (Issue #7).

Cobre modelos de dados, validacao de parametros, conversao para
milissegundos, agrupamento de segmentos, tratamento de erros e a
integracao real com o modelo ``tiny`` do Faster-Whisper (pode ser
desativada via ``VIDEO_ENGINE_SKIP_WHISPER_INTEGRATION``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from video_engine.captions.models import (
    CaptionSegment,
    TranscriberConfig,
    TranscriptionResult,
    WordTimestamp,
)
from video_engine.captions.transcriber import WhisperTranscriber

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "tests" / "data"
CONTROLLED_WAV = TEST_DATA / "controlled.wav"
MP3_SAMPLE = TEST_DATA / "sample.mp3"


# ---------------------------------------------------------------------------
# Stub do modelo Faster-Whisper (sem download de pesos nas suites rapidas)
# ---------------------------------------------------------------------------


def _word(start: float, end: float, text: str, probability: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(start=start, end=end, word=text, probability=probability)


def _segment(
    start: float,
    end: float,
    text: str,
    words: Optional[Iterable[types.SimpleNamespace]] = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(start=start, end=end, text=text, words=list(words or []))


def _info(
    language: str = "pt",
    language_probability: float = 0.95,
    duration: float = 7.0,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        language=language,
        language_probability=language_probability,
        duration=duration,
    )


class StubWhisperModel:
    """Miniatura da interface de ``faster_whisper.WhisperModel.transcribe``."""

    def __init__(
        self,
        segments: Optional[Iterable[types.SimpleNamespace]] = None,
        info: Optional[types.SimpleNamespace] = None,
    ) -> None:
        self._segments = list(segments or [])
        self._info = info or _info()
        self.calls: List[Dict[str, Any]] = []

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
        word_timestamps: Optional[bool] = None,
        vad_filter: Optional[bool] = None,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[Iterable[types.SimpleNamespace], types.SimpleNamespace]:
        self.calls.append(
            {
                "language": language,
                "beam_size": beam_size,
                "word_timestamps": word_timestamps,
                "vad_filter": vad_filter,
                "initial_prompt": initial_prompt,
            }
        )
        return iter(self._segments), self._info


def _seg_default() -> types.SimpleNamespace:
    return _segment(
        0.1,
        0.9,
        "  Olá   mundo!  ",
        [_word(0.1, 0.35, " Olá", 0.98), _word(0.4, 0.9, " mundo!", 0.9)],
    )


def _make_transcriber(
    segments: Optional[Iterable[types.SimpleNamespace]] = None,
    info: Optional[types.SimpleNamespace] = None,
    config: Optional[TranscriberConfig] = None,
) -> Tuple[WhisperTranscriber, StubWhisperModel]:
    stub = StubWhisperModel(segments, info)
    return WhisperTranscriber(config=config, model_instance=stub), stub


# ---------------------------------------------------------------------------
# Testes de modelo (Pydantic) e validacao de parametros
# ---------------------------------------------------------------------------


class TestModels:
    def test_word_timestamp_defaults(self) -> None:
        word = WordTimestamp(word="olá", start_ms=100, end_ms=200, probability=0.9)
        assert word.word == "olá"
        assert word.start_ms == 100
        assert word.end_ms == 200
        assert word.probability == 0.9

    def test_word_timestamp_negative_start_raises(self) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(word="olá", start_ms=-1, end_ms=200, probability=0.5)

    def test_word_timestamp_negative_end_raises(self) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(word="olá", start_ms=0, end_ms=-200, probability=0.5)

    def test_word_timestamp_probability_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(word="olá", start_ms=0, end_ms=100, probability=1.5)
        with pytest.raises(ValidationError):
            WordTimestamp(word="olá", start_ms=0, end_ms=100, probability=-0.1)

    def test_word_timestamp_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(word="olá", start_ms=0, end_ms=100, probability=0.5, extra="x")

    def test_caption_segment_default_words(self) -> None:
        seg = CaptionSegment(id=0, text="olá", start_ms=0, end_ms=500)
        assert seg.words == []

    def test_transcription_result_defaults(self) -> None:
        result = TranscriptionResult(text="", duration_ms=0)
        assert result.language == "pt"
        assert result.language_probability == 1.0
        assert result.segments == []
        assert result.words == []

    def test_transcription_result_nested_models(self) -> None:
        word = WordTimestamp(word="olá", start_ms=10, end_ms=50, probability=0.9)
        seg = CaptionSegment(id=0, text="olá", start_ms=10, end_ms=50, words=[word])
        result = TranscriptionResult(text="olá", duration_ms=50, segments=[seg], words=[word])
        assert result.segments[0].words[0].word == "olá"

    def test_transcriber_config_defaults(self) -> None:
        config = TranscriberConfig()
        assert config.model_size == "base"
        assert config.device == "auto"
        assert config.compute_type == "int8"
        assert config.language == "pt"
        assert config.word_timestamps is True
        assert config.vad_filter is True

    def test_transcriber_config_beam_size_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            TranscriberConfig(beam_size=0)
        with pytest.raises(ValidationError):
            TranscriberConfig(beam_size=11)

    def test_transcriber_config_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            TranscriberConfig(model_size="tiny", luz="alta")


# ---------------------------------------------------------------------------
# Testes unitarios com Stub Model
# ---------------------------------------------------------------------------


class TestWhisperTranscriberUnit:
    def test_seconds_to_integer_ms(self) -> None:
        transcriber, stub = _make_transcriber([_seg_default()])
        result = transcriber.transcribe_array(
            np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        assert result.duration_ms == 1000
        assert result.segments[0].start_ms == 100
        assert result.segments[0].end_ms == 900
        assert result.words[0].start_ms == 100
        assert result.words[0].end_ms == 350

    def test_words_chronological_flat_list(self) -> None:
        seg1 = _segment(0.1, 0.9, "ola mundo", [_word(0.1, 0.4, "ola", 0.9), _word(0.5, 0.9, "mundo", 0.8)])
        seg2 = _segment(1.2, 1.6, "teste final?", [_word(1.2, 1.4, "teste", 0.7), _word(1.4, 1.6, "final?", 0.6)])
        transcriber, _ = _make_transcriber([seg1, seg2])
        result = transcriber.transcribe_array(
            np.zeros(20000, dtype=np.float32), sample_rate=16000
        )
        assert [w.word for w in result.words] == ["ola", "mundo", "teste", "final?"]
        assert [w.start_ms for w in result.words] == [100, 500, 1200, 1400]
        assert result.segments[0].id == 0
        assert result.segments[1].id == 1

    def test_text_normalized(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        result = transcriber.transcribe_array(
            np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        assert result.segments[0].text == "Olá mundo!"
        assert result.text == "Olá mundo!"

    def test_consolidated_text_joins_segments(self) -> None:
        seg1 = _segment(0.0, 0.5, "primeira frase", [_word(0.0, 0.5, "primeira frase", 0.9)])
        seg2 = _segment(1.0, 1.5, "segunda frase", [_word(1.0, 1.5, "segunda frase", 0.8)])
        transcriber, _ = _make_transcriber([seg1, seg2])
        result = transcriber.transcribe_array(
            np.zeros(20000, dtype=np.float32), sample_rate=16000
        )
        assert result.text == "primeira frase\nsegunda frase"

    def test_all_silent_audio_returns_valid_empty_result(self) -> None:
        transcriber, _ = _make_transcriber([])
        result = transcriber.transcribe_array(
            np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        assert result.text == ""
        assert result.words == []
        assert result.segments == []
        assert result.duration_ms == 1000

    def test_empty_array_raises(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        with pytest.raises(ValueError, match="empty or invalid"):
            transcriber.transcribe_array(np.empty(0, dtype=np.float32))

    def test_nan_array_raises(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        audio = np.zeros(3200, dtype=np.float32)
        audio[10] = np.nan
        with pytest.raises(ValueError, match="empty or invalid"):
            transcriber.transcribe_array(audio)

    def test_inf_array_raises(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        audio = np.zeros(3200, dtype=np.float32)
        audio[10] = np.inf
        with pytest.raises(ValueError, match="empty or invalid"):
            transcriber.transcribe_array(audio)

    def test_invalid_sample_rate_raises(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        with pytest.raises(ValueError, match="sample_rate"):
            transcriber.transcribe_array(np.zeros(1600, dtype=np.float32), sample_rate=0)

    def test_stereo_array_downmix(self) -> None:
        transcriber, stub = _make_transcriber([_seg_default()])
        stereo = np.column_stack(
            [np.zeros(16000, dtype=np.float32), np.ones(16000, dtype=np.float32)]
        )
        result = transcriber.transcribe_array(stereo, sample_rate=16000)
        assert result.duration_ms == 1000
        assert len(stub.calls) == 1

    def test_array_resamples_8k(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        result = transcriber.transcribe_array(
            np.zeros(8000, dtype=np.float32), sample_rate=8000
        )
        assert result.duration_ms == 1000

    def test_model_receives_config_options(self) -> None:
        config = TranscriberConfig(
            model_size="tiny",
            language="pt",
            beam_size=3,
            word_timestamps=True,
            vad_filter=False,
            initial_prompt="Olá, bem-vindo.",
        )
        transcriber, stub = _make_transcriber([_seg_default()], config=config)
        transcriber.transcribe_array(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        call = stub.calls[0]
        assert call["language"] == "pt"
        assert call["beam_size"] == 3
        assert call["word_timestamps"] is True
        assert call["vad_filter"] is False
        assert call["initial_prompt"] == "Olá, bem-vindo."

    def test_language_probability_from_info(self) -> None:
        transcriber, _ = _make_transcriber(
            [_seg_default()], info=_info(language="pt", language_probability=0.77)
        )
        result = transcriber.transcribe_array(
            np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        assert result.language == "pt"
        assert result.language_probability == 0.77

    def test_word_probability_clamped(self) -> None:
        seg = _segment(
            0.0,
            0.5,
            "olá",
            [_word(0.0, 0.5, "olá", 1.7), _word(0.0, 0.2, "x", -0.5)],
        )
        transcriber, _ = _make_transcriber([seg])
        result = transcriber.transcribe_array(
            np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        assert result.words[0].probability == 1.0
        assert result.words[1].probability == 0.0

    def test_word_without_end_defaults_to_start(self) -> None:
        word = types.SimpleNamespace(start=0.3, end=None, word="olá", probability=0.5)
        transcriber, _ = _make_transcriber([_segment(0.1, 0.9, "olá", [word])])
        result = transcriber.transcribe_array(
            np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        assert result.words[0].start_ms == 300
        assert result.words[0].end_ms == 300

    def test_word_timestamps_disabled_yields_empty_words(self) -> None:
        config = TranscriberConfig(model_size="tiny", word_timestamps=False)
        transcriber, _ = _make_transcriber([_seg_default()], config=config)
        result = transcriber.transcribe_array(
            np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        assert result.segments
        assert result.words == []
        assert result.segments[0].words == []


# ---------------------------------------------------------------------------
# Testes de arquivos / casos de borda
# ---------------------------------------------------------------------------


class TestWhisperTranscriberFiles:
    def test_transcribe_file_wav_matches_array(self) -> None:
        segments = [
            _segment(0.1, 0.9, "olá mundo", [_word(0.1, 0.3, "olá", 0.9), _word(0.4, 0.9, "mundo", 0.8)])
        ]
        transcriber, _ = _make_transcriber(segments)
        file_result = transcriber.transcribe_file(CONTROLLED_WAV)
        assert file_result.duration_ms == 7000
        assert file_result.text == "olá mundo"

    def test_transcribe_file_mp3(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        result = transcriber.transcribe_file(MP3_SAMPLE)
        assert result.duration_ms == 2534

    def test_transcribe_file_missing_raises(self) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_file(ROOT / "does_not_exist.wav")

    def test_transcribe_file_unreadable_raises(self, tmp_path: Path) -> None:
        transcriber, _ = _make_transcriber([_seg_default()])
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"this is not audio data at all")
        with pytest.raises(RuntimeError):
            transcriber.transcribe_file(bogus)

    def test_transcribe_file_mp4_with_audio(self, tmp_path: Path) -> None:
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
        transcriber, _ = _make_transcriber([_seg_default()])
        result = transcriber.transcribe_file(mp4)
        assert abs(result.duration_ms - 7000) <= 50
        assert result.text == "Olá mundo!"

    def test_transcribe_file_video_without_audio_raises(self, tmp_path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            pytest.skip("ffmpeg/ffprobe nao disponiveis no PATH")
        muted = tmp_path / "muted.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x120:d=1",
                "-c:v",
                "mpeg4",
                str(muted),
            ],
            check=True,
            capture_output=True,
        )
        transcriber, _ = _make_transcriber([_seg_default()])
        with pytest.raises(ValueError, match="sem stream de audio"):
            transcriber.transcribe_file(muted)


# ---------------------------------------------------------------------------
# Integracao real com o modelo tiny (opcional via env var)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def real_transcriber() -> WhisperTranscriber:
    if os.environ.get("VIDEO_ENGINE_SKIP_WHISPER_INTEGRATION"):
        pytest.skip("Integracao Faster-Whisper desativada por env var")
    return WhisperTranscriber(TranscriberConfig(model_size="tiny", device="cpu"))


class TestRealWhisperIntegration:
    def test_real_transcription_portuguese_with_word_timestamps(
        self, real_transcriber: WhisperTranscriber
    ) -> None:
        result = real_transcriber.transcribe_file(CONTROLLED_WAV)

        assert result.duration_ms == 7000
        assert result.language == "pt"
        assert 0.0 <= result.language_probability <= 1.0

        assert result.words, "audio de fala deveria produzir palavras alinhadas"
        starts = [w.start_ms for w in result.words]
        assert starts == sorted(starts)
        for word in result.words:
            assert isinstance(word.word, str) and word.word.strip()
            assert 0 <= word.start_ms <= word.end_ms
            assert 0.0 <= word.probability <= 1.0

        for segment in result.segments:
            assert segment.start_ms <= segment.end_ms <= result.duration_ms
            assert segment.text.strip()
        assert result.text.strip()
