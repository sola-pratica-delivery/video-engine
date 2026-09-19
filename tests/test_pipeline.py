"""Testes da esteira audiovisual end-to-end (``worker.pipeline``).

Cobertura: orquestracao encadeada probe -> VAD -> splice -> loudness, casos de
borda (arquivo inexistente/corrompido, sem audio, sem fala), limpeza de
temporarios, criacao de ``output_dir`` e integracao leve com FFmpeg real
(quando disponivel).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import pytest

from video_engine.audio.loudness_models import LoudnessResult, LoudnormStats
from video_engine.audio.models import SpeechSegment, VadResult
from video_engine.editing.media_probe import MediaInfo
from video_engine.editing.models import SpliceResult
from video_engine.worker.models import VideoProcessingJobData, WorkerConfig
from video_engine.worker.pipeline import (
    InputFileInvalidError,
    NoAudioStreamError,
    NoSpeechDetectedError,
    VideoProcessingPipeline,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

MEASURED = LoudnormStats(
    input_i=-20.0,
    input_tp=-8.0,
    input_lra=5.0,
    input_thresh=-30.0,
    target_offset=0.5,
)


def job(upload_id: str = "upload-abc", file_path: Optional[str] = None) -> VideoProcessingJobData:
    return VideoProcessingJobData(
        jobId="job-001",
        uploadId=upload_id,
        filePath=file_path or "/tmp/raw.mp4",
        createdAt="2026-01-01T00:00:00Z",
    )


# --------------------------------------------------------------------------- #
# Stubs modulares
# --------------------------------------------------------------------------- #
class StubProbe:
    def __init__(self, has_audio: bool = True, error: Optional[Exception] = None) -> None:
        self.has_audio = has_audio
        self.error = error

    def probe(self, path) -> MediaInfo:
        if self.error is not None:
            raise self.error
        return MediaInfo(
            path=str(path),
            has_video=True,
            has_audio=self.has_audio,
            duration_ms=10000,
        )


class StubVAD:
    def __init__(self, segments: Sequence[SpeechSegment], total: int = 10000) -> None:
        self.segments = list(segments)
        self.total = total

    def detect_file(self, path) -> VadResult:
        return VadResult(
            total_duration_ms=self.total,
            speech_segments=self.segments,
            silence_segments=[],
        )


class StubSplicer:
    def __init__(self, duration_ms: int = 8000, error: Optional[Exception] = None) -> None:
        self.duration_ms = duration_ms
        self.error = error

    def splice_file(self, input_path, output_path, segments) -> SpliceResult:
        if self.error is not None:
            raise self.error
        Path(output_path).touch()
        return SpliceResult(
            output_path=str(output_path),
            total_duration_ms=self.duration_ms,
            num_segments=len(segments),
            num_junctions=max(0, len(segments) - 1),
            audio_sample_rate=44100,
            has_video=True,
        )


class StubNormalizer:
    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error

    def normalize_file(self, input_path, output_path) -> LoudnessResult:
        if self.error is not None:
            raise self.error
        Path(output_path).touch()
        return LoudnessResult(
            output_path=str(output_path),
            measured_input=MEASURED,
            measured_output=LoudnormStats(
                input_i=-14.1, input_tp=-1.02, input_lra=10.8, input_thresh=-25.0, target_offset=0.0
            ),
            target_i=-14.0,
            target_tp=-1.0,
            is_compliant=True,
            has_video=True,
        )


def make_pipeline(tmp_path, probe=None, vad=None, splicer=None, normalizer=None, **cfg_kwargs):
    cfg = WorkerConfig(output_dir=str(tmp_path / "out"), **cfg_kwargs)
    return VideoProcessingPipeline(
        config=cfg,
        probe=probe or StubProbe(),
        vad=vad or StubVAD([SpeechSegment(start_ms=500, end_ms=2500), SpeechSegment(start_ms=3000, end_ms=4000)]),
        splicer=splicer or StubSplicer(),
        normalizer=normalizer or StubNormalizer(),
    )


# --------------------------------------------------------------------------- #
# Sucesso
# --------------------------------------------------------------------------- #
def test_pipeline_success_writes_standardized_artifact(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(tmp_path)
    result = pipeline.process(job(file_path=str(source)))

    out = tmp_path / "out" / "upload-abc_processed.mp4"
    assert result.output_path == str(out)
    assert out.is_file()
    assert result.duration_sec == 8.0
    assert result.speech_segments_count == 2
    assert result.silence_removed_ms == 7000
    assert result.loudness.integrated_lufs == pytest.approx(-14.1)
    assert result.loudness.true_peak_dbtp == pytest.approx(-1.02)
    assert result.loudness.lra == pytest.approx(10.8)


def test_pipeline_success_single_segment_keeps_video_intact(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(
        tmp_path,
        vad=StubVAD([SpeechSegment(start_ms=0, end_ms=10000)]),
        splicer=StubSplicer(duration_ms=10000),
    )
    result = pipeline.process(job(file_path=str(source)))
    assert result.speech_segments_count == 1
    assert result.silence_removed_ms == 0
    assert result.duration_sec == 10.0


def test_pipeline_creates_output_dir_if_missing(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(tmp_path)
    pipeline.process(job(file_path=str(source)))
    assert (tmp_path / "out").is_dir()


# --------------------------------------------------------------------------- #
# Casos de borda
# --------------------------------------------------------------------------- #
def test_pipeline_missing_input_raises_input_file_invalid(tmp_path):
    pipeline = make_pipeline(tmp_path)
    with pytest.raises(InputFileInvalidError) as excinfo:
        pipeline.process(job(file_path=str(tmp_path / "nao_existe.mp4")))
    assert excinfo.value.error_code == "INPUT_FILE_INVALID"


def test_pipeline_probe_file_not_found_maps_to_input_invalid(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(tmp_path, probe=StubProbe(error=FileNotFoundError(source)))
    with pytest.raises(InputFileInvalidError) as excinfo:
        pipeline.process(job(file_path=str(source)))
    assert excinfo.value.error_code == "INPUT_FILE_INVALID"


def test_pipeline_corrupt_file_maps_to_input_invalid(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(tmp_path, probe=StubProbe(error=RuntimeError("ffprobe falhou")))
    with pytest.raises(InputFileInvalidError) as excinfo:
        pipeline.process(job(file_path=str(source)))
    assert excinfo.value.error_code == "INPUT_FILE_INVALID"


def test_pipeline_without_audio_raises_no_audio_stream(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(tmp_path, probe=StubProbe(has_audio=False))
    with pytest.raises(NoAudioStreamError) as excinfo:
        pipeline.process(job(file_path=str(source)))
    assert excinfo.value.error_code == "NO_AUDIO_STREAM"


def test_pipeline_no_speech_raises_no_speech_detected(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(tmp_path, vad=StubVAD([]))
    with pytest.raises(NoSpeechDetectedError) as excinfo:
        pipeline.process(job(file_path=str(source)))
    assert excinfo.value.error_code == "NO_SPEECH_DETECTED"


def test_pipeline_splicer_error_propagates_as_pipeline_error(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(tmp_path, splicer=StubSplicer(error=RuntimeError("ffmpeg falhou")))
    with pytest.raises(RuntimeError, match="ffmpeg falhou"):
        pipeline.process(job(file_path=str(source)))


# --------------------------------------------------------------------------- #
# Limpeza de temporarios
# --------------------------------------------------------------------------- #
def test_pipeline_cleans_intermediate_temp_files(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    work = tmp_path / "work"
    pipeline = make_pipeline(tmp_path, temp_dir=str(work))
    pipeline.process(job(file_path=str(source)))
    leftovers = list(work.rglob("*")) if work.is_dir() else []
    assert leftovers == []
    assert (tmp_path / "out" / "upload-abc_processed.mp4").is_file()


# --------------------------------------------------------------------------- #
# Integracao leve (FFmpeg real, quando disponivel)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
def test_pipeline_integration_with_real_ffmpeg(tmp_path):
    source = tmp_path / "input.wav"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-nostats",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=3",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    from video_engine.audio.loudness import LoudnessNormalizer
    from video_engine.editing.media_probe import MediaProbe
    from video_engine.editing.media_splicer import MediaSplicer

    pipeline = VideoProcessingPipeline(
        config=WorkerConfig(output_dir=str(tmp_path / "out")),
        probe=MediaProbe(),
        vad=StubVAD([SpeechSegment(start_ms=0, end_ms=3000)], total=3000),
        splicer=MediaSplicer(),
        normalizer=LoudnessNormalizer(),
    )
    result = pipeline.process(job(upload_id="upload-xyz", file_path=str(source)))
    out = tmp_path / "out" / "upload-xyz_processed.mp4"
    assert out.is_file()
    assert result.duration_sec == pytest.approx(3.0, abs=0.5)
    assert result.speech_segments_count == 1
