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

import numpy as np
import pytest
from PIL import Image

from video_engine.audio.loudness_models import LoudnessResult, LoudnormStats
from video_engine.audio.models import SilenceSegment, SpeechSegment, VadResult
from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.editing.media_probe import MediaInfo
from video_engine.editing.models import SpliceResult
from video_engine.thumbnail.composer import ThumbnailComposer
from video_engine.thumbnail.models import (
    FaceMetrics,
    FrameMetrics,
    KeyframeCandidate,
    KeyframeSelectorResult,
    SegmentationResult,
)
from video_engine.video.models import DynamicZoomResult, ZoomMode, ZoomShot
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
    def __init__(
        self,
        segments: Sequence[SpeechSegment],
        total: int = 10000,
        silence: Optional[Sequence] = None,
    ) -> None:
        self.segments = list(segments)
        self.total = total
        self.silence = list(silence) if silence is not None else []

    def detect_file(self, path) -> VadResult:
        return VadResult(
            total_duration_ms=self.total,
            speech_segments=self.segments,
            silence_segments=self.silence,
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
        self.inputs: list = []

    def normalize_file(self, input_path, output_path) -> LoudnessResult:
        self.inputs.append(str(input_path))
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


class StubTranscriber:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls: list = []

    def transcribe_file(self, media_path):
        self.calls.append(str(media_path))
        if self.error is not None:
            raise self.error
        return self.result


class StubZoom:
    def __init__(self, zoom_shots_count: int = 3, error: Optional[Exception] = None) -> None:
        self.zoom_shots_count = zoom_shots_count
        self.error = error
        self.calls: list = []

    def apply_zoom(
        self, input_video, output_video, pause_intervals=None, face_center=None, transcription=None
    ) -> DynamicZoomResult:
        self.calls.append((str(input_video), str(output_video), pause_intervals, face_center, transcription))
        if self.error is not None:
            raise self.error
        Path(output_video).touch()
        return DynamicZoomResult(
            output_path=str(output_video),
            total_duration_ms=8000,
            shots=[ZoomShot(start_ms=0, end_ms=8000, mode=ZoomMode.NORMAL, scale=1.0)],
            zoom_shots_count=self.zoom_shots_count,
            normal_shots_count=1,
            video_width=1920,
            video_height=1080,
            scaling_filter="lanczos",
        )


class StubKeyframeSelector:
    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.calls: list = []

    def select_best_keyframes(self, video_path, output_dir=None):
        self.calls.append((str(video_path), output_dir))
        if self.error is not None:
            raise self.error
        return self.result


class StubSegmenter:
    def __init__(self, error=None, empty_mask=False) -> None:
        self.error = error
        self.empty_mask = empty_mask
        self.calls: list = []

    def segment(self, frame_rgb):
        self.calls.append(np.asarray(frame_rgb).copy())
        if self.error is not None:
            raise self.error
        height, width = frame_rgb.shape[:2]
        mask = (
            np.zeros((height, width), dtype=np.uint8)
            if self.empty_mask
            else np.full((height, width), 255, dtype=np.uint8)
        )
        rgba = np.dstack([np.asarray(frame_rgb), mask]).astype(np.uint8)
        return SegmentationResult(width=width, height=height, foreground_rgba=rgba, alpha_mask=mask)


class RecordingComposer:
    def __init__(self) -> None:
        self.inner = ThumbnailComposer()
        self.compose_calls: list = []
        self.compose_from_frame_calls: list = []

    def compose(self, subject, headline, background=None, stroke=None, glow=None, output_path=None):
        self.compose_calls.append((subject, headline, output_path))
        return self.inner.compose(
            subject,
            headline,
            background=background,
            stroke=stroke,
            glow=glow,
            output_path=output_path,
        )

    def compose_from_frame(self, frame, headline, output_path=None, darken_factor=0.15):
        self.compose_from_frame_calls.append((frame, headline, output_path))
        return self.inner.compose_from_frame(
            frame, headline, output_path=output_path, darken_factor=darken_factor
        )


class BoomComposer:
    """Compositor que estoura em qualquer chamada (prova de isolamento de falhas)."""

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error or RuntimeError("falha devastadora na composicao")
        self.compose_calls: list = []
        self.compose_from_frame_calls: list = []

    def compose(self, *args, **kwargs):
        self.compose_calls.append((args, kwargs))
        raise self.error

    def compose_from_frame(self, *args, **kwargs):
        self.compose_from_frame_calls.append((args, kwargs))
        raise self.error


def _make_frame_png(path, width: int = 1280, height: int = 720) -> Path:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, : width // 2] = [60, 60, 200]
    rgb[:, width // 2 :] = [210, 100, 50]
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
    return Path(path)


def _make_candidate(image_path, timestamp_ms: int = 1000, score: float = 0.9) -> KeyframeCandidate:
    metrics = FrameMetrics(
        timestamp_ms=timestamp_ms,
        frame_index=0,
        sharpness_variance=150.0,
        is_blurry=False,
        luminance_mean=128.0,
        luminance_std=40.0,
        lighting_score=0.8,
        is_poor_lighting=False,
        face=FaceMetrics(detected=True),
        composite_score=score,
        is_valid=True,
        rejection_reasons=[],
    )
    return KeyframeCandidate(
        rank=1,
        timestamp_ms=timestamp_ms,
        frame_index=0,
        score=score,
        metrics=metrics,
        image_path=str(image_path),
    )


def _make_selection(image_path, score: float = 0.9) -> KeyframeSelectorResult:
    return KeyframeSelectorResult(
        video_path="video.mp4",
        video_duration_ms=10000,
        total_frames_sampled=5,
        valid_frames_count=1,
        discarded_blurry_count=0,
        discarded_closed_eyes_count=0,
        discarded_lighting_count=0,
        discarded_no_face_count=0,
        top_candidates=[_make_candidate(image_path, score=score)],
    )


def make_pipeline(
    tmp_path,
    probe=None,
    vad=None,
    splicer=None,
    normalizer=None,
    zoom=None,
    transcriber=None,
    keyframe_selector=None,
    segmenter=None,
    composer=None,
    **cfg_kwargs,
):
    cfg = WorkerConfig(output_dir=str(tmp_path / "out"), **cfg_kwargs)
    return VideoProcessingPipeline(
        config=cfg,
        probe=probe or StubProbe(),
        vad=vad or StubVAD([SpeechSegment(start_ms=500, end_ms=2500), SpeechSegment(start_ms=3000, end_ms=4000)]),
        splicer=splicer or StubSplicer(),
        normalizer=normalizer or StubNormalizer(),
        zoom_processor=zoom,
        transcriber=transcriber or StubTranscriber(),
        keyframe_selector=keyframe_selector,
        segmenter=segmenter,
        composer=composer,
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
# Dynamic Zoom (dinamica do pipeline)
# --------------------------------------------------------------------------- #
def test_pipeline_zoom_enabled_chains_zoom_after_splicer(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    zoom = StubZoom(zoom_shots_count=5)
    normalizer = StubNormalizer()
    pipeline = make_pipeline(tmp_path, zoom=zoom, normalizer=normalizer)
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = True

    result = pipeline.process(j)

    assert len(zoom.calls) == 1
    zoom_input, zoom_output, _pause, _face, _transcription = zoom.calls[0]
    assert zoom_input.endswith("spliced.mp4")
    assert zoom_output.endswith("zoomed.mp4")
    assert normalizer.inputs == [zoom_output]
    assert result.dynamic_zoom_applied is True
    assert result.zoom_shots_count == 5


def test_pipeline_zoom_forwards_pause_intervals_from_vad(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pauses = [
        SilenceSegment(start_ms=2500, end_ms=3000),
        SilenceSegment(start_ms=6000, end_ms=6800),
    ]
    vad = StubVAD(
        [
            SpeechSegment(start_ms=500, end_ms=2500),
            SpeechSegment(start_ms=3000, end_ms=6000),
            SpeechSegment(start_ms=6800, end_ms=10000),
        ],
        silence=pauses,
    )
    zoom = StubZoom()
    pipeline = make_pipeline(tmp_path, vad=vad, zoom=zoom)
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = True

    result = pipeline.process(j)

    assert len(zoom.calls) == 1
    assert list(zoom.calls[0][2]) == pauses
    assert result.dynamic_zoom_applied is True


def test_pipeline_zoom_disabled_skips_zoom_stage(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    zoom = StubZoom()
    normalizer = StubNormalizer()
    pipeline = make_pipeline(tmp_path, zoom=zoom, normalizer=normalizer)
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = False

    result = pipeline.process(j)

    assert zoom.calls == []
    assert normalizer.inputs[0].endswith("spliced.mp4")
    assert result.dynamic_zoom_applied is False
    assert result.zoom_shots_count == 0


def test_pipeline_zoom_missing_metadata_defaults_disabled(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    zoom = StubZoom()
    pipeline = make_pipeline(tmp_path, zoom=zoom)

    result = pipeline.process(job(file_path=str(source)))

    assert zoom.calls == []
    assert result.dynamic_zoom_applied is False


def test_pipeline_zoom_accepts_snake_case_and_string_values(tmp_path):
    cases = [("true", True), ("1", True), (1, True), ("false", False), ("0", False), (0, False)]
    for index, (raw_value, expected) in enumerate(cases):
        source = tmp_path / f"raw_{index}.mp4"
        source.write_bytes(b"fake")
        zoom = StubZoom()
        pipeline = make_pipeline(tmp_path, zoom=zoom)
        j = job(file_path=str(source))
        j.metadata["dynamic_zoom"] = raw_value

        result = pipeline.process(j)

        if expected:
            assert len(zoom.calls) == 1, f"esperado zoom para {raw_value!r}"
            assert result.dynamic_zoom_applied is True
        else:
            assert zoom.calls == [], f"esperado sem zoom para {raw_value!r}"
            assert result.dynamic_zoom_applied is False


def test_pipeline_zoom_short_video_zero_shots_does_not_fail(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    zoom = StubZoom(zoom_shots_count=0)
    pipeline = make_pipeline(tmp_path, zoom=zoom)
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = True

    result = pipeline.process(j)

    assert len(zoom.calls) == 1
    assert result.dynamic_zoom_applied is True
    assert result.zoom_shots_count == 0
    assert result.output_path.endswith("upload-abc_processed.mp4")


def test_pipeline_zoom_cleans_intermediate_zoomed_file(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    work = tmp_path / "work"
    zoom = StubZoom()
    pipeline = make_pipeline(tmp_path, zoom=zoom, temp_dir=str(work))
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = True

    pipeline.process(j)

    leftovers = list(work.rglob("*")) if work.is_dir() else []
    assert leftovers == []


def test_pipeline_zoom_error_propagates(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    zoom = StubZoom(error=RuntimeError("ffmpeg falhou no zoom"))
    pipeline = make_pipeline(tmp_path, zoom=zoom)
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = True

    with pytest.raises(RuntimeError, match="ffmpeg falhou no zoom"):
        pipeline.process(j)


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


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
def test_pipeline_integration_with_bgm(tmp_path):
    source = tmp_path / "voice.wav"
    bgm = tmp_path / "bgm.wav"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-nostats",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=44100:duration=4",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-nostats",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=2",
            "-c:a",
            "pcm_s16le",
            str(bgm),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    from video_engine.audio.bgm_ducking import BgmDucker
    from video_engine.audio.loudness import LoudnessNormalizer
    from video_engine.editing.media_probe import MediaProbe
    from video_engine.editing.media_splicer import MediaSplicer

    pipeline = VideoProcessingPipeline(
        config=WorkerConfig(output_dir=str(tmp_path / "out")),
        probe=MediaProbe(),
        vad=StubVAD([SpeechSegment(start_ms=0, end_ms=4000)], total=4000),
        splicer=MediaSplicer(),
        normalizer=LoudnessNormalizer(),
        bgm_ducker=BgmDucker(),
    )
    j = VideoProcessingJobData(
        jobId="job-bgm-01",
        uploadId="upload-bgm",
        filePath=str(source),
        metadata={"bgm_path": str(bgm)},
        createdAt="2026-01-01T00:00:00Z",
    )
    result = pipeline.process(j)
    out = tmp_path / "out" / "upload-bgm_processed.mp4"
    assert out.is_file()
    assert result.duration_sec == pytest.approx(4.0, abs=0.5)
    assert result.loudness.integrated_lufs <= -13.0


# --------------------------------------------------------------------------- #
# Thumbnail (Issue #21): geracao de capa integrada na esteira
# --------------------------------------------------------------------------- #
def test_pipeline_generates_thumbnail_and_populates_result(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    frame_png = _make_frame_png(tmp_path / "frame.png")
    composer = RecordingComposer()
    pipeline = make_pipeline(
        tmp_path,
        keyframe_selector=StubKeyframeSelector(result=_make_selection(frame_png, score=0.9)),
        segmenter=StubSegmenter(),
        composer=composer,
    )

    result = pipeline.process(job(file_path=str(source)))

    out_thumbnail = tmp_path / "out" / "upload-abc_thumbnail.jpg"
    assert result.thumbnail_path == str(out_thumbnail)
    assert result.thumbnail_score == pytest.approx(0.9)
    assert out_thumbnail.is_file()
    metadata = result.to_success_metadata()
    assert metadata["thumbnailPath"] == str(out_thumbnail)
    assert metadata["thumbnailScore"] == pytest.approx(0.9)
    assert len(composer.compose_calls) == 1
    assert composer.compose_from_frame_calls == []
    assert (tmp_path / "out" / "upload-abc_processed.mp4").is_file()


def test_pipeline_thumbnail_uses_metadata_headline(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    frame_png = _make_frame_png(tmp_path / "frame.png")
    composer = RecordingComposer()
    pipeline = make_pipeline(
        tmp_path,
        keyframe_selector=StubKeyframeSelector(result=_make_selection(frame_png)),
        segmenter=StubSegmenter(),
        composer=composer,
    )
    j = job(file_path=str(source))
    j.metadata["thumbnail_headline"] = "UMA META HEADLINE COM IMPACTO FORTE"

    result = pipeline.process(j)

    assert result.thumbnail_path is not None
    headline = composer.compose_calls[0][1]
    assert headline.text == "UMA META HEADLINE COM IMPACTO"


def test_pipeline_thumbnail_uses_transcription_headline_fallback(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    frame_png = _make_frame_png(tmp_path / "frame.png")
    composer = RecordingComposer()
    transcription = TranscriptionResult(
        text="vem aprender marketing digital mesmo",
        language="pt",
        duration_ms=4000,
        words=[WordTimestamp(word="vem", start_ms=0, end_ms=300, probability=0.9)],
    )
    transcriber = StubTranscriber(result=transcription)
    pipeline = make_pipeline(
        tmp_path,
        zoom=StubZoom(),
        transcriber=transcriber,
        keyframe_selector=StubKeyframeSelector(result=_make_selection(frame_png)),
        segmenter=StubSegmenter(),
        composer=composer,
    )
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = True

    result = pipeline.process(j)

    assert result.thumbnail_path is not None
    assert len(transcriber.calls) == 1
    headline = composer.compose_calls[0][1]
    assert headline.text == "vem aprender marketing digital mesmo"


def test_pipeline_thumbnail_falls_back_to_raw_frame_when_segmenter_fails(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    frame_png = _make_frame_png(tmp_path / "frame.png")
    composer = RecordingComposer()
    pipeline = make_pipeline(
        tmp_path,
        keyframe_selector=StubKeyframeSelector(result=_make_selection(frame_png, score=0.77)),
        segmenter=StubSegmenter(error=RuntimeError("modelo ONNX indisponivel")),
        composer=composer,
    )

    result = pipeline.process(job(file_path=str(source)))

    out_thumbnail = tmp_path / "out" / "upload-abc_thumbnail.jpg"
    assert result.thumbnail_path == str(out_thumbnail)
    assert result.thumbnail_score == pytest.approx(0.77)
    assert out_thumbnail.is_file()
    assert len(composer.compose_from_frame_calls) == 1
    assert composer.compose_calls == []


def test_pipeline_completes_video_even_if_thumbnail_generation_crashes(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = make_pipeline(
        tmp_path,
        keyframe_selector=StubKeyframeSelector(
            result=_make_selection(_make_frame_png(tmp_path / "frame.png"))
        ),
        segmenter=StubSegmenter(),
        composer=BoomComposer(),
    )

    result = pipeline.process(job(file_path=str(source)))

    assert (tmp_path / "out" / "upload-abc_processed.mp4").is_file()
    assert result.thumbnail_path is None
    assert result.thumbnail_score is None
    assert "thumbnailPath" not in result.to_success_metadata()
    assert "thumbnailScore" not in result.to_success_metadata()
    assert result.output_path.endswith("upload-abc_processed.mp4")

