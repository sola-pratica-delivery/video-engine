"""Testes de clean feed do video longo principal (Issue #9).

Validam o criterio de aceitacao CA5: a esteira do video longo
(``VideoProcessingPipeline``) permanece limpa, sem queima de legendas
hardsub, preservando o arquivo ``{upload_id}_processed.mp4`` intacto como
artefato cinematografico/instrucional.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

from video_engine.audio.loudness_models import LoudnessResult, LoudnormStats
from video_engine.audio.models import SpeechSegment
from video_engine.editing.media_probe import MediaInfo
from video_engine.editing.models import SpliceResult
from video_engine.worker.models import VideoProcessingJobData, WorkerConfig
from video_engine.worker.pipeline import VideoProcessingPipeline

MEASURED = LoudnormStats(
    input_i=-20.0,
    input_tp=-8.0,
    input_lra=5.0,
    input_thresh=-30.0,
    target_offset=0.5,
)


def _job(upload_id: str = "upload-clean", file_path: Optional[str] = None) -> VideoProcessingJobData:
    return VideoProcessingJobData(
        jobId="job-clean-001",
        uploadId=upload_id,
        filePath=file_path or "/tmp/raw.mp4",
        createdAt="2026-01-01T00:00:00Z",
    )


class StubProbe:
    def probe(self, path) -> MediaInfo:
        return MediaInfo(
            path=str(path),
            has_video=True,
            has_audio=True,
            duration_ms=10000,
        )


class StubVAD:
    def detect_file(self, path):
        from video_engine.audio.models import VadResult

        return VadResult(
            total_duration_ms=10000,
            speech_segments=[
                SpeechSegment(start_ms=1000, end_ms=3000),
                SpeechSegment(start_ms=3500, end_ms=6000),
            ],
            silence_segments=[],
        )


class StubSplicer:
    def splice_file(self, input_path, output_path, segments) -> SpliceResult:
        Path(output_path).write_bytes(b"CLEAN_FEED_SPLICED")
        return SpliceResult(
            output_path=str(output_path),
            total_duration_ms=4500,
            num_segments=2,
            num_junctions=1,
            audio_sample_rate=44100,
            has_video=True,
        )


class StubNormalizer:
    def normalize_file(self, input_path, output_path) -> LoudnessResult:
        source_bytes = Path(input_path).read_bytes()
        Path(output_path).write_bytes(source_bytes)
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
    def __init__(self) -> None:
        self.calls: list = []

    def transcribe_file(self, media_path):
        from video_engine.captions.models import TranscriptionResult

        self.calls.append(str(media_path))
        return TranscriptionResult(text="", language="pt", duration_ms=4500, words=[])


def _make_pipeline(tmp_path) -> VideoProcessingPipeline:
    cfg = WorkerConfig(
        output_dir=str(tmp_path / "out"),
        temp_dir=str(tmp_path / "work"),
    )
    return VideoProcessingPipeline(
        config=cfg,
        probe=StubProbe(),
        vad=StubVAD(),
        splicer=StubSplicer(),
        normalizer=StubNormalizer(),
        transcriber=StubTranscriber(),
    )


# --------------------------------------------------------------------------- #
# Garantia estrutural (o pipeline do video longo nao referencia queima)
# --------------------------------------------------------------------------- #
def test_pipeline_module_has_no_caption_burner_import():
    import video_engine.worker.pipeline as pipeline_module

    source = inspect.getsource(pipeline_module)
    assert "CaptionBurner" not in source
    assert "burn_cut_subtitles" not in source
    assert "video_engine.captions.burner" not in source


def test_pipeline_class_has_no_hardsub_filter():
    source = inspect.getsource(VideoProcessingPipeline)
    assert "-vf" not in source
    assert "-filter_complex" not in source or "ass" not in source


# --------------------------------------------------------------------------- #
# Clean feed funcional: artefato final intacto, sem legendas
# --------------------------------------------------------------------------- #
def test_pipeline_clean_feed_output_is_untouched(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = _make_pipeline(tmp_path)

    result = pipeline.process(_job(file_path=str(source)))

    out_file = tmp_path / "out" / "upload-clean_processed.mp4"
    assert result.output_path == str(out_file)
    assert out_file.read_bytes() == b"CLEAN_FEED_SPLICED"


def test_pipeline_clean_feed_does_not_create_ass_artifacts(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = _make_pipeline(tmp_path)

    pipeline.process(_job(file_path=str(source)))

    ass_files = list((tmp_path / "out").glob("*.ass"))
    assert ass_files == []
    ass_files_deep = list((tmp_path / "out").rglob("*.ass"))
    assert ass_files_deep == []


def test_pipeline_clean_feed_no_caption_data_in_metadata(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = _make_pipeline(tmp_path)

    result = pipeline.process(_job(file_path=str(source)))

    metadata = result.to_success_metadata()
    assert "caption" not in json_keys_lower(metadata)
    assert "subtitle" not in json_keys_lower(metadata)


def json_keys_lower(mapping) -> str:
    return " ".join(v for v in mapping.keys()).lower()


# --------------------------------------------------------------------------- #
# Garantia de desacoplamento: pipeline nao encadeia queima em momento algum
# --------------------------------------------------------------------------- #
def test_pipeline_clean_feed_zoom_enabled_still_clean(tmp_path):
    from video_engine.video.models import DynamicZoomResult, ZoomMode, ZoomShot

    calls: list = []

    class StubZoom:
        def apply_zoom(
            self, input_video, output_video, pause_intervals=None, face_center=None, transcription=None
        ) -> DynamicZoomResult:
            calls.append((str(input_video), str(output_video)))
            Path(output_video).write_bytes(b"CLEAN_FEED_ZOOMED")
            return DynamicZoomResult(
                output_path=str(output_video),
                total_duration_ms=4500,
                shots=[ZoomShot(start_ms=0, end_ms=4500, mode=ZoomMode.NORMAL, scale=1.0)],
                zoom_shots_count=0,
                normal_shots_count=1,
                video_width=1920,
                video_height=1080,
                scaling_filter="lanczos",
            )

    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = _make_pipeline(tmp_path)
    job = _job(file_path=str(source))
    job.metadata["dynamicZoom"] = True
    pipeline.zoom_processor = StubZoom()

    result = pipeline.process(job)

    out_file = tmp_path / "out" / "upload-clean_processed.mp4"
    assert out_file.read_bytes() == b"CLEAN_FEED_ZOOMED"
    assert result.dynamic_zoom_applied is True
    assert not list((tmp_path / "out").glob("*.ass"))
