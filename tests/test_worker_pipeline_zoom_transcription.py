"""Testes da integracao do WhisperTranscriber no pipeline do worker (Issue #20).

Cobertura:
- CA1: ``VideoProcessingPipeline`` aceita injeção de ``transcriber``.
- CA2/CA4: ``dynamicZoom`` habilitado -> ``transcribe_file(spliced_path)`` e
  ``TranscriptionResult`` repassado a ``zoom_processor.apply_zoom(..., transcription=...)``.
- CA3: ``dynamicZoom`` desabilitado -> transcritor nunca e acionado.
- CA5: falha no Whisper nao aborta a esteira: fallback heuristico com sucesso.
- CA6/CA7: ``zoom_strategy`` ("gemini" vs "heuristic") em ``ProcessingResult``
  e no payload ``to_success_metadata()`` (``zoomStrategy``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from video_engine.audio.loudness_models import LoudnessResult, LoudnormStats
from video_engine.audio.models import SilenceSegment, SpeechSegment, VadResult
from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.editing.media_probe import MediaInfo
from video_engine.editing.models import SpliceResult
from video_engine.video.models import DynamicZoomResult, ZoomMode, ZoomShot
from video_engine.worker.models import VideoProcessingJobData, WorkerConfig
from video_engine.worker.pipeline import VideoProcessingPipeline

MEASURED = LoudnormStats(
    input_i=-20.0,
    input_tp=-8.0,
    input_lra=5.0,
    input_thresh=-30.0,
    target_offset=0.5,
)


def _job(upload_id: str = "upload-zoom", file_path: Optional[str] = None) -> VideoProcessingJobData:
    return VideoProcessingJobData(
        jobId="job-zoom-001",
        uploadId=upload_id,
        filePath=file_path or "/tmp/raw.mp4",
        createdAt="2026-01-01T00:00:00Z",
    )


def _make_transcription(total_ms: int = 4500) -> TranscriptionResult:
    words = [
        WordTimestamp(word="oi", start_ms=0, end_ms=500, probability=0.9),
        WordTimestamp(word="bem", start_ms=800, end_ms=1300, probability=0.9),
    ]
    return TranscriptionResult(
        text="oi bem",
        language="pt",
        duration_ms=total_ms,
        words=words,
    )


# --------------------------------------------------------------------------- #
# Stubs modulares
# --------------------------------------------------------------------------- #
class StubProbe:
    def probe(self, path) -> MediaInfo:
        return MediaInfo(
            path=str(path),
            has_video=True,
            has_audio=True,
            duration_ms=10000,
        )


class StubVAD:
    def __init__(
        self,
        segments: Sequence[SpeechSegment],
        total: int = 10000,
        silence: Optional[Sequence[SilenceSegment]] = None,
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
    def splice_file(self, input_path, output_path, segments) -> SpliceResult:
        Path(output_path).write_bytes(b"SPLICED")
        return SpliceResult(
            output_path=str(output_path),
            total_duration_ms=4500,
            num_segments=len(segments),
            num_junctions=max(0, len(segments) - 1),
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
    def __init__(
        self,
        result: Optional[TranscriptionResult] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: List[str] = []

    def transcribe_file(self, media_path):
        self.calls.append(str(media_path))
        if self.error is not None:
            raise self.error
        return self.result


class BoomTranscriber:
    """Transcritor que falha se for acionado (prova que nao e usado)."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def transcribe_file(self, media_path):
        self.calls.append(str(media_path))
        raise AssertionError("transcriber.nao.deveria.ser.acionado")


class StubZoom:
    def __init__(self, strategy: str = "heuristic", zoom_shots_count: int = 2) -> None:
        self.strategy = strategy
        self.zoom_shots_count = zoom_shots_count
        self.calls: List[dict] = []

    def apply_zoom(
        self, input_video, output_video, pause_intervals=None, face_center=None, transcription=None
    ) -> DynamicZoomResult:
        self.calls.append(
            {
                "input": str(input_video),
                "output": str(output_video),
                "pause_intervals": pause_intervals,
                "face_center": face_center,
                "transcription": transcription,
            }
        )
        Path(output_video).write_bytes(b"ZOOMED")
        return DynamicZoomResult(
            output_path=str(output_video),
            total_duration_ms=4500,
            shots=[ZoomShot(start_ms=0, end_ms=4500, mode=ZoomMode.NORMAL, scale=1.0)],
            zoom_shots_count=self.zoom_shots_count,
            normal_shots_count=1,
            video_width=1920,
            video_height=1080,
            scaling_filter="lanczos",
            strategy_used=self.strategy,
        )


def _make_pipeline(tmp_path, transcriber=None, zoom=None, vad=None):
    cfg = WorkerConfig(
        output_dir=str(tmp_path / "out"),
        temp_dir=str(tmp_path / "work"),
    )
    return VideoProcessingPipeline(
        config=cfg,
        probe=StubProbe(),
        vad=vad or StubVAD(
            [
                SpeechSegment(start_ms=500, end_ms=2500),
                SpeechSegment(start_ms=3500, end_ms=6000),
            ]
        ),
        splicer=StubSplicer(),
        normalizer=StubNormalizer(),
        zoom_processor=zoom,
        transcriber=transcriber,
    )


# --------------------------------------------------------------------------- #
# CA1: injecao de dependencia do transcritor
# --------------------------------------------------------------------------- #
def test_pipeline_accepts_transcriber_injection():
    transcriber = StubTranscriber()
    pipeline = VideoProcessingPipeline(transcriber=transcriber)
    assert pipeline.transcriber is transcriber


# --------------------------------------------------------------------------- #
# CA2/CA4: dynamicZoom habilitado -> whisper + repasse ao zoom
# --------------------------------------------------------------------------- #
def test_pipeline_invokes_whisper_and_forwards_to_zoom_processor(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    transcription = _make_transcription()
    transcriber = StubTranscriber(result=transcription)
    zoom = StubZoom()
    pipeline = _make_pipeline(tmp_path, transcriber=transcriber, zoom=zoom)
    job = _job(file_path=str(source))
    job.metadata["dynamicZoom"] = True

    result = pipeline.process(job)

    assert len(transcriber.calls) == 1
    assert transcriber.calls[0].endswith("spliced.mp4")
    assert len(zoom.calls) == 1
    assert zoom.calls[0]["input"].endswith("spliced.mp4")
    assert zoom.calls[0]["output"].endswith("zoomed.mp4")
    assert zoom.calls[0]["transcription"] is transcription
    assert result.dynamic_zoom_applied is True


def test_pipeline_forwards_vad_pause_intervals_and_transcription(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    paused = [
        SilenceSegment(start_ms=2500, end_ms=3500),
        SilenceSegment(start_ms=6000, end_ms=6800),
    ]
    vad = StubVAD(
        [
            SpeechSegment(start_ms=500, end_ms=2500),
            SpeechSegment(start_ms=3500, end_ms=6000),
            SpeechSegment(start_ms=6800, end_ms=10000),
        ],
        silence=paused,
    )
    transcription = _make_transcription()
    transcriber = StubTranscriber(result=transcription)
    zoom = StubZoom()
    pipeline = _make_pipeline(tmp_path, transcriber=transcriber, zoom=zoom, vad=vad)
    job = _job(file_path=str(source))
    job.metadata["dynamicZoom"] = True

    pipeline.process(job)

    assert len(zoom.calls) == 1
    assert list(zoom.calls[0]["pause_intervals"]) == paused
    assert zoom.calls[0]["transcription"] is transcription


# --------------------------------------------------------------------------- #
# CA3: dynamicZoom desabilitado -> transcritor nao e acionado
# --------------------------------------------------------------------------- #
def test_pipeline_skips_whisper_when_dynamic_zoom_disabled(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    transcriber = BoomTranscriber()
    zoom = StubZoom()
    pipeline = _make_pipeline(tmp_path, transcriber=transcriber, zoom=zoom)
    job = _job(file_path=str(source))
    job.metadata["dynamicZoom"] = False

    result = pipeline.process(job)

    assert transcriber.calls == []
    assert zoom.calls == []
    assert result.dynamic_zoom_applied is False
    assert result.zoom_strategy is None
    assert "zoomStrategy" not in result.to_success_metadata()


# --------------------------------------------------------------------------- #
# CA5: fallback gracioso quando o Whisper falha
# --------------------------------------------------------------------------- #
def test_pipeline_graceful_fallback_when_whisper_fails(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    transcriber = StubTranscriber(error=RuntimeError("faster-whisper sem VRAM"))
    zoom = StubZoom()
    pipeline = _make_pipeline(tmp_path, transcriber=transcriber, zoom=zoom)
    job = _job(file_path=str(source))
    job.metadata["dynamicZoom"] = True

    result = pipeline.process(job)

    assert len(transcriber.calls) == 1
    assert len(zoom.calls) == 1
    assert zoom.calls[0]["transcription"] is None
    assert result.dynamic_zoom_applied is True
    assert result.zoom_strategy == "heuristic"
    assert result.to_success_metadata()["zoomStrategy"] == "heuristic"
    assert result.output_path.endswith("upload-zoom_processed.mp4")
    assert Path(result.output_path).is_file()


# --------------------------------------------------------------------------- #
# CA6/CA7: estrategia no relatorio e no metadata de sucesso
# --------------------------------------------------------------------------- #
def test_pipeline_reports_zoom_strategy_in_metadata(tmp_path):
    gemini_source = tmp_path / "raw_gemini.mp4"
    gemini_source.write_bytes(b"fake")
    gemini_transcriber = StubTranscriber(result=_make_transcription())
    gemini_zoom = StubZoom(strategy="gemini")
    pipeline = _make_pipeline(tmp_path, transcriber=gemini_transcriber, zoom=gemini_zoom)
    gemini_job = _job(upload_id="upload-gemini", file_path=str(gemini_source))
    gemini_job.metadata["dynamicZoom"] = True

    gemini_result = pipeline.process(gemini_job)

    assert gemini_result.zoom_strategy == "gemini"
    assert gemini_result.to_success_metadata()["zoomStrategy"] == "gemini"

    heuristic_source = tmp_path / "raw_heur.mp4"
    heuristic_source.write_bytes(b"fake")
    heuristic_transcriber = StubTranscriber(result=_make_transcription())
    heuristic_zoom = StubZoom(strategy="heuristic")
    pipeline2 = _make_pipeline(tmp_path, transcriber=heuristic_transcriber, zoom=heuristic_zoom)
    heuristic_job = _job(upload_id="upload-heur", file_path=str(heuristic_source))
    heuristic_job.metadata["dynamicZoom"] = True

    heuristic_result = pipeline2.process(heuristic_job)

    assert heuristic_result.zoom_strategy == "heuristic"
    assert heuristic_result.to_success_metadata()["zoomStrategy"] == "heuristic"


def test_pipeline_metadata_omits_zoom_strategy_when_zoom_not_applied(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    pipeline = _make_pipeline(tmp_path, transcriber=BoomTranscriber())

    result = pipeline.process(_job(file_path=str(source)))

    assert result.zoom_strategy is None
    assert "zoomStrategy" not in result.to_success_metadata()
