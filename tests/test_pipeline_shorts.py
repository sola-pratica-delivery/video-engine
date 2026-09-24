"""Testes da integracao de Shorts no pipeline do worker (Issue #24).

Cobertura:
- CA1: flag desabilitada por padrao -> detector/renderer nunca acionados e
  ``result.shorts == []`` (sem chave ``"shorts"`` no metadata).
- CA2: flag habilitada -> deteccao + renderizacao com nomenclatura
  ``{upload_id}_short_{index}.mp4`` e ``{upload_id}_short_{index}_metadata.json``.
- CA3: reuso da transcricao do Dynamic Zoom (1 chamada ao transcritor).
- CA4: transcricao disparada quando apenas ``generate_shorts`` esta ativo.
- CA5: falha no detector nao quebra o video principal (``shorts=[]``).
- CA6: falha no renderer e capturada e o video principal conclui.
- CA7: detector retornando 0 cortes resulta em ``shorts=[]``.
- CA8: parsing heterogeneo da flag e fallback para o ``WorkerConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytest

from video_engine.audio.loudness_models import LoudnessResult, LoudnormStats
from video_engine.audio.models import SpeechSegment, VadResult
from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.editing.media_probe import MediaInfo
from video_engine.editing.models import SpliceResult
from video_engine.shorts.models import HookDetectionResult, ShortCandidateCut
from video_engine.shorts.renderer_models import ShortsPublishPackage
from video_engine.video.models import DynamicZoomResult, ZoomMode, ZoomShot
from video_engine.worker.models import VideoProcessingJobData, WorkerConfig
from video_engine.worker.pipeline import VideoProcessingPipeline, parse_generate_shorts

MEASURED = LoudnormStats(
    input_i=-20.0,
    input_tp=-8.0,
    input_lra=5.0,
    input_thresh=-30.0,
    target_offset=0.5,
)


def job(upload_id: str = "upload-abc", file_path: Optional[str] = None) -> VideoProcessingJobData:
    return VideoProcessingJobData(
        jobId="job-shorts-001",
        uploadId=upload_id,
        filePath=file_path or "/tmp/raw.mp4",
        createdAt="2026-01-01T00:00:00Z",
    )


def _make_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        text="venha aprender com este gancho",
        language="pt",
        duration_ms=30000,
        words=[
            WordTimestamp(word="venha", start_ms=0, end_ms=500, probability=0.9),
            WordTimestamp(word="aprender", start_ms=800, end_ms=1300, probability=0.9),
            WordTimestamp(word="com", start_ms=1500, end_ms=1800, probability=0.9),
            WordTimestamp(word="este", start_ms=2000, end_ms=2400, probability=0.9),
            WordTimestamp(word="gancho", start_ms=2500, end_ms=3000, probability=0.9),
        ],
    )


def make_cut(index: int = 1) -> ShortCandidateCut:
    return ShortCandidateCut(
        id=f"cut_{index:02d}",
        start_ms=0,
        end_ms=30000,
        duration_ms=30000,
        hook_text="O segredo que ninguem conta",
        summary="Resumo do trecho de alto impacto",
        reason="Gancho emocional com pico de energia",
        semantic_score=0.85,
        energy_score=0.7,
        virality_score=0.78,
        words=[
            WordTimestamp(word="gancho", start_ms=2500, end_ms=3000, probability=0.9),
        ],
    )


# --------------------------------------------------------------------------- #
# Stubs modulares
# --------------------------------------------------------------------------- #
class StubProbe:
    def probe(self, path) -> MediaInfo:
        return MediaInfo(path=str(path), has_video=True, has_audio=True, duration_ms=30000)


class StubVAD:
    def __init__(self, segments: Sequence[SpeechSegment], total: int = 30000) -> None:
        self.segments = list(segments)
        self.total = total

    def detect_file(self, path) -> VadResult:
        return VadResult(
            total_duration_ms=self.total,
            speech_segments=self.segments,
            silence_segments=[],
        )


class StubSplicer:
    def splice_file(self, input_path, output_path, segments) -> SpliceResult:
        Path(output_path).write_bytes(b"SPLICED")
        return SpliceResult(
            output_path=str(output_path),
            total_duration_ms=30000,
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
        raise AssertionError("transcriber.shorts.nao.deveria.ser.acionado")


class StubZoom:
    def __init__(self, zoom_shots_count: int = 2) -> None:
        self.zoom_shots_count = zoom_shots_count
        self.calls: List[dict] = []

    def apply_zoom(self, input_video, output_video, pause_intervals=None, face_center=None, transcription=None):
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
            total_duration_ms=30000,
            shots=[ZoomShot(start_ms=0, end_ms=30000, mode=ZoomMode.NORMAL, scale=1.0)],
            zoom_shots_count=self.zoom_shots_count,
            normal_shots_count=1,
            video_width=1920,
            video_height=1080,
            scaling_filter="lanczos",
            strategy_used="heuristic",
        )


class StubShortsDetector:
    def __init__(self, cuts: Optional[Sequence[ShortCandidateCut]] = None, error: Optional[Exception] = None) -> None:
        self.cuts = list(cuts) if cuts else []
        self.error = error
        self.calls: List[tuple] = []

    def detect_cuts(self, transcription, audio_source=None, total_duration_ms=None) -> HookDetectionResult:
        self.calls.append((transcription, audio_source, total_duration_ms))
        if self.error is not None:
            raise self.error
        return HookDetectionResult(cuts=self.cuts, total_cuts_found=len(self.cuts), strategy_used="heuristic_fallback")


class BoomShortsDetector:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def detect_cuts(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("detector.shorts.nao.deveria.ser.acionado")


class StubShortsRenderer:
    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error
        self.calls: List[tuple] = []

    def render_cut(
        self,
        source_video,
        cut,
        output_dir,
        transcription=None,
        video_title=None,
    ) -> ShortsPublishPackage:
        self.calls.append(
            (str(source_video), cut, str(output_dir), transcription, video_title)
        )
        if self.error is not None:
            raise self.error
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"{cut.id}.mp4"
        metadata_path = out_dir / f"{cut.id}_metadata.json"
        video_path.write_bytes(b"fake-mp4")
        metadata_path.write_text('{"ok": true}', encoding="utf-8")
        return ShortsPublishPackage(
            id=cut.id,
            title="Titulo otimizado de alto CTR",
            description=f"{cut.hook_text}\n\n#shorts #cortes",
            hashtags=["#shorts", "#cortes"],
            video_path=str(video_path),
            metadata_path=str(metadata_path),
            duration_sec=30.0,
            start_ms=cut.start_ms,
            end_ms=cut.end_ms,
            resolution="1080x1920",
            virality_score=cut.virality_score,
            hook_text=cut.hook_text,
        )


class BoomShortsRenderer:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def render_cut(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("renderer.shorts.nao.deveria.ser.acionado")


def make_pipeline(
    tmp_path,
    detector: Optional[Any] = None,
    renderer: Optional[Any] = None,
    transcriber: Optional[Any] = None,
    zoom: Optional[Any] = None,
    config: Optional[WorkerConfig] = None,
) -> VideoProcessingPipeline:
    cfg = config or WorkerConfig(output_dir=str(tmp_path / "out"))
    return VideoProcessingPipeline(
        config=cfg,
        probe=StubProbe(),
        vad=StubVAD(
            [
                SpeechSegment(start_ms=500, end_ms=2500),
                SpeechSegment(start_ms=3000, end_ms=6000),
            ]
        ),
        splicer=StubSplicer(),
        normalizer=StubNormalizer(),
        zoom_processor=zoom,
        transcriber=transcriber,
        shorts_detector=detector,
        shorts_renderer=renderer,
    )


# --------------------------------------------------------------------------- #
# CA1: flag desabilitada por padrao
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_disabled_by_default(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    detector = BoomShortsDetector()
    renderer = BoomShortsRenderer()
    transcriber = BoomTranscriber()
    pipeline = make_pipeline(tmp_path, detector=detector, renderer=renderer, transcriber=transcriber)

    result = pipeline.process(job(file_path=str(source)))

    assert detector.calls == []
    assert renderer.calls == []
    assert transcriber.calls == []
    assert result.shorts == []
    assert "shorts" not in result.to_success_metadata()
    assert result.output_path.endswith("upload-abc_processed.mp4")


def test_pipeline_shorts_disabled_explicitly_via_metadata(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    detector = BoomShortsDetector()
    renderer = BoomShortsRenderer()
    pipeline = make_pipeline(
        tmp_path,
        detector=detector,
        renderer=renderer,
        config=WorkerConfig(output_dir=str(tmp_path / "out"), generate_shorts=True),
    )
    j = job(file_path=str(source))
    j.metadata["generate_shorts"] = False

    result = pipeline.process(j)

    assert detector.calls == []
    assert renderer.calls == []
    assert result.shorts == []


# --------------------------------------------------------------------------- #
# CA2: flag habilitada -> geracao e renderizacao com nomenclatura padrao
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_enabled_generates_and_renders_cuts(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    cuts = [make_cut(1), make_cut(2)]
    detector = StubShortsDetector(cuts=cuts)
    renderer = StubShortsRenderer()
    transcriber = StubTranscriber(result=_make_transcription())
    pipeline = make_pipeline(tmp_path, detector=detector, renderer=renderer, transcriber=transcriber)
    j = job(file_path=str(source))
    j.metadata["generate_shorts"] = True

    result = pipeline.process(j)

    assert len(detector.calls) == 1
    assert len(renderer.calls) == 2
    first_cut = renderer.calls[0][1]
    second_cut = renderer.calls[1][1]
    assert first_cut.id == "upload-abc_short_1"
    assert second_cut.id == "upload-abc_short_2"

    out = tmp_path / "out"
    assert (out / "upload-abc_short_1.mp4").is_file()
    assert (out / "upload-abc_short_1_metadata.json").is_file()
    assert (out / "upload-abc_short_2.mp4").is_file()
    assert (out / "upload-abc_short_2_metadata.json").is_file()

    assert len(result.shorts) == 2
    assert result.shorts[0]["id"] == "upload-abc_short_1"
    assert result.shorts[0]["videoPath"].endswith("upload-abc_short_1.mp4")
    assert result.shorts[0]["metadataPath"].endswith("upload-abc_short_1_metadata.json")
    assert result.shorts[0]["title"]
    assert result.shorts[0]["hashtags"] == ["#shorts", "#cortes"]
    assert result.shorts[0]["durationSec"] == 30.0
    assert result.shorts[0]["startMs"] == 0
    assert result.shorts[0]["endMs"] == 30000
    assert result.shorts[0]["viralityScore"] == pytest.approx(0.78)
    assert result.shorts[0]["resolution"] == "1080x1920"

    metadata = result.to_success_metadata()
    assert len(metadata["shorts"]) == 2
    assert result.output_path.endswith("upload-abc_processed.mp4")


def test_pipeline_shorts_enabled_via_worker_config(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    detector = StubShortsDetector(cuts=[make_cut(1)])
    renderer = StubShortsRenderer()
    pipeline = make_pipeline(
        tmp_path,
        detector=detector,
        renderer=renderer,
        transcriber=StubTranscriber(result=_make_transcription()),
        config=WorkerConfig(output_dir=str(tmp_path / "out"), generate_shorts=True),
    )

    result = pipeline.process(job(file_path=str(source)))

    assert len(detector.calls) == 1
    assert len(renderer.calls) == 1
    assert len(result.shorts) == 1


# --------------------------------------------------------------------------- #
# CA3: reuso da transcricao do Dynamic Zoom
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_reuses_transcription_from_zoom(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    transcription = _make_transcription()
    transcriber = StubTranscriber(result=transcription)
    cuts = [make_cut(1)]
    detector = StubShortsDetector(cuts=cuts)
    renderer = StubShortsRenderer()
    zoom = StubZoom()
    pipeline = make_pipeline(tmp_path, detector=detector, renderer=renderer, transcriber=transcriber, zoom=zoom)
    j = job(file_path=str(source))
    j.metadata["dynamicZoom"] = True
    j.metadata["generate_shorts"] = True

    result = pipeline.process(j)

    assert len(transcriber.calls) == 1
    assert len(zoom.calls) == 1
    assert zoom.calls[0]["transcription"] is transcription
    detected_transcription = detector.calls[0][0]
    assert detected_transcription is transcription
    assert len(result.shorts) == 1


# --------------------------------------------------------------------------- #
# CA4: transcricao disparada quando apenas generate_shorts esta ativo
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_runs_transcription_when_zoom_disabled(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    transcriber = StubTranscriber(result=_make_transcription())
    detector = StubShortsDetector(cuts=[make_cut(1)])
    renderer = StubShortsRenderer()
    pipeline = make_pipeline(tmp_path, detector=detector, renderer=renderer, transcriber=transcriber)
    j = job(file_path=str(source))
    j.metadata["generate_shorts"] = True

    result = pipeline.process(j)

    assert len(transcriber.calls) == 1
    assert transcriber.calls[0].endswith("spliced.mp4")
    assert len(detector.calls) == 1
    assert len(result.shorts) == 1
    assert result.dynamic_zoom_applied is False


def test_pipeline_shorts_aborts_when_transcription_fails(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    transcriber = StubTranscriber(error=RuntimeError("faster-whisper sem VRAM"))
    detector = StubShortsDetector(cuts=[make_cut(1)])
    renderer = StubShortsRenderer()
    pipeline = make_pipeline(tmp_path, detector=detector, renderer=renderer, transcriber=transcriber)
    j = job(file_path=str(source))
    j.metadata["generate_shorts"] = True

    result = pipeline.process(j)

    assert len(transcriber.calls) == 1
    assert detector.calls == []
    assert renderer.calls == []
    assert result.shorts == []
    assert result.output_path.endswith("upload-abc_processed.mp4")
    assert Path(result.output_path).is_file()


# --------------------------------------------------------------------------- #
# CA5: resiliencia do detector
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_resilience_on_detector_failure(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    detector = StubShortsDetector(error=RuntimeError("gemini fora do ar"))
    renderer = StubShortsRenderer()
    pipeline = make_pipeline(
        tmp_path,
        detector=detector,
        renderer=renderer,
        transcriber=StubTranscriber(result=_make_transcription()),
    )
    j = job(file_path=str(source))
    j.metadata["generate_shorts"] = True

    result = pipeline.process(j)

    assert len(detector.calls) == 1
    assert renderer.calls == []
    assert result.shorts == []
    assert "shorts" not in result.to_success_metadata()
    assert Path(result.output_path).is_file()


# --------------------------------------------------------------------------- #
# CA6: resiliencia do renderer
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_resilience_on_renderer_failure(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    detector = StubShortsDetector(cuts=[make_cut(1), make_cut(2)])
    renderer = StubShortsRenderer(error=RuntimeError("ffmpeg falhou no encoding"))
    pipeline = make_pipeline(
        tmp_path,
        detector=detector,
        renderer=renderer,
        transcriber=StubTranscriber(result=_make_transcription()),
    )
    j = job(file_path=str(source))
    j.metadata["generate_shorts"] = True

    result = pipeline.process(j)

    assert len(renderer.calls) == 2
    assert result.shorts == []
    assert result.output_path.endswith("upload-abc_processed.mp4")
    assert Path(result.output_path).is_file()


def test_pipeline_shorts_partial_render_keeps_successful_cuts(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    cuts = [make_cut(1), make_cut(2)]

    class FailingSecondRenderer(StubShortsRenderer):
        def __init__(self) -> None:
            super().__init__()
            self.attempts: List[str] = []

        def render_cut(self, source_video, cut, output_dir, transcription=None, video_title=None):
            self.attempts.append(cut.id)
            if cut.id.endswith("_short_2"):
                raise RuntimeError("falha isolada no segundo corte")
            return super().render_cut(
                source_video, cut, output_dir, transcription=transcription, video_title=video_title
            )

    detector = StubShortsDetector(cuts=cuts)
    renderer = FailingSecondRenderer()
    pipeline = make_pipeline(
        tmp_path,
        detector=detector,
        renderer=renderer,
        transcriber=StubTranscriber(result=_make_transcription()),
    )
    j = job(file_path=str(source))
    j.metadata["shorts"] = "true"

    result = pipeline.process(j)

    assert renderer.attempts == ["upload-abc_short_1", "upload-abc_short_2"]
    assert len(renderer.calls) == 1
    assert len(result.shorts) == 1
    assert result.shorts[0]["id"] == "upload-abc_short_1"
    assert (tmp_path / "out" / "upload-abc_short_1.mp4").is_file()


# --------------------------------------------------------------------------- #
# CA7: nenhum gancho detectado
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_empty_cuts_handled_cleanly(tmp_path):
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"fake")
    detector = StubShortsDetector(cuts=[])
    renderer = StubShortsRenderer()
    pipeline = make_pipeline(
        tmp_path,
        detector=detector,
        renderer=renderer,
        transcriber=StubTranscriber(result=_make_transcription()),
    )
    j = job(file_path=str(source))
    j.metadata["generate_shorts"] = True

    result = pipeline.process(j)

    assert len(detector.calls) == 1
    assert renderer.calls == []
    assert result.shorts == []
    assert "shorts" not in result.to_success_metadata()
    assert Path(result.output_path).is_file()


# --------------------------------------------------------------------------- #
# CA8: parsing heterogeneo da flag
# --------------------------------------------------------------------------- #
def test_pipeline_shorts_flag_parses_heterogeneous_values(tmp_path):
    cases = [
        ("true", True),
        ("1", True),
        (1, True),
        ("yes", True),
        ("false", False),
        ("0", False),
        (0, False),
        (None, False),
    ]
    for index, (raw_value, expected) in enumerate(cases):
        source = tmp_path / f"raw_{index}.mp4"
        source.write_bytes(b"fake")
        detector = StubShortsDetector(cuts=[make_cut(1)])
        renderer = StubShortsRenderer()
        pipeline = make_pipeline(
            tmp_path,
            detector=detector,
            renderer=renderer,
            transcriber=StubTranscriber(result=_make_transcription()),
        )
        j = job(file_path=str(source))
        j.metadata["generate_shorts"] = raw_value

        result = pipeline.process(j)

        if expected:
            assert len(renderer.calls) == 1, f"esperado render para {raw_value!r}"
            assert len(result.shorts) == 1
        else:
            assert renderer.calls == [], f"esperado sem render para {raw_value!r}"
            assert result.shorts == []


def test_parse_generate_shorts_unit():
    assert parse_generate_shorts(None) is False
    assert parse_generate_shorts(None, default=True) is True
    assert parse_generate_shorts(True) is True
    assert parse_generate_shorts("true") is True
    assert parse_generate_shorts("TRUE") is True
    assert parse_generate_shorts("1") is True
    assert parse_generate_shorts(1) is True
    assert parse_generate_shorts("yes") is True
    assert parse_generate_shorts(False) is False
    assert parse_generate_shorts("false") is False
    assert parse_generate_shorts("0") is False
    assert parse_generate_shorts(0) is False
    assert parse_generate_shorts("qualquercoisa") is False
