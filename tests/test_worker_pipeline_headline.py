"""Testes de integracao do GeminiHeadlineSynthesizer na esteira do worker (Issue #22)."""

from __future__ import annotations

from unittest.mock import MagicMock

from video_engine.captions.models import TranscriptionResult
from video_engine.thumbnail.headline_synthesizer import GeminiHeadlineSynthesizer
from video_engine.worker.models import VideoProcessingJobData
from video_engine.worker.pipeline import VideoProcessingPipeline


def test_resolve_headline_with_explicit_metadata_override() -> None:
    pipeline = VideoProcessingPipeline()
    job = VideoProcessingJobData(
        jobId="job-1",
        uploadId="up-1",
        filePath="dummy.mp4",
        metadata={"thumbnail_headline": "TITULO MANUAL ESPECIFICO"},
        createdAt="2026-09-24T00:00:00Z",
    )
    headline = pipeline._resolve_headline(job, None)
    assert headline == "TITULO MANUAL ESPECIFICO"


def test_resolve_headline_delegates_to_synthesizer() -> None:
    synthesizer_mock = MagicMock(spec=GeminiHeadlineSynthesizer)
    synthesizer_mock.synthesize.return_value = "O SEGREDO REVELADO"

    pipeline = VideoProcessingPipeline(headline_synthesizer=synthesizer_mock)
    job = VideoProcessingJobData(
        jobId="job-2",
        uploadId="up-2",
        filePath="dummy.mp4",
        metadata={"title": "Descubra Como Fazer Sucesso"},
        createdAt="2026-09-24T00:00:00Z",
    )
    transcription = TranscriptionResult(text="fala do video", duration_ms=1000)

    headline = pipeline._resolve_headline(job, transcription)
    assert headline == "O SEGREDO REVELADO"
    synthesizer_mock.synthesize.assert_called_once_with(
        title="Descubra Como Fazer Sucesso",
        transcription=transcription,
    )


def test_resolve_headline_falls_back_when_synthesizer_raises() -> None:
    synthesizer_mock = MagicMock(spec=GeminiHeadlineSynthesizer)
    synthesizer_mock.synthesize.side_effect = RuntimeError("Gemini error")
    synthesizer_mock.fallback_headline.return_value = "METODO INFALIVEL AGORA"

    pipeline = VideoProcessingPipeline(headline_synthesizer=synthesizer_mock)
    job = VideoProcessingJobData(
        jobId="job-3",
        uploadId="up-3",
        filePath="dummy.mp4",
        metadata={"title": "Metodo Infalivel Agora"},
        createdAt="2026-09-24T00:00:00Z",
    )

    headline = pipeline._resolve_headline(job, None)
    assert headline == "METODO INFALIVEL AGORA"
