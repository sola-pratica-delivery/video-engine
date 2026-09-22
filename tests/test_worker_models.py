"""Testes dos modelos de dados do worker (Spec Issue #18).

Cobertura: schemas e serializacao de ``VideoProcessingJobData`` (aliases
camelCase via ``populate_by_name``), ``QueueMessage``, ``WorkerConfig``
(defaults, ``extra="ignore"``) e ``ProcessingResult`` (grade de metadados de
sucesso para o ``delivery-core``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_engine.worker.models import (
    LoudnessReport,
    ProcessingResult,
    QueueMessage,
    VideoProcessingJobData,
    WorkerConfig,
)

RAW_JOB = {
    "jobId": "job-001",
    "uploadId": "upload-abc",
    "filePath": "/tmp/raw.mp4",
    "metadata": {"source": "mobile"},
    "createdAt": "2026-01-01T00:00:00Z",
}


# --------------------------------------------------------------------------- #
# VideoProcessingJobData
# --------------------------------------------------------------------------- #
def test_job_data_parses_camel_case_aliases():
    job = VideoProcessingJobData.model_validate(RAW_JOB)
    assert job.job_id == "job-001"
    assert job.upload_id == "upload-abc"
    assert job.file_path == "/tmp/raw.mp4"
    assert job.metadata == {"source": "mobile"}
    assert job.created_at == "2026-01-01T00:00:00Z"


def test_job_data_parses_by_field_name():
    job = VideoProcessingJobData(
        job_id="job-002",
        upload_id="upload-def",
        file_path="/x.mp4",
        created_at="2026-01-01T00:00:00Z",
    )
    assert job.job_id == "job-002"
    assert job.metadata == {}


def test_job_data_requires_required_fields():
    with pytest.raises(ValidationError):
        VideoProcessingJobData.model_validate(
            {"jobId": "x", "uploadId": "y", "filePath": "/x.mp4"}
        )


def test_job_data_serializes_with_aliases():
    job = VideoProcessingJobData.model_validate(RAW_JOB)
    dumped = job.model_dump(by_alias=True)
    assert dumped["jobId"] == "job-001"
    assert dumped["uploadId"] == "upload-abc"
    assert "fileName" not in dumped


# --------------------------------------------------------------------------- #
# QueueMessage
# --------------------------------------------------------------------------- #
def test_queue_message_default_attempts():
    msg = QueueMessage(
        id="job-001",
        queue="video-processing",
        data=VideoProcessingJobData.model_validate(RAW_JOB),
        raw_payload="{}",
    )
    assert msg.attempts == 1


def test_queue_message_roundtrip_with_attempts():
    msg = QueueMessage(
        id="job-001",
        queue="video-processing",
        data=VideoProcessingJobData.model_validate(RAW_JOB),
        raw_payload="{}",
        attempts=3,
    )
    assert msg.attempts == 3


# --------------------------------------------------------------------------- #
# WorkerConfig
# --------------------------------------------------------------------------- #
def test_worker_config_defaults_match_spec():
    cfg = WorkerConfig()
    assert cfg.redis_host == "127.0.0.1"
    assert cfg.redis_port == 6379
    assert cfg.redis_password is None
    assert cfg.redis_db == 0
    assert cfg.queue_name == "video-processing"
    assert cfg.delivery_core_url == "http://127.0.0.1:3000"
    assert cfg.upload_endpoint_pattern == "/uploads/{upload_id}/transition"
    assert cfg.output_dir == "./storage/processed"
    assert cfg.job_timeout_seconds == 600.0
    assert cfg.poll_interval_seconds == 1.0
    assert cfg.success_status == "AUTO_QA"
    assert cfg.max_retries == 3


def test_worker_config_supports_validating_qa_alias():
    cfg = WorkerConfig(success_status="VALIDATING_QA")
    assert cfg.success_status == "VALIDATING_QA"


def test_worker_config_ignores_extra_fields():
    cfg = WorkerConfig(nao_existe="x", redis_host="10.0.0.1")
    assert cfg.redis_host == "10.0.0.1"
    assert not hasattr(cfg, "nao_existe")


def test_worker_config_coerces_numbers_from_strings():
    cfg = WorkerConfig(redis_port="6380", max_retries="5", job_timeout_seconds="30.5")
    assert cfg.redis_port == 6380
    assert cfg.max_retries == 5
    assert cfg.job_timeout_seconds == 30.5


# --------------------------------------------------------------------------- #
# ProcessingResult
# --------------------------------------------------------------------------- #
def test_processing_result_defaults():
    result = ProcessingResult(
        output_path="/storage/a_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
    )
    assert result.output_path == "/storage/a_processed.mp4"
    assert result.duration_sec == 42.5
    assert result.speech_segments_count == 8
    assert result.silence_removed_ms == 5200
    assert result.dynamic_zoom_applied is False
    assert result.zoom_shots_count == 0


def test_processing_result_dynamic_zoom_fields():
    result = ProcessingResult(
        output_path="/storage/a_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
        dynamic_zoom_applied=True,
        zoom_shots_count=4,
    )
    assert result.dynamic_zoom_applied is True
    assert result.zoom_shots_count == 4


def test_processing_result_success_metadata_includes_dynamic_zoom():
    result = ProcessingResult(
        output_path="/storage/uuid_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
        dynamic_zoom_applied=True,
        zoom_shots_count=6,
    )
    metadata = result.to_success_metadata()
    assert metadata["dynamicZoomApplied"] is True
    assert metadata["zoomShotsCount"] == 6


def test_processing_result_success_metadata_dynamic_zoom_defaults():
    result = ProcessingResult(
        output_path="/storage/uuid_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
    )
    metadata = result.to_success_metadata()
    assert metadata["dynamicZoomApplied"] is False
    assert metadata["zoomShotsCount"] == 0


def test_processing_result_success_metadata_matches_contract():
    result = ProcessingResult(
        output_path="/storage/uuid_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
    )
    metadata = result.to_success_metadata()
    assert metadata["processedVideoPath"] == "/storage/uuid_processed.mp4"
    assert metadata["duration"] == 42.5
    assert metadata["speechSegmentsCount"] == 8
    assert metadata["silenceRemovedMs"] == 5200
    assert metadata["loudness"] == {
        "integratedLufs": -14.1,
        "truePeakDbtp": -1.02,
        "lra": 10.8,
    }


# --------------------------------------------------------------------------- #
# Thumbnail (Issue #21): campos e metadata de sucesso
# --------------------------------------------------------------------------- #
def test_processing_result_thumbnail_fields():
    result = ProcessingResult(
        output_path="/storage/uuid_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
    )
    assert result.thumbnail_path is None
    assert result.thumbnail_score is None

    result = ProcessingResult(
        output_path="/storage/uuid_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
        thumbnail_path="/storage/upload-abc_thumbnail.jpg",
        thumbnail_score=0.87,
    )
    assert result.thumbnail_path == "/storage/upload-abc_thumbnail.jpg"
    assert result.thumbnail_score == pytest.approx(0.87)


def test_processing_result_success_metadata_includes_thumbnail():
    result = ProcessingResult(
        output_path="/storage/uuid_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
        thumbnail_path="/storage/upload-abc_thumbnail.jpg",
        thumbnail_score=0.91,
    )
    metadata = result.to_success_metadata()
    assert metadata["thumbnailPath"] == "/storage/upload-abc_thumbnail.jpg"
    assert metadata["thumbnailScore"] == pytest.approx(0.91)


def test_processing_result_success_metadata_omits_thumbnail_when_none():
    result = ProcessingResult(
        output_path="/storage/uuid_processed.mp4",
        duration_sec=42.5,
        speech_segments_count=8,
        silence_removed_ms=5200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.8),
    )
    metadata = result.to_success_metadata()
    assert "thumbnailPath" not in metadata
    assert "thumbnailScore" not in metadata
