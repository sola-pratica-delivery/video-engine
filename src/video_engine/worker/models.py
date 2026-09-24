"""Modelos de dados do worker consumidor de jobs de video.

Contratos utilizados na comunicacao com a fila ``video-processing`` e com a
API de lifecycle do ``delivery-core`` (Spec Issue #18).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class VideoProcessingJobData(BaseModel):
    """Dados do job enfileirado pelo ``delivery-core``."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    upload_id: str = Field(alias="uploadId")
    file_path: str = Field(alias="filePath")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(alias="createdAt")


class QueueMessage(BaseModel):
    """Mensagem desenvelopada da fila pronta para processamento."""

    id: str
    queue: str
    data: VideoProcessingJobData
    raw_payload: str
    attempts: int = 1


class WorkerConfig(BaseModel):
    """Configuracao do worker consumidor (enviavel via variaveis de ambiente)."""

    model_config = ConfigDict(extra="ignore")

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    queue_name: str = "video-processing"

    delivery_core_url: str = "http://127.0.0.1:3000"
    upload_endpoint_pattern: str = "/uploads/{upload_id}/transition"
    api_token: Optional[str] = None
    request_timeout_seconds: float = 10.0

    output_dir: str = "./storage/processed"
    temp_dir: Optional[str] = None

    job_timeout_seconds: float = 600.0
    poll_interval_seconds: float = 1.0
    success_status: str = "AUTO_QA"
    max_retries: int = 3
    consumer_backend: str = "redis"
    generate_shorts: bool = False


class LoudnessReport(BaseModel):
    """Metricas EBU R128 do artefato processado."""

    integrated_lufs: float
    true_peak_dbtp: float
    lra: float


class ProcessingResult(BaseModel):
    """Relatorio de um processamento de video concluido com sucesso."""

    output_path: str
    duration_sec: float
    speech_segments_count: int
    silence_removed_ms: int
    loudness: LoudnessReport
    dynamic_zoom_applied: bool = False
    zoom_shots_count: int = 0
    zoom_strategy: Optional[str] = None
    thumbnail_path: Optional[str] = None
    thumbnail_score: Optional[float] = None
    shorts: List[Dict[str, Any]] = Field(default_factory=list)

    def to_success_metadata(self) -> Dict[str, Any]:
        """Monta o payload ``metadata`` do contrato de transicao de sucesso."""
        data = {
            "processedVideoPath": self.output_path,
            "duration": self.duration_sec,
            "speechSegmentsCount": self.speech_segments_count,
            "silenceRemovedMs": self.silence_removed_ms,
            "loudness": {
                "integratedLufs": self.loudness.integrated_lufs,
                "truePeakDbtp": self.loudness.true_peak_dbtp,
                "lra": self.loudness.lra,
            },
            "dynamicZoomApplied": self.dynamic_zoom_applied,
            "zoomShotsCount": self.zoom_shots_count,
        }
        if self.zoom_strategy is not None:
            data["zoomStrategy"] = self.zoom_strategy
        if self.thumbnail_path is not None:
            data["thumbnailPath"] = self.thumbnail_path
        if self.thumbnail_score is not None:
            data["thumbnailScore"] = self.thumbnail_score
        if self.shorts:
            data["shorts"] = self.shorts
        return data


__all__ = [
    "LoudnessReport",
    "ProcessingResult",
    "QueueMessage",
    "VideoProcessingJobData",
    "WorkerConfig",
]
