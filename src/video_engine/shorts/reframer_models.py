"""Modelos de dados do reenquadramento vertical 9:16 (Issue #16).

Contratos derivados da Spec da Issue #16 (SDD): conversao e adaptacao
geometrica de videos horizontais (16:9) para o formato vertical nativo
das plataformas moveis (9:16, 1080x1920), mantendo o apresentador
enquadrado de forma estavel e fluida.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CropMode(str, Enum):
    """Modo de reenquadramento vertical aplicado ao video."""

    SMART_CROP = "smart_crop"
    AESTHETIC_FILL = "aesthetic_fill"
    AUTO = "auto"


class FaceTrackingPoint(BaseModel):
    """Ponto de rastreamento facial em um instante do tempo."""

    model_config = ConfigDict(extra="forbid")

    timestamp_ms: int = Field(ge=0, description="Timestamp do frame em milissegundos.")
    face_detected: bool = Field(description="Se a face foi detectada no frame.")
    raw_center_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Centro X normalizado original (0.0 a 1.0).",
    )
    raw_center_y: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Centro Y normalizado original.",
    )
    smoothed_center_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Centro X apos filtro deadband + suavizacao.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TrackingTrajectory(BaseModel):
    """Trajetoria completa de rastreamento do apresentador ao longo do video."""

    model_config = ConfigDict(extra="forbid")

    points: List[FaceTrackingPoint] = Field(default_factory=list)
    face_detected_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Proporcao de frames com face detectada.",
    )
    average_center_x: float = Field(
        ge=0.0,
        le=1.0,
        description="Posicao horizontal media.",
    )
    dominant_mode: CropMode = Field(default=CropMode.SMART_CROP)


class ReframerConfig(BaseModel):
    """Configuracoes do reenquadrador vertical 9:16."""

    model_config = ConfigDict(extra="forbid")

    target_width: int = Field(default=1080, description="Largura do video vertical.")
    target_height: int = Field(default=1920, description="Altura do video vertical.")
    crop_mode: CropMode = Field(default=CropMode.AUTO)

    # Parametros de suavizacao e estabilizacao
    sample_interval_ms: int = Field(
        default=200,
        ge=50,
        description="Intervalo de amostragem facial.",
    )
    smoothing_factor: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        description="Fator de suavizacao EMA (menor = mais suave).",
    )
    deadband_threshold: float = Field(
        default=0.03,
        ge=0.0,
        le=0.20,
        description="Deslocamento minimo antes de mover a camera.",
    )
    min_face_detection_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Limiar para manter smart_crop em modo AUTO.",
    )

    # Qualidade de renderizacao FFmpeg
    scaling_filter: str = Field(
        default="lanczos",
        description="Filtro de interpolacao Lanczos ou Bicubic.",
    )
    blur_radius: int = Field(
        default=25,
        ge=5,
        le=100,
        description="Raio de desfoque para o fundo em aesthetic_fill.",
    )

    video_codec: str = Field(default="libx264")
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = Field(default="veryfast")


class ReframerResult(BaseModel):
    """Resultado do processo de reenquadramento vertical."""

    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(description="Caminho do arquivo 9:16 gerado.")
    duration_sec: float = Field(ge=0.0)
    width: int = Field(default=1080)
    height: int = Field(default=1920)
    mode_used: CropMode = Field(
        description="Modo efetivamente aplicado (smart_crop ou aesthetic_fill)."
    )
    face_detected_ratio: float = Field(ge=0.0, le=1.0)
    trajectory: Optional[TrackingTrajectory] = None

    @model_validator(mode="after")
    def reject_auto_mode(self) -> ReframerResult:
        """O modo efetivamente aplicado nunca pode permanecer ``AUTO``."""
        if self.mode_used == CropMode.AUTO:
            raise ValueError(
                "mode_used deve ser um modo efetivo (SMART_CROP ou "
                "AESTHETIC_FILL), nunca AUTO"
            )
        return self


__all__ = [
    "CropMode",
    "FaceTrackingPoint",
    "ReframerConfig",
    "ReframerResult",
    "TrackingTrajectory",
]
