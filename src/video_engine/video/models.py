"""Modelos de dados para Dynamic Punch-in Zoom.

Contratos derivados da Spec da Issue #8 (SDD).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ZoomMode(str, Enum):
    """Modo do enquadramento do plano."""

    NORMAL = "normal"  # Escala 1.0 (100%)
    ZOOM = "zoom"      # Escala aumentada (ex: 1.15x / 115%)


class ZoomShot(BaseModel):
    """Representa um trecho temporal continuo com enquadramento especifico."""

    model_config = ConfigDict(extra="forbid")

    start_ms: int = Field(ge=0, description="Inicio do plano em milissegundos.")
    end_ms: int = Field(ge=0, description="Fim do plano em milissegundos.")
    mode: ZoomMode = Field(description="Modo do plano (normal ou zoom).")
    scale: float = Field(ge=1.0, le=2.0, description="Fator de escala aplicado (ex: 1.0 ou 1.15).")
    anchor_x: float = Field(default=0.5, ge=0.0, le=1.0, description="Centro relativo horizontal do crop.")
    anchor_y: float = Field(default=0.40, ge=0.0, le=1.0, description="Centro relativo vertical do crop.")

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


class DynamicZoomConfig(BaseModel):
    """Parametros de configuracao do Dynamic Punch-in Zoom."""

    model_config = ConfigDict(extra="forbid")

    zoom_scale: float = Field(
        default=1.15,
        ge=1.05,
        le=1.50,
        description="Fator de escala do punch-in zoom (ex: 1.15 = 115%).",
    )
    min_shot_duration_s: float = Field(
        default=8.0,
        ge=2.0,
        le=30.0,
        description="Duracao minima de um plano antes de alternar (padrao 8s conforme criterio).",
    )
    max_shot_duration_s: float = Field(
        default=15.0,
        ge=4.0,
        le=60.0,
        description="Duracao maxima de um plano antes de forcar alternancia (padrao 15s).",
    )
    target_shot_duration_s: float = Field(
        default=10.0,
        ge=3.0,
        le=45.0,
        description="Duracao alvo tipica de cada enquadramento.",
    )
    anchor_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Posicao horizontal padrao do rosto (0.5 = centro).",
    )
    anchor_y: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Posicao vertical padrao do rosto (0.40 = terco superior / nivel dos olhos).",
    )
    scaling_filter: Literal["lanczos", "bicubic"] = Field(
        default="lanczos",
        description="Algoritmo de interpolacao para nitidez sem degradacao.",
    )
    video_codec: str = Field(
        default="libx264",
        description="Codec de video para codificacao.",
    )
    crf: int = Field(
        default=18,
        ge=0,
        le=35,
        description="Fator de qualidade constante (CRF 18 = visualmente sem perdas).",
    )
    preset: str = Field(
        default="veryfast",
        description="Preset de velocidade de compressao FFmpeg.",
    )

    @model_validator(mode="after")
    def validate_durations(self) -> DynamicZoomConfig:
        if self.min_shot_duration_s > self.max_shot_duration_s:
            raise ValueError("min_shot_duration_s nao pode ser maior que max_shot_duration_s")
        if not (self.min_shot_duration_s <= self.target_shot_duration_s <= self.max_shot_duration_s):
            raise ValueError("target_shot_duration_s deve estar entre min e max")
        return self


class DynamicZoomResult(BaseModel):
    """Relatorio do processamento de zoom dinamico."""

    model_config = ConfigDict(extra="forbid")

    output_path: str
    total_duration_ms: int
    shots: List[ZoomShot]
    zoom_shots_count: int
    normal_shots_count: int
    video_width: int
    video_height: int
    scaling_filter: str


__all__ = [
    "DynamicZoomConfig",
    "DynamicZoomResult",
    "ZoomMode",
    "ZoomShot",
]
