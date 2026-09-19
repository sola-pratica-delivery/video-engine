"""Modelos de dados da selecao automatica de keyframes (Issue #11).

Contratos derivados da Spec da Issue #11 (SDD): descarte de frames
degradados (borrao de movimento, olhos fechados, iluminacao pobre) e
ranqueamento deterministico dos 5 melhores frames candidatos para
thumbnails de alto CTR.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RejectionReason(str, Enum):
    """Criterios tecnicos que invalidam um frame como candidato a thumbnail."""

    BLURRY = "blurry"
    CLOSED_EYES = "closed_eyes"
    NO_FACE_DETECTED = "no_face_detected"
    POOR_LIGHTING_DARK = "poor_lighting_dark"
    POOR_LIGHTING_OVEREXPOSED = "poor_lighting_overexposed"


class FaceBoundingBox(BaseModel):
    """Bounding box da face em coordenadas normalizadas (0-1)."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0, le=1.0, description="Posicao X normalizada (0-1)")
    y: float = Field(ge=0.0, le=1.0, description="Posicao Y normalizada (0-1)")
    width: float = Field(ge=0.0, le=1.0, description="Largura normalizada")
    height: float = Field(ge=0.0, le=1.0, description="Altura normalizada")


class FaceMetrics(BaseModel):
    """Metricas faciais e de expressividade de um frame."""

    model_config = ConfigDict(extra="forbid")

    detected: bool
    bounding_box: Optional[FaceBoundingBox] = None
    eye_openness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="1.0 = olhos bem abertos, 0.0 = fechados",
    )
    eyes_closed: bool = Field(
        default=False,
        description="True se eye_openness < min_eye_openness",
    )
    mouth_articulation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Abertura/articulacao da boca",
    )
    expression_intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Expressao marcante/enfase",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FrameMetrics(BaseModel):
    """Metricas completas de avaliacao de um frame candidato."""

    model_config = ConfigDict(extra="forbid")

    timestamp_ms: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    sharpness_variance: float = Field(ge=0.0, description="Variancia do Laplaciano")
    is_blurry: bool
    luminance_mean: float = Field(ge=0.0, le=255.0)
    luminance_std: float = Field(ge=0.0, le=255.0)
    lighting_score: float = Field(ge=0.0, le=1.0)
    is_poor_lighting: bool
    face: FaceMetrics
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_valid: bool = Field(default=True, description="False se descartado por qualquer criterio")
    rejection_reasons: List[RejectionReason] = Field(default_factory=list)


class KeyframeCandidate(BaseModel):
    """Frame ranqueado e selecionado como candidato final a thumbnail."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, le=5)
    timestamp_ms: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    metrics: FrameMetrics
    image_path: Optional[str] = None  # Caminho do frame salvo em disco (se solicitado)


class KeyframeSelectorConfig(BaseModel):
    """Parametros de descarte, amostragem e ranqueamento de keyframes.

    Os pesos do score composto ``weight_*`` devem somar exatamente 1.0.
    """

    model_config = ConfigDict(extra="forbid")

    # Nitidez (Laplacian variance)
    min_sharpness_threshold: float = Field(default=80.0, ge=0.0)
    discard_blurry: bool = Field(default=True)

    # Olhos e Face
    min_eye_openness: float = Field(default=0.25, ge=0.0, le=1.0)
    discard_closed_eyes: bool = Field(default=True)
    require_face: bool = Field(default=True, description="Descarta frames sem face se True")

    # Iluminacao
    min_luminance: float = Field(default=35.0, ge=0.0, le=255.0)
    max_luminance: float = Field(default=225.0, ge=0.0, le=255.0)
    discard_poor_lighting: bool = Field(default=True)

    # Amostragem
    sample_interval_s: float = Field(default=1.0, gt=0.0, description="Intervalo entre frames amostrados")
    start_offset_s: float = Field(default=1.0, ge=0.0, description="Ignorar intro pre-apresentacao")
    end_offset_s: float = Field(default=1.0, ge=0.0, description="Ignorar vinheta final")

    # Selecao & Ranqueamento
    top_n: int = Field(default=5, ge=1, le=20)
    min_candidate_distance_ms: int = Field(
        default=1500,
        ge=0,
        description="Distancia temporal minima entre Top frames",
    )

    # Pesos do Score Composto (devem somar 1.0)
    weight_sharpness: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_expression: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_lighting: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_face_prominence: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> KeyframeSelectorConfig:
        total = (
            self.weight_sharpness
            + self.weight_expression
            + self.weight_lighting
            + self.weight_face_prominence
        )
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"Pesos do score composto devem somar 1.0 (recebido: {total:.4f})"
            )
        if self.min_luminance > self.max_luminance:
            raise ValueError("min_luminance nao pode ser maior que max_luminance")
        return self


class KeyframeSelectorResult(BaseModel):
    """Relatorio estruturado da selecao de keyframes de um video."""

    model_config = ConfigDict(extra="forbid")

    video_path: str
    video_duration_ms: int
    total_frames_sampled: int
    valid_frames_count: int
    discarded_blurry_count: int
    discarded_closed_eyes_count: int
    discarded_lighting_count: int
    discarded_no_face_count: int
    top_candidates: List[KeyframeCandidate] = Field(default_factory=list)


__all__ = [
    "FaceBoundingBox",
    "FaceMetrics",
    "FrameMetrics",
    "KeyframeCandidate",
    "KeyframeSelectorConfig",
    "KeyframeSelectorResult",
    "RejectionReason",
]
