"""Modelos de dados da selecao automatica de keyframes (Issue #11).

Contratos derivados da Spec da Issue #11 (SDD): descarte de frames
degradados (borrao de movimento, olhos fechados, iluminacao pobre) e
ranqueamento deterministico dos 5 melhores frames candidatos para
thumbnails de alto CTR.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class StrokeConfig(BaseModel):
    """Configuracao de contorno (stroke) ao redor do sujeito segmentado."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    width: int = Field(default=8, ge=1, le=50, description="Largura do contorno em pixels")
    color: Tuple[int, int, int] = Field(default=(255, 255, 255), description="Cor RGB (0-255)")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Opacidade do stroke")

    @field_validator("color")
    @classmethod
    def validate_rgb(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(v) != 3 or any(c < 0 or c > 255 for c in v):
            raise ValueError("Cor RGB deve ser uma tupla (R, G, B) com valores entre 0 e 255")
        return v


class GlowConfig(BaseModel):
    """Configuracao de brilho difuso suave (glow) ao redor do sujeito."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    radius: int = Field(default=16, ge=1, le=100, description="Raio de difusao do glow")
    color: Tuple[int, int, int] = Field(default=(255, 255, 255), description="Cor RGB (0-255)")
    intensity: float = Field(default=0.8, ge=0.0, le=1.0, description="Intensidade maxima do glow")

    @field_validator("color")
    @classmethod
    def validate_rgb(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(v) != 3 or any(c < 0 or c > 255 for c in v):
            raise ValueError("Cor RGB deve ser uma tupla (R, G, B) com valores entre 0 e 255")
        return v


class SegmenterConfig(BaseModel):
    """Parametros de configuracao do segmentador semantico."""

    model_config = ConfigDict(extra="forbid")

    model_path: Optional[str] = None
    target_size: Tuple[int, int] = Field(default=(1024, 1024), description="Resolucao de entrada da rede")
    feather_radius: int = Field(default=2, ge=0, le=20, description="Raio de suavizacao da borda alfa")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Limiar binarizador de probabilidade")


class SegmentationResult(BaseModel):
    """Resultado da segmentacao contendo o sujeito e mascaras alfas."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    foreground_rgba: np.ndarray = Field(description="Array HxWx4 uint8 com canal alfa transparente")
    alpha_mask: np.ndarray = Field(description="Array HxW uint8 com a mascara alfa (0-255)")

    def apply_enhancements(
        self,
        stroke: Optional[StrokeConfig] = None,
        glow: Optional[GlowConfig] = None,
    ) -> np.ndarray:
        """Aplica contorno e brilho suave ao redor do sujeito segmentado."""
        from video_engine.thumbnail.edge_enhancer import apply_stroke_and_glow

        return apply_stroke_and_glow(self.foreground_rgba, stroke=stroke, glow=glow)


class SubjectPosition(str, Enum):
    """Posicionamento lateral do sujeito (regra dos tercos)."""

    LEFT = "left"
    RIGHT = "right"


class BackgroundType(str, Enum):
    """Tipo de fundo visual da thumbnail."""

    GRADIENT = "gradient"
    SOLID = "solid"
    IMAGE = "image"


class BackgroundConfig(BaseModel):
    """Configuracao do fundo visual da thumbnail."""

    model_config = ConfigDict(extra="forbid")

    type: BackgroundType = BackgroundType.GRADIENT
    color_start: Tuple[int, int, int] = Field(default=(15, 23, 42), description="Cor inicial RGB (#0F172A)")
    color_end: Tuple[int, int, int] = Field(default=(30, 41, 59), description="Cor final RGB (#1E293B)")
    direction: Literal["horizontal", "vertical", "diagonal"] = "diagonal"
    image_path: Optional[str] = None
    blur_radius: int = Field(default=8, ge=0, le=50)
    darken_factor: float = Field(default=0.25, ge=0.0, le=1.0)


class HeadlineConfig(BaseModel):
    """Configuracao tipografica da headline de alto impacto."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    font_size: int = Field(default=72, ge=24, le=180)
    text_color: Tuple[int, int, int] = Field(default=(255, 242, 0), description="Amarelo vibrante (#FFF200)")
    stroke_color: Tuple[int, int, int] = Field(default=(0, 0, 0), description="Contorno preto")
    stroke_width: int = Field(default=6, ge=0, le=30)
    shadow_color: Tuple[int, int, int] = Field(default=(0, 0, 0))
    shadow_offset: Tuple[int, int] = Field(default=(5, 5))
    all_caps: bool = Field(default=True)
    max_words: int = Field(default=5, ge=1, le=10)
    strict_word_limit: bool = Field(default=True, description="Levanta ValueError se exceder max_words")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Texto da headline nao pode ser vazio")
        return clean

    @model_validator(mode="after")
    def validate_word_count(self) -> HeadlineConfig:
        words = self.text.split()
        if self.strict_word_limit and len(words) > self.max_words:
            raise ValueError(
                f"Headline excede o limite recomendado de {self.max_words} palavras "
                f"para legibilidade mobile (recebido {len(words)} palavras: '{self.text}')"
            )
        return self


class ThumbnailConfig(BaseModel):
    """Parametros gerais de layout e exportacao da thumbnail 1280x720."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=1280, ge=640)
    height: int = Field(default=720, ge=360)
    subject_position: SubjectPosition = SubjectPosition.RIGHT
    subject_scale: float = Field(default=0.88, ge=0.5, le=1.2)
    subject_margin_x: int = Field(default=40, ge=0)
    jpeg_quality: int = Field(default=90, ge=50, le=100)
    max_file_size_bytes: int = Field(default=2 * 1024 * 1024, description="Limite estrito de 2MB do YouTube")


class ThumbnailCompositionResult(BaseModel):
    """Relatorio estruturado da thumbnail gerada."""

    model_config = ConfigDict(extra="forbid")

    output_path: str
    file_size_bytes: int
    width: int = 1280
    height: int = 720
    headline: str
    word_count: int
    subject_position: SubjectPosition


class GeminiHeadlineConfig(BaseModel):
    """Configuracao do sintetizador de headlines de alto CTR via Google Gemini."""

    model_config = ConfigDict(extra="forbid")

    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    base_url: str = "https://generativelanguage.googleapis.com"
    timeout_s: float = Field(default=8.0, gt=0.0)
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)
    min_words: int = Field(default=2, ge=1)
    max_words: int = Field(default=4, ge=1)
    fallback_headline: str = "ASSISTA AGORA"


class GeminiHeadlineResponse(BaseModel):
    """Payload estruturado retornado pelo Gemini para thumbnail."""

    model_config = ConfigDict(extra="forbid")

    headline: str


__all__ = [
    "BackgroundConfig",
    "BackgroundType",
    "FaceBoundingBox",
    "FaceMetrics",
    "FrameMetrics",
    "GeminiHeadlineConfig",
    "GeminiHeadlineResponse",
    "GlowConfig",
    "HeadlineConfig",
    "KeyframeCandidate",
    "KeyframeSelectorConfig",
    "KeyframeSelectorResult",
    "RejectionReason",
    "SegmentationResult",
    "SegmenterConfig",
    "StrokeConfig",
    "SubjectPosition",
    "ThumbnailCompositionResult",
    "ThumbnailConfig",
]

