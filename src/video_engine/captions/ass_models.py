"""Modelos de dados das legendas animadas ASS para cortes (Issue #9).

Define o estilo tipografico de alto impacto para Shorts/Reels
(``SubtitleStyleConfig``), os blocos de exibicao sincronizados
(``SubtitleCue``), o modo de destaque kataoke e o contrato de resultado
da queima de legendas em um corte (``CutCaptionBurnResult``).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from video_engine.captions.models import WordTimestamp


class KaraokeHighlightMode(str, Enum):
    """Modo de animacao de destaque palavra a palavra."""

    WORD_HIGHLIGHT = "word_highlight"  # Destaque com troca de cor/escala na palavra ativa
    KARAOKE_TAG = "karaoke_tag"  # Tags nativas ASS de karaoke (\\k / \\kf)


class SubtitleStyleConfig(BaseModel):
    """Parametros tipograficos e visuais para cortes verticais/shorts."""

    model_config = ConfigDict(extra="forbid")

    font_name: str = Field(default="Montserrat", description="Nome da fonte tipografica.")
    font_size: int = Field(default=52, ge=12, le=140, description="Tamanho da fonte otimizado para cortes.")
    primary_color: str = Field(default="#FFFFFF", description="Cor do texto inativo/base (HEX ou ASS).")
    highlight_color: str = Field(default="#FFF200", description="Cor vibrante de destaque da palavra ativa.")
    outline_color: str = Field(default="#000000", description="Cor do contorno/borda.")
    outline_width: float = Field(default=3.5, ge=0.0, le=10.0, description="Espessura da borda.")
    shadow_depth: float = Field(default=1.5, ge=0.0, le=10.0, description="Profundidade da sombra.")
    shadow_color: str = Field(default="#80000000", description="Cor e transparencia da sombra.")
    bold: bool = Field(default=True, description="Negrito ativado.")
    alignment: int = Field(default=2, ge=1, le=9, description="Alinhamento ASS (2 = Bottom-Center).")
    margin_v: int = Field(
        default=160,
        ge=0,
        le=800,
        description="Margem vertical inferior (safe area para Shorts/Reels).",
    )
    margin_l: int = Field(default=50, ge=0, le=500, description="Margem horizontal esquerda.")
    margin_r: int = Field(default=50, ge=0, le=500, description="Margem horizontal direita.")
    max_words_per_line: int = Field(default=4, ge=1, le=10, description="Maximo de palavras por linha no corte.")
    highlight_mode: KaraokeHighlightMode = Field(
        default=KaraokeHighlightMode.WORD_HIGHLIGHT,
        description="Modo de animacao de karaoke.",
    )
    active_word_scale: float = Field(
        default=1.05,
        ge=1.0,
        le=1.3,
        description="Escala de punch-in sutil da palavra ativa.",
    )
    play_res_x: int = Field(default=1080, ge=100, le=4096)
    play_res_y: int = Field(default=1920, ge=100, le=4096)


class SubtitleCue(BaseModel):
    """Representa um bloco/chunk de exibicao sincronizado com palavras individuais."""

    start_ms: int
    end_ms: int
    words: List[WordTimestamp]
    text: str


class CutCaptionBurnResult(BaseModel):
    """Resultado da queima de legendas em um corte."""

    output_path: str
    ass_path: Optional[str] = None
    cues_count: int
    total_words_count: int
    duration_sec: float


__all__ = [
    "CutCaptionBurnResult",
    "KaraokeHighlightMode",
    "SubtitleCue",
    "SubtitleStyleConfig",
]
