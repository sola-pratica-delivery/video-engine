"""Modelos de dados da camada de edicao (dominio da Issue #3).

Tipos compartilhados pela camada de sinal pura NumPy
(``video_engine.audio.splicer``) e pela camada de midia FFmpeg
(``video_engine.editing.media_splicer``).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FadeCurve(str, Enum):
    """Lei de atenuacao aplicada em cada ponto de emenda."""

    LINEAR = "linear"
    EQUAL_POWER = "equal_power"


class SplicerConfig(BaseModel):
    """Configuracao do splicer (sinal e midia).

    ``crossfade_ms`` e o micro-crossfade da spec (10-15ms; default 15ms).
    """

    crossfade_ms: float = Field(
        default=15.0,
        ge=1.0,
        le=50.0,
        description="Duracao do micro-crossfade em milissegundos (default 15ms)",
    )
    curve: FadeCurve = Field(
        default=FadeCurve.EQUAL_POWER,
        description="Lei de atenuacao das emendas (default: equal power)",
    )
    video_codec: str = Field(default="libx264")
    audio_codec: str = Field(default="aac")
    audio_bitrate: str = Field(default="192k")
    preset: str = Field(default="fast")
    crf: int = Field(default=18, ge=0, le=51)


class SpliceResult(BaseModel):
    """Resultado de uma operacao de emenda (splice) de arquivo de midia."""

    output_path: str
    total_duration_ms: int = Field(ge=0)
    num_segments: int = Field(ge=0)
    num_junctions: int = Field(ge=0)
    audio_sample_rate: int = Field(default=16000, ge=0)
    has_video: bool = Field(default=False)


__all__ = ["FadeCurve", "SpliceResult", "SplicerConfig"]
