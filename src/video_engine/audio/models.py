"""Modelos de dados (Pydantic v2) do motor audiovisual.

Define os contratos temporais e os resultados da analise de Voz
(``SileroVadDetector``), incluindo segmentos de fala com padding e
pausas/silencios elegiveis para corte.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TimeInterval(BaseModel):
    """Representacao de um intervalo temporal com precisao em milissegundos e segundos."""

    start_ms: int = Field(..., ge=0, description="Timestamp de inicio em milissegundos")
    end_ms: int = Field(..., ge=0, description="Timestamp de fim em milissegundos")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


class SpeechSegment(TimeInterval):
    """Segmento de fala ativa identificado e ajustado com padding."""

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confianca media estimada do VAD",
    )


class SilenceSegment(TimeInterval):
    """Intervalo de silencio/pausa elegivel para corte (duracao >= min_silence_duration_ms)."""


class VadConfig(BaseModel):
    """Configuracao do detector Silero VAD e pos-processamento de intervalos."""

    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Limiar de probabilidade de fala do modelo",
    )
    min_speech_duration_ms: int = Field(
        100,
        ge=0,
        description="Duracao minima de fala para filtrar ruidos breves",
    )
    min_silence_duration_ms: int = Field(
        500,
        ge=0,
        description="Threshold minimo de silencio para registrar como pausa (padrao: 500ms)",
    )
    padding_ms: int = Field(
        80,
        ge=0,
        description="Margem de seguranca antes e depois de cada fala (padrao: 80ms)",
    )
    sample_rate: int = Field(
        16000,
        description="Taxa de amostragem padrao requerida pelo modelo VAD (16kHz)",
    )


class VadResult(BaseModel):
    """Resultado completo da analise VAD."""

    total_duration_ms: int = Field(
        ...,
        ge=0,
        description="Duracao total do audio em milissegundos",
    )
    speech_segments: List[SpeechSegment] = Field(
        default_factory=list,
        description="Trechos de fala com padding e fusoes aplicadas",
    )
    silence_segments: List[SilenceSegment] = Field(
        default_factory=list,
        description="Pausas detectadas que atendem ao threshold",
    )
