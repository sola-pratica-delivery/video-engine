"""Modelos de dados da deteccao de ganchos virais para Shorts 9:16 (Issue #15).

Contratos derivados da Spec SDD da Issue #15: ``DetectionMode``,
``HookDetectorConfig``, ``ShortCandidateCut`` e ``HookDetectionResult``, alem
dos contratos auxiliares de energia vocal (``EnergyPeak``/``EnergyAnalysisResult``)
e da decisao semantica (``SemanticHookInterval``/``SemanticHookResponse``).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from video_engine.captions.models import WordTimestamp


class DetectionMode(str, Enum):
    """Estrategia de decisao da deteccao de ganchos virais."""

    HYBRID = "hybrid"  # Semantico (Gemini) + Picos de Energia Acustica
    SEMANTIC_ONLY = "semantic"  # Apenas Semantico (Gemini)
    HEURISTIC = "heuristic"  # Regras lexicais + Energia acustica (offline deterministico)


class HookDetectorConfig(BaseModel):
    """Parametros do detector de ganchos e cortes verticais autocontidos."""

    model_config = ConfigDict(extra="forbid")

    min_duration_ms: int = Field(
        default=25000,
        ge=10000,
        description="Duracao minima do corte (25s padrao Shorts).",
    )
    max_duration_ms: int = Field(
        default=58000,
        le=60000,
        description="Duracao maxima do corte (58s limite de seguranca do Shorts).",
    )
    target_cuts: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Quantidade maxima de cortes a extrair (1 a 3 tipico).",
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Score minimo de corte.",
    )
    decision_mode: DetectionMode = Field(default=DetectionMode.HYBRID)

    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    energy_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    snap_to_sentence_boundaries: bool = Field(default=True)
    max_overlap_iou: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Sobreposicao maxima permitida entre cortes (IoU temporal).",
    )

    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_timeout_s: float = Field(default=15.0, ge=1.0)
    gemini_max_retries: int = Field(default=1, ge=0)
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta")


class ShortCandidateCut(BaseModel):
    """Corte vertical (9:16) autocontido candidato a viralizacao."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Identificador unico do corte (ex: cut_01).")
    start_ms: int = Field(ge=0, description="Inicio do corte em milissegundos.")
    end_ms: int = Field(ge=0, description="Fim do corte em milissegundos.")
    duration_ms: int = Field(
        ge=25000,
        le=58000,
        description="Duracao exata do corte em ms (25s a 58s).",
    )

    hook_text: str = Field(description="Frase de gancho inicial (primeiros 3 a 5 segundos).")
    summary: str = Field(description="Resumo do raciocinio autocontido coberto no corte.")
    reason: str = Field(description="Justificativa da viralidade/relevancia.")

    semantic_score: float = Field(ge=0.0, le=1.0, description="Pontuacao semantica.")
    energy_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Pontuacao de energia vocal.")
    virality_score: float = Field(ge=0.0, le=1.0, description="Score composto final ponderado.")

    energy_peak_ms: Optional[int] = Field(default=None, description="Instante do maior pico vocal no corte.")
    words: List[WordTimestamp] = Field(default_factory=list, description="Palavras alinhadas dentro do corte.")


class HookDetectionResult(BaseModel):
    """Resultado da deteccao de ganchos virais em um video de formato longo."""

    model_config = ConfigDict(extra="forbid")

    cuts: List[ShortCandidateCut] = Field(default_factory=list)
    total_cuts_found: int = Field(ge=0)
    strategy_used: str = Field(description="hybrid, semantic ou heuristic_fallback")
    audio_energy_analyzed: bool = Field(default=False)


class SemanticHookInterval(BaseModel):
    """Gancho sugerido (semantico ou heuristico) antes do ajuste de duracao/snapping."""

    model_config = ConfigDict(extra="forbid")

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    hook_text: str = Field(default="")
    summary: str = Field(default="")
    reason: str = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SemanticHookResponse(BaseModel):
    """Contrato de structured output retornado pelo Gemini."""

    model_config = ConfigDict(extra="forbid")

    hook_candidates: List[SemanticHookInterval] = Field(default_factory=list)


class EnergyPeak(BaseModel):
    """Pico de intensidade vocal (janela de maior RMS relativo)."""

    model_config = ConfigDict(extra="forbid")

    start_ms: int = Field(ge=0, description="Inicio da janela do pico em ms.")
    end_ms: int = Field(ge=0, description="Fim da janela do pico em ms.")
    peak_ms: int = Field(ge=0, description="Instante de maior intensidade dentro da janela.")
    intensity: float = Field(ge=0.0, le=1.0, description="Intensidade normalizada do pico (0.0 a 1.0).")


class EnergyAnalysisResult(BaseModel):
    """Perfil de energia RMS por janelas temporais do audio (0.0 a 1.0)."""

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    window_ms: int = Field(ge=1)
    energies: List[float] = Field(
        default_factory=list,
        description="RMS relativo normalizado (0.0 a 1.0) por janela.",
    )
    peaks: List[EnergyPeak] = Field(default_factory=list)
    mean_energy: float = Field(ge=0.0, le=1.0)
    peak_energy: float = Field(ge=0.0, le=1.0)
    peak_ms: Optional[int] = Field(default=None, description="Instante do maior pico vocal do arquivo.")

    def window_energy(self, start_ms: int, end_ms: int) -> float:
        """Energia media (0.0 a 1.0) no intervalo temporal ``[start_ms, end_ms)``."""
        if not self.energies or end_ms <= start_ms:
            return 0.0
        step = max(self.window_ms, 1)
        i_start = max(0, start_ms // step)
        i_end = max(i_start, (end_ms + step - 1) // step)
        window = self.energies[i_start:i_end]
        if not window:
            return 0.0
        return float(sum(window) / len(window))

    def max_window_energy(self, start_ms: int, end_ms: int) -> float:
        """Pico de energia (0.0 a 1.0) dentro do intervalo ``[start_ms, end_ms)``."""
        if not self.energies or end_ms <= start_ms:
            return 0.0
        step = max(self.window_ms, 1)
        i_start = max(0, start_ms // step)
        i_end = max(i_start, (end_ms + step - 1) // step)
        window = self.energies[i_start:i_end]
        return max(window) if window else 0.0

    def strongest_peak(self, start_ms: int, end_ms: int) -> Optional[EnergyPeak]:
        """Retorna o pico mais intenso dentro do intervalo, ou ``None``."""
        candidates = [p for p in self.peaks if p.start_ms >= start_ms and p.end_ms <= end_ms]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.intensity)


EnergyProfile = EnergyAnalysisResult


__all__ = [
    "DetectionMode",
    "EnergyAnalysisResult",
    "EnergyPeak",
    "EnergyProfile",
    "HookDetectionResult",
    "HookDetectorConfig",
    "SemanticHookInterval",
    "SemanticHookResponse",
    "ShortCandidateCut",
]
