"""Modelos de dados e configuracao para mixagem de BGM e sidechain ducking.

Contratos derivados da Spec da Issue #5 (SDD).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BgmDuckingConfig(BaseModel):
    """Configuracao dos parametros de ducking sidechain e mixagem da BGM."""

    model_config = ConfigDict(extra="forbid")

    bgm_volume_db: float = Field(
        default=-14.0,
        description="Volume nominal da BGM quando nao ha fala (dB relativo a faixa original).",
    )
    ducking_attenuation_db: float = Field(
        default=-18.0,
        ge=-30.0,
        le=-6.0,
        description="Atenuacao alvo da BGM durante a presenca da voz (deve atender -18dB a -22dB).",
    )
    threshold: float = Field(
        default=0.03,
        gt=0.0001,
        le=1.0,
        description="Threshold linear do compressor sidechain para deteccao da voz.",
    )
    ratio: float = Field(
        default=10.0,
        ge=1.0,
        le=20.0,
        description="Razao de compressao aplicada sobre a BGM quando a voz ultrapassa o threshold.",
    )
    attack_ms: float = Field(
        default=30.0,
        ge=5.0,
        le=500.0,
        description="Tempo de ataque (ms) para inicio suave da atenuacao sem estalos.",
    )
    release_ms: float = Field(
        default=500.0,
        ge=50.0,
        le=2000.0,
        description="Tempo de release (ms) para fade-up suave nos momentos de respiro e pausas.",
    )
    knee: float = Field(
        default=2.8,
        ge=1.0,
        le=8.0,
        description="Curvatura de transicao (soft knee) da curva de compressao.",
    )
    loop_crossfade_ms: float = Field(
        default=2000.0,
        ge=100.0,
        description="Duracao da sobreposicao de crossfade entre iteracoes de loop da BGM.",
    )
    fade_in_ms: float = Field(
        default=1000.0,
        ge=0.0,
        description="Fade-in inicial suave da BGM na introducao da midia.",
    )
    fade_out_ms: float = Field(
        default=1500.0,
        ge=0.0,
        description="Fade-out final suave da BGM na conclusao da midia.",
    )
    audio_codec: str = Field(
        default="aac",
        description="Codec de audio para saida (ex: aac para MP4/M4A, pcm_s16le para WAV).",
    )
    audio_bitrate: str = Field(
        default="192k",
        description="Taxa de bits do audio de saida.",
    )


class BgmDuckingResult(BaseModel):
    """Metadados e estatisticas resultantes do processo de ducking e mixagem."""

    model_config = ConfigDict(extra="forbid")

    output_path: str
    primary_duration_ms: int
    bgm_original_duration_ms: int
    loops_applied: int
    has_video: bool
    effective_attenuation_db: float


__all__ = ["BgmDuckingConfig", "BgmDuckingResult"]
