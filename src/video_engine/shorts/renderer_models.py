"""Modelos de dados da renderizacao final de Shorts 9:16 (Issue #17).

Contratos derivados da Spec SDD da Issue #17: ``ShortsRendererConfig``
(parametros de renderizacao e estilizacao na safe area do YouTube Shorts),
``ShortsPublishPackage`` (metadados prontos para upload) e
``ShortsRenderResult`` (resultado da renderizacao individual e em lote).
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class ShortsRendererConfig(BaseModel):
    """Parametros de renderizacao e estilizacao de Shorts 9:16."""

    model_config = ConfigDict(extra="forbid")

    target_width: int = Field(default=1080)
    target_height: int = Field(default=1920)

    # Safe area do YouTube Shorts no terco medio (canvas 1080x1920)
    margin_v: int = Field(
        default=750,
        ge=400,
        le=1200,
        description="Distancia vertical inferior para situar as legendas no terco medio.",
    )
    margin_l: int = Field(
        default=80,
        ge=20,
        le=300,
        description="Margem lateral esquerda (longe dos botoes de UI do Shorts).",
    )
    margin_r: int = Field(
        default=150,
        ge=100,
        le=400,
        description="Margem lateral direita para evitar botoes de like/comentario.",
    )
    font_size: int = Field(default=52, ge=24, le=100)
    font_name: str = Field(default="Montserrat")
    highlight_color: str = Field(default="#FFF200")
    outline_width: float = Field(default=3.5, ge=0.0, le=10.0)

    # Parametros de compressao de video e audio
    video_codec: str = Field(default="libx264")
    audio_codec: str = Field(default="aac")
    audio_bitrate: str = Field(default="192k")
    crf: int = Field(default=20, ge=10, le=35)
    preset: str = Field(default="fast")
    pix_fmt: str = Field(default="yuv420p")
    faststart: bool = Field(
        default=True,
        description="Adiciona -movflags +faststart para reproducao instantanea mobile.",
    )

    max_title_chars: int = Field(
        default=100,
        ge=1,
        le=200,
        description="Limite de caracteres do titulo otimizado para o YouTube.",
    )
    default_hashtags: List[str] = Field(
        default_factory=lambda: ["#shorts", "#cortes", "#viral"]
    )


class ShortsPublishPackage(BaseModel):
    """Pacote de metadados pronto para publicacao no YouTube Shorts / Reels / TikTok."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Identificador do corte (ex: cut_01).")
    title: str = Field(description="Titulo de alto CTR otimizado para o Shorts (<= 100 caracteres).")
    description: str = Field(description="Descricao completa acompanhada de hashtags.")
    hashtags: List[str] = Field(default_factory=list, description="Lista de hashtags formatadas.")
    video_path: str = Field(description="Caminho do arquivo final 1080x1920.")
    metadata_path: str = Field(description="Caminho do arquivo JSON de metadados gerado.")
    duration_sec: float = Field(ge=0.0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    resolution: str = Field(default="1080x1920")
    video_codec: str = Field(default="h264")
    audio_codec: str = Field(default="aac")
    virality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hook_text: str = Field(default="")


class ShortsRenderResult(BaseModel):
    """Resultado da renderizacao de um ou mais cortes."""

    model_config = ConfigDict(extra="forbid")

    packages: List[ShortsPublishPackage] = Field(default_factory=list)
    total_rendered: int = Field(ge=0)
    output_dir: str


__all__ = [
    "ShortsPublishPackage",
    "ShortsRenderResult",
    "ShortsRendererConfig",
]
