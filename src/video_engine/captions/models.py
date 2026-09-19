"""Modelos de dados da transcricao fonetica com timestamps por palavra.

Define os contratos de saída do ``WhisperTranscriber`` (Issue #7):
``WordTimestamp`` (alinha palavras individuais em milissegundos),
``CaptionSegment`` (frases com as palavras do segmento) e
``TranscriptionResult`` (resultado completo da transcricao), alem da
configuracao ``TranscriberConfig`` para o Faster-Whisper.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class WordTimestamp(BaseModel):
    """Palavra individual com alinhamento temporal em milissegundos."""

    model_config = ConfigDict(extra="forbid")

    word: str = Field(description="Texto da palavra individual.")
    start_ms: int = Field(ge=0, description="Tempo inicial em milissegundos.")
    end_ms: int = Field(ge=0, description="Tempo final em milissegundos.")
    probability: float = Field(ge=0.0, le=1.0, description="Confianca do modelo (0.0 a 1.0).")


class CaptionSegment(BaseModel):
    """Segmento de fala contendo oracao/frase e suas palavras alinhadas."""

    model_config = ConfigDict(extra="forbid")

    id: int
    text: str = Field(description="Texto transcrito do segmento.")
    start_ms: int = Field(ge=0, description="Inicio do segmento em milissegundos.")
    end_ms: int = Field(ge=0, description="Fim do segmento em milissegundos.")
    words: List[WordTimestamp] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Resultado completo da transcricao de audio com timestamps por palavra."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Texto consolidado completo da transcricao.")
    language: str = Field(default="pt", description="Idioma detectado ou configurado.")
    language_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    duration_ms: int = Field(ge=0, description="Duracao total do audio processado em ms.")
    segments: List[CaptionSegment] = Field(default_factory=list)
    words: List[WordTimestamp] = Field(default_factory=list)


class TranscriberConfig(BaseModel):
    """Parametros de configuracao do Faster-Whisper."""

    model_config = ConfigDict(extra="forbid")

    model_size: str = Field(default="base", description="Tamanho do modelo (ex: tiny, base, small).")
    device: str = Field(default="auto", description="Dispositivo de computacao ('auto', 'cpu', 'cuda').")
    compute_type: str = Field(
        default="int8",
        description="Tipo de quantizacao ('int8', 'float16', 'default').",
    )
    language: str = Field(default="pt", description="Codigo do idioma alvo.")
    beam_size: int = Field(default=5, ge=1, le=10)
    word_timestamps: bool = Field(
        default=True,
        description="Habilitar alinhamento no nivel de palavras.",
    )
    vad_filter: bool = Field(
        default=True,
        description="Filtrar silencios com VAD interno do Whisper.",
    )
    initial_prompt: Optional[str] = Field(
        default=None,
        description="Prompt inicial para guiar vocabulario.",
    )
    download_root: Optional[str] = Field(
        default=None,
        description="Diretorio de cache dos modelos.",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Caminho direto para modelo local CTranslate2.",
    )


__all__ = [
    "CaptionSegment",
    "TranscriberConfig",
    "TranscriptionResult",
    "WordTimestamp",
]
