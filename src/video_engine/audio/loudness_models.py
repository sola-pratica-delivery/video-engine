"""Modelos de dados da normalizacao de loudness EBU R128 (YouTube / Broadcast).

Contratos Pydantic v2 usados pelo motor ``video_engine.audio.loudness``:
``LoudnormStats`` (estatisticas do filtro ``loudnorm``), ``LoudnessConfig``
(alvos de -14.0 LUFS / -1.0 dBTP) e ``LoudnessResult`` (relatorio com QA de
conformidade).
"""

from pydantic import BaseModel, Field


class LoudnormStats(BaseModel):
    """Estatisticas EBU R128 extraidas pelo filtro loudnorm do FFmpeg."""

    input_i: float = Field(..., description="Integrated Loudness em LUFS")
    input_tp: float = Field(..., description="True Peak maximo em dBTP")
    input_lra: float = Field(..., description="Loudness Range em LU")
    input_thresh: float = Field(..., description="Threshold de medicao em LUFS")
    target_offset: float = Field(
        0.0,
        description="Offset de ganho calculado pelo loudnorm em dB",
    )


class LoudnessConfig(BaseModel):
    """Configuracao da normalizacao de loudness para YouTube / Broadcast."""

    target_i: float = Field(
        -14.0,
        ge=-70.0,
        le=-5.0,
        description="Integrated Loudness alvejado em LUFS (padrao YouTube: -14.0 LUFS)",
    )
    target_tp: float = Field(
        -1.0,
        ge=-9.0,
        le=0.0,
        description="True Peak maximo em dBTP (padrao YouTube: -1.0 dBTP)",
    )
    target_lra: float = Field(
        11.0,
        ge=1.0,
        le=50.0,
        description="Loudness Range alvejado em LU (padrao vocal/conteudo falado: 11.0 LU)",
    )
    linear: bool = Field(
        True,
        description="Aplica ganho linear no 2o passe para preservar dinamica natural sem pumping",
    )
    tolerance_lufs: float = Field(
        0.5,
        ge=0.1,
        le=2.0,
        description="Tolerancia maxima permitida para aprovacao no QA (+/- 0.5 LUFS)",
    )
    audio_codec: str = Field("aac", description="Codec de audio para conteneires de video (MP4/MKV)")
    audio_bitrate: str = Field("192k", description="Bitrate de audio para exportacao")


class LoudnessResult(BaseModel):
    """Resultado da normalizacao e relatorio de conformidade QA."""

    output_path: str = Field(..., description="Caminho do arquivo final gerado")
    measured_input: LoudnormStats = Field(
        ..., description="Estatisticas medidas no arquivo original (1o passe)"
    )
    measured_output: LoudnormStats = Field(
        ..., description="Estatisticas medidas no arquivo normalizado (QA pos-processamento)"
    )
    target_i: float = Field(-14.0, description="Alvo configurado de Integrated Loudness")
    target_tp: float = Field(-1.0, description="Alvo configurado de True Peak")
    is_compliant: bool = Field(
        ...,
        description=(
            "Indica se o arquivo gerado atende estritamente a -14 LUFS (+/- 0.5) "
            "e TP <= -1.0 dBTP"
        ),
    )
    has_video: bool = Field(False, description="Indica se o arquivo contem stream de video copiado")


__all__ = ["LoudnessConfig", "LoudnessResult", "LoudnormStats"]
