"""Esteira audiovisual de ponta a ponta do worker.

Orquestra a sequencia MediaProbe -> SileroVadDetector -> MediaSplicer
(micro-crossfades de 15ms) -> LoudnessNormalizer (EBU R128, -14 LUFS / -1.0
dBTP), gravando o artefato final em ``output_dir`` com nomenclatura
``{upload_id}_processed.mp4`` e garantindo a limpeza dos temporarios
intermediários.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from video_engine.audio.loudness import LoudnessNormalizer
from video_engine.audio.silero_vad import SileroVadDetector
from video_engine.editing.media_probe import MediaProbe
from video_engine.editing.media_splicer import MediaSplicer
from video_engine.worker.models import (
    LoudnessReport,
    ProcessingResult,
    VideoProcessingJobData,
    WorkerConfig,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Erro de negocios da esteira audiovisual com codigo de diagnostico."""

    error_code = "MEDIA_PROCESSING_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class InputFileInvalidError(PipelineError):
    """Arquivo de entrada inexistente, corrompido ou ilegivel."""

    error_code = "INPUT_FILE_INVALID"


class NoAudioStreamError(PipelineError):
    """Arquivo sem trilha de audio (nunca processar GPU/CPU sem audio)."""

    error_code = "NO_AUDIO_STREAM"


class NoSpeechDetectedError(PipelineError):
    """Nenhuma fala detectada pelo VAD (video 100% silencioso)."""

    error_code = "NO_SPEECH_DETECTED"


class VideoProcessingPipeline:
    """Orquestra a esteira audiovisual encadeada modular a modular."""

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        probe: Optional[MediaProbe] = None,
        vad: Optional[SileroVadDetector] = None,
        splicer: Optional[MediaSplicer] = None,
        normalizer: Optional[LoudnessNormalizer] = None,
    ) -> None:
        self.config = config or WorkerConfig()
        self.probe = probe or MediaProbe()
        self.vad = vad or SileroVadDetector()
        self.splicer = splicer or MediaSplicer()
        self.normalizer = normalizer or LoudnessNormalizer()

    def process(self, job: VideoProcessingJobData) -> ProcessingResult:
        """Executa a esteira completa e retorna o relatorio final.

        Raises:
            InputFileInvalidError: entrada inexistente/corrompida.
            NoAudioStreamError: entrada sem trilha de audio.
            NoSpeechDetectedError: nenhuma fala detectada no video.
            PipelineError: qualquer outro erro de processamento de midia.
        """
        source = Path(job.file_path)
        if not source.is_file():
            raise InputFileInvalidError(f"Arquivo de entrada nao encontrado: {source}")

        try:
            info = self.probe.probe(source)
        except FileNotFoundError as exc:
            raise InputFileInvalidError(f"Arquivo de entrada invalido: {exc}") from exc
        except RuntimeError as exc:
            raise InputFileInvalidError(f"Arquivo de entrada corrompido ou ilegivel: {exc}") from exc

        if not info.has_audio:
            raise NoAudioStreamError("Entrada sem stream de audio")

        vad_result = self.vad.detect_file(source)
        if not vad_result.speech_segments:
            raise NoSpeechDetectedError("Nenhum segmento de fala detectado no video")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{job.upload_id}_processed.mp4"

        temp_base = Path(self.config.temp_dir) if self.config.temp_dir else None
        if temp_base is not None:
            temp_base.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            dir=temp_base, prefix="video_engine_pipeline_"
        ) as tmp_dir:
            spliced_path = Path(tmp_dir) / "spliced.mp4"
            splice_result = self.splicer.splice_file(source, spliced_path, vad_result.speech_segments)
            loudness = self.normalizer.normalize_file(spliced_path, final_path)

        speech_total_ms = sum(
            (seg.end_ms - seg.start_ms) for seg in vad_result.speech_segments
        )
        silence_removed_ms = max(0, vad_result.total_duration_ms - speech_total_ms)

        return ProcessingResult(
            output_path=str(final_path),
            duration_sec=splice_result.total_duration_ms / 1000.0,
            speech_segments_count=len(vad_result.speech_segments),
            silence_removed_ms=silence_removed_ms,
            loudness=LoudnessReport(
                integrated_lufs=loudness.measured_output.input_i,
                true_peak_dbtp=loudness.measured_output.input_tp,
                lra=loudness.measured_output.input_lra,
            ),
        )


__all__ = [
    "InputFileInvalidError",
    "NoAudioStreamError",
    "NoSpeechDetectedError",
    "PipelineError",
    "VideoProcessingPipeline",
]
