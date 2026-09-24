"""Esteira audiovisual de ponta a ponta do worker.

Orquestra a sequencia MediaProbe -> SileroVadDetector -> MediaSplicer
(micro-crossfades de 15ms) -> DynamicZoomProcessor (condicional, quando
``dynamicZoom`` habilitado) -> LoudnessNormalizer (EBU R128, -14 LUFS / -1.0
dBTP), gravando o artefato final em ``output_dir`` com nomenclatura
``{upload_id}_processed.mp4`` e garantindo a limpeza dos temporarios
intermediários.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from video_engine.audio.bgm_ducking import BgmDucker
from video_engine.audio.loudness import LoudnessNormalizer
from video_engine.audio.silero_vad import SileroVadDetector
from video_engine.editing.media_probe import MediaProbe
from video_engine.editing.media_splicer import MediaSplicer
from video_engine.thumbnail.models import HeadlineConfig, KeyframeSelectorResult
from video_engine.video.zoom import DynamicZoomProcessor
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


def parse_dynamic_zoom(value: Any) -> bool:
    """Interpreta representações heterogêneas do flag de dynamic zoom.

    ``true``, ``"true"``, ``"1"``, ``1`` -> ``True``
    ``false``, ``"false"``, ``"0"``, ``0``, ``None`` -> ``False``
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "1"}


def parse_generate_shorts(value: Any, default: bool = False) -> bool:
    """Interpreta representações heterogêneas do flag de geracao de shorts.

    ``true``, ``"true"``, ``"1"``, ``1``, ``"yes"`` -> ``True``
    ``false``, ``"false"``, ``"0"``, ``0``, ``None`` -> ``default``
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "1", "yes"}


class VideoProcessingPipeline:
    """Orquestra a esteira audiovisual encadeada modular a modular."""

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        probe: Optional[MediaProbe] = None,
        vad: Optional[SileroVadDetector] = None,
        splicer: Optional[MediaSplicer] = None,
        normalizer: Optional[LoudnessNormalizer] = None,
        bgm_ducker: Optional[BgmDucker] = None,
        zoom_processor: Optional[DynamicZoomProcessor] = None,
        transcriber: Optional[Any] = None,
        keyframe_selector: Optional[Any] = None,
        segmenter: Optional[Any] = None,
        composer: Optional[Any] = None,
        shorts_detector: Optional[Any] = None,
        shorts_renderer: Optional[Any] = None,
        headline_synthesizer: Optional[Any] = None,
    ) -> None:
        self.config = config or WorkerConfig()
        self.probe = probe or MediaProbe()
        self.vad = vad or SileroVadDetector()
        self.splicer = splicer or MediaSplicer()
        self.normalizer = normalizer or LoudnessNormalizer()
        self.bgm_ducker = bgm_ducker or BgmDucker()
        self.zoom_processor = zoom_processor or DynamicZoomProcessor()
        self._transcriber = transcriber
        self._keyframe_selector = keyframe_selector
        self._segmenter = segmenter
        self._composer = composer
        self._shorts_detector = shorts_detector
        self._shorts_renderer = shorts_renderer
        self._headline_synthesizer = headline_synthesizer
        self._frame_extractor = None

    @property
    def headline_synthesizer(self) -> Any:
        """Sintetizador de headlines lazy: injetado ou instanciado sob demanda."""
        if self._headline_synthesizer is None:
            from video_engine.thumbnail.headline_synthesizer import GeminiHeadlineSynthesizer

            self._headline_synthesizer = GeminiHeadlineSynthesizer()
        return self._headline_synthesizer

    @property
    def transcriber(self) -> Any:
        """Transcritor lazy: injetado via construtor ou carregado sob demanda."""
        if self._transcriber is None:
            from video_engine.captions.transcriber import WhisperTranscriber

            self._transcriber = WhisperTranscriber()
        return self._transcriber

    @property
    def keyframe_selector(self) -> Any:
        """Seletor de keyframes lazy: injetado ou instanciado sob demanda."""

        if self._keyframe_selector is None:
            from video_engine.thumbnail.selector import KeyframeSelector

            self._keyframe_selector = KeyframeSelector()
        return self._keyframe_selector

    @property
    def segmenter(self) -> Any:
        """Segmentador lazy: injetado ou ``OnnxBackgroundSegmenter`` sob demanda."""
        if self._segmenter is None:
            from video_engine.thumbnail.segmenter import OnnxBackgroundSegmenter

            self._segmenter = OnnxBackgroundSegmenter()
        return self._segmenter

    @property
    def composer(self) -> Any:
        """Compositor de thumbnail lazy: injetado ou instanciado sob demanda."""
        if self._composer is None:
            from video_engine.thumbnail.composer import ThumbnailComposer

            self._composer = ThumbnailComposer()
        return self._composer

    @property
    def frame_extractor(self) -> Any:
        """Extrator de frames lazy para extracao de emergencia de keyframes."""
        if self._frame_extractor is None:
            from video_engine.thumbnail.frame_extractor import VideoFrameExtractor

            self._frame_extractor = VideoFrameExtractor()
        return self._frame_extractor

    @property
    def shorts_detector(self) -> Any:
        """Detector de ganchos virais lazy: injetado ou carregado sob demanda."""
        if self._shorts_detector is None:
            from video_engine.shorts.hook_detector import HookDetector

            self._shorts_detector = HookDetector()
        return self._shorts_detector

    @property
    def shorts_renderer(self) -> Any:
        """Renderer de Shorts 9:16 lazy: injetado ou carregado sob demanda."""
        if self._shorts_renderer is None:
            from video_engine.shorts.shorts_renderer import ShortsRenderer

            self._shorts_renderer = ShortsRenderer()
        return self._shorts_renderer

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

            dynamic_zoom_applied = False
            zoom_shots_count = 0
            zoom_strategy: Optional[str] = None
            zoom_target = spliced_path
            transcription_result = None
            if self._is_dynamic_zoom_enabled(job.metadata):
                try:
                    transcription_result = self.transcriber.transcribe_file(spliced_path)
                except Exception as exc:
                    logger.warning(
                        "Falha na transcricao com Whisper; prosseguindo com zoom heuristico: %s",
                        exc,
                    )

                zoomed_path = Path(tmp_dir) / "zoomed.mp4"
                zoom_result = self.zoom_processor.apply_zoom(
                    spliced_path,
                    zoomed_path,
                    pause_intervals=vad_result.silence_segments,
                    transcription=transcription_result,
                )
                zoom_target = zoomed_path
                dynamic_zoom_applied = True
                zoom_shots_count = zoom_result.zoom_shots_count
                zoom_strategy = zoom_result.strategy_used

            shorts: List[Dict[str, Any]] = []
            if self._is_shorts_enabled(job.metadata):
                shorts = self._generate_shorts(job, spliced_path, transcription_result)

            bgm_candidate = job.metadata.get("bgm_path") or job.metadata.get("bgmPath")
            if bgm_candidate and Path(bgm_candidate).is_file():
                ducked_path = Path(tmp_dir) / "ducked.mp4"
                self.bgm_ducker.mix(zoom_target, bgm_candidate, ducked_path)
                target_for_loudness = ducked_path
            else:
                target_for_loudness = zoom_target

            loudness = self.normalizer.normalize_file(target_for_loudness, final_path)

            thumbnail_path, thumbnail_score = self._generate_thumbnail(
                job, zoom_target, Path(tmp_dir), transcription_result
            )

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
            dynamic_zoom_applied=dynamic_zoom_applied,
            zoom_shots_count=zoom_shots_count,
            zoom_strategy=zoom_strategy,
            thumbnail_path=thumbnail_path,
            thumbnail_score=thumbnail_score,
            shorts=shorts,
        )

    @staticmethod
    def _is_dynamic_zoom_enabled(metadata: Dict[str, Any]) -> bool:
        """Resolve o flag ``dynamicZoom``/``dynamic_zoom`` com valores heterogêneos."""
        raw_value = metadata.get("dynamicZoom", metadata.get("dynamic_zoom"))
        return parse_dynamic_zoom(raw_value)

    def _is_shorts_enabled(self, metadata: Dict[str, Any]) -> bool:
        """Resolve a ativacao de shorts com override por metadata ou config."""
        for key in ("generate_shorts", "generateShorts", "shorts"):
            if key in metadata:
                return parse_generate_shorts(
                    metadata[key], default=self.config.generate_shorts
                )
        return self.config.generate_shorts

    # ------------------------------------------------------------------ #
    # Shorts 9:16 (Issue #24): deteccao de ganchos + renderizacao vertical
    # ------------------------------------------------------------------ #
    def _generate_shorts(
        self,
        job: VideoProcessingJobData,
        video_path: Path,
        transcription_result: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """Gera os artefatos de Shorts com isolamento total de falhas.

        Qualquer erro em deteccao, corte ou renderizacao e logado como warning
        e resulta em lista vazia (ou parcial), sem interromper a conclusao do
        video principal nem a geracao da thumbnail.
        """
        try:
            return self._do_generate_shorts(job, video_path, transcription_result)
        except Exception as exc:  # noqa: BLE001 - isolamento de falhas por design
            logger.warning(
                "Falha geral na geracao de shorts do job %s; prosseguindo sem shorts: %s",
                job.job_id,
                exc,
                exc_info=True,
            )
            return []

    def _do_generate_shorts(
        self,
        job: VideoProcessingJobData,
        video_path: Path,
        transcription_result: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """Reusa/dispara a transcricao, detecta ganchos e renderiza os cortes."""
        if transcription_result is None:
            try:
                transcription_result = self.transcriber.transcribe_file(video_path)
            except Exception as exc:  # noqa: BLE001 - aborta so a geracao de shorts
                logger.warning(
                    "Falha na transcricao para geracao de shorts do job %s; "
                    "abortando geracao: %s",
                    job.job_id,
                    exc,
                    exc_info=True,
                )
                return []

        try:
            detection = self.shorts_detector.detect_cuts(
                transcription_result,
                audio_source=video_path,
            )
        except Exception as exc:  # noqa: BLE001 - isolamento do detector
            logger.warning(
                "Falha na deteccao de ganchos virais do job %s; "
                "prosseguindo sem shorts: %s",
                job.job_id,
                exc,
                exc_info=True,
            )
            return []

        cuts = detection.cuts or []
        if not cuts:
            logger.info(
                "Nenhum gancho viral detectado para o job %s; sem shorts gerados.",
                job.job_id,
            )
            return []

        logger.info(
            "Detectados %d cortes virais para o job %s; renderizando shorts.",
            len(cuts),
            job.job_id,
        )

        output_dir = Path(self.config.output_dir)
        shorts: List[Dict[str, Any]] = []
        for index, cut in enumerate(cuts, start=1):
            try:
                package = self._render_one_short(
                    job, cut, index, video_path, output_dir, transcription_result
                )
                if package is not None:
                    shorts.append(package)
            except Exception as exc:  # noqa: BLE001 - isola cada corte individual
                logger.warning(
                    "Falha ao renderizar o short %d do job %s; ignorando corte: %s",
                    index,
                    job.job_id,
                    exc,
                    exc_info=True,
                )
        return shorts

    def _render_one_short(
        self,
        job: VideoProcessingJobData,
        cut: Any,
        index: int,
        video_path: Path,
        output_dir: Path,
        transcription_result: Optional[Any],
    ) -> Dict[str, Any]:
        """Renderiza um corte renomeado na nomenclatura padrao do contrato."""
        renamed = cut.model_copy(update={"id": f"{job.upload_id}_short_{index}"})
        package = self.shorts_renderer.render_cut(
            video_path,
            renamed,
            output_dir,
            transcription=transcription_result,
        )
        return {
            "id": package.id,
            "videoPath": package.video_path,
            "metadataPath": package.metadata_path,
            "title": package.title,
            "description": package.description,
            "hashtags": package.hashtags,
            "durationSec": package.duration_sec,
            "startMs": package.start_ms,
            "endMs": package.end_ms,
            "viralityScore": package.virality_score,
            "resolution": package.resolution,
        }

    # ------------------------------------------------------------------ #
    # Geracao de thumbnail (Issue #21)
    # ------------------------------------------------------------------ #
    def _generate_thumbnail(
        self,
        job: VideoProcessingJobData,
        video_path: Path,
        tmp_dir: Path,
        transcription_result: Optional[Any] = None,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Extrai keyframe, recorta apresentador (com fallback) e compoe capa.

        Isolamento de falhas: qualquer erro (segmentador, selector, I/O,
        Pillow) e logado como warning e retorna ``(None, None)`` sem derrubar a
        esteira nem interromper a geracao do video principal.
        """
        try:
            return self._do_generate_thumbnail(job, video_path, tmp_dir, transcription_result)
        except Exception as exc:  # noqa: BLE001 - isolamento de falhas
            logger.warning(
                "Falha na geracao da thumbnail; prosseguindo sem capa: %s",
                exc,
                exc_info=True,
            )
            return (None, None)

    def _do_generate_thumbnail(
        self,
        job: VideoProcessingJobData,
        video_path: Path,
        tmp_dir: Path,
        transcription_result: Optional[Any],
    ) -> Tuple[Optional[str], Optional[float]]:
        headline_cfg = HeadlineConfig(  # sanitizado antes: nunca excede 5 palavras
            text=self._resolve_headline(job, transcription_result),
            strict_word_limit=False,
        )

        keyframes_dir = Path(tmp_dir) / "keyframes"
        score: Optional[float] = None
        frame_rgb: Optional[np.ndarray] = None

        selection: Optional[KeyframeSelectorResult] = None
        try:
            selection = self.keyframe_selector.select_best_keyframes(
                video_path, output_dir=str(keyframes_dir)
            )
        except Exception as exc:  # noqa: BLE001 - fallback controlado
            logger.warning("Selecao de keyframes indisponivel (%s); usando emergencia", exc)

        if selection is not None and selection.top_candidates:
            candidate = selection.top_candidates[0]
            score = candidate.score
            frame_rgb = self._load_candidate_frame(video_path, candidate, keyframes_dir)

        segmented_subject = self._segment_or_fallback(frame_rgb) if frame_rgb is not None else None

        if frame_rgb is None:
            frame_rgb = self._emergency_frame(video_path)
            score = 0.0

        thumbnail_path = Path(self.config.output_dir) / f"{job.upload_id}_thumbnail.jpg"
        if segmented_subject is not None:
            composition = self.composer.compose(
                segmented_subject, headline_cfg, output_path=thumbnail_path
            )
        else:
            composition = self.composer.compose_from_frame(
                frame_rgb, headline_cfg, output_path=thumbnail_path
            )
        return (str(composition.output_path), score)

    def _segment_or_fallback(self, frame_rgb: np.ndarray) -> Optional[Any]:
        """Segmenta o sujeito com fallback para o frame bruto em caso de falha."""
        try:
            segmented = self.segmenter.segment(frame_rgb)
        except Exception as exc:  # noqa: BLE001 - fallback controlado
            logger.warning(
                "Segmentacao falhou (%s); compondo capa com frame bruto", exc
            )
            return None
        if segmented.alpha_mask is None or segmented.alpha_mask.max() == 0:
            logger.warning(
                "Segmentador produziu mascara alfa vazia; compondo capa com frame bruto"
            )
            return None
        return segmented

    def _load_candidate_frame(
        self,
        video_path: Path,
        candidate: Any,
        keyframes_dir: Path,
    ) -> Optional[np.ndarray]:
        """Carrega o frame do keyframe selecionado (PNG persistido ou re-decodificacao)."""
        if candidate.image_path and Path(candidate.image_path).is_file():
            return self._read_image_rgb(Path(candidate.image_path))
        fallback = keyframes_dir / f"keyframe_{candidate.timestamp_ms:07d}ms.png"
        if fallback.is_file():
            return self._read_image_rgb(fallback)
        return np.asarray(self.frame_extractor.extract_at(video_path, candidate.timestamp_ms))

    def _emergency_frame(self, video_path: Path) -> np.ndarray:
        """Extracao de emergencia de frame unico (0s/1s) quando o selector falha."""
        for timestamp_ms in (0, 1000):
            try:
                return np.asarray(self.frame_extractor.extract_at(video_path, timestamp_ms))
            except Exception as exc:  # noqa: BLE001 - tentativa seguinte
                logger.warning(
                    "Falha ao extrair frame de emergencia em %dms: %s", timestamp_ms, exc
                )
        raise RuntimeError("Nao foi possivel extrair nenhum frame de emergencia")

    @staticmethod
    def _read_image_rgb(path: Path) -> np.ndarray:
        """Le imagem em disco como array RGB (HxWx3, uint8)."""
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def _resolve_headline(
        self,
        job: VideoProcessingJobData,
        transcription_result: Optional[Any],
    ) -> str:
        """Resolve a headline com prioridade metadata explicita > Gemini Synthesizer > fallback."""
        metadata = job.metadata or {}
        for key in ("thumbnail_headline", "thumbnailHeadline", "headline"):
            raw = metadata.get(key)
            if isinstance(raw, str) and raw.strip():
                return self._sanitize_headline(raw)

        title = metadata.get("title")
        try:
            return self.headline_synthesizer.synthesize(
                title=title,
                transcription=transcription_result,
            )
        except Exception as exc:
            logger.warning(
                "Falha ao sintetizar headline com Gemini (%s); aplicando fallback no pipeline.",
                exc,
            )
            return self._fallback_headline(title, transcription_result)

    def _fallback_headline(
        self,
        title: Optional[str],
        transcription_result: Optional[Any],
    ) -> str:
        """Fallback seguro derivado de titulo/transcricao quando a sintese falha."""
        try:
            return self.headline_synthesizer.fallback_headline(
                title=title,
                transcription=transcription_result,
            )
        except Exception:
            return self._sanitize_headline(title or "ASSISTA AGORA")

    @staticmethod
    def _sanitize_headline(text: str) -> str:
        """Normaliza e limita o texto da headline a 5 palavras (mobile CTR)."""
        words = str(text).split()
        if not words:
            return "ASSISTA AGORA"
        return " ".join(words[:5])




__all__ = [
    "InputFileInvalidError",
    "NoAudioStreamError",
    "NoSpeechDetectedError",
    "PipelineError",
    "VideoProcessingPipeline",
    "parse_dynamic_zoom",
    "parse_generate_shorts",
]
