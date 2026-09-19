"""Motor de selecao automatica do keyframe mais expressivo (Issue #11).

Implementa ``KeyframeSelector``: amostra frames do video, descarta os
degradados (borrao, olhos fechados, iluminacao pobre, ausencia de face),
ranqueia os validos por ``composite_score`` ponderado e seleciona o top-N
respeitando a diversidade temporal minima (``min_candidate_distance_ms``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from video_engine.thumbnail.face_analyzer import FaceAnalyzer, LandmarkFaceAnalyzer
from video_engine.thumbnail.frame_extractor import SampledFrame, VideoFrameExtractor, compute_uniform_timestamps
from video_engine.thumbnail.lighting import lighting_score, luminance_channel, luminance_stats
from video_engine.thumbnail.models import (
    FaceMetrics,
    FrameMetrics,
    KeyframeCandidate,
    KeyframeSelectorConfig,
    KeyframeSelectorResult,
    RejectionReason,
)
from video_engine.thumbnail.sharpness import is_blurry, laplacian_variance

logger = logging.getLogger(__name__)


class KeyframeSelector:
    """Inspeciona frames de um video e ranqueia os melhores candidatos.

    Args:
        config: Configuracao de descarte, amostragem e pesos do score.
        face_analyzer: Analisador facial injetavel (determinismo em testes).
        frame_extractor: Extrator de frames injetavel (PyAV por padrao).
    """

    def __init__(
        self,
        config: Optional[KeyframeSelectorConfig] = None,
        face_analyzer: Optional[FaceAnalyzer] = None,
        frame_extractor: Optional[VideoFrameExtractor] = None,
    ) -> None:
        self.config = config or KeyframeSelectorConfig()
        self.face_analyzer = face_analyzer or LandmarkFaceAnalyzer(
            min_eye_openness=self.config.min_eye_openness
        )
        self.frame_extractor = frame_extractor or VideoFrameExtractor()

    # ------------------------------------------------------------------ #
    # Avaliacao de um unico frame
    # ------------------------------------------------------------------ #
    def evaluate_frame_array(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int = 0,
        frame_index: int = 0,
    ) -> FrameMetrics:
        """Avalia nitidez, iluminacao e expressividade de um frame RGB.

        Raises:
            ValueError: se ``frame_rgb`` for nulo, vazio ou de dimensoes
                incorretas (esperado HxWx3).
        """
        array = np.asarray(frame_rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
            raise ValueError("Array de frame vazio ou dimensões inválidas")
        if not np.isfinite(array).all():
            raise ValueError("Array de frame contém valores NaN/Inf")

        luma = luminance_channel(array)
        variance = laplacian_variance(array)
        blurry = is_blurry(variance, self.config.min_sharpness_threshold)
        is_poor, is_dark, is_over = self._lighting_state(array)
        lighting = lighting_score(
            array,
            min_luminance=self.config.min_luminance,
            max_luminance=self.config.max_luminance,
        )
        face = self.face_analyzer.analyze(array, timestamp_ms)

        reasons = self._rejection_reasons(blurry, face, is_dark, is_over)
        composite = self._composite_score(variance, lighting, face)

        return FrameMetrics(
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            sharpness_variance=variance,
            is_blurry=blurry,
            luminance_mean=float(np.mean(luma)),
            luminance_std=float(np.std(luma)),
            lighting_score=lighting,
            is_poor_lighting=is_poor,
            face=face,
            composite_score=composite,
            is_valid=not reasons,
            rejection_reasons=reasons,
        )

    # ------------------------------------------------------------------ #
    # Selecao de keyframes de um video
    # ------------------------------------------------------------------ #
    def select_best_keyframes(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> KeyframeSelectorResult:
        """Seleciona os melhores keyframes de ``video_path``.

        Raises:
            FileNotFoundError: se ``video_path`` nao existir.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de video nao encontrado: {path}")

        duration_ms = self.frame_extractor.duration_ms(path)
        interval_ms = int(round(self.config.sample_interval_s * 1000))
        start_ms = int(round(self.config.start_offset_s * 1000))
        end_ms = int(round(self.config.end_offset_s * 1000))
        timestamps = compute_uniform_timestamps(
            duration_ms=duration_ms,
            sample_interval_ms=interval_ms,
            start_offset_ms=start_ms,
            end_offset_ms=end_ms,
            min_samples=3,
        )
        sampled = self.frame_extractor.extract_frames(path, timestamps)
        frames_by_timestamp: Dict[int, SampledFrame] = {
            frame.timestamp_ms: frame for frame in sampled
        }

        valid: List[FrameMetrics] = []
        counters = {
            "blurry": 0,
            "closed_eyes": 0,
            "lighting": 0,
            "no_face": 0,
        }

        for frame in sampled:
            metrics = self.evaluate_frame_array(
                frame.frame,
                timestamp_ms=frame.timestamp_ms,
                frame_index=frame.frame_index,
            )
            if not metrics.is_valid:
                self._count_rejections(metrics, counters)
                continue
            valid.append(metrics)

        ranked = sorted(valid, key=lambda m: -m.composite_score)
        selected = self._filter_temporal_diversity(ranked)

        candidates: List[KeyframeCandidate] = []
        for rank, metrics in enumerate(selected, start=1):
            image_path = None
            if output_dir is not None:
                image_path = self._persist_frame(
                    output_dir,
                    frames_by_timestamp.get(metrics.timestamp_ms),
                )
            candidates.append(
                KeyframeCandidate(
                    rank=rank,
                    timestamp_ms=metrics.timestamp_ms,
                    frame_index=metrics.frame_index,
                    score=metrics.composite_score,
                    metrics=metrics,
                    image_path=image_path,
                )
            )

        return KeyframeSelectorResult(
            video_path=str(path),
            video_duration_ms=duration_ms,
            total_frames_sampled=len(sampled),
            valid_frames_count=len(valid),
            discarded_blurry_count=counters["blurry"],
            discarded_closed_eyes_count=counters["closed_eyes"],
            discarded_lighting_count=counters["lighting"],
            discarded_no_face_count=counters["no_face"],
            top_candidates=candidates,
        )

    # ------------------------------------------------------------------ #
    # Helpers de avaliacao
    # ------------------------------------------------------------------ #
    def _lighting_state(self, array: np.ndarray) -> tuple:
        mean, _ = luminance_stats(array)
        is_dark = mean < self.config.min_luminance
        is_over = mean > self.config.max_luminance
        return (is_dark or is_over), is_dark, is_over

    def _rejection_reasons(
        self,
        blurry: bool,
        face: FaceMetrics,
        is_dark: bool,
        is_over: bool,
    ) -> List[RejectionReason]:
        reasons: List[RejectionReason] = []
        if blurry and self.config.discard_blurry:
            reasons.append(RejectionReason.BLURRY)
        if face.eyes_closed and self.config.discard_closed_eyes:
            reasons.append(RejectionReason.CLOSED_EYES)
        if not face.detected and self.config.require_face:
            reasons.append(RejectionReason.NO_FACE_DETECTED)
        if is_dark and self.config.discard_poor_lighting:
            reasons.append(RejectionReason.POOR_LIGHTING_DARK)
        if is_over and self.config.discard_poor_lighting:
            reasons.append(RejectionReason.POOR_LIGHTING_OVEREXPOSED)
        return reasons

    def _composite_score(self, variance: float, lighting: float, face: FaceMetrics) -> float:
        """Score ponderado normalizado (0.0-1.0) dos criterios disponiveis.

        Frames validos tem ``variance >= min_sharpness_threshold``, portanto a
        componente de nitidez e normalizada entre o limiar e o triplo do limiar
        para diferenciar frames mais ou menos nididos. Componentes indisponiveis
        (ex.: face ausente) tem seus pesos renormalizados para manter soma 1.
        """
        threshold = self.config.min_sharpness_threshold
        sharpness_range = max(threshold * 3.0 - threshold, 1e-9)
        sharpness_norm = float(
            np.clip((variance - threshold) / sharpness_range, 0.0, 1.0)
        )

        components = [
            ("sharpness", sharpness_norm, self.config.weight_sharpness),
            ("expression", face.expression_intensity, self.config.weight_expression),
            ("lighting", lighting, self.config.weight_lighting),
        ]
        if face.detected and face.bounding_box is not None:
            area = face.bounding_box.width * face.bounding_box.height
            prominence = float(np.clip(face.confidence * area * 2.0, 0.0, 1.0))
            components.append(("face", prominence, self.config.weight_face_prominence))

        total_weight = sum(weight for _, _, weight in components)
        if total_weight <= 0:
            return 0.0
        score = sum(value * weight for _, value, weight in components) / total_weight
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _count_rejections(metrics: FrameMetrics, counters: dict) -> None:
        for reason in metrics.rejection_reasons:
            if reason == RejectionReason.BLURRY:
                counters["blurry"] += 1
            elif reason == RejectionReason.CLOSED_EYES:
                counters["closed_eyes"] += 1
            elif reason in (
                RejectionReason.POOR_LIGHTING_DARK,
                RejectionReason.POOR_LIGHTING_OVEREXPOSED,
            ):
                counters["lighting"] += 1
            elif reason == RejectionReason.NO_FACE_DETECTED:
                counters["no_face"] += 1

    def _filter_temporal_diversity(
        self,
        ranked: List[FrameMetrics],
    ) -> List[FrameMetrics]:
        """Seleciona os top-N garantindo distância temporal minima.

        Percorre os candidatos ordenados por score decrescente e mantem um
        frame somente se estiver a pelo menos ``min_candidate_distance_ms`` do
        ultimo selecionado. Nao reordena: preserva o ranqueamento por score.
        """
        selected: List[FrameMetrics] = []
        for metrics in ranked:
            if len(selected) >= self.config.top_n:
                break
            if any(
                abs(metrics.timestamp_ms - prev.timestamp_ms) < self.config.min_candidate_distance_ms
                for prev in selected
            ):
                continue
            selected.append(metrics)
        return selected

    # ------------------------------------------------------------------ #
    # Persistencia de frames
    # ------------------------------------------------------------------ #
    def _persist_frame(
        self,
        output_dir: Union[str, Path],
        frame: Optional[SampledFrame],
    ) -> Optional[str]:
        if frame is None:
            return None
        try:
            destination = Path(output_dir) / f"keyframe_{frame.timestamp_ms:07d}ms.png"
            path = self.frame_extractor.save_frame(frame.frame, destination)
            return str(path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Falha ao salvar frame do keyframe: %s", exc)
            return None


__all__ = ["KeyframeSelector"]
