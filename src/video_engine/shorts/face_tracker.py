"""Rastreador facial com estabilizacao temporal e histerese (Issue #16).

Amostra frames do video em intervalos uniformes via PyAV, executa uma
deteccao facial injetavel e suaviza a trajetoria do centro da face com
filtro EMA (exponential moving average) combinado a zona morta (deadband),
garantindo enquadramento estavel sem tremores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from video_engine.shorts.reframer_models import (
    CropMode,
    FaceTrackingPoint,
    TrackingTrajectory,
)
from video_engine.thumbnail.frame_extractor import VideoFrameExtractor

DEFAULT_CENTER: Tuple[float, float] = (0.5, 0.4)


@dataclass(frozen=True)
class RawFaceSample:
    """Amostra bruta de deteccao facial em um instante do video."""

    timestamp_ms: int
    detected: bool
    x: Optional[float] = None
    y: Optional[float] = None


def smooth_centers(
    samples: Sequence[RawFaceSample],
    alpha: float = 0.15,
    deadband: float = 0.03,
    default_center: Tuple[float, float] = DEFAULT_CENTER,
) -> List[Tuple[float, float]]:
    """Estabiliza a trajetoria do centro da face com deadband + EMA.

    O filtro EMA amortece transicoes rapidas de posicao (movimento de
    grua/pan suave). A zona morta (deadband) mantem a camera estatica
    enquanto o deslocamento suavizado for inferior a ``deadband``, evitando
    micro-vibracoes de respiracao ou fala. Frames sem face preservam a
    ultima posicao suavizada sem deriva da camera.
    """
    hold_x, hold_y = default_center
    ema_x: Optional[float] = None
    ema_y: Optional[float] = None
    had_detection = False

    output: List[Tuple[float, float]] = []
    for sample in samples:
        if sample.detected and sample.x is not None and sample.y is not None:
            raw_x = min(1.0, max(0.0, float(sample.x)))
            raw_y = min(1.0, max(0.0, float(sample.y)))
            if not had_detection:
                ema_x, ema_y = raw_x, raw_y
            else:
                ema_x = ema_x + alpha * (raw_x - ema_x)
                ema_y = ema_y + alpha * (raw_y - ema_y)
            if abs(ema_x - hold_x) >= deadband:
                hold_x = ema_x
            if abs(ema_y - hold_y) >= deadband:
                hold_y = ema_y
            had_detection = True
        else:
            had_detection = False
        output.append((hold_x, hold_y))
    return output


class FaceTracker:
    """Rastreador facial com estabilizacao temporal e histerese.

    A deteccao facial e injetavel via ``face_detector`` (callable que recebe
    um frame RGB HxWx3 uint8 e devolve ``(x, y)`` normalizados do centro da
    face ou ``None``) ou via ``face_analyzer`` compatível com a interface
    ``FaceAnalyzer`` da camada de thumbnails.
    """

    def __init__(
        self,
        sample_interval_ms: int = 200,
        smoothing_factor: float = 0.15,
        deadband_threshold: float = 0.03,
        face_detector: Optional[Callable[[np.ndarray], Optional[Tuple[float, float]]]] = None,
        face_analyzer: Optional[object] = None,
        frame_extractor: Optional[VideoFrameExtractor] = None,
        min_face_detection_ratio: float = 0.25,
    ) -> None:
        self.sample_interval_ms = sample_interval_ms
        self.smoothing_factor = smoothing_factor
        self.deadband_threshold = deadband_threshold
        self.face_detector = face_detector
        self.face_analyzer = face_analyzer
        self._frame_extractor = frame_extractor or VideoFrameExtractor()
        self._min_face_detection_ratio = min_face_detection_ratio

    def track_video(
        self,
        video_path: Union[str, Path],
        start_ms: int = 0,
        end_ms: Optional[int] = None,
    ) -> TrackingTrajectory:
        """Rastreia a face ao longo do video e devolve a trajetoria suavizada.

        Raises:
            FileNotFoundError: se ``video_path`` nao existir.
            ValueError: se ``start_ms`` for negativo ou o arquivo nao possuir
                stream de video.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de video nao encontrado: {path}")
        if start_ms < 0:
            raise ValueError(f"start_ms nao pode ser negativo: {start_ms}")

        extractor = self._frame_extractor
        duration_ms = extractor.duration_ms(path)
        window_end = duration_ms if end_ms is None else min(max(0, end_ms), duration_ms)
        window_start = min(start_ms, window_end)

        timestamps = self._sampling_timestamps(window_start, window_end)
        frames = {
            frame.timestamp_ms: frame
            for frame in extractor.extract_frames(path, timestamps)
        }

        raw_samples: List[RawFaceSample] = []
        for timestamp_ms in timestamps:
            frame = frames.get(timestamp_ms)
            detected = False
            center: Optional[Tuple[float, float]] = None
            if frame is not None:
                center = self._detect_center(frame.frame)
                detected = center is not None
            raw_samples.append(
                RawFaceSample(
                    timestamp_ms=timestamp_ms,
                    detected=detected,
                    x=center[0] if center is not None else None,
                    y=center[1] if center is not None else None,
                )
            )

        smoothed = smooth_centers(
            raw_samples,
            alpha=self.smoothing_factor,
            deadband=self.deadband_threshold,
            default_center=DEFAULT_CENTER,
        )
        return self._build_trajectory(raw_samples, smoothed)

    # ------------------------------------------------------------------ #
    # Helpers internos
    # ------------------------------------------------------------------ #
    def _sampling_timestamps(self, start_ms: int, end_ms: int) -> List[int]:
        """Gera timestamps a cada ``sample_interval_ms`` dentro da janela."""
        timestamps: List[int] = []
        current = start_ms
        while current < end_ms:
            timestamps.append(current)
            current += self.sample_interval_ms
        if not timestamps:
            timestamps = [start_ms]
        return timestamps

    def _detect_center(
        self, frame_rgb: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Executa a deteccao injetavel e normaliza o centro da face."""
        if self.face_detector is not None:
            try:
                result = self.face_detector(frame_rgb)
            except Exception:  # noqa: BLE001 - detector opcional nunca derruba o rastreio
                result = None
            if result is None:
                return None
            raw_x, raw_y = float(result[0]), float(result[1])
            return (min(1.0, max(0.0, raw_x)), min(1.0, max(0.0, raw_y)))

        if self.face_analyzer is not None:
            try:
                metrics = self.face_analyzer.analyze(frame_rgb)
            except Exception:  # noqa: BLE001 - detector opcional nunca derruba o rastreio
                metrics = None
            if metrics is None or not metrics.detected or metrics.bounding_box is None:
                return None
            box = metrics.bounding_box
            center_x = box.x + box.width / 2.0
            center_y = box.y + box.height / 2.0
            return (min(1.0, max(0.0, center_x)), min(1.0, max(0.0, center_y)))

        return None

    def _build_trajectory(
        self,
        raw_samples: Sequence[RawFaceSample],
        smoothed: Sequence[Tuple[float, float]],
    ) -> TrackingTrajectory:
        """Consolida amostras brutas e suavizadas em uma ``TrackingTrajectory``."""
        points: List[FaceTrackingPoint] = []
        detected_count = 0
        smoothed_xs: List[float] = []
        for sample, (smooth_x, _smooth_y) in zip(raw_samples, smoothed):
            points.append(
                FaceTrackingPoint(
                    timestamp_ms=sample.timestamp_ms,
                    face_detected=sample.detected,
                    raw_center_x=DEFAULT_CENTER[0] if sample.x is None else sample.x,
                    raw_center_y=DEFAULT_CENTER[1] if sample.y is None else sample.y,
                    smoothed_center_x=smooth_x,
                    confidence=1.0 if sample.detected else 0.0,
                )
            )
            if sample.detected:
                detected_count += 1
                smoothed_xs.append(smooth_x)

        total = len(raw_samples)
        ratio = (detected_count / total) if total else 0.0
        average_x = float(np.mean(smoothed_xs)) if smoothed_xs else DEFAULT_CENTER[0]
        return TrackingTrajectory(
            points=points,
            face_detected_ratio=ratio,
            average_center_x=average_x,
            dominant_mode=(
                CropMode.SMART_CROP
                if ratio >= self._min_face_detection_ratio
                else CropMode.AESTHETIC_FILL
            ),
        )


__all__ = ["FaceTracker", "RawFaceSample", "smooth_centers"]
