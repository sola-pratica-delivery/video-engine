"""Analise facial e de expressividade para thumbnails (Issue #11).

Define a interface extensivel ``FaceAnalyzer`` (aberta para enriquecimento
multimodal via Gemini no futuro) e a implementacao local deterministica
``LandmarkFaceAnalyzer``, que calcula abertura ocular (EAR) e articulacao
labial (MAR) a partir de landmarks faciais de 68 pontos (formato dlib).

A extracao dos landmarks e injetavel (``landmark_detector``), permitindo
mapear futuramente modelos ONNX ultralight (ex.: a face), hoje torna a suite
de testes 100% deterministica e offline com landmarks sinteticos.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from video_engine.thumbnail.models import FaceBoundingBox, FaceMetrics

# Indices dos landmarks de 68 pontos (formato dlib)
_LEFT_EYE = (36, 37, 38, 39, 40, 41)
_RIGHT_EYE = (42, 43, 44, 45, 46, 47)
_MOUTH = (48, 51, 54, 57)  # canto esq., topo, canto dir., base

LandmarkLike = Union[np.ndarray, Sequence[Sequence[float]], Dict[int, Sequence[float]]]


def _to_points(landmarks: LandmarkLike) -> np.ndarray:
    """Normaliza landmarks para um array (N, 2) (pontos ausentes = NaN)."""
    if isinstance(landmarks, dict):
        if not landmarks:
            return np.empty((0, 2), dtype=np.float64)
        size = max(max(landmarks) + 1, 58)
        array = np.full((size, 2), np.nan, dtype=np.float64)
        for index, point in landmarks.items():
            array[index] = point
        return array
    array = np.asarray(landmarks, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Landmarks devem ser um array (N, 2) ou dict {indice: (x, y)}")
    return array


def _dist(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


class FaceAnalyzer(ABC):
    """Interface extensivel de analise facial de frames.

    Implementacoes podem agregar heuristica offline deterministica
    (``LandmarkFaceAnalyzer``) ou enriquecimento multimodal externo
    (ex.: Gemini), desde que produzam um :class:`FaceMetrics`.
    """

    @abstractmethod
    def analyze(self, frame_rgb: np.ndarray, timestamp_ms: int = 0) -> FaceMetrics:
        """Analisa um frame RGB e retorna suas metricas faciais."""


class LandmarkFaceAnalyzer(FaceAnalyzer):
    """Analisa olhos (EAR) e boca (MAR) a partir de landmarks de 68 pontos.

    Args:
        min_eye_openness: Limiar de ``eye_openness`` abaixo do qual os olhos
            sao considerados fechados (piscada).
        reference_eye_ear: EAR de referencia de olho totalmente aberto.
        closed_eye_ear: EAR tipico de olho fechado (piso de normalizacao).
        neutral_mouth_ratio: MAR de boca relaxada (piso da articulacao).
        max_expression_mar: MAR de boca totalmente articulada (teto).
        landmark_detector: Callable opcional ``frame_rgb -> landmarks``.
            Quando ``None`` (sem modelo), ``analyze`` reporta ``detected=False``.
        confidence: Confianca padrao das deteccoes.
    """

    def __init__(
        self,
        min_eye_openness: float = 0.25,
        reference_eye_ear: float = 0.35,
        closed_eye_ear: float = 0.12,
        neutral_mouth_ratio: float = 0.15,
        max_expression_mar: float = 0.45,
        landmark_detector: Optional[Callable[[np.ndarray], Optional[LandmarkLike]]] = None,
        confidence: float = 0.95,
    ) -> None:
        self.min_eye_openness = min_eye_openness
        self.reference_eye_ear = reference_eye_ear
        self.closed_eye_ear = closed_eye_ear
        self.neutral_mouth_ratio = neutral_mouth_ratio
        self.max_expression_mar = max_expression_mar
        self._landmark_detector = landmark_detector
        self.confidence = confidence

    def detect_landmarks(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extrai landmarks do frame (ou ``None`` quando sem detector)."""
        if self._landmark_detector is None:
            return None
        landmarks = self._landmark_detector(frame_rgb)
        return _to_points(landmarks) if landmarks is not None else None

    def analyze(self, frame_rgb: np.ndarray, timestamp_ms: int = 0) -> FaceMetrics:
        """Analisa o frame: sem detector disponivel, reporta ausencia de face."""
        landmarks = self.detect_landmarks(frame_rgb)
        if landmarks is None or landmarks.shape[0] == 0:
            return FaceMetrics(detected=False)
        frame_shape: Optional[Tuple[int, int]] = None
        if getattr(frame_rgb, "ndim", None) == 3:
            frame_shape = (frame_rgb.shape[0], frame_rgb.shape[1])
        return self.analyze_landmarks(
            landmarks,
            timestamp_ms=timestamp_ms,
            frame_shape=frame_shape,
            confidence=self.confidence,
        )

    def analyze_landmarks(
        self,
        landmarks: LandmarkLike,
        timestamp_ms: int = 0,
        frame_shape: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
    ) -> FaceMetrics:
        """Calcula metricas deterministicas a partir de landmarks.

        ``frame_shape`` normaliza o bounding box; quando omitido, as
        coordenadas dos landmarks sao tratadas como ja normalizadas (0-1).
        """
        points = _to_points(landmarks)
        ear_left = self._eye_aspect_ratio(points, _LEFT_EYE)
        ear_right = self._eye_aspect_ratio(points, _RIGHT_EYE)
        mar = self._mouth_aspect_ratio(points)

        ear = min(ear_left, ear_right)
        eye_openness = _clip(
            (ear - self.closed_eye_ear) / max(self.reference_eye_ear - self.closed_eye_ear, 1e-9)
        )
        eyes_closed = eye_openness < self.min_eye_openness

        mouth_articulation = _clip(
            (mar - self.neutral_mouth_ratio)
            / max(self.max_expression_mar - self.neutral_mouth_ratio, 1e-9)
        )
        expression_intensity = _clip(0.65 * mouth_articulation + 0.35 * eye_openness)

        return FaceMetrics(
            detected=True,
            bounding_box=self._bounding_box(points, frame_shape),
            eye_openness=eye_openness,
            eyes_closed=eyes_closed,
            mouth_articulation=mouth_articulation,
            expression_intensity=expression_intensity,
            confidence=confidence if confidence is not None else self.confidence,
        )

    # ------------------------------------------------------------------ #
    # Formulas EAR / MAR
    # ------------------------------------------------------------------ #
    @staticmethod
    def _eye_aspect_ratio(points: np.ndarray, indices: Tuple[int, int, int, int, int, int]) -> float:
        try:
            p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
        except IndexError:
            return 0.0
        vertical = _dist(p2, p6) + _dist(p3, p5)
        horizontal = _dist(p1, p4)
        if horizontal <= 0:
            return 0.0
        return vertical / (2.0 * horizontal)

    @staticmethod
    def _mouth_aspect_ratio(points: np.ndarray) -> float:
        try:
            left, top, right, bottom = (points[i] for i in _MOUTH)
        except IndexError:
            return 0.0
        width = _dist(left, right)
        height = _dist(top, bottom)
        if width <= 0:
            return 0.0
        return height / width

    @staticmethod
    def _bounding_box(
        points: np.ndarray,
        frame_shape: Optional[Tuple[int, int]],
    ) -> FaceBoundingBox:
        xs = points[:, 0]
        ys = points[:, 1]
        x0 = float(np.nanmin(xs))
        y0 = float(np.nanmin(ys))
        x1 = float(np.nanmax(xs))
        y1 = float(np.nanmax(ys))
        if frame_shape is not None:
            height, width = float(frame_shape[0]), float(frame_shape[1])
            if width > 0:
                x0 /= width
                x1 /= width
            if height > 0:
                y0 /= height
                y1 /= height
        x0, x1 = _clip(x0), _clip(x1)
        y0, y1 = _clip(y0), _clip(y1)
        return FaceBoundingBox(
            x=x0,
            y=y0,
            width=max(0.0, x1 - x0),
            height=max(0.0, y1 - y0),
        )


__all__ = ["FaceAnalyzer", "LandmarkFaceAnalyzer"]
