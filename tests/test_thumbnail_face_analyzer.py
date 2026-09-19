"""Testes de analise facial e expressividade (Issue #11)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest

from video_engine.thumbnail.face_analyzer import LandmarkFaceAnalyzer, _to_points
from video_engine.thumbnail.models import FaceMetrics


def _build_synthetic_landmarks(
    left_eye_open: bool = True,
    right_eye_open: bool = True,
    mouth_open: bool = False,
    face_origin: Tuple[float, float] = (100.0, 100.0),
    face_size: float = 200.0,
) -> Dict[int, Tuple[float, float]]:
    """Gera dicionario de 68 landmarks sinteticos com estados controlados."""
    ox, oy = face_origin
    points: Dict[int, Tuple[float, float]] = {}

    # Bounding basico da face (pontos 0 a 16 e contorno)
    points[0] = (ox, oy)
    points[16] = (ox + face_size, oy)
    points[8] = (ox + face_size / 2.0, oy + face_size)  # queixo

    # Olho esquerdo (indices 36-41)
    # p1=36, p2=37, p3=38, p4=39, p5=40, p6=41
    # vertical: (p2-p6) + (p3-p5), horizontal: (p1-p4)
    eye_w = 40.0
    eye_h = 15.0 if left_eye_open else 2.0  # EAR alto vs baixo
    lx = ox + 40.0
    ly = oy + 60.0
    points[36] = (lx, ly)
    points[39] = (lx + eye_w, ly)
    points[37] = (lx + eye_w * 0.33, ly - eye_h / 2.0)
    points[38] = (lx + eye_w * 0.66, ly - eye_h / 2.0)
    points[41] = (lx + eye_w * 0.33, ly + eye_h / 2.0)
    points[40] = (lx + eye_w * 0.66, ly + eye_h / 2.0)

    # Olho direito (indices 42-47)
    # p1=42, p2=43, p3=44, p4=45, p5=46, p6=47
    eye_h_r = 15.0 if right_eye_open else 2.0
    rx = ox + 120.0
    ry = oy + 60.0
    points[42] = (rx, ry)
    points[45] = (rx + eye_w, ry)
    points[43] = (rx + eye_w * 0.33, ry - eye_h_r / 2.0)
    points[44] = (rx + eye_w * 0.66, ry - eye_h_r / 2.0)
    points[47] = (rx + eye_w * 0.33, ry + eye_h_r / 2.0)
    points[46] = (rx + eye_w * 0.66, ry + eye_h_r / 2.0)

    # Boca (indices 48: left, 51: top, 54: right, 57: bottom)
    mouth_w = 60.0
    mouth_h = 28.0 if mouth_open else 5.0  # MAR alto vs neutro
    mx = ox + 70.0
    my = oy + 140.0
    points[48] = (mx, my)
    points[54] = (mx + mouth_w, my)
    points[51] = (mx + mouth_w / 2.0, my - mouth_h / 2.0)
    points[57] = (mx + mouth_w / 2.0, my + mouth_h / 2.0)

    # Preenche outros pontos genericos para cobrir indice 0-57
    for i in range(58):
        if i not in points:
            points[i] = (ox + 10.0, oy + 10.0)

    return points


class TestFaceAnalyzer:
    def test_no_detector_returns_undetected(self) -> None:
        analyzer = LandmarkFaceAnalyzer(landmark_detector=None)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        metrics = analyzer.analyze(frame)

        assert isinstance(metrics, FaceMetrics)
        assert metrics.detected is False
        assert metrics.bounding_box is None

    def test_open_eyes_detection(self) -> None:
        landmarks = _build_synthetic_landmarks(left_eye_open=True, right_eye_open=True)
        analyzer = LandmarkFaceAnalyzer(min_eye_openness=0.25)
        metrics = analyzer.analyze_landmarks(landmarks, frame_shape=(400, 400))

        assert metrics.detected is True
        assert metrics.eyes_closed is False
        assert metrics.eye_openness >= 0.50

    def test_closed_eyes_detection(self) -> None:
        landmarks = _build_synthetic_landmarks(left_eye_open=False, right_eye_open=False)
        analyzer = LandmarkFaceAnalyzer(min_eye_openness=0.25)
        metrics = analyzer.analyze_landmarks(landmarks, frame_shape=(400, 400))

        assert metrics.detected is True
        assert metrics.eyes_closed is True
        assert metrics.eye_openness < 0.25

    def test_one_eye_closed_is_considered_closed(self) -> None:
        # Quando um olho esta piscando/fechado (winking)
        landmarks = _build_synthetic_landmarks(left_eye_open=True, right_eye_open=False)
        analyzer = LandmarkFaceAnalyzer(min_eye_openness=0.25)
        metrics = analyzer.analyze_landmarks(landmarks, frame_shape=(400, 400))

        assert metrics.detected is True
        assert metrics.eyes_closed is True

    def test_articulated_mouth_and_expression_intensity(self) -> None:
        neutral = _build_synthetic_landmarks(left_eye_open=True, right_eye_open=True, mouth_open=False)
        expressive = _build_synthetic_landmarks(left_eye_open=True, right_eye_open=True, mouth_open=True)

        analyzer = LandmarkFaceAnalyzer()
        m_neutral = analyzer.analyze_landmarks(neutral)
        m_expressive = analyzer.analyze_landmarks(expressive)

        assert m_expressive.mouth_articulation > m_neutral.mouth_articulation
        assert m_expressive.expression_intensity > m_neutral.expression_intensity

    def test_bounding_box_normalized(self) -> None:
        landmarks = _build_synthetic_landmarks(face_origin=(50.0, 100.0), face_size=200.0)
        analyzer = LandmarkFaceAnalyzer()
        metrics = analyzer.analyze_landmarks(landmarks, frame_shape=(500, 500))

        bbox = metrics.bounding_box
        assert bbox is not None
        assert 0.0 <= bbox.x <= 1.0
        assert 0.0 <= bbox.y <= 1.0
        assert 0.0 <= bbox.width <= 1.0
        assert 0.0 <= bbox.height <= 1.0
        assert bbox.x == pytest.approx(50.0 / 500.0, abs=0.01)
        assert bbox.y == pytest.approx(100.0 / 500.0, abs=0.01)

    def test_custom_detector_injected_in_analyze(self) -> None:
        landmarks_data = _build_synthetic_landmarks(left_eye_open=True, right_eye_open=True, mouth_open=True)

        def detector(img: np.ndarray) -> dict:
            return landmarks_data

        analyzer = LandmarkFaceAnalyzer(landmark_detector=detector)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        metrics = analyzer.analyze(frame, timestamp_ms=5000)

        assert metrics.detected is True
        assert metrics.eyes_closed is False
        assert metrics.mouth_articulation > 0.50

    def test_invalid_landmarks_raise_error(self) -> None:
        with pytest.raises(ValueError, match="Landmarks devem ser um array"):
            _to_points(np.zeros((10, 3)))  # 3 colunas em vez de 2
