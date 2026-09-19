"""Testes do seletor e ranqueador de keyframes (Issue #11)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from video_engine.thumbnail.frame_extractor import SampledFrame, VideoFrameExtractor
from video_engine.thumbnail.models import (
    FaceBoundingBox,
    FaceMetrics,
    KeyframeSelectorConfig,
    KeyframeSelectorResult,
    RejectionReason,
)
from video_engine.thumbnail.selector import KeyframeSelector


# --------------------------------------------------------------------------- #
# Helpers e Mocks
# --------------------------------------------------------------------------- #
def _make_frame(
    size: int = 120,
    checker_block: Optional[int] = 10,
    base_luma: int = 128,
) -> np.ndarray:
    """Gera frame configuravel: xadrez (alta nitidez) ou uniforme (blur)."""
    if checker_block is not None:
        grid = (np.indices((size, size)) // checker_block).sum(axis=0) % 2
        rgb = np.full((size, size, 3), base_luma, dtype=np.uint8)
        rgb[grid == 1] = min(255, base_luma + 60)
        rgb[grid == 0] = max(0, base_luma - 60)
        return rgb
    return np.full((size, size, 3), base_luma, dtype=np.uint8)


def _make_face_metrics(
    detected: bool = True,
    eyes_closed: bool = False,
    eye_openness: float = 0.8,
    mouth_articulation: float = 0.7,
    expression_intensity: float = 0.75,
) -> FaceMetrics:
    if not detected:
        return FaceMetrics(detected=False)
    return FaceMetrics(
        detected=True,
        bounding_box=FaceBoundingBox(x=0.3, y=0.2, width=0.4, height=0.5),
        eye_openness=eye_openness if not eyes_closed else 0.1,
        eyes_closed=eyes_closed,
        mouth_articulation=mouth_articulation,
        expression_intensity=expression_intensity,
        confidence=0.98,
    )


class MockFaceAnalyzer:
    """Analisador facial com respostas configuraveis por timestamp ou retorno fixo."""

    def __init__(self, metrics: Optional[FaceMetrics] = None) -> None:
        self.default_metrics = metrics or _make_face_metrics()
        self.by_timestamp: Dict[int, FaceMetrics] = {}

    def analyze(self, frame_rgb: np.ndarray, timestamp_ms: int = 0) -> FaceMetrics:
        return self.by_timestamp.get(timestamp_ms, self.default_metrics)


class MockFrameExtractor(VideoFrameExtractor):
    """Extrator em memoria para testes unitarios ultrarapidos e deterministicos."""

    def __init__(self, frames: List[SampledFrame], duration_ms: int = 10000, fps: float = 25.0) -> None:
        self.frames = frames
        self._duration_ms_val = duration_ms
        self._fps_val = fps

    def probe(self, video_path: Union[str, Path]) -> Tuple[int, float]:
        return self._duration_ms_val, self._fps_val

    def duration_ms(self, video_path: Union[str, Path]) -> int:
        return self._duration_ms_val

    def extract_frames(
        self,
        video_path: Union[str, Path],
        timestamps_ms: Optional[object] = None,
    ) -> List[SampledFrame]:
        return self.frames

    def sample_uniform(
        self,
        video_path: Union[str, Path],
        sample_interval_ms: int = 1000,
        start_offset_ms: int = 1000,
        end_offset_ms: int = 1000,
        min_samples: int = 3,
    ) -> List[SampledFrame]:
        return self.frames

    def save_frame(
        self,
        frame_rgb: np.ndarray,
        output_path: Union[str, Path],
    ) -> Path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x89PNG\r\n\x1a\n")
        return dest


# --------------------------------------------------------------------------- #
# Testes de Configuracao
# --------------------------------------------------------------------------- #
class TestKeyframeSelectorConfig:
    def test_weights_sum_validation(self) -> None:
        with pytest.raises(ValueError, match="Pesos do score composto devem somar 1.0"):
            KeyframeSelectorConfig(
                weight_sharpness=0.5,
                weight_expression=0.5,
                weight_lighting=0.2,  # soma 1.2
                weight_face_prominence=0.0,
            )

    def test_luminance_min_max_validation(self) -> None:
        with pytest.raises(ValueError, match="min_luminance nao pode ser maior"):
            KeyframeSelectorConfig(min_luminance=200.0, max_luminance=100.0)


# --------------------------------------------------------------------------- #
# Testes de Avaliacao de Frame Individual (evaluate_frame_array)
# --------------------------------------------------------------------------- #
class TestEvaluateFrameArray:
    def test_valid_sharp_expressive_frame(self) -> None:
        selector = KeyframeSelector(face_analyzer=MockFaceAnalyzer(_make_face_metrics()))
        frame = _make_frame(checker_block=8, base_luma=128)

        metrics = selector.evaluate_frame_array(frame, timestamp_ms=2000, frame_index=5)

        assert metrics.is_valid is True
        assert metrics.is_blurry is False
        assert metrics.face.eyes_closed is False
        assert metrics.is_poor_lighting is False
        assert metrics.rejection_reasons == []
        assert metrics.composite_score > 0.60

    def test_discard_blurry_frame(self) -> None:
        selector = KeyframeSelector(
            config=KeyframeSelectorConfig(min_sharpness_threshold=100.0, discard_blurry=True),
            face_analyzer=MockFaceAnalyzer(_make_face_metrics()),
        )
        uniform_blurry = _make_frame(checker_block=None, base_luma=128)

        metrics = selector.evaluate_frame_array(uniform_blurry)

        assert metrics.is_valid is False
        assert metrics.is_blurry is True
        assert RejectionReason.BLURRY in metrics.rejection_reasons

    def test_discard_closed_eyes_frame(self) -> None:
        closed_eyes_face = _make_face_metrics(eyes_closed=True, eye_openness=0.05)
        selector = KeyframeSelector(
            config=KeyframeSelectorConfig(discard_closed_eyes=True),
            face_analyzer=MockFaceAnalyzer(closed_eyes_face),
        )
        sharp_frame = _make_frame(checker_block=8)

        metrics = selector.evaluate_frame_array(sharp_frame)

        assert metrics.is_valid is False
        assert metrics.face.eyes_closed is True
        assert RejectionReason.CLOSED_EYES in metrics.rejection_reasons

    def test_discard_poor_lighting_dark(self) -> None:
        selector = KeyframeSelector(
            config=KeyframeSelectorConfig(min_luminance=40.0, discard_poor_lighting=True),
            face_analyzer=MockFaceAnalyzer(_make_face_metrics()),
        )
        dark_frame = _make_frame(checker_block=8, base_luma=20)

        metrics = selector.evaluate_frame_array(dark_frame)

        assert metrics.is_valid is False
        assert metrics.is_poor_lighting is True
        assert RejectionReason.POOR_LIGHTING_DARK in metrics.rejection_reasons

    def test_discard_poor_lighting_overexposed(self) -> None:
        selector = KeyframeSelector(
            config=KeyframeSelectorConfig(max_luminance=220.0, discard_poor_lighting=True),
            face_analyzer=MockFaceAnalyzer(_make_face_metrics()),
        )
        blown_frame = _make_frame(checker_block=None, base_luma=245)

        metrics = selector.evaluate_frame_array(blown_frame)

        assert metrics.is_valid is False
        assert metrics.is_poor_lighting is True
        assert RejectionReason.POOR_LIGHTING_OVEREXPOSED in metrics.rejection_reasons

    def test_discard_no_face_when_required(self) -> None:
        selector = KeyframeSelector(
            config=KeyframeSelectorConfig(require_face=True),
            face_analyzer=MockFaceAnalyzer(_make_face_metrics(detected=False)),
        )
        sharp_frame = _make_frame(checker_block=8)

        metrics = selector.evaluate_frame_array(sharp_frame)

        assert metrics.is_valid is False
        assert RejectionReason.NO_FACE_DETECTED in metrics.rejection_reasons

    def test_invalid_array_inputs_raise_value_error(self) -> None:
        selector = KeyframeSelector()
        with pytest.raises(ValueError, match="vazio ou dimensões inválidas"):
            selector.evaluate_frame_array(np.ones((50, 50), dtype=np.uint8))

        with pytest.raises(ValueError, match="vazio ou dimensões inválidas"):
            selector.evaluate_frame_array(np.empty((0, 0, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="NaN/Inf"):
            selector.evaluate_frame_array(np.full((10, 10, 3), np.nan))


# --------------------------------------------------------------------------- #
# Testes de Selecao e Ranqueamento (select_best_keyframes)
# --------------------------------------------------------------------------- #
class TestSelectBestKeyframes:
    def test_ranking_top_5_with_temporal_diversity(self, tmp_path: Path) -> None:
        # Prepara 10 frames com variadas pontuacoes e timestamps a cada 500ms
        sampled_frames: List[SampledFrame] = []
        face_mock = MockFaceAnalyzer()

        for i in range(10):
            ts = i * 1000  # 0s, 1s, 2s, ..., 9s
            # frames pares mais expressivos que impares
            is_expressive = (i % 2 == 0)
            mouth = 0.9 if is_expressive else 0.2
            intensity = 0.85 if is_expressive else 0.3
            face_mock.by_timestamp[ts] = _make_face_metrics(
                detected=True,
                eyes_closed=False,
                mouth_articulation=mouth,
                expression_intensity=intensity,
            )
            frame_img = _make_frame(checker_block=8, base_luma=128)
            sampled_frames.append(SampledFrame(timestamp_ms=ts, frame_index=i, frame=frame_img))

        extractor = MockFrameExtractor(frames=sampled_frames, duration_ms=10000)
        config = KeyframeSelectorConfig(
            top_n=5,
            min_candidate_distance_ms=1500,  # requer pelo menos 1.5s entre candidatos
        )
        selector = KeyframeSelector(config=config, face_analyzer=face_mock, frame_extractor=extractor)

        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        result = selector.select_best_keyframes(fake_video)

        assert isinstance(result, KeyframeSelectorResult)
        assert len(result.top_candidates) == 5
        assert result.valid_frames_count == 10

        # Verifica ordenacao decrescente de score
        scores = [c.score for c in result.top_candidates]
        assert scores == sorted(scores, reverse=True)

        # Ranks de 1 a 5
        ranks = [c.rank for c in result.top_candidates]
        assert ranks == [1, 2, 3, 4, 5]

        # Verifica restricao de diversidade temporal: diferenca >= 1500ms entre qualquer par
        timestamps = [c.timestamp_ms for c in result.top_candidates]
        for i in range(len(timestamps)):
            for j in range(i + 1, len(timestamps)):
                assert abs(timestamps[i] - timestamps[j]) >= 1500

    def test_discarded_blurry_and_closed_eyes_counts(self, tmp_path: Path) -> None:
        face_mock = MockFaceAnalyzer()
        sampled_frames: List[SampledFrame] = []

        # 3 frames normais, 2 borrados, 2 com olhos fechados
        # Frame 0, 1, 2: bons
        for i in range(3):
            ts = i * 2000
            face_mock.by_timestamp[ts] = _make_face_metrics(eyes_closed=False)
            sampled_frames.append(SampledFrame(ts, i, _make_frame(checker_block=8)))

        # Frame 3, 4: borrados (checker_block=None)
        for i in range(3, 5):
            ts = i * 2000
            face_mock.by_timestamp[ts] = _make_face_metrics(eyes_closed=False)
            sampled_frames.append(SampledFrame(ts, i, _make_frame(checker_block=None)))

        # Frame 5, 6: olhos fechados
        for i in range(5, 7):
            ts = i * 2000
            face_mock.by_timestamp[ts] = _make_face_metrics(eyes_closed=True)
            sampled_frames.append(SampledFrame(ts, i, _make_frame(checker_block=8)))

        extractor = MockFrameExtractor(frames=sampled_frames, duration_ms=14000)
        selector = KeyframeSelector(face_analyzer=face_mock, frame_extractor=extractor)

        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        result = selector.select_best_keyframes(fake_video)

        assert result.total_frames_sampled == 7
        assert result.discarded_blurry_count == 2
        assert result.discarded_closed_eyes_count == 2
        assert result.valid_frames_count == 3
        assert len(result.top_candidates) == 3

    def test_export_images_to_output_dir(self, tmp_path: Path) -> None:
        sampled_frames = [
            SampledFrame(1000, 1, _make_frame(checker_block=8)),
            SampledFrame(3000, 2, _make_frame(checker_block=8)),
        ]
        extractor = MockFrameExtractor(frames=sampled_frames, duration_ms=5000)
        selector = KeyframeSelector(face_analyzer=MockFaceAnalyzer(), frame_extractor=extractor)

        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        out_dir = tmp_path / "thumbnails_out"

        result = selector.select_best_keyframes(fake_video, output_dir=out_dir)

        assert len(result.top_candidates) == 2
        for candidate in result.top_candidates:
            assert candidate.image_path is not None
            exported_file = Path(candidate.image_path)
            assert exported_file.is_file()
            assert exported_file.suffix == ".png"
            assert exported_file.stat().st_size > 0

    def test_all_frames_blurry_returns_empty_candidates(self, tmp_path: Path) -> None:
        blurry_frames = [
            SampledFrame(1000, 1, _make_frame(checker_block=None)),
            SampledFrame(2000, 2, _make_frame(checker_block=None)),
        ]
        extractor = MockFrameExtractor(frames=blurry_frames, duration_ms=3000)
        selector = KeyframeSelector(face_analyzer=MockFaceAnalyzer(), frame_extractor=extractor)

        fake_video = tmp_path / "video.mp4"
        fake_video.touch()
        result = selector.select_best_keyframes(fake_video)

        assert result.top_candidates == []
        assert result.discarded_blurry_count == 2
        assert result.valid_frames_count == 0

    def test_missing_video_raises_file_not_found(self, tmp_path: Path) -> None:
        selector = KeyframeSelector()
        with pytest.raises(FileNotFoundError):
            selector.select_best_keyframes(tmp_path / "nao_existe.mp4")
