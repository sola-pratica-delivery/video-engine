"""Testes do reenquadramento vertical 9:16 com face tracking (Issue #16).

Cobertura (TDD, seccao 4 da Spec #16): validacao dos modelos Pydantic,
algoritmo de suavizacao EMA + deadband, clamping estrito de geometria,
construcao do filtergraph FFmpeg, comutacao AUTO e renderizacao ponta a
ponta com FFmpeg (MP4 1080x1920 com audio copiado).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import types
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from video_engine.editing.media_probe import MediaInfo, MediaProbe
from video_engine.shorts.face_tracker import (
    FaceTracker,
    RawFaceSample,
    smooth_centers,
)
from video_engine.shorts.reframer_models import (
    CropMode,
    FaceTrackingPoint,
    ReframerConfig,
    ReframerResult,
    TrackingTrajectory,
)
from video_engine.shorts.vertical_reframer import ReframerError, VerticalReframer

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _numbers(text: str) -> List[float]:
    """Extrai os numeros (incl. decimais) presentes em ``text``."""
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


# --------------------------------------------------------------------------- #
# Fixture: clipe horizontal sintetico (16:9) com audio
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def horizontal_clip(tmp_path_factory) -> Path:
    if not FFMPEG or not FFPROBE:
        pytest.skip("ffmpeg/ffprobe nao disponivel")
    path = tmp_path_factory.mktemp("reframer") / "horizontal.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=3:size=320x180:rate=15",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=3",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return path


class ScriptedDetector:
    """Deteccao falsa retornando posicoes pre-programadas por chamada."""

    def __init__(self, positions: List[Optional[Tuple[float, float]]]) -> None:
        self.positions = positions
        self.calls = 0

    def __call__(self, frame_rgb: np.ndarray) -> Optional[Tuple[float, float]]:
        self.calls += 1
        index = self.calls - 1
        if index < len(self.positions):
            return self.positions[index]
        return None


def trajectory(positions: List[Tuple[float, float]]) -> TrackingTrajectory:
    """Trajetoria deterministica com um ponto por posicao fornecida."""
    return TrackingTrajectory(
        points=[
            FaceTrackingPoint(
                timestamp_ms=i * 200,
                face_detected=True,
                raw_center_x=x,
                smoothed_center_x=x,
            )
            for i, (x, _y) in enumerate(positions)
        ],
        face_detected_ratio=1.0,
        average_center_x=positions[0][0],
    )


def empty_trajectory() -> TrackingTrajectory:
    return TrackingTrajectory(points=[], face_detected_ratio=0.0, average_center_x=0.5)


# --------------------------------------------------------------------------- #
# 1. Modelos & validacoes
# --------------------------------------------------------------------------- #
class TestModels:
    def test_reframer_config_defaults(self) -> None:
        config = ReframerConfig()
        assert config.target_width == 1080
        assert config.target_height == 1920
        assert config.crop_mode == CropMode.AUTO
        assert config.sample_interval_ms == 200
        assert config.smoothing_factor == 0.15
        assert config.deadband_threshold == 0.03
        assert config.scaling_filter == "lanczos"

    def test_crop_mode_values(self) -> None:
        assert CropMode.SMART_CROP.value == "smart_crop"
        assert CropMode.AESTHETIC_FILL.value == "aesthetic_fill"
        assert CropMode.AUTO.value == "auto"

    def test_config_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ReframerConfig(mode_extra="forbiden_field")

    def test_config_rejects_samples_below_minimum(self) -> None:
        with pytest.raises(ValidationError):
            ReframerConfig(sample_interval_ms=10)

    def test_config_rejects_invalid_ratio(self) -> None:
        with pytest.raises(ValidationError):
            ReframerConfig(min_face_detection_ratio=1.5)

    def test_face_tracking_point_requires_nonnegative_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            FaceTrackingPoint(timestamp_ms=-1, face_detected=True)

    def test_face_tracking_point_clamps_coordinate_range(self) -> None:
        with pytest.raises(ValidationError):
            FaceTrackingPoint(
                timestamp_ms=0, face_detected=True, raw_center_x=1.5
            )

    def test_trajectory_rejects_ratio_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TrackingTrajectory(points=[], face_detected_ratio=1.1)

    def test_reframer_result_requires_mode(self) -> None:
        with pytest.raises(ValidationError):
            ReframerResult(
                output_path="x.mp4",
                duration_sec=1.0,
                mode_used=CropMode.AUTO,
                face_detected_ratio=0.0,
            )


# --------------------------------------------------------------------------- #
# 2. Suavizacao temporale deadband
# --------------------------------------------------------------------------- #
class TestSmoothing:
    def test_deadband_eliminates_micro_jitter(self) -> None:
        samples = [
            RawFaceSample(
                timestamp_ms=ts,
                detected=True,
                x=0.5 + (0.01 if ts % 400 == 0 else -0.01),
                y=0.4,
            )
            for ts in range(0, 4000, 200)
        ]
        output = smooth_centers(samples, alpha=0.15, deadband=0.05)
        xs = [x for x, _y in output]
        assert max(xs) - min(xs) < 1e-6
        assert all(x == pytest.approx(0.5) for x in xs)

    def test_rapid_transition_is_continuous_and_monotonic(self) -> None:
        before = [RawFaceSample(ts, True, 0.5, 0.4) for ts in range(0, 2400, 200)]
        after = [RawFaceSample(ts, True, 0.9, 0.4) for ts in range(2400, 6000, 200)]
        output = smooth_centers(before + after, alpha=0.3, deadband=0.0)
        xs = [x for x, _y in output]

        assert xs[0] == pytest.approx(0.5)
        assert xs == sorted(xs)
        assert xs[-1] > 0.85
        deltas = [b - a for a, b in zip(xs, xs[1:])]
        assert max(deltas) <= 0.3

    def test_deadband_holds_camera_until_threshold_exceeded(self) -> None:
        before = [RawFaceSample(ts, True, 0.5, 0.4) for ts in range(0, 3000, 300)]
        after = [RawFaceSample(ts, True, 0.8, 0.4) for ts in range(3000, 9000, 300)]
        output = smooth_centers(before + after, alpha=0.3, deadband=0.10)
        xs = [x for x, _y in output]

        assert xs[0] == pytest.approx(0.5)
        assert all(x >= 0.5 - 1e-9 for x in xs)
        assert xs[-1] > 0.7

    def test_undetected_frames_preserve_last_position(self) -> None:
        samples = [
            RawFaceSample(0, True, 0.7, 0.4),
            RawFaceSample(200, False),
            RawFaceSample(400, False),
        ]
        output = smooth_centers(samples, alpha=0.5, deadband=0.0)
        assert output[0] == (0.7, 0.4)
        assert output[1] == (0.7, 0.4)
        assert output[2] == (0.7, 0.4)

    def test_out_of_range_centers_are_clamped(self) -> None:
        samples = [
            RawFaceSample(ts, True, x, y)
            for ts, (x, y) in enumerate([(-0.5, 0.4), (0.0, -0.2), (1.5, 1.2)])
        ]
        output = smooth_centers(samples, alpha=1.0, deadband=0.0)
        for x, y in output:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
        assert output[0][0] == 0.0
        assert output[2][0] == 1.0


# --------------------------------------------------------------------------- #
# 3. Clamping estrito de geometria
# --------------------------------------------------------------------------- #
class TestClamping:
    def test_crop_x_never_exceeds_frame_bounds(self) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        width, height = 1920, 1080
        crop_w, _crop_h = reframer._smart_crop_dims(width, height)
        max_x = width - crop_w

        for raw_x in (0.0, 1.0):
            traj = TrackingTrajectory(
                points=[
                    FaceTrackingPoint(
                        timestamp_ms=0,
                        face_detected=True,
                        smoothed_center_x=raw_x,
                    ),
                    FaceTrackingPoint(
                        timestamp_ms=1000,
                        face_detected=True,
                        smoothed_center_x=raw_x,
                    ),
                ],
                face_detected_ratio=1.0,
                average_center_x=raw_x,
            )
            expr = reframer._crop_x_expression(traj, width, crop_w)
            assert min(_numbers(expr)) >= 0.0
            assert max(_numbers(expr)) <= max_x + 0.005

    def test_crop_x_leading_and_tail_knots(self) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        width, height = 1920, 1080
        crop_w, _crop_h = reframer._smart_crop_dims(width, height)
        max_x = width - crop_w

        traj = TrackingTrajectory(
            points=[
                FaceTrackingPoint(timestamp_ms=0, face_detected=True, smoothed_center_x=0.0),
                FaceTrackingPoint(timestamp_ms=1000, face_detected=True, smoothed_center_x=1.0),
            ],
            face_detected_ratio=1.0,
            average_center_x=0.5,
        )
        expr = reframer._crop_x_expression(traj, width, crop_w)
        # knot central X=0.0 forcado para 0; X=1.0 forcado para W-W_crop
        assert expr.startswith("if(between(t,0.000,1.000),(0.00+")
        assert expr.endswith(f",{max_x:.2f})")

    def test_smart_crop_dims_on_16x9(self) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        crop_w, crop_h = reframer._smart_crop_dims(1920, 1080)
        assert crop_h == 1080
        assert crop_w % 2 == 0
        assert abs((crop_w / crop_h) - (9 / 16)) < 0.05


# --------------------------------------------------------------------------- #
# 4. Construcao de filtergraph
# --------------------------------------------------------------------------- #
class TestFilterGraph:
    def test_smart_crop_filter_structure(self) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        traj = trajectory([(0.5, 0.4), (0.6, 0.4)])
        graph = reframer.build_filter_complex(1920, 1080, traj, CropMode.SMART_CROP)

        assert graph.startswith("[0:v]crop=608:1080:")
        assert "scale=1080:1920:flags=lanczos[vout]" in graph

    def test_aesthetic_fill_filter_structure(self) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.AESTHETIC_FILL))
        graph = reframer.build_filter_complex(
            1920, 1080, empty_trajectory(), CropMode.AESTHETIC_FILL
        )

        assert "boxblur=lr=25:lp=25" in graph
        assert "force_original_aspect_ratio=increase" in graph
        assert "force_original_aspect_ratio=decrease" in graph
        assert "overlay=(W-w)/2:(H-h)/2[vout]" in graph
        assert graph.startswith("[0:v]split=2[bg][fg];")

    def test_vertical_input_uses_scale_only(self) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        traj = trajectory([(0.5, 0.4)])
        graph = reframer.build_filter_complex(1080, 1920, traj, CropMode.SMART_CROP)
        assert graph == "[0:v]scale=1080:1920:flags=lanczos[vout]"

    def test_vertical_input_short_side(self) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        traj = trajectory([(0.5, 0.4)])
        graph = reframer.build_filter_complex(720, 1280, traj, CropMode.SMART_CROP)
        assert "scale=1080:1920:flags=lanczos[vout]" in graph


# --------------------------------------------------------------------------- #
# 5. Comutacao em modo AUTO
# --------------------------------------------------------------------------- #
class TestAutoMode:
    def test_auto_without_face_uses_aesthetic_fill(self) -> None:
        reframer = VerticalReframer(
            ReframerConfig(crop_mode=CropMode.AUTO),
            face_tracker=_FakeTracker(empty_trajectory()),
        )
        info = MediaInfo(path="x.mp4", has_video=True, has_audio=False, duration_ms=5000)
        mode, traj = reframer._resolve_mode(info, 0, 5000)
        assert mode == CropMode.AESTHETIC_FILL
        assert traj.dominant_mode == CropMode.AESTHETIC_FILL  # type: ignore[union-attr]

    def test_auto_with_face_uses_smart_crop(self) -> None:
        reframer = VerticalReframer(
            ReframerConfig(crop_mode=CropMode.AUTO),
            face_tracker=_FakeTracker(trajectory([(0.5, 0.4)])),
        )
        info = MediaInfo(path="x.mp4", has_video=True, has_audio=False, duration_ms=5000)
        mode, _traj = reframer._resolve_mode(info, 0, 5000)
        assert mode == CropMode.SMART_CROP

    def test_explicit_aesthetic_fill_skips_tracking(self) -> None:
        reframer = VerticalReframer(
            ReframerConfig(crop_mode=CropMode.AESTHETIC_FILL),
            face_tracker=_FakeTracker(empty_trajectory()),
        )
        info = MediaInfo(path="x.mp4", has_video=True, has_audio=False, duration_ms=5000)
        mode, traj = reframer._resolve_mode(info, 0, 5000)
        assert mode == CropMode.AESTHETIC_FILL
        assert traj is None


class _FakeTracker:
    """Stub de FaceTracker devolvendo uma trajetoria pre-configurada."""

    def __init__(self, track: TrackingTrajectory) -> None:
        self._track = track

    def track_video(
        self,
        video_path: object,
        start_ms: int = 0,
        end_ms: Optional[int] = None,
    ) -> TrackingTrajectory:
        return self._track


class _FakeProbe:
    """Stub de MediaProbe devolvendo informacoes de midia estaticas."""

    def probe(self, path: object) -> MediaInfo:
        return MediaInfo(
            path=str(path),
            has_video=True,
            has_audio=False,
            duration_ms=5000,
        )


# --------------------------------------------------------------------------- #
# 6. FaceTracker real (PyAV) e renderizacao ponta a ponta
# --------------------------------------------------------------------------- #
class TestFaceTrackerPyAV:
    def test_track_video_with_scripted_detector(self, horizontal_clip: Path) -> None:
        detector = ScriptedDetector([(0.5, 0.4)] * 2 + [None, (0.7, 0.4)] * 3)
        tracker = FaceTracker(
            sample_interval_ms=300,
            smoothing_factor=0.2,
            deadband_threshold=0.05,
            face_detector=detector,
        )
        traj = tracker.track_video(horizontal_clip)

        assert detector.calls > 0
        assert 0.5 <= traj.face_detected_ratio <= 0.9
        assert all(
            0.0 <= p.smoothed_center_x <= 1.0 for p in traj.points
        )
        assert all(p.timestamp_ms >= 0 for p in traj.points)

    def test_track_video_without_detector_reports_no_face(
        self, horizontal_clip: Path
    ) -> None:
        tracker = FaceTracker(sample_interval_ms=300)
        traj = tracker.track_video(horizontal_clip)
        assert traj.face_detected_ratio == 0.0
        assert traj.dominant_mode == CropMode.AESTHETIC_FILL
        assert len(traj.points) > 0
        assert all(not p.face_detected for p in traj.points)

    def test_track_video_missing_file(self) -> None:
        tracker = FaceTracker()
        with pytest.raises(FileNotFoundError):
            tracker.track_video(Path("nao_existe.mp4"))

    def test_track_video_rejects_negative_start(self, horizontal_clip: Path) -> None:
        tracker = FaceTracker()
        with pytest.raises(ValueError, match="start_ms"):
            tracker.track_video(horizontal_clip, start_ms=-10)


class TestVerticalReframerE2E:
    def test_smart_crop_renders_9x16_with_audio(
        self, horizontal_clip: Path, tmp_path: Path
    ) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        out = tmp_path / "short.mp4"
        result = reframer.reframe(horizontal_clip, out)

        assert result.mode_used == CropMode.SMART_CROP
        info = MediaProbe(ffprobe_path=FFPROBE).probe(out)
        assert info.has_video
        assert info.video_width == 1080
        assert info.video_height == 1920
        assert info.has_audio
        assert abs(result.duration_sec - 3.0) < 0.5

    def test_auto_without_face_uses_aesthetic_fill(
        self, horizontal_clip: Path, tmp_path: Path
    ) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.AUTO))
        out = tmp_path / "auto.mp4"
        result = reframer.reframe(horizontal_clip, out)

        assert result.mode_used == CropMode.AESTHETIC_FILL
        assert result.face_detected_ratio == 0.0
        assert result.trajectory is not None
        info = MediaProbe(ffprobe_path=FFPROBE).probe(out)
        assert info.video_width == 1080
        assert info.video_height == 1920
        assert info.has_audio

    def test_explicit_aesthetic_fill_no_tracking(
        self, horizontal_clip: Path, tmp_path: Path
    ) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.AESTHETIC_FILL))
        out = tmp_path / "fill.mp4"
        result = reframer.reframe(horizontal_clip, out)

        assert result.mode_used == CropMode.AESTHETIC_FILL
        assert result.trajectory is None
        info = MediaProbe(ffprobe_path=FFPROBE).probe(out)
        assert info.video_width == 1080
        assert info.video_height == 1920

    def test_temporal_cut_respects_window(
        self, horizontal_clip: Path, tmp_path: Path
    ) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.SMART_CROP))
        out = tmp_path / "cut.mp4"
        result = reframer.reframe(horizontal_clip, out, cut_start_ms=1000, cut_end_ms=3000)

        assert abs(result.duration_sec - 2.0) < 0.3
        info = MediaProbe(ffprobe_path=FFPROBE).probe(out)
        assert info.video_width == 1080
        assert info.video_height == 1920
        assert info.has_audio

    def test_reframe_missing_input(self, tmp_path: Path) -> None:
        reframer = VerticalReframer()
        with pytest.raises(FileNotFoundError):
            reframer.reframe(Path("nao_existe.mp4"), tmp_path / "out.mp4")

    def test_reframe_invalid_window(self, horizontal_clip: Path, tmp_path: Path) -> None:
        reframer = VerticalReframer()
        with pytest.raises(ValueError, match="cut_start_ms"):
            reframer.reframe(horizontal_clip, tmp_path / "out.mp4", cut_start_ms=5000)

    def test_reframe_without_video_stream(self, tmp_path: Path) -> None:
        if not FFMPEG:
            pytest.skip("ffmpeg nao disponivel")
        audio_only = tmp_path / "audio_only.m4a"
        cmd = [
            FFMPEG,
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1",
            "-c:a",
            "aac",
            str(audio_only),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        reframer = VerticalReframer()
        with pytest.raises(ValueError, match="sem stream de video"):
            reframer.reframe(audio_only, tmp_path / "out.mp4")

    def test_reframe_ffmpeg_failure_raises_reframer_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reframer = VerticalReframer(ReframerConfig(crop_mode=CropMode.AESTHETIC_FILL))
        reframer.probe_service = _FakeProbe()
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stderr="boom", stdout=""),
        )
        src = tmp_path / "fake.mp4"
        src.write_bytes(b"")
        with pytest.raises(ReframerError, match="FFmpeg falhou"):
            reframer.reframe(src, tmp_path / "out.mp4")
