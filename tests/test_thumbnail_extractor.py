"""Testes de extracao de frames de video via PyAV (Issue #11)."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from video_engine.thumbnail.frame_extractor import (
    SampledFrame,
    VideoFrameExtractor,
    compute_uniform_timestamps,
)


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    """Gera um arquivo de video MP4 sintetico de 3 segundos (25 fps, 320x240)."""
    file_path = tmp_path / "test_video.mp4"
    width, height = 320, 240
    fps = 25
    total_frames = fps * 3  # 75 frames = 3000ms

    with av.open(str(file_path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for i in range(total_frames):
            # Frame colorido gradiente que muda ao longo do tempo
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 0] = int((i / total_frames) * 255)
            img[:, :, 1] = 128
            img[:, :, 2] = 200

            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return file_path


class TestUniformTimestamps:
    def test_standard_video(self) -> None:
        # Video de 10s (10000ms), amostragem a cada 2s (2000ms), offsets de 1s
        timestamps = compute_uniform_timestamps(
            duration_ms=10000,
            sample_interval_ms=2000,
            start_offset_ms=1000,
            end_offset_ms=1000,
        )
        assert len(timestamps) >= 4
        assert timestamps[0] >= 1000
        assert timestamps[-1] <= 9000
        # Ordenado monotonicamente crescente
        assert timestamps == sorted(timestamps)

    def test_short_video_edge_case(self) -> None:
        # Video de apenas 1.5s (1500ms) onde offsets padrao de 1s + 1s > duracao
        timestamps = compute_uniform_timestamps(
            duration_ms=1500,
            sample_interval_ms=500,
            start_offset_ms=1000,
            end_offset_ms=1000,
            min_samples=3,
        )
        # Deve ter ajustado os offsets e gerado pelo menos 3 amostras
        assert len(timestamps) >= 3
        assert timestamps[0] >= 0
        assert timestamps[-1] <= 1500

    def test_zero_or_negative_returns_empty(self) -> None:
        assert compute_uniform_timestamps(0, 1000, 100, 100) == []
        assert compute_uniform_timestamps(5000, 0, 100, 100) == []


class TestVideoFrameExtractor:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        extractor = VideoFrameExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.probe(tmp_path / "non_existent.mp4")

    def test_probe_synthetic_video(self, synthetic_video_path: Path) -> None:
        extractor = VideoFrameExtractor()
        duration_ms, fps = extractor.probe(synthetic_video_path)

        assert 2800 <= duration_ms <= 3200
        assert fps == pytest.approx(25.0, abs=1.0)
        assert extractor.duration_ms(synthetic_video_path) == duration_ms

    def test_extract_at_timestamps(self, synthetic_video_path: Path) -> None:
        extractor = VideoFrameExtractor()
        requested_ts = [500, 1500, 2500]

        frames = extractor.extract_frames(synthetic_video_path, requested_ts)

        assert len(frames) == len(requested_ts)
        for i, item in enumerate(frames):
            assert isinstance(item, SampledFrame)
            assert item.frame.shape == (240, 320, 3)
            assert item.frame.dtype == np.uint8
            assert abs(item.timestamp_ms - requested_ts[i]) <= 200

    def test_sample_uniform(self, synthetic_video_path: Path) -> None:
        extractor = VideoFrameExtractor()
        frames = extractor.sample_uniform(
            synthetic_video_path,
            sample_interval_ms=1000,
            start_offset_ms=200,
            end_offset_ms=200,
        )
        assert len(frames) >= 2

    def test_extract_at(self, synthetic_video_path: Path) -> None:
        extractor = VideoFrameExtractor()
        rgb = extractor.extract_at(synthetic_video_path, 1000)
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (240, 320, 3)
        assert rgb.dtype == np.uint8
