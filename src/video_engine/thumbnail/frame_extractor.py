"""Extracao de frames de videos via PyAV/FFmpeg (Issue #11).

Implementa ``VideoFrameExtractor`` com amostragem temporal uniforme
(ex: 1 frame por segundo) e extracao em timestamps arbitrarios, decodificando
frames RGB (HxWx3, uint8). Frames corrompidos sao logados como warning e
pulados sem interromper a esteira.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

try:
    import av
except ImportError as exc:  # pragma: no cover - dependencia opcional no carregamento
    raise ImportError(
        "PyAV (av) is required for VideoFrameExtractor. Install with 'pip install av'."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampledFrame:
    """Frame RGB amostrado com seus metadados temporais."""

    timestamp_ms: int
    frame_index: int
    frame: np.ndarray


def compute_uniform_timestamps(
    duration_ms: int,
    sample_interval_ms: int,
    start_offset_ms: int,
    end_offset_ms: int,
    min_samples: int = 3,
) -> List[int]:
    """Calcula timestamps uniformemente espacados dentro da janela util.

    Respeita os offsets de inicio/fim; para videos curtos os offsets sao
    reduzidos proporcionalmente a duracao, garantindo no minimo
    ``min_samples`` inspecoes (edge case de videos < 2s).
    """
    if duration_ms <= 0 or sample_interval_ms <= 0:
        return []

    start = start_offset_ms
    end = end_offset_ms
    if start + end >= duration_ms:
        fraction = 0.10 * duration_ms
        start = int(round(fraction))
        end = int(round(fraction))

    window_start = start
    window_end = duration_ms - end
    if window_end - window_start <= 0:
        window_start = min(start, max(0, duration_ms - 1))
        window_end = duration_ms

    grid_count = int(round((window_end - window_start) / sample_interval_ms)) + 1
    count = max(grid_count, min_samples)

    points = np.linspace(window_start, window_end, count, dtype=np.float64)
    points = np.clip(points, 0.0, float(max(0, duration_ms - 1)))
    timestamps = [int(round(float(ts))) for ts in points]

    deduplicated: List[int] = []
    for ts in timestamps:
        if not deduplicated or ts != deduplicated[-1]:
            deduplicated.append(ts)
    return deduplicated


class VideoFrameExtractor:
    """Decodifica frames de um arquivo de video com PyAV."""

    def probe(self, video_path: Union[str, Path]) -> tuple:
        """Inspeciona o video e retorna ``(duration_ms, fps)``.

        Raises:
            FileNotFoundError: se ``video_path`` nao existir.
            ValueError: se o arquivo nao possuir stream de video.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de video nao encontrado: {path}")
        with av.open(str(path)) as container:
            stream = self._video_stream(container, path)
            duration_ms = self._duration_ms(container, stream)
            fps = self._fps(stream)
            return duration_ms, fps

    def duration_ms(self, video_path: Union[str, Path]) -> int:
        """Retorna a duracao do video em milissegundos."""
        duration_ms, _ = self.probe(video_path)
        return duration_ms

    def extract_at(self, video_path: Union[str, Path], timestamp_ms: int) -> np.ndarray:
        """Extrai o frame RGB mais proximo de ``timestamp_ms``.

        Raises:
            FileNotFoundError: se ``video_path`` nao existir.
            ValueError: se o arquivo nao possuir stream de video.
            RuntimeError: se nenhum frame puder ser decodificado.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de video nao encontrado: {path}")

        with av.open(str(path)) as container:
            stream = self._video_stream(container, path)
            if timestamp_ms <= 0:
                frame = next(container.decode(stream), None)
                if frame is None:
                    raise RuntimeError(f"Falha ao decodificar o frame inicial de {path}")
                return frame.to_ndarray(format="rgb24")

            target_pts = int(round((timestamp_ms / 1000.0) / stream.time_base))
            container.seek(target_pts, stream=stream)

            last_frame = None
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target_pts:
                    return frame.to_ndarray(format="rgb24")
                last_frame = frame

            if last_frame is not None:
                return last_frame.to_ndarray(format="rgb24")

        raise RuntimeError(
            f"Nao foi possivel decodificar o frame em {timestamp_ms}ms de {path}"
        )

    def extract_frames(
        self,
        video_path: Union[str, Path],
        timestamps_ms: Iterable[int],
    ) -> List[SampledFrame]:
        """Extrai frames em timestamps arbitrarios, pulando codificacoes falhas."""
        path = Path(video_path)
        _, fps = self.probe(path)
        sampled: List[SampledFrame] = []
        for index, timestamp_ms in enumerate(sorted(timestamps_ms)):
            try:
                rgb = self.extract_at(path, timestamp_ms)
            except (RuntimeError, av.FFmpegError) as exc:  # pragma: no cover
                logger.warning("Frame corrompido em %dms de %s: %s", timestamp_ms, path, exc)
                continue
            frame_index = int(round((timestamp_ms / 1000.0) * fps)) if fps else index
            sampled.append(
                SampledFrame(
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    frame=rgb,
                )
            )
        return sampled

    def sample_uniform(
        self,
        video_path: Union[str, Path],
        sample_interval_ms: int = 1000,
        start_offset_ms: int = 1000,
        end_offset_ms: int = 1000,
        min_samples: int = 3,
    ) -> List[SampledFrame]:
        """Amostra frames uniformemente dentro da janela util do video."""
        duration_ms = self.duration_ms(video_path)
        timestamps = compute_uniform_timestamps(
            duration_ms=duration_ms,
            sample_interval_ms=sample_interval_ms,
            start_offset_ms=start_offset_ms,
            end_offset_ms=end_offset_ms,
            min_samples=min_samples,
        )
        return self.extract_frames(video_path, timestamps)

    def save_frame(
        self,
        frame_rgb: np.ndarray,
        output_path: Union[str, Path],
    ) -> Path:
        """Grava um frame RGB como imagem PNG via PyAV."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame_rgb.shape[:2]
        with av.open(str(destination), mode="w", format="image2") as output:
            stream = output.add_stream("png", rate=1)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "rgb24"
            video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            for packet in stream.encode(video_frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        return destination

    # ------------------------------------------------------------------ #
    # Helpers internos
    # ------------------------------------------------------------------ #
    @staticmethod
    def _video_stream(container, path: Path):
        streams = container.streams.video
        if not streams:
            raise ValueError(f"Arquivo de midia sem stream de video: {path}")
        return streams[0]

    @staticmethod
    def _duration_ms(container, stream) -> int:
        if container.duration:
            return int(round(container.duration / av.time_base * 1000))
        if stream.duration:
            return int(round(stream.duration * stream.time_base * 1000))
        # Fallback: estima pela media de duracao dos frames restantes
        frames = list(container.decode(stream))
        if not frames:
            return 0
        last = frames[-1]
        pts = last.pts or 0.0
        return int(round(pts * stream.time_base * 1000))

    @staticmethod
    def _fps(stream) -> Optional[float]:
        try:
            rate = stream.average_rate
        except Exception:  # noqa: BLE001
            return None
        if not rate:
            return None
        try:
            return float(rate)
        except (TypeError, ValueError):
            return None


__all__ = [
    "SampledFrame",
    "VideoFrameExtractor",
    "compute_uniform_timestamps",
]
