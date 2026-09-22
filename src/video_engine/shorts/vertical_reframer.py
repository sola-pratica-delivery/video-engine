"""Reenquadrador vertical 9:16 com face tracking ativo (Issue #16).

Converte videos horizontais (16:9) para o formato vertical nativo
(9:16, 1080x1920) gerando um filtergraph FFmpeg otimizado com
interpolacao Lanczos/Bicubic, seguindo a trajetoria suavizada da face
(``SMART_CROP``), compondo sobre fundo desfocado (``AESTHETIC_FILL``)
ou decidindo automaticamente (``AUTO``). O stream de audio e preservado
via stream copy (``-c:a copy``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

from video_engine.editing.media_probe import MediaInfo, MediaProbe
from video_engine.shorts.face_tracker import FaceTracker
from video_engine.shorts.reframer_models import (
    CropMode,
    ReframerConfig,
    ReframerResult,
    TrackingTrajectory,
)


class ReframerError(RuntimeError):
    """Falha na execucao do FFmpeg durante o reenquadramento vertical."""


def _to_even(value: float) -> int:
    """Arredonda para o inteiro par mais proximo (compativel com yuv420p)."""
    even = int(round(value)) // 2 * 2
    return max(0, even)


class VerticalReframer:
    """Motor de conversao horizontal 16:9 para vertical 9:16 com face tracking."""

    def __init__(
        self,
        config: Optional[ReframerConfig] = None,
        face_tracker: Optional[FaceTracker] = None,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ) -> None:
        self.config = config or ReframerConfig()
        self.face_tracker = face_tracker or FaceTracker(
            sample_interval_ms=self.config.sample_interval_ms,
            smoothing_factor=self.config.smoothing_factor,
            deadband_threshold=self.config.deadband_threshold,
            min_face_detection_ratio=self.config.min_face_detection_ratio,
        )
        self._ffmpeg_path = ffmpeg_path
        self.probe_service = MediaProbe(ffprobe_path=ffprobe_path)

    # ------------------------------------------------------------------ #
    # Construcao de filtergraph
    # ------------------------------------------------------------------ #
    def build_filter_complex(
        self,
        width: int,
        height: int,
        trajectory: TrackingTrajectory,
        mode: CropMode,
    ) -> str:
        """Monta o filtro FFmpeg ``-filter_complex`` para o modo escolhido."""
        target_width = self.config.target_width
        target_height = self.config.target_height
        scaling = self.config.scaling_filter

        if self._is_vertical_input(width, height):
            return (
                f"[0:v]scale={target_width}:{target_height}:flags={scaling}[vout]"
            )

        if mode == CropMode.AESTHETIC_FILL:
            return self._aesthetic_fill_filter(target_width, target_height)

        if mode == CropMode.SMART_CROP:
            crop_w, crop_h = self._smart_crop_dims(width, height)
            crop_x = self._crop_x_expression(
                trajectory,
                src_width=width,
                crop_w=crop_w,
            )
            crop_y = (height - crop_h) // 2
            return (
                f"[0:v]crop={crop_w}:{crop_h}:'{crop_x}':{crop_y},"
                f"scale={target_width}:{target_height}:flags={scaling}[vout]"
            )

        return f"[0:v]scale={target_width}:{target_height}:flags={scaling}[vout]"

    def _is_vertical_input(self, width: int, height: int) -> bool:
        """Detecta entrada ja vertical (9:16 ou proporcao similar)."""
        if width <= 0 or height <= 0:
            return False
        source_aspect = width / height
        target_aspect = self.config.target_width / self.config.target_height
        return source_aspect <= target_aspect * 1.1

    def _smart_crop_dims(self, width: int, height: int) -> Tuple[int, int]:
        """Calcula dimensoes (w, h) do crop 9:16 dentro do quadro original."""
        target_aspect = self.config.target_width / self.config.target_height
        if width / height >= target_aspect:
            crop_w = _to_even(height * target_aspect)
            crop_h = height
        else:
            crop_w = width
            crop_h = _to_even(width / target_aspect)
        return crop_w, crop_h

    def _crop_x_expression(
        self,
        trajectory: TrackingTrajectory,
        src_width: int,
        crop_w: int,
    ) -> str:
        """Expressao piecewise-linear de ``x`` para o filtro ``crop``.

        Cada knot da trajetoria suavizada vira um ramo ``if(between(t,...))``
        com interpolacao linear, mantendo a camera em movimento de grua/pan
        continuo. O valor final (cauda) permanece no ultimo knot e cada knot
        e clampeado estritamente em ``[0, W - W_crop]``.

        A expressao depende de virgulas (funcoes ``if``/``between``), portanto
        deve ser envolvida em aspas simples no filtergraph (ver
        :meth:`build_filter_complex`), evitando que o parser do FFmpeg trate
        as virgulas como separadores de filtros.
        """
        points = trajectory.points
        if not points:
            return f"{(src_width - crop_w) / 2.0:.2f}"

        max_x = max(0, src_width - crop_w)
        ts = [p.timestamp_ms / 1000.0 for p in points]
        xs = [
            max(0.0, min(float(max_x), p.smoothed_center_x * src_width - crop_w / 2.0))
            for p in points
        ]

        # Knot predecessor em t=0 para cobrir o trecho anterior ao primeiro novo
        seg_ts: List[float] = []
        seg_xs: List[float] = []
        if ts[0] > 0:
            seg_ts.append(0.0)
            seg_xs.append(xs[0])
        seg_ts.extend(ts)
        seg_xs.extend(xs)

        if len(seg_ts) == 1:
            return f"{seg_xs[-1]:.2f}"

        expr = f"{seg_xs[-1]:.2f}"
        for i in range(len(seg_ts) - 2, -1, -1):
            t0, t1 = seg_ts[i], seg_ts[i + 1]
            x0, x1 = seg_xs[i], seg_xs[i + 1]
            if t1 - t0 <= 0:
                continue
            slope = (x1 - x0) / (t1 - t0)
            branch = f"({x0:.2f}+{slope:.4f}*(t-{t0:.3f}))"
            expr = f"if(between(t,{t0:.3f},{t1:.3f}),{branch},{expr})"
        return expr

    def _aesthetic_fill_filter(self, target_width: int, target_height: int) -> str:
        """Fundo desfocado 9:16 com o video original nítido centralizado."""
        blur = self.config.blur_radius
        return (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height},"
            f"boxblur=lr={blur}:lp={blur}[bgblur];"
            f"[fg]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease[fgfit];"
            f"[bgblur][fgfit]overlay=(W-w)/2:(H-h)/2[vout]"
        )

    # ------------------------------------------------------------------ #
    # Renderizacao
    # ------------------------------------------------------------------ #
    def reframe(
        self,
        input_video: Union[str, Path],
        output_video: Union[str, Path],
        cut_start_ms: Optional[int] = None,
        cut_end_ms: Optional[int] = None,
    ) -> ReframerResult:
        """Converte o video horizontal para 9:16 e salva em ``output_video``.

        Raises:
            FileNotFoundError: se ``input_video`` nao existir.
            ValueError: se o arquivo nao possuir stream de video ou a janela
                temporal for invalida.
            ReframerError: se o FFmpeg falhar durante a renderizacao.
        """
        src = Path(input_video)
        out = Path(output_video)

        if not src.is_file():
            raise FileNotFoundError(f"Arquivo de video nao encontrado: {src}")

        info = self.probe_service.probe(src)
        if not info.has_video:
            raise ValueError(f"Arquivo de midia sem stream de video: {src}")

        width = info.video_width or 1920
        height = info.video_height or 1080
        start_ms, end_ms = self._resolve_window(info, cut_start_ms, cut_end_ms)

        mode, trajectory = self._resolve_mode(info, start_ms, end_ms)
        relative = self._relative_trajectory(trajectory, start_ms, end_ms)
        filter_complex = self.build_filter_complex(width, height, relative, mode)

        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = self._resolve_ffmpeg()

        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_ms / 1000.0:.3f}",
            "-to",
            f"{end_ms / 1000.0:.3f}",
            "-i",
            str(src),
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
        ]

        if info.has_audio:
            cmd.extend(["-map", "0:a", "-c:a", "copy"])

        cmd.extend([
            "-c:v",
            self.config.video_codec,
            "-crf",
            str(self.config.crf),
            "-preset",
            self.config.preset,
            "-pix_fmt",
            "yuv420p",
            str(out),
        ])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ReframerError(
                f"FFmpeg falhou ao reenquadrar {src} ({proc.returncode}):\n{proc.stderr}"
            )

        ratio = trajectory.face_detected_ratio if trajectory else 0.0
        return ReframerResult(
            output_path=str(out),
            duration_sec=(end_ms - start_ms) / 1000.0,
            width=self.config.target_width,
            height=self.config.target_height,
            mode_used=mode,
            face_detected_ratio=ratio,
            trajectory=trajectory,
        )

    # ------------------------------------------------------------------ #
    # Decisao de modo e janela temporal
    # ------------------------------------------------------------------ #
    def _resolve_window(
        self,
        info: MediaInfo,
        cut_start_ms: Optional[int],
        cut_end_ms: Optional[int],
    ) -> Tuple[int, int]:
        """Normaliza a janela temporal ``[start, end]`` do recorte."""
        duration_ms = info.duration_ms
        start = 0 if cut_start_ms is None else max(0, cut_start_ms)
        end = duration_ms if cut_end_ms is None else min(duration_ms, cut_end_ms)
        if start >= duration_ms:
            raise ValueError(
                f"cut_start_ms ({start}) excede a duracao do video ({duration_ms}ms)"
            )
        if end <= start:
            raise ValueError(
                f"Janela temporal invalida: start={start}ms, end={end}ms"
            )
        return start, end

    def _resolve_mode(
        self,
        info: MediaInfo,
        start_ms: int,
        end_ms: int,
    ) -> Tuple[CropMode, Optional[TrackingTrajectory]]:
        """Decide o modo efetivo (resolvendo ``AUTO``) e rastreia a face."""
        requested = self.config.crop_mode
        if requested == CropMode.AESTHETIC_FILL:
            return CropMode.AESTHETIC_FILL, None

        trajectory = self.face_tracker.track_video(
            info.path,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if requested == CropMode.SMART_CROP:
            return CropMode.SMART_CROP, trajectory

        resolved = (
            CropMode.SMART_CROP
            if trajectory.face_detected_ratio >= self.config.min_face_detection_ratio
            else CropMode.AESTHETIC_FILL
        )
        trajectory = trajectory.model_copy(update={"dominant_mode": resolved})
        return resolved, trajectory

    @staticmethod
    def _relative_trajectory(
        trajectory: Optional[TrackingTrajectory],
        start_ms: int,
        end_ms: int,
    ) -> TrackingTrajectory:
        """Desloca os timestamps para a timeline do recorte (``t`` relativo)."""
        if trajectory is None:
            return TrackingTrajectory(
                points=[],
                face_detected_ratio=0.0,
                average_center_x=0.5,
                dominant_mode=CropMode.AESTHETIC_FILL,
            )
        points = [
            point
            for point in trajectory.points
            if start_ms <= point.timestamp_ms <= end_ms
        ]
        relative = [
            point.model_copy(update={"timestamp_ms": point.timestamp_ms - start_ms})
            for point in points
        ]
        return trajectory.model_copy(update={"points": relative})

    def _resolve_ffmpeg(self) -> str:
        binary = self._ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise RuntimeError("ffmpeg nao encontrado no PATH")
        return binary


__all__ = ["ReframerError", "VerticalReframer"]
