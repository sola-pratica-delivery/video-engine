"""Processamento de Dynamic Punch-in Zoom em cortes e transicoes.

Implementa a Spec da Issue #8 (SDD):
- Alternancia harmonica entre plano normal (100%) e zoom punch-in (ex: 115%) a cada 8 a 15 segundos.
- Sincronizacao de cortes com pausas estruturais da fala.
- Enquadramento centralizado na face do apresentador (ou ponto de ancoragem customizado).
- Interpolacao de alta fidelidade Lanczos ou Bicubic sem perda perceptivel de resolucao.
- Preservacao integral do stream de audio via direct stream copy (-c:a copy).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from video_engine.audio.models import TimeInterval
from video_engine.editing.media_probe import MediaProbe
from video_engine.video.models import (
    DynamicZoomConfig,
    DynamicZoomResult,
    ZoomMode,
    ZoomShot,
)


class DynamicZoomProcessor:
    """Motor de Dynamic Punch-in Zoom com alinhamento facial e interpolacao Lanczos."""

    def __init__(
        self,
        config: Optional[DynamicZoomConfig] = None,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ) -> None:
        self.config = config or DynamicZoomConfig()
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.probe_service = MediaProbe(ffprobe_path=ffprobe_path)

    def _resolve_ffmpeg(self) -> str:
        binary = self.ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise RuntimeError("ffmpeg nao encontrado no PATH")
        return binary

    def plan_zoom_shots(
        self,
        total_duration_ms: int,
        pause_intervals: Optional[Sequence[TimeInterval]] = None,
        anchor_override: Optional[Tuple[float, float]] = None,
    ) -> List[ZoomShot]:
        """Planeja a sequencia de planos alternando NORMAL e ZOOM a cada 8 a 15s ou em pausas."""
        if total_duration_ms <= 0:
            return []

        anchor_x = anchor_override[0] if anchor_override else self.config.anchor_x
        anchor_y = anchor_override[1] if anchor_override else self.config.anchor_y

        min_shot_ms = int(round(self.config.min_shot_duration_s * 1000))
        max_shot_ms = int(round(self.config.max_shot_duration_s * 1000))
        target_shot_ms = int(round(self.config.target_shot_duration_s * 1000))

        # Se a duracao total for menor que a duracao minima de um plano, mantem plano unico normal
        if total_duration_ms <= min_shot_ms:
            return [
                ZoomShot(
                    start_ms=0,
                    end_ms=total_duration_ms,
                    mode=ZoomMode.NORMAL,
                    scale=1.0,
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                )
            ]

        # Normalizar pausas
        sorted_pauses = (
            sorted(pause_intervals, key=lambda p: (p.start_ms, p.end_ms))
            if pause_intervals
            else []
        )

        shots: List[ZoomShot] = []
        current_time = 0
        current_mode = ZoomMode.NORMAL

        while current_time < total_duration_ms:
            remaining = total_duration_ms - current_time
            if remaining <= min_shot_ms:
                # O restante e menor que o minimo: absorve no plano anterior se houver
                if shots:
                    prev = shots.pop()
                    shots.append(
                        ZoomShot(
                            start_ms=prev.start_ms,
                            end_ms=total_duration_ms,
                            mode=prev.mode,
                            scale=prev.scale,
                            anchor_x=prev.anchor_x,
                            anchor_y=prev.anchor_y,
                        )
                    )
                else:
                    shots.append(
                        ZoomShot(
                            start_ms=0,
                            end_ms=total_duration_ms,
                            mode=ZoomMode.NORMAL,
                            scale=1.0,
                            anchor_x=anchor_x,
                            anchor_y=anchor_y,
                        )
                    )
                break

            min_cut = current_time + min_shot_ms
            max_cut = min(total_duration_ms, current_time + max_shot_ms)
            ideal_cut = min(total_duration_ms, current_time + target_shot_ms)

            # Procurar pausa na janela [min_cut, max_cut]
            chosen_cut: Optional[int] = None
            candidate_pauses = [
                p
                for p in sorted_pauses
                if (min_cut <= p.end_ms <= max_cut) or (min_cut <= p.start_ms <= max_cut)
            ]

            if candidate_pauses:
                # Escolhe a pausa mais proxima do tempo ideal de corte
                def _dist(p: TimeInterval) -> float:
                    mid = (p.start_ms + p.end_ms) / 2.0
                    return abs(mid - ideal_cut)

                best_pause = min(candidate_pauses, key=_dist)
                # Corta no final da pausa para o proximo plano iniciar junto com a proxima fala
                pause_cut = best_pause.end_ms
                chosen_cut = min(max(min_cut, pause_cut), max_cut)
            else:
                chosen_cut = ideal_cut

            # Se o restante apos o corte for menor que min_shot_ms e estiver dentro de max_cut,
            # estende o corte ate o final
            if total_duration_ms - chosen_cut < min_shot_ms and total_duration_ms - current_time <= max_shot_ms:
                chosen_cut = total_duration_ms

            scale = self.config.zoom_scale if current_mode == ZoomMode.ZOOM else 1.0
            shots.append(
                ZoomShot(
                    start_ms=current_time,
                    end_ms=chosen_cut,
                    mode=current_mode,
                    scale=scale,
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                )
            )

            current_time = chosen_cut
            current_mode = ZoomMode.ZOOM if current_mode == ZoomMode.NORMAL else ZoomMode.NORMAL

        return shots

    def build_filter_complex(
        self,
        shots: Sequence[ZoomShot],
        width: int,
        height: int,
    ) -> str:
        """Monta o filtro FFmpeg para crop dinamico e reescalonamento Lanczos/Bicubic."""
        zoom_shots = [s for s in shots if s.mode == ZoomMode.ZOOM and s.scale > 1.0]
        scaling = self.config.scaling_filter

        if not zoom_shots:
            return f"[0:v]scale={width}:{height}:flags={scaling}[vout]"

        # Expressoes de intervalo de tempo para zoom
        between_exprs = [
            f"between(t,{s.start_ms / 1000.0:.3f},{s.end_ms / 1000.0:.3f})"
            for s in zoom_shots
        ]
        active_zoom_condition = "+".join(between_exprs)

        # Suporte a escala e ancoragem do primeiro zoom (ou representativo)
        first_zoom = zoom_shots[0]
        scale = first_zoom.scale
        anchor_x = first_zoom.anchor_x
        anchor_y = first_zoom.anchor_y

        crop_w = int(round(width / scale))
        # Garantir paridade (multiplo de 2) para compatibilidade YUV420p
        if crop_w % 2 != 0:
            crop_w -= 1
        crop_h = int(round(height / scale))
        if crop_h % 2 != 0:
            crop_h -= 1

        # Ancoragem centralizada ou customizada com clamping estrito
        crop_x = int(round(max(0, min(width - crop_w, anchor_x * width - crop_w / 2))))
        crop_y = int(round(max(0, min(height - crop_h, anchor_y * height - crop_h / 2))))
        if crop_x % 2 != 0:
            crop_x = max(0, crop_x - 1)
        if crop_y % 2 != 0:
            crop_y = max(0, crop_y - 1)

        split_filter = "[0:v]split[base][for_zoom]"
        zoom_chain = (
            f"[for_zoom]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={width}:{height}:flags={scaling}[zoomed]"
        )
        overlay_filter = (
            f"[base][zoomed]overlay=0:0:enable='{active_zoom_condition}'[vout]"
        )

        return f"{split_filter};{zoom_chain};{overlay_filter}"

    def apply_zoom(
        self,
        input_video: Union[str, Path],
        output_video: Union[str, Path],
        pause_intervals: Optional[Sequence[TimeInterval]] = None,
        face_center: Optional[Tuple[float, float]] = None,
    ) -> DynamicZoomResult:
        """Aplica o zoom dinamico no video de entrada gerando o arquivo final processado."""
        src = Path(input_video)
        out = Path(output_video)

        if not src.is_file():
            raise FileNotFoundError(f"Arquivo de video nao encontrado: {src}")

        info = self.probe_service.probe(src)
        if not info.has_video:
            raise ValueError(f"Arquivo de midia sem stream de video: {src}")

        width = info.video_width or 1920
        height = info.video_height or 1080

        shots = self.plan_zoom_shots(
            total_duration_ms=info.duration_ms,
            pause_intervals=pause_intervals,
            anchor_override=face_center,
        )

        filter_complex = self.build_filter_complex(shots, width=width, height=height)

        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = self._resolve_ffmpeg()

        cmd = [
            ffmpeg,
            "-y",
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
            raise RuntimeError(
                f"FFmpeg falhou ao aplicar dynamic zoom ({proc.returncode}):\n{proc.stderr}"
            )

        zoom_count = sum(1 for s in shots if s.mode == ZoomMode.ZOOM)
        normal_count = sum(1 for s in shots if s.mode == ZoomMode.NORMAL)

        return DynamicZoomResult(
            output_path=str(out),
            total_duration_ms=info.duration_ms,
            shots=shots,
            zoom_shots_count=zoom_count,
            normal_shots_count=normal_count,
            video_width=width,
            video_height=height,
            scaling_filter=self.config.scaling_filter,
        )


__all__ = ["DynamicZoomProcessor"]
