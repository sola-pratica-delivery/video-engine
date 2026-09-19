"""Processamento de Dynamic Punch-in Zoom em cortes e transicoes.

Implementa a Spec da Issue #8 (SDD) e a decisao semantica da Issue #19:
- Alternancia harmonica entre plano normal (100%) e zoom punch-in (ex: 115%)
  a cada 8 a 15 segundos (heuristica temporal com pausas do VAD).
- Decisao semantica opcional via Google AI Studio (``decision_mode="gemini"``):
  punch-in zoom aplicado em momentos de alto impacto retorico (argumentos
  cencentrais, alertas, ganchos e punchlines), com fallback gracioso para a
  heuristica temporal em qualquer falha de chave/rede/API.
- Sincronizacao de cortes com pausas estruturais da fala.
- Enquadramento centralizado na face do apresentador (ou ponto customizado).
- Interpolacao de alta fidelidade Lanczos ou Bicubic.
- Preservacao integral do stream de audio via direct stream copy (-c:a copy).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

from video_engine.audio.models import TimeInterval
from video_engine.captions.models import TranscriptionResult
from video_engine.editing.media_probe import MediaProbe
from video_engine.video.models import (
    DecisionMode,
    DynamicZoomConfig,
    DynamicZoomResult,
    SemanticZoomInterval,
    ZoomMode,
    ZoomShot,
)
from video_engine.video.semantic_analyzer import (
    SemanticZoomAnalyzer,
    SemanticZoomError,
)

logger = logging.getLogger(__name__)


class DynamicZoomProcessor:
    """Motor de Dynamic Punch-in Zoom com decisao temporal e semantica."""

    def __init__(
        self,
        config: Optional[DynamicZoomConfig] = None,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        semantic_analyzer: Optional[SemanticZoomAnalyzer] = None,
    ) -> None:
        self.config = config or DynamicZoomConfig()
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.probe_service = MediaProbe(ffprobe_path=ffprobe_path)
        self.semantic_analyzer = semantic_analyzer or SemanticZoomAnalyzer(config=self.config.gemini)

    def _resolve_ffmpeg(self) -> str:
        binary = self.ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise RuntimeError("ffmpeg nao encontrado no PATH")
        return binary

    # ------------------------------------------------------------------ #
    # Helpers de plano
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normal_shot(
        start_ms: int,
        end_ms: int,
        anchor: Tuple[float, float],
    ) -> ZoomShot:
        return ZoomShot(
            start_ms=start_ms,
            end_ms=end_ms,
            mode=ZoomMode.NORMAL,
            scale=1.0,
            anchor_x=anchor[0],
            anchor_y=anchor[1],
        )

    def _min_shot_ms(self) -> int:
        return int(round(self.config.min_shot_duration_s * 1000))

    def _max_shot_ms(self) -> int:
        return int(round(self.config.max_shot_duration_s * 1000))

    def _target_shot_ms(self) -> int:
        return int(round(self.config.target_shot_duration_s * 1000))

    def _semantic_enabled(
        self,
        total_duration_ms: int,
        transcription: Optional[TranscriptionResult],
    ) -> bool:
        """Define se vale tentar a decisao semantica (Gemini) para este video."""
        if self.config.decision_mode != DecisionMode.GEMINI:
            return False
        if not transcription or not transcription.words:
            return False
        if total_duration_ms <= self._min_shot_ms():
            return False
        return True

    # ------------------------------------------------------------------ #
    # Plano heuristico (Issue #8)
    # ------------------------------------------------------------------ #
    def _plan_heuristic_shots(
        self,
        total_duration_ms: int,
        pause_intervals: Optional[Sequence[TimeInterval]],
        anchor: Tuple[float, float],
    ) -> List[ZoomShot]:
        """Planeja a sequencia alternando NORMAL/ZOOM a cada 8 a 15s ou em pausas."""
        if total_duration_ms <= 0:
            return []

        min_shot_ms = self._min_shot_ms()
        max_shot_ms = self._max_shot_ms()
        target_shot_ms = self._target_shot_ms()

        # Se a duracao total for menor que a duracao minima, mantem plano normal unico
        if total_duration_ms <= min_shot_ms:
            return [self._normal_shot(0, total_duration_ms, anchor)]

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
                if shots:
                    prev = shots.pop()
                    shots.append(
                        prev.model_copy(update={"end_ms": total_duration_ms})
                    )
                else:
                    shots.append(self._normal_shot(0, total_duration_ms, anchor))
                break

            min_cut = current_time + min_shot_ms
            max_cut = min(total_duration_ms, current_time + max_shot_ms)
            ideal_cut = min(total_duration_ms, current_time + target_shot_ms)

            chosen_cut: Optional[int] = None
            candidate_pauses = [
                p
                for p in sorted_pauses
                if (min_cut <= p.end_ms <= max_cut) or (min_cut <= p.start_ms <= max_cut)
            ]

            if candidate_pauses:
                def _dist(p: TimeInterval) -> float:
                    mid = (p.start_ms + p.end_ms) / 2.0
                    return abs(mid - ideal_cut)

                best_pause = min(candidate_pauses, key=_dist)
                pause_cut = best_pause.end_ms
                chosen_cut = min(max(min_cut, pause_cut), max_cut)
            else:
                chosen_cut = ideal_cut

            if (
                total_duration_ms - chosen_cut < min_shot_ms
                and total_duration_ms - current_time <= max_shot_ms
            ):
                chosen_cut = total_duration_ms

            scale = self.config.zoom_scale if current_mode == ZoomMode.ZOOM else 1.0
            shots.append(
                ZoomShot(
                    start_ms=current_time,
                    end_ms=chosen_cut,
                    mode=current_mode,
                    scale=scale,
                    anchor_x=anchor[0],
                    anchor_y=anchor[1],
                )
            )

            current_time = chosen_cut
            current_mode = ZoomMode.ZOOM if current_mode == ZoomMode.NORMAL else ZoomMode.NORMAL

        return shots

    # ------------------------------------------------------------------ #
    # Plano semantico (Issue #19)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _align_windows_to_pauses(
        windows: List[SemanticZoomInterval],
        pause_intervals: Sequence[TimeInterval],
        total_ms: int,
        tolerance_ms: int,
    ) -> List[SemanticZoomInterval]:
        """Desloca bordas das janelas para pausas de fala proximas (respeitando tolerancia)."""
        pauses = sorted(pause_intervals, key=lambda p: (p.start_ms, p.end_ms))
        aligned: List[SemanticZoomInterval] = []
        for window in windows:
            start = window.start_ms
            end = window.end_ms

            candidates_in = [p for p in pauses if 0 <= start - p.end_ms <= tolerance_ms]
            if candidates_in:
                best = min(candidates_in, key=lambda p: start - p.end_ms)
                start = max(0, best.end_ms)

            candidates_out = [p for p in pauses if 0 <= p.start_ms - end <= tolerance_ms]
            if candidates_out:
                best = min(candidates_out, key=lambda p: p.start_ms - end)
                end = min(total_ms, best.start_ms)

            if end <= start:
                aligned.append(window.model_copy())
            else:
                aligned.append(
                    window.model_copy(update={"start_ms": start, "end_ms": end})
                )
        return aligned

    @classmethod
    def _fit_semantic_windows(
        cls,
        intervals: Sequence[SemanticZoomInterval],
        total_ms: int,
        min_ms: int,
        max_ms: int,
    ) -> List[SemanticZoomInterval]:
        """Clampa e ajusta intervalos semantico para durar entre min_ms e max_ms."""
        windows: List[SemanticZoomInterval] = []
        for interval in intervals:
            if not isinstance(interval, SemanticZoomInterval):
                try:
                    interval = SemanticZoomInterval.model_validate(interval)
                except Exception:  # noqa: BLE001
                    continue
            start = max(0, interval.start_ms)
            end = min(total_ms, interval.end_ms)
            if end <= start or start >= total_ms:
                continue

            duration = end - start
            if duration < min_ms:
                center = (start + end) // 2
                half = min_ms // 2
                start = max(0, center - half)
                end = min(total_ms, start + min_ms)
                if end - start < min_ms:
                    if start == 0:
                        end = min(total_ms, min_ms)
                    elif end == total_ms:
                        start = max(0, total_ms - min_ms)
                    else:
                        continue
            elif duration > max_ms:
                center = (start + end) // 2
                half = max_ms // 2
                start = max(0, center - half)
                end = min(total_ms, start + max_ms)

            if end - start < min_ms:
                continue
            windows.append(
                SemanticZoomInterval(
                    start_ms=start,
                    end_ms=end,
                    reason=interval.reason,
                    confidence=interval.confidence,
                )
            )

        # Uniao de sobreposicoes, capping em max_ms para nao criar zoom continuo longo
        merged: List[SemanticZoomInterval] = []
        for window in sorted(windows, key=lambda w: (w.start_ms, w.end_ms)):
            if merged and window.start_ms < merged[-1].end_ms:
                last = merged[-1]
                new_end = max(last.end_ms, window.end_ms)
                if new_end - last.start_ms > max_ms:
                    new_end = last.start_ms + max_ms
                merged[-1] = last.model_copy(update={"end_ms": new_end})
            else:
                merged.append(window.model_copy())
        return merged

    def _build_semantic_shots(
        self,
        total_duration_ms: int,
        intervals: Sequence[SemanticZoomInterval],
        pause_intervals: Optional[Sequence[TimeInterval]],
        anchor: Tuple[float, float],
    ) -> List[ZoomShot]:
        """Converte intervalos semanticos em planos respeitando os guardrails ergonomicos."""
        min_ms = self._min_shot_ms()
        max_ms = self._max_shot_ms()

        if total_duration_ms <= min_ms:
            return [self._normal_shot(0, total_duration_ms, anchor)]

        windows = self._fit_semantic_windows(intervals, total_duration_ms, min_ms, max_ms)
        if not windows:
            return []

        if pause_intervals:
            windows = self._align_windows_to_pauses(
                windows,
                pause_intervals,
                total_duration_ms,
                tolerance_ms=min_ms,
            )

        scale = self.config.zoom_scale
        shots: List[ZoomShot] = []
        current = 0
        index = 0

        while current < total_duration_ms:
            remaining = total_duration_ms - current
            if remaining <= min_ms:
                if shots:
                    prev = shots.pop()
                    shots.append(prev.model_copy(update={"end_ms": total_duration_ms}))
                else:
                    shots.append(self._normal_shot(0, total_duration_ms, anchor))
                break

            window = windows[index] if index < len(windows) else None
            if window is None:
                end = min(total_duration_ms, current + max_ms)
                shots.append(self._normal_shot(current, end, anchor))
                current = end
                continue

            if window.end_ms <= current:
                index += 1
                continue

            zoom_start = max(current, window.start_ms)
            zoom_end = min(total_duration_ms, window.end_ms)

            if zoom_end - zoom_start > max_ms:
                zoom_end = zoom_start + max_ms
            if zoom_end <= zoom_start:
                index += 1
                continue

            # Bordas curtas antes do zoom sao absorvidas no proprio plano de zoom
            if zoom_start - current < min_ms:
                zoom_start = current
                if zoom_end - zoom_start < min_ms:
                    zoom_end = min(total_duration_ms, zoom_start + min_ms)

            if zoom_start > current:
                lead_end = min(zoom_start, current + max_ms)
                shots.append(self._normal_shot(current, lead_end, anchor))
                current = lead_end
                if current < zoom_start:
                    continue

            # Cauda curta apos o zoom e absorvida para fechar o plano sem interrupcao
            if total_duration_ms - zoom_end < min_ms:
                zoom_end = total_duration_ms

            shots.append(
                ZoomShot(
                    start_ms=zoom_start,
                    end_ms=zoom_end,
                    mode=ZoomMode.ZOOM,
                    scale=scale,
                    anchor_x=anchor[0],
                    anchor_y=anchor[1],
                )
            )
            current = zoom_end
            index += 1

        return shots

    # ------------------------------------------------------------------ #
    # Planejamento publico
    # ------------------------------------------------------------------ #
    def plan_zoom_shots(
        self,
        total_duration_ms: int,
        pause_intervals: Optional[Sequence[TimeInterval]] = None,
        anchor_override: Optional[Tuple[float, float]] = None,
        transcription: Optional[TranscriptionResult] = None,
    ) -> List[ZoomShot]:
        """Planeja a sequencia de planos com decisao semantica ou temporal.

        Quando ``decision_mode="gemini"`` e uma transcricao e fornecida, tenta
        a decisao semantica via ``semantic_analyzer``; qualquer falha (chave
        ausente, timeout, 429, JSON invalido) ativa fallback transparente para
        a heuristica temporal da Issue #8.
        """
        shots, _ = self.plan_zoom_shots_with_strategy(
            total_duration_ms=total_duration_ms,
            pause_intervals=pause_intervals,
            anchor_override=anchor_override,
            transcription=transcription,
        )
        return shots

    def plan_zoom_shots_with_strategy(
        self,
        total_duration_ms: int,
        pause_intervals: Optional[Sequence[TimeInterval]] = None,
        anchor_override: Optional[Tuple[float, float]] = None,
        transcription: Optional[TranscriptionResult] = None,
    ) -> Tuple[List[ZoomShot], Literal["gemini", "heuristic"]]:
        """Retorna a lista de planos e a estrategia utilizada ('gemini' ou 'heuristic').

        Compatível com a heuristica da Issue #8; quando ``decision_mode="gemini"``
        e a transcricao possuir palavras alinhadas, tenta a decisao semantica via
        ``semantic_analyzer``. Qualquer falha (chave ausente, timeout, 429, JSON
        invalido, lista vazia) ativa fallback transparente para a heuristica.
        """
        anchor = (
            (anchor_override[0], anchor_override[1])
            if anchor_override is not None
            else (self.config.anchor_x, self.config.anchor_y)
        )

        if self._semantic_enabled(total_duration_ms, transcription):
            try:
                intervals = self.semantic_analyzer.analyze(transcription, total_duration_ms)
                shots = self._build_semantic_shots(
                    total_duration_ms,
                    intervals,
                    pause_intervals,
                    anchor,
                )
                if shots:
                    logger.info("Dynamic zoom planejado semanticamente: %d planos", len(shots))
                    return shots, "gemini"
            except SemanticZoomError as exc:
                logger.warning("Falha na decisao semantica de zoom; usando heuristica: %s", exc)
            except Exception as exc:  # pragma: no cover - salvaguarda extra
                logger.warning("Falha inesperada na decisao semantica; usando heuristica: %s", exc)

        return self._plan_heuristic_shots(total_duration_ms, pause_intervals, anchor), "heuristic"

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

        first_zoom = zoom_shots[0]
        scale = first_zoom.scale
        anchor_x = first_zoom.anchor_x
        anchor_y = first_zoom.anchor_y

        crop_w = int(round(width / scale))
        if crop_w % 2 != 0:
            crop_w -= 1
        crop_h = int(round(height / scale))
        if crop_h % 2 != 0:
            crop_h -= 1

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
        transcription: Optional[TranscriptionResult] = None,
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

        shots, strategy = self.plan_zoom_shots_with_strategy(
            total_duration_ms=info.duration_ms,
            pause_intervals=pause_intervals,
            anchor_override=face_center,
            transcription=transcription,
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
            strategy_used=strategy,
        )


__all__ = ["DynamicZoomProcessor"]
