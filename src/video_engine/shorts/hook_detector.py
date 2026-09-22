"""Detector principal de ganchos virais e cortes autocontidos (Issue #15).

Orquestra a analise semantica (Google AI Studio) com a acustica (energia/RMS)
e aplica o fallback deterministico para heuristaca lexical/acustica quando o
Gemini esta indisponivel. Garante cortes de 25s a 58s, snapping a oracoes
(``CaptionSegment``), subconjunto de palavras (``WordTimestamp``) com
timestamps originais preservados e Non-Maximum Suppression (NMS) temporal.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from video_engine.captions.models import CaptionSegment, TranscriptionResult, WordTimestamp
from video_engine.shorts.energy_analyzer import EnergyAnalyzer
from video_engine.shorts.heuristic_hook_detector import HeuristicHookDetector
from video_engine.shorts.models import (
    DetectionMode,
    EnergyAnalysisResult,
    HookDetectionResult,
    HookDetectorConfig,
    SemanticHookInterval,
    ShortCandidateCut,
)
from video_engine.shorts.semantic_hook_analyzer import SemanticHookAnalyzer

logger = logging.getLogger(__name__)


def temporal_iou(
    a_start_ms: int,
    a_end_ms: int,
    b_start_ms: int,
    b_end_ms: int,
) -> float:
    """Intersecao sobre uniao (IoU) temporal entre dois intervalos."""
    intersection = max(0, min(a_end_ms, b_end_ms) - max(a_start_ms, b_start_ms))
    union = (a_end_ms - a_start_ms) + (b_end_ms - b_start_ms) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def suppress_overlaps(
    candidates: Sequence[ShortCandidateCut],
    max_overlap_iou: float,
    max_cuts: int,
) -> List[ShortCandidateCut]:
    """NMS temporal: mantem cortes ranqueados por ``virality_score`` sem sobreposicao.

    Os cortes sao ordenados por viralidade (desc); para cada candidato, ele e
    mantido somente se sua IoU com todos os ja mantidos for menor ou igual a
    ``max_overlap_iou`` e se ainda houver espaco ate ``max_cuts``. IDs
    ``cut_01..cut_N`` sao atribuidos em ordem de ranking.
    """
    ranked = sorted(candidates, key=lambda c: c.virality_score, reverse=True)
    kept: List[ShortCandidateCut] = []
    for candidate in ranked:
        if len(kept) >= max_cuts:
            break
        if all(
            temporal_iou(candidate.start_ms, candidate.end_ms, k.start_ms, k.end_ms)
            <= max_overlap_iou
            for k in kept
        ):
            kept.append(candidate)
    kept = sorted(kept, key=lambda c: c.virality_score, reverse=True)
    for i, cut in enumerate(kept, start=1):
        cut.id = f"cut_{i:02d}"
    return kept


class HookDetector:
    """Motor de deteccao de ganchos virais e cortes verticais autocontidos."""

    def __init__(
        self,
        config: Optional[HookDetectorConfig] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        self.config = config or HookDetectorConfig()
        self.semantic_analyzer = SemanticHookAnalyzer(self.config, http_client=http_client)
        self._strategy = "heuristic_fallback"

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def detect_cuts(
        self,
        transcription: TranscriptionResult,
        audio_source: Optional[Union[str, Path, "Any"]] = None,
        total_duration_ms: Optional[int] = None,
    ) -> HookDetectionResult:
        """Detecta 1 a 3 cortes virais autocontidos de 25s a 58s.

        Raises:
            ValueError: transcricao nula ou sem palavras (esperado pelo contrato)
                retorna resultado vazio, sem erros.
        """
        if transcription is None or not transcription.words:
            return HookDetectionResult(cuts=[], total_cuts_found=0, strategy_used=self._strategy)

        duration = self._resolve_duration(transcription, total_duration_ms)
        if duration < self.config.min_duration_ms:
            logger.info(
                "Video com duracao %d ms menor que o minimo de %d ms; nenhum corte.",
                duration,
                self.config.min_duration_ms,
            )
            return HookDetectionResult(cuts=[], total_cuts_found=0, strategy_used=self._strategy)

        energy_profile, audio_analyzed = self._analyze_energy(audio_source)

        intervals = self._collect_intervals(transcription, duration, energy_profile)
        cuts = self._build_cuts(transcription, intervals, duration, energy_profile)

        return HookDetectionResult(
            cuts=cuts,
            total_cuts_found=len(cuts),
            strategy_used=self._strategy,
            audio_energy_analyzed=audio_analyzed,
        )

    # ------------------------------------------------------------------ #
    # Energia acustica
    # ------------------------------------------------------------------ #
    def _analyze_energy(
        self,
        audio_source: Optional[Union[str, Path, "Any"]],
    ) -> Tuple[Optional[EnergyAnalysisResult], bool]:
        if audio_source is None:
            return None, False
        analyzer = EnergyAnalyzer()
        try:
            profile = analyzer.analyze_audio(audio_source)
        except Exception as exc:  # noqa: BLE001 - fallback gracioso e transparente
            logger.warning(
                "Falha ao analisar energia do audio (%s); usando score neutro 0.5.",
                exc,
            )
            return None, False
        return profile, True

    # ------------------------------------------------------------------ #
    # Coleta de intervalos (semantico + fallback heuristico)
    # ------------------------------------------------------------------ #
    def _collect_intervals(
        self,
        transcription: TranscriptionResult,
        duration_ms: int,
        energy_profile: Optional[EnergyAnalysisResult],
    ) -> List[SemanticHookInterval]:
        mode = self.config.decision_mode
        self._strategy = "heuristic_fallback"

        if mode is DetectionMode.HEURISTIC:
            return self._heuristic_detect(transcription, energy_profile)

        try:
            intervals = self.semantic_analyzer.analyze(transcription, duration_ms)
        except Exception as exc:  # noqa: BLE001 - rede, timeout, 429, schema etc
            logger.warning(
                "Decisao semantica indisponivel (%s); fallback heuristico automatico.",
                exc,
            )
            return self._heuristic_detect(transcription, energy_profile)

        if not intervals:
            logger.warning(
                "Decisao semantica sem ganchos; fallback heuristico automatico."
            )
            return self._heuristic_detect(transcription, energy_profile)

        self._strategy = "hybrid" if mode is DetectionMode.HYBRID else "semantic"
        return intervals

    def _heuristic_detect(
        self,
        transcription: TranscriptionResult,
        energy_profile: Optional[EnergyAnalysisResult],
    ) -> List[SemanticHookInterval]:
        detector = HeuristicHookDetector(energy_profile=energy_profile)
        return detector.detect(transcription)

    # ------------------------------------------------------------------ #
    # Construcao dos cortes (snapping + duracao + NMS)
    # ------------------------------------------------------------------ #
    def _build_cuts(
        self,
        transcription: TranscriptionResult,
        intervals: Sequence[SemanticHookInterval],
        total_duration_ms: int,
        energy_profile: Optional[EnergyAnalysisResult],
    ) -> List[ShortCandidateCut]:
        words = sorted(transcription.words, key=lambda w: (w.start_ms, w.end_ms))
        segments = list(transcription.segments)
        b_starts, b_ends = self._boundaries(words, segments)

        candidates: List[ShortCandidateCut] = []
        for interval in intervals:
            cut = self._build_cut(
                interval, words, segments, b_starts, b_ends, total_duration_ms, energy_profile
            )
            if cut is not None:
                candidates.append(cut)

        max_cuts = min(self.config.target_cuts, 3)
        return suppress_overlaps(candidates, self.config.max_overlap_iou, max_cuts)

    def _build_cut(
        self,
        interval: SemanticHookInterval,
        words: Sequence[WordTimestamp],
        segments: Sequence[CaptionSegment],
        b_starts: Sequence[int],
        b_ends: Sequence[int],
        total_duration_ms: int,
        energy_profile: Optional[EnergyAnalysisResult],
    ) -> Optional[ShortCandidateCut]:
        start = max(0, min(interval.start_ms, total_duration_ms))
        end = max(start, min(interval.end_ms, total_duration_ms))
        if end <= start:
            return None

        snapped_start, snapped_end = self._snap_boundaries(start, end, words, segments)
        fitted = self._fit_duration(
            snapped_start, snapped_end, b_starts, b_ends, total_duration_ms
        )
        if fitted is None:
            return None
        cut_start, cut_end = fitted

        cut_words = [w for w in words if w.start_ms >= cut_start and w.end_ms <= cut_end]
        if not cut_words:
            return None

        semantic_score = max(0.0, min(1.0, interval.confidence))
        if energy_profile is not None:
            energy_score = energy_profile.max_window_energy(cut_start, cut_end)
            peak = energy_profile.strongest_peak(cut_start, cut_end)
            peak_ms = peak.peak_ms if peak else None
        else:
            energy_score = 0.5  # score neutro sem analise acustica
            peak_ms = None

        if self.config.decision_mode is DetectionMode.SEMANTIC_ONLY:
            virality = semantic_score
        else:
            virality = (
                self.config.semantic_weight * semantic_score
                + self.config.energy_weight * energy_score
            )
        virality = max(0.0, min(1.0, virality))
        if virality < self.config.min_confidence:
            return None

        return ShortCandidateCut(
            id="",
            start_ms=cut_start,
            end_ms=cut_end,
            duration_ms=cut_end - cut_start,
            hook_text=interval.hook_text.strip() or self._hook_text(cut_words, cut_start),
            summary=interval.summary.strip() or self._window_text(cut_words),
            reason=interval.reason.strip() or "trecho de alto impacto selecionado",
            semantic_score=round(semantic_score, 6),
            energy_score=round(energy_score, 6),
            virality_score=round(virality, 6),
            energy_peak_ms=peak_ms,
            words=cut_words,
        )

    # ------------------------------------------------------------------ #
    # Snapping a oracoes e ajuste de duracao
    # ------------------------------------------------------------------ #
    def _snap_boundaries(
        self,
        start_ms: int,
        end_ms: int,
        words: Sequence[WordTimestamp],
        segments: Sequence[CaptionSegment],
    ) -> Tuple[int, int]:
        starts = [w.start_ms for w in words]
        ends = [w.end_ms for w in words]

        i0 = min(bisect.bisect_right(ends, start_ms), len(words) - 1)
        new_start = words[i0].start_ms
        i1 = max(bisect.bisect_left(starts, end_ms) - 1, i0)
        new_end = words[i1].end_ms

        if self.config.snap_to_sentence_boundaries and segments:
            seg0 = self._segment_at(segments, words[i0].start_ms)
            seg1 = self._segment_at(segments, words[i1].start_ms)
            if seg0 is not None:
                new_start = seg0.start_ms
            if seg1 is not None:
                new_end = seg1.end_ms
        return new_start, new_end

    def _fit_duration(
        self,
        start_ms: int,
        end_ms: int,
        b_starts: Sequence[int],
        b_ends: Sequence[int],
        total_duration_ms: int,
    ) -> Optional[Tuple[int, int]]:
        """Expande/contrai o corte aos limites validos de duracao.

        Retorna ``(start, end)`` dentro de ``[min_duration_ms, max_duration_ms]``
        ou ``None`` quando nao ha material suficiente/elegivel.
        """
        min_d = self.config.min_duration_ms
        max_d = self.config.max_duration_ms
        start, end = start_ms, end_ms

        while end - start < min_d:
            prev_s = self._previous(b_starts, start)
            next_e = self._next(b_ends, end)
            advanced = False
            if prev_s is not None and end - prev_s <= max_d:
                start = prev_s
                advanced = True
            elif next_e is not None and next_e - start <= max_d:
                end = next_e
                advanced = True
            elif prev_s is not None and next_e is not None and next_e - prev_s <= max_d:
                start, end = prev_s, next_e
                advanced = True
            if not advanced:
                break

        if end - start < min_d:
            return None

        while end - start > max_d:
            prev_e = self._previous(b_ends, end)
            next_s = self._next(b_starts, start)
            shrunk = False
            if prev_e is not None and prev_e - start >= min_d:
                end = prev_e
                shrunk = True
            elif next_s is not None and end - next_s >= min_d:
                start = next_s
                shrunk = True
            if not shrunk:
                break

        if end - start > max_d or end - start < min_d:
            return None
        return start, end

    @staticmethod
    def _previous(values: Sequence[int], time_ms: int) -> Optional[int]:
        idx = bisect.bisect_left(values, time_ms) - 1
        return values[idx] if idx >= 0 else None

    @staticmethod
    def _next(values: Sequence[int], time_ms: int) -> Optional[int]:
        idx = bisect.bisect_right(values, time_ms)
        return values[idx] if idx < len(values) else None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _boundaries(
        self,
        words: Sequence[WordTimestamp],
        segments: Sequence[CaptionSegment],
    ) -> Tuple[List[int], List[int]]:
        if self.config.snap_to_sentence_boundaries and segments:
            b_starts = sorted({s.start_ms for s in segments})
            b_ends = sorted({s.end_ms for s in segments})
        else:
            b_starts = sorted({w.start_ms for w in words})
            b_ends = sorted({w.end_ms for w in words})
        return b_starts, b_ends

    @staticmethod
    def _segment_at(segments: Sequence[CaptionSegment], time_ms: int) -> Optional[CaptionSegment]:
        for segment in segments:
            if segment.start_ms <= time_ms < segment.end_ms:
                return segment
        return None

    @staticmethod
    def _resolve_duration(
        transcription: TranscriptionResult,
        total_duration_ms: Optional[int],
    ) -> int:
        if total_duration_ms is not None and total_duration_ms > 0:
            return int(total_duration_ms)
        if transcription.duration_ms > 0:
            return int(transcription.duration_ms)
        if transcription.words:
            return max(w.end_ms for w in transcription.words)
        return 0

    @staticmethod
    def _hook_text(words: Sequence[WordTimestamp], start_ms: int) -> str:
        parts = [w.word for w in words if w.end_ms <= start_ms + 5000]
        return " ".join(parts)

    @staticmethod
    def _window_text(words: Sequence[WordTimestamp]) -> str:
        parts = [w.word for w in words]
        text = " ".join(parts)
        if len(text) > 220:
            text = text[:217] + "..."
        return text


__all__ = ["HookDetector", "suppress_overlaps", "temporal_iou"]
