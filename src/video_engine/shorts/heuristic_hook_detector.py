"""Deteccao offline/deterministica de ganchos virais (Issue #15).

Implementa ``HeuristicHookDetector``, usado como fallback automatico quando o
Google AI Studio nao esta disponivel (sem chave, timeout, HTTP 429/5xx) ou
quando ``DecisionMode.HEURISTIC`` e selecionado. Localiza ancas de alto impacto
através de regras lexicais/estruturais: perguntas retoricas, palavras de alto
impacto, quebras adversativas (\"mas\", \"porém\", \"o segredo\", \"ninguém te
conta\") e frases provocativas, associadas a picos de energia vocal quando o
perfil acustico e fornecido.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.shorts.models import EnergyAnalysisResult, SemanticHookInterval

_IMPACT_WORDS = frozenset(
    {
        "segredo",
        "secreto",
        "atencao",
        "alerta",
        "perceba",
        "repare",
        "descobri",
        "revelo",
        "revela",
        "revelar",
        "revelacao",
        "inacreditavel",
        "incrivel",
        "surpreendente",
        "importante",
        "proibido",
        "perigoso",
        "ninguem",
        "sabia",
        "verdade",
        "mudanca",
    }
)

_ADVERSATIVES = frozenset({"mas", "porem", "contudo", "entretanto", "todavia"})

_PHRASES = frozenset(
    {
        "ninguem te conta",
        "ninguem vai te contar",
        "ninguem te contou",
        "o segredo e",
        "a verdade e",
        "a verdade e que",
        "preste atencao",
        "presta atencao",
        "isso vai mudar",
        "isso vai mudar tudo",
        "nao acredite em mim",
        "vou te mostrar",
        "quer saber por que",
        "voce sabia",
        "o motivo e",
        "por que voce",
        "eu vou te contar",
        "prossiga com atencao",
    }
)


def _norm_fragment(text: str) -> str:
    """Lowercase sem acentos/pontuacao para comparacao deterministica."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())


class HeuristicHookDetector:
    """Localiza ganchos virais com regras lexicais/estruturais offline."""

    def __init__(self, energy_profile: Optional[EnergyAnalysisResult] = None) -> None:
        self.energy_profile = energy_profile

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def detect(self, transcription: TranscriptionResult) -> List[SemanticHookInterval]:
        """Retorna candidatos a gancho (por oracao) ancorados em sinais lexicos.

        Cada candidato cobre uma oracao completa (``CaptionSegment``) e carrega
        ``confidence`` em [0.0, 1.0]. O ``HookDetector`` aplica snapping,
        ajuste de duracao e NMS sobre essa lista.
        """
        if not transcription or not transcription.words:
            return []

        words = sorted(transcription.words, key=lambda w: (w.start_ms, w.end_ms))
        segments = list(transcription.segments)
        anchors = self._find_anchors(words, segments)

        per_segment: Dict[int, Tuple[float, int, str]] = {}
        for seg_idx, word_idx, score, reason in anchors:
            key = seg_idx if seg_idx is not None else -1
            current = per_segment.get(key)
            if current is None or score > current[0]:
                per_segment[key] = (score, word_idx, reason)

        candidates: List[SemanticHookInterval] = []
        for seg_idx, (score, word_idx, reason) in per_segment.items():
            if seg_idx is not None and 0 <= seg_idx < len(segments):
                segment = segments[seg_idx]
                start_ms, end_ms = segment.start_ms, segment.end_ms
                hook_text = segment.text
            else:
                word = words[word_idx]
                start_ms, end_ms = word.start_ms, word.end_ms
                hook_text = word.word

            final_score = self._apply_energy_boost(score, start_ms, end_ms)
            candidates.append(
                SemanticHookInterval(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    hook_text=hook_text,
                    summary=self._window_text(words, start_ms, end_ms),
                    reason=reason,
                    confidence=round(max(0.0, min(1.0, final_score)), 6),
                )
            )
        return candidates

    # ------------------------------------------------------------------ #
    # Deteccao de ancas
    # ------------------------------------------------------------------ #
    def _find_anchors(
        self,
        words: Sequence[WordTimestamp],
        segments: Sequence,
    ) -> List[Tuple[Optional[int], int, float, str]]:
        anchors: List[Tuple[Optional[int], int, float, str]] = []

        for i, word in enumerate(words):
            raw = word.word
            norm = _norm_fragment(raw)
            tokens = norm.split()
            reasons: List[str] = []
            score = 0.50

            if "?" in raw:
                reasons.append("interrogacao retorica")
                score += 0.25

            if tokens and tokens[0] in _ADVERSATIVES:
                reasons.append(f"quebra adversativa ('{tokens[0]}')")
                score += 0.25

            impact_hit = next((t for t in tokens if t in _IMPACT_WORDS), None)
            if impact_hit:
                reasons.append(f"palavra de alto impacto ('{impact_hit}')")
                score += 0.15

            phrase_hit = self._phrase_starting_at(words, i)
            if phrase_hit:
                reasons.append(f"frase provocativa ('{phrase_hit}')")
                score += 0.30

            if not reasons:
                continue

            seg_idx = self._segment_index(segments, word.start_ms)
            anchors.append(
                (seg_idx, i, round(max(0.0, min(1.0, score)), 6), "; ".join(reasons))
            )
        return anchors

    def _phrase_starting_at(
        self,
        words: Sequence[WordTimestamp],
        index: int,
    ) -> Optional[str]:
        """Procura uma frase provocativa iniciada no indice ``index`` (ate 4 palavras)."""
        tokens: List[str] = []
        for j in range(index, min(index + 4, len(words))):
            tokens.extend(_norm_fragment(words[j].word).split())
        for size in range(2, min(4, len(tokens)) + 1):
            candidate = " ".join(tokens[:size])
            if candidate in _PHRASES:
                return candidate
        return None

    @staticmethod
    def _segment_index(segments: Sequence, time_ms: int) -> Optional[int]:
        for idx, segment in enumerate(segments):
            if segment.start_ms <= time_ms < segment.end_ms:
                return idx
        return None

    def _apply_energy_boost(self, score: float, start_ms: int, end_ms: int) -> float:
        if self.energy_profile is None:
            return score
        peak = self.energy_profile.max_window_energy(start_ms, end_ms)
        if peak >= 0.85:
            return score + 0.15
        if peak >= 0.60:
            return score + 0.10
        return score

    @staticmethod
    def _window_text(words: Sequence[WordTimestamp], start_ms: int, end_ms: int) -> str:
        parts = [w.word for w in words if w.start_ms >= start_ms and w.end_ms <= end_ms]
        text = " ".join(parts)
        if len(text) > 220:
            text = text[:217] + "..."
        return text


__all__ = ["HeuristicHookDetector"]
