"""Gerador de legendas animadas ASS (stilo karaoke) para cortes.

Implementa ``AssSubtitleGenerator`` (Issue #9), responsavel por transformar
transcricoes com timestamps por palavra em arquivos ``.ass`` com tipografia
de alto impacto para Shorts/Reels (Montserrat/TheBoldFont, contorno escuro
espesso, sombra e destaque colorido palavra a palavra).

Suporta dois modos de destaque:
- ``WORD_HIGHLIGHT``: um ``Dialogue`` por palavra com a palavra ativa em cor
  vibrante e escala de punch-in sutil, mantendo as demais legiveis.
- ``KARAOKE_TAG``: tags nativas ``\\k``/``\\kf`` que preenchem a frase em
  tempo real conforme o audio.

Suporta recortes temporais com reindexacao automatica (``start_time_ms``/``end_time_ms``),
ideal para cortes extraidos de videos longos (t = 0 no corte).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

from video_engine.captions.ass_models import (
    KaraokeHighlightMode,
    SubtitleCue,
    SubtitleStyleConfig,
)
from video_engine.captions.models import (
    CaptionSegment,
    TranscriptionResult,
    WordTimestamp,
)

TranscriptionInput = Union[TranscriptionResult, List[CaptionSegment], List[WordTimestamp]]

MIN_WORD_DURATION_MS = 50


def hex_to_ass_color(hex_or_ass: str) -> str:
    """Converte ``#RRGGBB``/``#AARRGGBB`` para o formato ASS ``&HAABBGGRR&``.

    Valores já no formato ASS (prefixo ``&H``) sao devolvidos intactos.
    """
    token = hex_or_ass.strip()
    if token.upper().startswith("&H"):
        return token
    digits = token.lstrip("#")
    if len(digits) == 6:
        r, g, b = digits[0:2], digits[2:4], digits[4:6]
        return f"&H00{b}{g}{r}&"
    if len(digits) == 8:
        a, r, g, b = digits[0:2], digits[2:4], digits[4:6], digits[6:8]
        return f"&H{a}{b}{g}{r}&"
    raise ValueError(
        f"Cor invalida para ASS: {hex_or_ass!r}. Use #RRGGBB, #AARRGGBB ou &HAABBGGRR&."
    )


def format_ass_time(ms: int) -> str:
    """Formata milissegundos como ``H:MM:SS.cc`` (centesimos de segundo)."""
    total_cs = max(0, int(round(ms / 10.0)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    seconds, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escapa caracteres especiais do texto para nao quebrar overrides ASS."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\uff5b")
        .replace("}", "\uff5d")
        .replace("\r", " ")
        .replace("\n", " ")
    )


class AssSubtitleGenerator:
    """Motor de formatacao de legendas animadas ASS para cortes (Issue #9)."""

    def __init__(
        self,
        style_config: Optional[SubtitleStyleConfig] = None,
        karaoke_tag: Literal["k", "kf"] = "kf",
    ) -> None:
        self.style = style_config or SubtitleStyleConfig()
        if karaoke_tag not in ("k", "kf"):
            raise ValueError(f"karaoke_tag invalido: {karaoke_tag!r} (use 'k' ou 'kf')")
        self.karaoke_tag = karaoke_tag

    # ------------------------------------------------------------------ #
    # Coleta, reindexacao e particionamento
    # ------------------------------------------------------------------ #
    def _extract_words(self, transcription: TranscriptionInput) -> List[WordTimestamp]:
        if isinstance(transcription, TranscriptionResult):
            if transcription.words:
                return list(transcription.words)
            flat: List[WordTimestamp] = []
            for segment in transcription.segments:
                flat.extend(segment.words)
            return flat
        if isinstance(transcription, list):
            if not transcription:
                return []
            first = transcription[0]
            if isinstance(first, WordTimestamp):
                return [
                    w for w in transcription if isinstance(w, WordTimestamp)
                ]
            if isinstance(first, CaptionSegment):
                flat = []
                for segment in transcription:
                    if isinstance(segment, CaptionSegment):
                        flat.extend(segment.words)
                return flat
        raise TypeError(
            "transcription deve ser TranscriptionResult, lista de CaptionSegment "
            "ou lista de WordTimestamp"
        )

    @staticmethod
    def _filter_and_reindex(
        words: List[WordTimestamp],
        start_time_ms: int,
        end_time_ms: Optional[int],
    ) -> List[WordTimestamp]:
        """Filtra palavras fora do intervalo e rebaseia os timestamps para t=0."""
        result: List[WordTimestamp] = []
        for word in words:
            raw_end = (
                word.end_ms
                if word.end_ms > word.start_ms
                else word.start_ms + MIN_WORD_DURATION_MS
            )
            if raw_end <= start_time_ms:
                continue
            if end_time_ms is not None and word.start_ms >= end_time_ms:
                continue
            new_start = max(0, word.start_ms - start_time_ms)
            new_end = max(new_start, raw_end - start_time_ms)
            result.append(
                WordTimestamp(
                    word=word.word,
                    start_ms=new_start,
                    end_ms=new_end,
                    probability=word.probability,
                )
            )
        return result

    @staticmethod
    def _ensure_word_duration(word: WordTimestamp) -> WordTimestamp:
        """Garante duracao minima (>= 50ms) para palavras instantaneas."""
        if word.end_ms <= word.start_ms:
            return word.model_copy(
                update={"end_ms": word.start_ms + MIN_WORD_DURATION_MS}
            )
        return word

    def _partition_cues(self, words: List[WordTimestamp]) -> List[SubtitleCue]:
        """Agrupa palavras em blocos de ate ``max_words_per_line`` palavras."""
        cues: List[SubtitleCue] = []
        limit = self.style.max_words_per_line
        for index in range(0, len(words), limit):
            chunk = words[index : index + limit]
            cues.append(
                SubtitleCue(
                    start_ms=chunk[0].start_ms,
                    end_ms=chunk[-1].end_ms,
                    words=list(chunk),
                    text=" ".join(w.word for w in chunk),
                )
            )
        return cues

    # ------------------------------------------------------------------ #
    # Geracao de eventos
    # ------------------------------------------------------------------ #
    def build_cues(
        self,
        transcription: TranscriptionInput,
        start_time_ms: int = 0,
        end_time_ms: Optional[int] = None,
    ) -> List[SubtitleCue]:
        """Constroi blocos de exibicao rebaseados a partir do trecho do corte."""
        words = self._extract_words(transcription)
        reindexed = self._filter_and_reindex(words, start_time_ms, end_time_ms)
        normalized = [self._ensure_word_duration(w) for w in reindexed]
        return self._partition_cues(normalized)

    @staticmethod
    def _dialogue_line(start_ms: int, end_ms: int, text: str) -> str:
        return (
            f"Dialogue: 0,{format_ass_time(start_ms)},{format_ass_time(end_ms)},"
            f"Default,,0,0,0,,{text}"
        )

    def _word_highlight_line(self, cue: SubtitleCue, active_index: int) -> str:
        scale = int(round(self.style.active_word_scale * 100))
        highlight = hex_to_ass_color(self.style.highlight_color)
        primary = hex_to_ass_color(self.style.primary_color)
        parts = ["{\\r}"]
        for index, word in enumerate(cue.words):
            token = _escape_ass_text(word.word)
            if index == active_index:
                parts.append(f"{{\\c{highlight}\\fscx{scale}\\fscy{scale}}}{token}")
            else:
                parts.append(f"{{\\c{primary}\\fscx100\\fscy100}}{token}")
        return " ".join(parts)

    def _karaoke_line(self, cue: SubtitleCue) -> str:
        parts = []
        for word in cue.words:
            duration = max(MIN_WORD_DURATION_MS, word.end_ms - word.start_ms)
            centisecs = max(1, int(round(duration / 10.0)))
            parts.append(
                f"{{\\{self.karaoke_tag}{centisecs}}}{_escape_ass_text(word.word)}"
            )
        return " ".join(parts)

    def _build_dialogues(self, cues: List[SubtitleCue]) -> List[str]:
        dialogues: List[str] = []
        if self.style.highlight_mode == KaraokeHighlightMode.KARAOKE_TAG:
            for cue in cues:
                dialogues.append(
                    self._dialogue_line(cue.start_ms, cue.end_ms, self._karaoke_line(cue))
                )
            return dialogues

        for cue in cues:
            for index in range(len(cue.words)):
                word = cue.words[index]
                dialogues.append(
                    self._dialogue_line(
                        word.start_ms,
                        word.end_ms,
                        self._word_highlight_line(cue, index),
                    )
                )
        return dialogues

    def _ass_document(self, res_x: int, res_y: int, dialogues: List[str]) -> str:
        style = self.style
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {res_x}",
            f"PlayResY: {res_y}",
            "ScaledBorderAndShadow: yes",
            "WrapStyle: 0",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
                "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
                "MarginL, MarginR, MarginV, Encoding"
            ),
            (
                f"Style: Default,{style.font_name},{style.font_size},"
                f"{hex_to_ass_color(style.primary_color)},"
                f"{hex_to_ass_color(style.highlight_color)},"
                f"{hex_to_ass_color(style.outline_color)},"
                f"{hex_to_ass_color(style.shadow_color)},"
                f"{1 if style.bold else 0},0,0,0,100,100,0,0,1,"
                f"{style.outline_width},{style.shadow_depth},{style.alignment},"
                f"{style.margin_l},{style.margin_r},{style.margin_v},1"
            ),
            "",
            "[Events]",
            (
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
                "MarginV, Effect, Text"
            ),
        ]
        lines.extend(dialogues)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def generate_ass_content(
        self,
        transcription: TranscriptionInput,
        start_time_ms: int = 0,
        end_time_ms: Optional[int] = None,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> str:
        """Gera o conteudo ASS completo do trecho/transcricao informados."""
        cues = self.build_cues(transcription, start_time_ms, end_time_ms)
        res_x = video_width or self.style.play_res_x
        res_y = video_height or self.style.play_res_y
        return self._ass_document(res_x, res_y, self._build_dialogues(cues))

    def write_ass_file(
        self,
        transcription: TranscriptionInput,
        output_ass_path: Union[str, Path],
        start_time_ms: int = 0,
        end_time_ms: Optional[int] = None,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> Path:
        """Gera e grava o arquivo ASS em ``output_ass_path``."""
        content = self.generate_ass_content(
            transcription,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            video_width=video_width,
            video_height=video_height,
        )
        output_path = Path(output_ass_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path


__all__ = [
    "AssSubtitleGenerator",
    "MIN_WORD_DURATION_MS",
    "TranscriptionInput",
    "format_ass_time",
    "hex_to_ass_color",
]
