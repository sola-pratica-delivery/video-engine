"""Pacote de transcricao fonetica e legendas animadas.

Inclui a transcricao com Faster-Whisper e timestamps por palavra (Issue #7)
e a geracao/queima de legendas animadas ASS em cortes (Issue #9).
"""

from video_engine.captions.ass_generator import AssSubtitleGenerator
from video_engine.captions.ass_models import (
    CutCaptionBurnResult,
    KaraokeHighlightMode,
    SubtitleCue,
    SubtitleStyleConfig,
)
from video_engine.captions.burner import CaptionBurner, CaptionBurnError
from video_engine.captions.models import (
    CaptionSegment,
    TranscriberConfig,
    TranscriptionResult,
    WordTimestamp,
)
from video_engine.captions.transcriber import WhisperTranscriber

__all__ = [
    "AssSubtitleGenerator",
    "CaptionBurnError",
    "CaptionBurner",
    "CaptionSegment",
    "CutCaptionBurnResult",
    "KaraokeHighlightMode",
    "SubtitleCue",
    "SubtitleStyleConfig",
    "TranscriberConfig",
    "TranscriptionResult",
    "WhisperTranscriber",
    "WordTimestamp",
]
