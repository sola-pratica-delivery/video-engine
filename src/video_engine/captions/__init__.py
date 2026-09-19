"""Pacote de transcricao fonetica com timestamps por palavra (Issue #7)."""

from video_engine.captions.models import (
    CaptionSegment,
    TranscriberConfig,
    TranscriptionResult,
    WordTimestamp,
)
from video_engine.captions.transcriber import WhisperTranscriber

__all__ = [
    "CaptionSegment",
    "TranscriberConfig",
    "TranscriptionResult",
    "WhisperTranscriber",
    "WordTimestamp",
]
