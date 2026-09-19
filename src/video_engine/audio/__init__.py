"""Pacote de processamento de audio: VAD, silencios e pausas."""

from video_engine.audio.models import (
    SilenceSegment,
    SpeechSegment,
    TimeInterval,
    VadConfig,
    VadResult,
)
from video_engine.audio.silero_vad import SileroVadDetector
from video_engine.audio.splicer import merge_intervals, splice_audio_array
from video_engine.audio.vad_utils import (
    apply_padding_and_merge,
    extract_silence_intervals,
)

__all__ = [
    "SileroVadDetector",
    "TimeInterval",
    "SpeechSegment",
    "SilenceSegment",
    "VadConfig",
    "VadResult",
    "apply_padding_and_merge",
    "extract_silence_intervals",
    "merge_intervals",
    "splice_audio_array",
]
