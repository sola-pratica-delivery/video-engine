"""Pacote de processamento de audio: VAD, silencios, pausas e loudness."""

from video_engine.audio.loudness import LoudnessNormalizer
from video_engine.audio.loudness_models import LoudnessConfig, LoudnessResult
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
    "LoudnessConfig",
    "LoudnessNormalizer",
    "LoudnessResult",
    "apply_padding_and_merge",
    "extract_silence_intervals",
    "merge_intervals",
    "splice_audio_array",
]
