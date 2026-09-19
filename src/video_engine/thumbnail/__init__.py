"""Selecao automatica de keyframes expressivos para thumbnails (Issue #11).

Motor deterministico que inspeciona frames de um video, descarta frames
degradados ou inexpressivos (borrao de movimento, olhos fechados, exposicao
pobre, ausencia de face) e ranqueia os 5 melhores candidatos com base em
nitidez, iluminacao e expressividade facial.
"""

from video_engine.thumbnail.face_analyzer import FaceAnalyzer, LandmarkFaceAnalyzer
from video_engine.thumbnail.frame_extractor import (
    SampledFrame,
    VideoFrameExtractor,
    compute_uniform_timestamps,
)
from video_engine.thumbnail.models import (
    FaceBoundingBox,
    FaceMetrics,
    FrameMetrics,
    KeyframeCandidate,
    KeyframeSelectorConfig,
    KeyframeSelectorResult,
    RejectionReason,
)
from video_engine.thumbnail.selector import KeyframeSelector

__all__ = [
    "FaceAnalyzer",
    "FaceBoundingBox",
    "FaceMetrics",
    "FrameMetrics",
    "KeyframeCandidate",
    "KeyframeSelector",
    "KeyframeSelectorConfig",
    "KeyframeSelectorResult",
    "LandmarkFaceAnalyzer",
    "RejectionReason",
    "SampledFrame",
    "VideoFrameExtractor",
    "compute_uniform_timestamps",
]
