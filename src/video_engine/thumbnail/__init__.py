"""Selecao automatica de keyframes expressivos para thumbnails (Issue #11).

Motor deterministico que inspeciona frames de um video, descarta frames
degradados ou inexpressivos (borrao de movimento, olhos fechados, exposicao
pobre, ausencia de face) e ranqueia os 5 melhores candidatos com base em
nitidez, iluminacao e expressividade facial.
"""

from video_engine.thumbnail.edge_enhancer import (
    apply_stroke_and_glow,
    create_glow_mask,
    create_stroke_mask,
    dilate_mask,
    feather_mask,
)
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
    GlowConfig,
    KeyframeCandidate,
    KeyframeSelectorConfig,
    KeyframeSelectorResult,
    RejectionReason,
    SegmentationResult,
    SegmenterConfig,
    StrokeConfig,
)
from video_engine.thumbnail.segmenter import BackgroundSegmenter, OnnxBackgroundSegmenter
from video_engine.thumbnail.selector import KeyframeSelector

__all__ = [
    "BackgroundSegmenter",
    "FaceAnalyzer",
    "FaceBoundingBox",
    "FaceMetrics",
    "FrameMetrics",
    "GlowConfig",
    "KeyframeCandidate",
    "KeyframeSelector",
    "KeyframeSelectorConfig",
    "KeyframeSelectorResult",
    "LandmarkFaceAnalyzer",
    "OnnxBackgroundSegmenter",
    "RejectionReason",
    "SampledFrame",
    "SegmentationResult",
    "SegmenterConfig",
    "StrokeConfig",
    "VideoFrameExtractor",
    "apply_stroke_and_glow",
    "compute_uniform_timestamps",
    "create_glow_mask",
    "create_stroke_mask",
    "dilate_mask",
    "feather_mask",
]
