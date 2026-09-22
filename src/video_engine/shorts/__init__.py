"""Pacote de deteccao de ganchos virais e cortes verticais (Shorts 9:16).

Inclui a inteligencia da Issue #15 (Epic #14): modelos Pydantic, analisador
de energia vocal (RMS), analisador semantico via Google AI Studio, heuristica
offline deterministic a e o detector principal ``HookDetector`` com snapping a
oracoes, validacao de duracao (25s a 58s) e NMS temporal. Tambem inclui o
reenquadramento vertical 9:16 da Issue #16: rastreamento facial suavizado
(EMA + deadband) e o motor ``VerticalReframer`` (SMART_CROP/AESTHETIC_FILL).
"""

from video_engine.shorts.energy_analyzer import EnergyAnalysisError, EnergyAnalyzer
from video_engine.shorts.face_tracker import FaceTracker, RawFaceSample, smooth_centers
from video_engine.shorts.heuristic_hook_detector import HeuristicHookDetector
from video_engine.shorts.hook_detector import HookDetector, suppress_overlaps, temporal_iou
from video_engine.shorts.models import (
    DetectionMode,
    EnergyAnalysisResult,
    EnergyPeak,
    EnergyProfile,
    HookDetectionResult,
    HookDetectorConfig,
    SemanticHookInterval,
    SemanticHookResponse,
    ShortCandidateCut,
)
from video_engine.shorts.reframer_models import (
    CropMode,
    FaceTrackingPoint,
    ReframerConfig,
    ReframerResult,
    TrackingTrajectory,
)
from video_engine.shorts.semantic_hook_analyzer import SemanticHookAnalyzer, SemanticHookError
from video_engine.shorts.vertical_reframer import ReframerError, VerticalReframer

__all__ = [
    "CropMode",
    "DetectionMode",
    "EnergyAnalysisError",
    "EnergyAnalysisResult",
    "EnergyAnalyzer",
    "EnergyPeak",
    "EnergyProfile",
    "FaceTracker",
    "FaceTrackingPoint",
    "HeuristicHookDetector",
    "HookDetectionResult",
    "HookDetector",
    "HookDetectorConfig",
    "RawFaceSample",
    "ReframerConfig",
    "ReframerError",
    "ReframerResult",
    "SemanticHookAnalyzer",
    "SemanticHookError",
    "SemanticHookInterval",
    "SemanticHookResponse",
    "ShortCandidateCut",
    "TrackingTrajectory",
    "VerticalReframer",
    "suppress_overlaps",
    "smooth_centers",
    "temporal_iou",
]
