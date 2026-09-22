"""Pacote de deteccao de ganchos virais e cortes verticais (Shorts 9:16).

Inclui a inteligencia da Issue #15 (Epic #14): modelos Pydantic, analisador
de energia vocal (RMS), analisador semantico via Google AI Studio, heuristica
offline deterministic a e o detector principal ``HookDetector`` com snapping a
oracoes, validacao de duracao (25s a 58s) e NMS temporal.
"""

from video_engine.shorts.energy_analyzer import EnergyAnalysisError, EnergyAnalyzer
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
from video_engine.shorts.semantic_hook_analyzer import SemanticHookAnalyzer, SemanticHookError

__all__ = [
    "DetectionMode",
    "EnergyAnalysisError",
    "EnergyAnalysisResult",
    "EnergyAnalyzer",
    "EnergyPeak",
    "EnergyProfile",
    "HeuristicHookDetector",
    "HookDetectionResult",
    "HookDetector",
    "HookDetectorConfig",
    "SemanticHookAnalyzer",
    "SemanticHookError",
    "SemanticHookInterval",
    "SemanticHookResponse",
    "ShortCandidateCut",
    "suppress_overlaps",
    "temporal_iou",
]
