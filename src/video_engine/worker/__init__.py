"""Worker consumidor de jobs da fila ``video-processing`` (Spec Issue #18).

Consome mensagens da fila, orquestra a esteira audiovisual (probe, VAD,
splice com micro-crossfades e normalizacao EBU R128) e notifica o
``delivery-core`` sobre a transicao de lifecycle do job.
"""

from __future__ import annotations

from video_engine.worker.models import (
    LoudnessReport,
    ProcessingResult,
    QueueMessage,
    VideoProcessingJobData,
    WorkerConfig,
)
from video_engine.worker.notifier import DeliveryCoreNotifier
from video_engine.worker.pipeline import (
    InputFileInvalidError,
    NoAudioStreamError,
    NoSpeechDetectedError,
    PipelineError,
    VideoProcessingPipeline,
)
from video_engine.worker.queue import (
    InMemoryQueueConsumer,
    QueueConsumer,
    RedisError,
    RedisQueueConsumer,
    decode_job_payload,
    enqueue_payload,
)
from video_engine.worker.worker import JobTimeoutError, VideoProcessingWorker

__all__ = [
    "DeliveryCoreNotifier",
    "InMemoryQueueConsumer",
    "InputFileInvalidError",
    "JobTimeoutError",
    "LoudnessReport",
    "NoAudioStreamError",
    "NoSpeechDetectedError",
    "PipelineError",
    "ProcessingResult",
    "QueueConsumer",
    "QueueMessage",
    "RedisError",
    "RedisQueueConsumer",
    "VideoProcessingJobData",
    "VideoProcessingPipeline",
    "VideoProcessingWorker",
    "WorkerConfig",
    "decode_job_payload",
    "enqueue_payload",
]
