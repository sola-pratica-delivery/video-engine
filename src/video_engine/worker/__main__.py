"""Ponto de entrada executavel do worker consumidor.

Uso: ``python -m video_engine.worker``.

A configuracao pode ser fornecida via variaveis de ambiente com prefixo
``VIDEO_ENGINE_`` (ex.: ``VIDEO_ENGINE_REDIS_HOST``, ``VIDEO_ENGINE_API_TOKEN``).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from video_engine.worker.models import WorkerConfig
from video_engine.worker.notifier import DeliveryCoreNotifier
from video_engine.worker.pipeline import VideoProcessingPipeline
from video_engine.worker.queue import InMemoryQueueConsumer, RedisQueueConsumer
from video_engine.worker.worker import VideoProcessingWorker

_ENV_PREFIX = "VIDEO_ENGINE_"

# env var -> field do WorkerConfig (valores numericos sao convertidos por pydantic)
_ENV_MAPPING: Dict[str, str] = {
    "REDIS_HOST": "redis_host",
    "REDIS_PORT": "redis_port",
    "REDIS_PASSWORD": "redis_password",
    "REDIS_DB": "redis_db",
    "QUEUE_NAME": "queue_name",
    "DELIVERY_CORE_URL": "delivery_core_url",
    "API_TOKEN": "api_token",
    "OUTPUT_DIR": "output_dir",
    "TEMP_DIR": "temp_dir",
    "JOB_TIMEOUT_SECONDS": "job_timeout_seconds",
    "POLL_INTERVAL_SECONDS": "poll_interval_seconds",
    "SUCCESS_STATUS": "success_status",
    "MAX_RETRIES": "max_retries",
    "CONSUMER_BACKEND": "consumer_backend",
}


def config_from_env() -> WorkerConfig:
    """Le ``WorkerConfig`` a partir de variaveis de ambiente (prefixo ``VIDEO_ENGINE_``)."""
    from video_engine.env import load_env

    load_env()
    values: Dict[str, object] = {}
    for env_name, field_name in _ENV_MAPPING.items():
        raw = os.environ.get(_ENV_PREFIX + env_name)
        if raw is not None:
            values[field_name] = raw
    return WorkerConfig(**values)


def build_consumer(config: WorkerConfig):
    """Instancia o consumidor conforme ``config.consumer_backend``."""
    if config.consumer_backend == "inmemory":
        return InMemoryQueueConsumer(config.queue_name)
    if config.consumer_backend == "redis":
        return RedisQueueConsumer(config)
    raise ValueError(f"consumer_backend desconhecido: {config.consumer_backend}")


def build_worker(config: Optional[WorkerConfig] = None) -> VideoProcessingWorker:
    """Monta o worker completo a partir de ``WorkerConfig``."""
    config = config or config_from_env()
    consumer = build_consumer(config)
    pipeline = VideoProcessingPipeline(config)
    notifier = DeliveryCoreNotifier(config)
    return VideoProcessingWorker(config, consumer=consumer, pipeline=pipeline, notifier=notifier)


def main() -> int:
    """Executa o worker ate receber SIGINT/SIGTERM."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker = build_worker()
    logger = logging.getLogger("video_engine.worker")
    logger.info(
        "Worker iniciado (fila=%s, backend=%s, output_dir=%s)",
        worker.config.queue_name,
        worker.config.consumer_backend,
        worker.config.output_dir,
    )
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
