"""Worker consumidor de jobs de ``video-processing``.

Loop continuo com interceptacao de sinais (``SIGINT``/``SIGTERM``), controle
de timeout por job via ``concurrent.futures``, isolamento de erros por mensagem
e relatorios de metricas. Excecoes de um job individual nunca derrubam o
processo.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, Optional

from video_engine.worker.models import QueueMessage, WorkerConfig
from video_engine.worker.notifier import DeliveryCoreNotifier
from video_engine.worker.pipeline import PipelineError, VideoProcessingPipeline
from video_engine.worker.queue import QueueConsumer

logger = logging.getLogger(__name__)


class JobTimeoutError(PipelineError):
    """O job excedeu ``job_timeout_seconds`` durante o processamento."""

    error_code = "TIMEOUT_EXCEEDED"


class VideoProcessingWorker:
    """Loop continuo de processamento de mensagens da fila."""

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        consumer: Optional[QueueConsumer] = None,
        pipeline: Optional[VideoProcessingPipeline] = None,
        notifier: Optional[DeliveryCoreNotifier] = None,
        poll_interval: Optional[float] = None,
        job_timeout: Optional[float] = None,
    ) -> None:
        self.config = config or WorkerConfig()
        self.consumer = consumer
        self.pipeline = pipeline or VideoProcessingPipeline(self.config)
        self.notifier = notifier or DeliveryCoreNotifier(self.config)
        self.poll_interval = (
            poll_interval if poll_interval is not None else self.config.poll_interval_seconds
        )
        self.job_timeout = (
            job_timeout if job_timeout is not None else self.config.job_timeout_seconds
        )

        self._stop = False
        self._metrics: Dict[str, Any] = {
            "jobs_processed": 0,
            "jobs_succeeded": 0,
            "jobs_failed": 0,
            "jobs_timed_out": 0,
            "started_at": None,
        }

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Solicita parada graciosa do loop."""
        self._stop = True

    def _install_signal_handlers(self) -> None:
        def _handle(signum, frame):
            logger.info("Sinal %d recebido, encerrando graciosamente...", signum)
            self.stop()

        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError):
                pass

    def run(self, max_jobs: Optional[int] = None) -> int:
        """Executa o loop consumidor ate parada ou ``max_jobs`` processados.

        Returns:
            Numero de mensagens consumidas.
        """
        self._metrics["started_at"] = time.monotonic()
        self._install_signal_handlers()
        processed = 0
        try:
            while not self._stop and (max_jobs is None or processed < max_jobs):
                message = self.consumer.poll(timeout=self.poll_interval)
                if message is None:
                    continue
                self._handle(message)
                processed += 1
        finally:
            if self.consumer is not None:
                self.consumer.close()
        return processed

    # ------------------------------------------------------------------ #
    # Processamento
    # ------------------------------------------------------------------ #
    def _handle(self, message: QueueMessage) -> bool:
        """Processa uma mensagem com timeout e isolamento de erros."""
        self._metrics["jobs_processed"] += 1
        try:
            result = self._run_with_timeout(lambda: self.pipeline.process(message.data), self.job_timeout)
        except Exception as exc:  # noqa: BLE001 - isolamento por design
            error_code = getattr(exc, "error_code", "MEDIA_PROCESSING_ERROR")
            details = dict(getattr(exc, "details", None) or {})
            details.setdefault("errorType", type(exc).__name__)
            notify_ok = self.notifier.notify_failure(
                message.data.upload_id, error_code, str(exc), details
            )
            self._metrics["jobs_failed"] += 1
            if error_code == "TIMEOUT_EXCEEDED":
                self._metrics["jobs_timed_out"] += 1
            if self.consumer is not None:
                self.consumer.ack(message)
            logger.error(
                "Job %s (upload %s) falhou [%s]: %s (notificacao=%s)",
                message.id,
                message.data.upload_id,
                error_code,
                exc,
                notify_ok,
            )
            return False

        metadata = result.to_success_metadata()
        notify_ok = self.notifier.notify_success(message.data.upload_id, result.output_path, metadata)
        self._metrics["jobs_succeeded"] += 1
        if self.consumer is not None:
            self.consumer.ack(message)
        logger.info(
            "Job %s (upload %s) processado com sucesso (duracao=%.2fs, notificacao=%s)",
            message.id,
            message.data.upload_id,
            result.duration_sec,
            notify_ok,
        )
        return True

    def _run_with_timeout(self, fn, timeout: float):
        """Executa ``fn`` respeitando o limite de ``timeout`` segundos."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video_job")
        future = executor.submit(fn)
        try:
            return future.result(timeout=max(0.01, timeout))
        except FutureTimeoutError:
            raise JobTimeoutError(
                f"Job excedeu o limite de processamento de {timeout:.1f}s"
            ) from None
        finally:
            executor.shutdown(wait=False)

    def metrics(self) -> Dict[str, Any]:
        """Copia das metricas acumuladas do worker."""
        return dict(self._metrics)


__all__ = ["JobTimeoutError", "VideoProcessingWorker"]
