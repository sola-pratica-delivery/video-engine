"""Testes de orquestracao do loop do worker consumidor.

Cobertura: fluxo completo de sucesso, isolamento de falhas por job, relatorio
de metricas, timeout por job via ``concurrent.futures``, parada graciosa
(``stop()``) e fechamento do consumidor.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

from video_engine.worker.models import (
    LoudnessReport,
    ProcessingResult,
    QueueMessage,
    WorkerConfig,
)
from video_engine.worker.pipeline import NoSpeechDetectedError
from video_engine.worker.queue import decode_job_payload
from video_engine.worker.worker import VideoProcessingWorker

JOB_OK = {
    "jobId": "job-001",
    "uploadId": "upload-abc",
    "filePath": "/tmp/raw.mp4",
    "createdAt": "2026-01-01T00:00:00Z",
}
JOB_FAIL = {**JOB_OK, "jobId": "job-002", "uploadId": "upload-fail"}


def render_result(upload_id: str = "upload-abc") -> ProcessingResult:
    return ProcessingResult(
        output_path=f"/storage/{upload_id}_processed.mp4",
        duration_sec=42.0,
        speech_segments_count=2,
        silence_removed_ms=1200,
        loudness=LoudnessReport(integrated_lufs=-14.1, true_peak_dbtp=-1.02, lra=10.0),
    )


# --------------------------------------------------------------------------- #
# Stubs de apoio
# --------------------------------------------------------------------------- #
class StubConsumer:
    def __init__(self, raw_payloads: List[str]):
        self.queue = list(raw_payloads)
        self.acked: List[QueueMessage] = []
        self.nacked: List[Any] = []
        self.closed = False

    def poll(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        if self.queue:
            raw = self.queue.pop(0)
            return decode_job_payload(raw, "video-processing")
        time.sleep(0.005)
        return None

    def ack(self, message: QueueMessage) -> None:
        self.acked.append(message)

    def nack(self, message: QueueMessage, reason: str) -> None:
        self.nacked.append((message, reason))

    def close(self) -> None:
        self.closed = True


class FakePipeline:
    def __init__(self, fn) -> None:
        self.fn = fn

    def process(self, data):
        return self.fn(data)


class FakeNotifier:
    def __init__(self) -> None:
        self.success_calls: List[tuple] = []
        self.failure_calls: List[tuple] = []
        self.success_return = True
        self.failure_return = True

    def notify_success(self, upload_id: str, processed_path: str, metadata: Dict[str, Any]) -> bool:
        self.success_calls.append((upload_id, processed_path, metadata))
        return self.success_return

    def notify_failure(self, upload_id: str, error_code: str, error_message: str, details=None) -> bool:
        self.failure_calls.append((upload_id, error_code, error_message, details))
        return self.failure_return


def make_worker(pipeline_fn, payloads, notifier=None, **kwargs):
    consumer = StubConsumer(payloads)
    worker = VideoProcessingWorker(
        config=WorkerConfig(),
        consumer=consumer,
        pipeline=FakePipeline(pipeline_fn),
        notifier=notifier or FakeNotifier(),
        **kwargs,
    )
    return worker, consumer, worker.notifier


# --------------------------------------------------------------------------- #
# Sucesso
# --------------------------------------------------------------------------- #
def test_worker_success_flow_notifies_and_acks():
    worker, consumer, notifier = make_worker(
        lambda data: render_result(data.upload_id), [json.dumps(JOB_OK)]
    )
    processed = worker.run(max_jobs=1)
    assert processed == 1
    assert len(notifier.success_calls) == 1
    upload_id, path, metadata = notifier.success_calls[0]
    assert upload_id == "upload-abc"
    assert path == "/storage/upload-abc_processed.mp4"
    assert metadata["processedVideoPath"] == "/storage/upload-abc_processed.mp4"
    assert metadata["duration"] == 42.0
    assert len(consumer.acked) == 1
    assert consumer.closed is True
    assert worker.metrics()["jobs_succeeded"] == 1
    assert worker.metrics()["jobs_failed"] == 0


# --------------------------------------------------------------------------- #
# Falhas isoladas por job
# --------------------------------------------------------------------------- #
def test_worker_pipeline_error_notifies_failure_and_acks():
    def fail(data):
        raise NoSpeechDetectedError("Nenhum segmento de fala detectado no video")

    worker, consumer, notifier = make_worker(fail, [json.dumps(JOB_OK)])
    processed = worker.run(max_jobs=1)
    assert processed == 1
    assert len(notifier.failure_calls) == 1
    upload_id, code, message, details = notifier.failure_calls[0]
    assert upload_id == "upload-abc"
    assert code == "NO_SPEECH_DETECTED"
    assert message == "Nenhum segmento de fala detectado no video"
    assert details.get("errorType") == "NoSpeechDetectedError"
    assert len(notifier.success_calls) == 0
    assert len(consumer.acked) == 1
    assert worker.metrics()["jobs_failed"] == 1


def test_worker_unknown_exception_maps_to_generic_error():
    def boom(data):
        raise ValueError("audio corrompido")

    worker, consumer, notifier = make_worker(boom, [json.dumps(JOB_OK)])
    worker.run(max_jobs=1)
    upload_id, code, message, details = notifier.failure_calls[0]
    assert code == "MEDIA_PROCESSING_ERROR"
    assert details.get("errorType") == "ValueError"
    assert message == "audio corrompido"


def test_worker_isolates_failures_between_jobs():
    def behavior(data):
        if data.job_id == "job-fail":
            raise ValueError("falha sintetica")
        return render_result(data.upload_id)

    payloads = [json.dumps(JOB_OK), json.dumps({**JOB_OK, "jobId": "job-fail", "uploadId": "upload-fail"})]
    worker, consumer, notifier = make_worker(behavior, payloads)
    processed = worker.run(max_jobs=2)
    assert processed == 2
    assert len(notifier.success_calls) == 1
    assert len(notifier.failure_calls) == 1
    assert len(consumer.acked) == 2
    metrics = worker.metrics()
    assert metrics["jobs_succeeded"] == 1
    assert metrics["jobs_failed"] == 1
    assert metrics["jobs_processed"] == 2
    assert metrics["jobs_timed_out"] == 0


# --------------------------------------------------------------------------- #
# Timeout por job
# --------------------------------------------------------------------------- #
def test_worker_job_timeout_notifies_timed_out():
    def slow(data):
        time.sleep(0.5)
        return render_result(data.upload_id)

    worker, consumer, notifier = make_worker(slow, [json.dumps(JOB_OK)], job_timeout=0.05)
    started = time.monotonic()
    processed = worker.run(max_jobs=1)
    elapsed = time.monotonic() - started
    assert processed == 1
    assert elapsed < 0.4  # nao espera o job lento terminar
    assert len(notifier.failure_calls) == 1
    assert notifier.failure_calls[0][1] == "TIMEOUT_EXCEEDED"
    assert len(consumer.acked) == 1
    assert worker.metrics()["jobs_timed_out"] == 1


# --------------------------------------------------------------------------- #
# Parada graciosa
# --------------------------------------------------------------------------- #
def test_worker_stop_returns_and_closes_consumer():
    def fast(data):
        return render_result(data.upload_id)

    worker, consumer, notifier = make_worker(
        fast, [json.dumps(JOB_OK)], poll_interval=0.005
    )
    results = []

    def runner():
        results.append(worker.run())

    thread = threading.Thread(target=runner)
    thread.start()
    time.sleep(0.1)
    worker.stop()
    thread.join(timeout=2.0)
    assert results == [1]
    assert consumer.closed is True
    assert not thread.is_alive()


def test_worker_stop_without_messages_exits_cleanly():
    consumer = StubConsumer([])
    worker = VideoProcessingWorker(
        config=WorkerConfig(),
        consumer=consumer,
        pipeline=FakePipeline(lambda data: render_result()),
        notifier=FakeNotifier(),
        poll_interval=0.005,
    )
    results = []

    def runner():
        results.append(worker.run())

    thread = threading.Thread(target=runner)
    thread.start()
    time.sleep(0.02)
    worker.stop()
    thread.join(timeout=1.0)
    assert results == [0]
    assert consumer.closed is True
