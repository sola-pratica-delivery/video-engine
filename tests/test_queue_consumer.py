"""Testes do consumo de mensagens da fila ``video-processing``.

Cobertura: decodificacao JSON robusta (formato nativo e BullMQ/Redis),
``InMemoryQueueConsumer`` (thread-safe, ack/nack, FIFO, bloqueio/despertar) e
``RedisQueueConsumer`` (parsing RESP minimo, comando ``BRPOP`` e tratamento de
falhas).
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from video_engine.worker.models import WorkerConfig
from video_engine.worker.queue import (
    InMemoryQueueConsumer,
    RedisError,
    RedisQueueConsumer,
    decode_job_payload,
    enqueue_payload,
)

JOB = {
    "jobId": "job-001",
    "uploadId": "upload-abc",
    "filePath": "/tmp/raw.mp4",
    "metadata": {},
    "createdAt": "2026-01-01T00:00:00Z",
}


# --------------------------------------------------------------------------- #
# decode_job_payload
# --------------------------------------------------------------------------- #
def test_decode_native_format():
    raw = json.dumps(JOB)
    msg = decode_job_payload(raw, "video-processing")
    assert msg is not None
    assert msg.id == "job-001"
    assert msg.queue == "video-processing"
    assert msg.raw_payload == raw
    assert msg.data.upload_id == "upload-abc"
    assert msg.data.file_path == "/tmp/raw.mp4"
    assert msg.attempts == 1


def test_decode_bullmq_style():
    bullmq = {"id": "job-100", "name": "video", "data": dict(JOB), "opts": {"attempts": 4}}
    msg = decode_job_payload(json.dumps(bullmq), "video-processing")
    assert msg is not None
    assert msg.id == "job-001"
    assert msg.attempts == 4
    assert msg.data.job_id == "job-001"


def test_decode_preserves_attempts_from_top_level():
    envelope = dict(JOB)
    envelope["attempts"] = 2
    msg = decode_job_payload(json.dumps(envelope), "video-processing")
    assert msg is not None
    assert msg.attempts == 2


def test_decode_invalid_json_returns_none():
    assert decode_job_payload("isso nao e json", "video-processing") is None
    assert decode_job_payload("", "video-processing") is None


def test_decode_non_dict_json_returns_none():
    assert decode_job_payload("[1, 2, 3]", "video-processing") is None


def test_decode_missing_job_fields_returns_none():
    assert decode_job_payload('{"jobId": "x"}', "video-processing") is None
    assert decode_job_payload('{"data": {"foo": 1}}', "video-processing") is None


def test_enqueue_payload_roundtrip():
    raw = enqueue_payload(JOB, attempts=3)
    msg = decode_job_payload(raw, "video-processing")
    assert msg is not None
    assert msg.attempts == 3
    assert msg.data.job_id == "job-001"


# --------------------------------------------------------------------------- #
# InMemoryQueueConsumer
# --------------------------------------------------------------------------- #
def test_inmemory_fifo_poll_and_ack():
    consumer = InMemoryQueueConsumer("video-processing")
    consumer.enqueue(json.dumps(JOB))
    consumer.enqueue(json.dumps({**JOB, "jobId": "job-002", "uploadId": "upload-2"}))

    first = consumer.poll(timeout=0)
    second = consumer.poll(timeout=0)
    assert first is not None and first.data.job_id == "job-001"
    assert second is not None and second.data.job_id == "job-002"

    consumer.ack(first)
    consumer.ack(second)
    assert [m.data.job_id for m in consumer.acked] == ["job-001", "job-002"]
    assert consumer.pending_count == 0


def test_inmemory_poll_empty_returns_none():
    consumer = InMemoryQueueConsumer("video-processing")
    assert consumer.poll(timeout=0) is None
    assert consumer.poll(timeout=0.01) is None


def test_inmemory_nack_records_reason():
    consumer = InMemoryQueueConsumer("video-processing")
    consumer.enqueue(json.dumps(JOB))
    msg = consumer.poll(timeout=0)
    assert msg is not None
    consumer.nack(msg, "media invalida")
    assert consumer.nacked == [{"message": msg, "reason": "media invalida"}]


def test_inmemory_poll_blocks_and_wakes_on_enqueue():
    consumer = InMemoryQueueConsumer("video-processing")
    results = []

    def waiter():
        results.append(consumer.poll(timeout=2.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    consumer.enqueue(json.dumps(JOB))
    thread.join(timeout=1.5)
    assert len(results) == 1
    assert results[0] is not None and results[0].data.job_id == "job-001"


def test_inmemory_invalid_payload_counts_failure():
    consumer = InMemoryQueueConsumer("video-processing")
    consumer.enqueue("payload invalido")
    assert consumer.poll(timeout=0) is None
    assert consumer.decode_failures == 1
    assert consumer.pending_count == 0


def test_inmemory_close_wakes_poll():
    consumer = InMemoryQueueConsumer("video-processing")
    results = []

    def waiter():
        results.append(consumer.poll(timeout=5.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    consumer.close()
    thread.join(timeout=1.0)
    assert results == [None]
    assert not thread.is_alive()


# --------------------------------------------------------------------------- #
# RedisQueueConsumer: parsing RESP
# --------------------------------------------------------------------------- #
class _MemorySocket:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def recv(self, size: int) -> bytes:
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def _parse(data: bytes):
    consumer = _redis_consumer()
    consumer._socket = _MemorySocket(data)  # type: ignore[attr-defined]
    return consumer._parse_reply()


def test_resp_parses_simple_string():
    assert _parse(b"+OK\r\n") == "OK"


def test_resp_parses_bulk_string():
    assert _parse(b"$5\r\nhello\r\n") == b"hello"


def test_resp_parses_bulk_string_empty():
    assert _parse(b"$0\r\n\r\n") == b""


def test_resp_parses_nil_bulk():
    assert _parse(b"$-1\r\n") is None


def test_resp_parses_integer():
    assert _parse(b":42\r\n") == 42


def test_resp_parses_array_mixed():
    assert _parse(b"*2\r\n$2\r\nab\r\n:7\r\n") == [b"ab", 7]


def test_resp_parses_nil_array():
    assert _parse(b"*-1\r\n") is None


def test_resp_error_raises_redis_error():
    with pytest.raises(RedisError, match="ERR usuario nao autorizado"):
        _parse(b"-ERR usuario nao autorizado\r\n")


def test_resp_unknown_marker_raises():
    with pytest.raises(RedisError, match="RESP desconhecida"):
        _parse(b"~oops\r\n")


def test_resp_raises_on_closed_connection():
    with pytest.raises(RedisError, match="encerrada"):
        _parse(b"$5\r\n")


# --------------------------------------------------------------------------- #
# RedisQueueConsumer: poll / ack / nack
# --------------------------------------------------------------------------- #
def _redis_consumer() -> RedisQueueConsumer:
    return RedisQueueConsumer(WorkerConfig(queue_name="video-processing"))


def test_redis_poll_decodes_message_and_sends_brpop():
    consumer = _redis_consumer()
    calls = []

    def fake_send(*args):
        calls.append(args)
        raw = json.dumps(JOB).encode()
        return [b"video-processing", raw]

    consumer._send_command = fake_send  # type: ignore[method-assign]
    msg = consumer.poll(timeout=1.0)
    assert msg is not None
    assert msg.data.job_id == "job-001"
    assert calls == [("BRPOP", "video-processing", "1")]


def test_redis_poll_nil_returns_none():
    consumer = _redis_consumer()
    consumer._send_command = lambda *args: None  # type: ignore[method-assign]
    assert consumer.poll(timeout=1.0) is None


def test_redis_poll_invalid_payload_returns_none():
    consumer = _redis_consumer()
    consumer._send_command = lambda *args: [b"video-processing", b"{corrompido"]  # type: ignore[method-assign]
    assert consumer.poll(timeout=1.0) is None


def test_redis_poll_exception_returns_none():
    consumer = _redis_consumer()

    def boom(*args):
        raise RedisError("Redis indisponivel")

    consumer._send_command = boom  # type: ignore[method-assign]
    assert consumer.poll(timeout=1.0) is None


def test_redis_poll_wrong_reply_shape_returns_none():
    consumer = _redis_consumer()
    consumer._send_command = lambda *args: b"estranho"  # type: ignore[method-assign]
    assert consumer.poll(timeout=1.0) is None


def test_redis_ack_and_nack_are_safe_noops():
    consumer = _redis_consumer()
    msg = decode_job_payload(json.dumps(JOB), "video-processing")
    assert msg is not None
    consumer.ack(msg)  # nao deve levantar
    consumer.nack(msg, "motivo")  # nao deve levantar


def test_redis_poll_closed_returns_none():
    consumer = _redis_consumer()
    consumer.close()
    assert consumer.poll(timeout=1.0) is None
