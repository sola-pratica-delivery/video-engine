"""Consumo de mensagens da fila ``video-processing``.

Define o protocolo ``QueueConsumer`` (interfaces ``poll``/``ack``/``nack``),
o adaptador determinístico ``InMemoryQueueConsumer`` (testes locais e CI) e o
``RedisQueueConsumer`` baseado em ``BRPOP`` com um cliente RESP mínimo
(biblioteca padrão, sem dependências adicionais) e decodificacao JSON robusta
compatível com o formato BullMQ/Redis.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Protocol

from pydantic import ValidationError

from video_engine.worker.models import QueueMessage, VideoProcessingJobData

logger = logging.getLogger(__name__)


class RedisError(Exception):
    """Erro de comunicacao ou protocolo com o servidor Redis."""


class QueueConsumer(Protocol):
    """Protocolo abstrato para consumo de jobs de ``video-processing``."""

    def poll(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Bloqueia ate ``timeout`` segundos e retorna uma mensagem (ou None)."""
        ...

    def ack(self, message: QueueMessage) -> None:
        """Confirma que ``message`` foi tratada com sucesso."""
        ...

    def nack(self, message: QueueMessage, reason: str) -> None:
        """Sinaliza que ``message`` nao pode ser processada (``reason``)."""
        ...

    def close(self) -> None:
        """Fecha conexoes/recursos do consumidor."""
        ...


# --------------------------------------------------------------------------- #
# Decodificacao de payloads (formato nativo e BullMQ/Redis)
# --------------------------------------------------------------------------- #
def _extract_job_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "jobId" in payload and "uploadId" in payload:
        return payload
    data = payload.get("data")
    if isinstance(data, dict) and "jobId" in data and "uploadId" in data:
        return data
    return None


def _extract_attempts(payload: Dict[str, Any]) -> int:
    for source in (payload, payload.get("opts") if isinstance(payload.get("opts"), dict) else payload):
        raw = source.get("attempts")
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 1


def decode_job_payload(raw_payload: str, queue: str) -> Optional[QueueMessage]:
    """Converte o JSON bruto de um item da fila em :class:`QueueMessage`.

    Suporta o formato nativo (campos ``jobId``/``uploadId`` no topo) e o
    formato BullMQ/Redis (job com chave ``data``). Retorna ``None`` para
    payloads invalidos ou incompletos.
    """
    try:
        payload = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    data = _extract_job_payload(payload)
    if data is None:
        return None
    try:
        job = VideoProcessingJobData.model_validate(data)
    except ValidationError:
        return None
    return QueueMessage(
        id=data.get("jobId", "unknown"),
        queue=queue,
        data=job,
        raw_payload=raw_payload,
        attempts=_extract_attempts(payload),
    )


def enqueue_payload(job_data: Dict[str, Any], attempts: int = 1) -> str:
    """Serializa um job no formato nativo pronto para ser enfileirado."""
    envelope = dict(job_data)
    envelope["attempts"] = attempts
    return json.dumps(envelope)


# --------------------------------------------------------------------------- #
# Consumidor em memoria (deterministico / testes)
# --------------------------------------------------------------------------- #
class InMemoryQueueConsumer:
    """Fila em memoria thread-safe para execucao deterministica e testes."""

    def __init__(self, queue_name: str = "video-processing") -> None:
        self.queue_name = queue_name
        self._condition = threading.Condition()
        self._pending: Deque[str] = deque()
        self._delivered: List[QueueMessage] = []
        self._acked: List[QueueMessage] = []
        self._nacked: List[Dict[str, Any]] = []
        self._decode_failures = 0
        self._closed = False

    # Configuracoes de teste
    def enqueue(self, raw_payload: str) -> None:
        """Enfileira um payload JSON bruto."""
        with self._condition:
            self._pending.append(raw_payload)
            self._condition.notify_all()

    def enqueue_job(self, job_data: Dict[str, Any], attempts: int = 1) -> None:
        """Enfileira um job serializando ``job_data`` no formato nativo."""
        self.enqueue(enqueue_payload(job_data, attempts=attempts))

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def delivered(self) -> List[QueueMessage]:
        with self._condition:
            return list(self._delivered)

    @property
    def acked(self) -> List[QueueMessage]:
        with self._condition:
            return list(self._acked)

    @property
    def nacked(self) -> List[Dict[str, Any]]:
        with self._condition:
            return list(self._nacked)

    @property
    def decode_failures(self) -> int:
        with self._condition:
            return self._decode_failures

    # Interfaces do protocolo QueueConsumer
    def poll(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                if self._pending:
                    raw = self._pending.popleft()
                    message = decode_job_payload(raw, self.queue_name)
                    if message is None:
                        self._decode_failures += 1
                        return None
                    self._delivered.append(message)
                    return message
                if self._closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def ack(self, message: QueueMessage) -> None:
        with self._condition:
            self._acked.append(message)

    def nack(self, message: QueueMessage, reason: str) -> None:
        with self._condition:
            self._nacked.append({"message": message, "reason": reason})

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


# --------------------------------------------------------------------------- #
# Consumidor Redis (cliente RESP minimo + BRPOP)
# --------------------------------------------------------------------------- #
class RedisQueueConsumer:
    """Consumidor da fila ``video-processing`` via comando bloqueante ``BRPOP``.

    Implementa um cliente RESP mínimo sobre ``socket`` da biblioteca padrao,
    dispensando dependencias externas. ``BRPOP`` remove atomicamente a mensagem
    da lista, de modo que ``ack``/``nack`` sao operacoes no-op (logging).
    """

    _COMMAND_TIMEOUT = 10.0

    def __init__(self, config, queue_name: Optional[str] = None) -> None:
        self.config = config
        self.queue_name = queue_name or config.queue_name
        self._lock = threading.RLock()
        self._socket: Optional[socket.socket] = None
        self._buffer = bytearray()
        self._closed = False

    # ------------------------------------------------------------------ #
    # Conexao / protocolo RESP
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        if self._socket is not None or self._closed:
            return
        sock = socket.create_connection(
            (self.config.redis_host, self.config.redis_port),
            timeout=self._COMMAND_TIMEOUT,
        )
        sock.settimeout(self._COMMAND_TIMEOUT)
        try:
            self._socket = sock
            self._buffer = bytearray()
            if self.config.redis_password:
                reply = self._raw_command("AUTH", self.config.redis_password)
                if str(reply).upper() != "OK":
                    raise RedisError(f"AUTH rejeitado pelo Redis: {reply!r}")
            if self.config.redis_db:
                reply = self._raw_command("SELECT", str(self.config.redis_db))
                if str(reply).upper() != "OK":
                    raise RedisError(f"SELECT {self.config.redis_db} rejeitado pelo Redis: {reply!r}")
        except Exception:
            self._socket = None
            self._buffer = bytearray()
            sock.close()
            raise

    def _raw_command(self, *args: str) -> Any:
        if self._socket is None:
            raise RedisError("conexao Redis nao estabelecida")
        parts: List[bytes] = [f"*{len(args)}\r\n".encode("ascii")]
        for arg in args:
            encoded = str(arg).encode("utf-8")
            parts.append(f"${len(encoded)}\r\n".encode("ascii"))
            parts.append(encoded)
            parts.append(b"\r\n")
        self._socket.sendall(b"".join(parts))
        return self._parse_reply()

    def _read_line(self) -> bytes:
        while True:
            index = self._buffer.find(b"\r\n")
            if index >= 0:
                line = bytes(self._buffer[:index])
                del self._buffer[: index + 2]
                return line
            chunk = self._socket.recv(4096)
            if not chunk:
                raise RedisError("conexao Redis encerrada durante a leitura")
            self._buffer.extend(chunk)

    def _read_exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            chunk = self._socket.recv(size - len(self._buffer))
            if not chunk:
                raise RedisError("conexao Redis encerrada durante a leitura do bulk")
            self._buffer.extend(chunk)
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def _parse_reply(self) -> Any:
        line = self._read_line()
        if not line:
            raise RedisError("resposta RESP vazia do servidor Redis")
        marker = line[:1]
        body = line[1:]
        if marker == b"*":
            count = int(body)
            if count < 0:
                return None
            return [self._parse_reply() for _ in range(count)]
        if marker == b"$":
            length = int(body)
            if length < 0:
                return None  # bulk string nil
            data = self._read_exact(length)
            self._read_exact(2)  # CRLF de terminacao do bulk
            return data
        if marker == b"+":
            return body.decode("utf-8", errors="replace")
        if marker == b":":
            return int(body)
        if marker == b"-":
            raise RedisError(body.decode("utf-8", errors="replace"))
        raise RedisError(f"resposta RESP desconhecida: {line.decode('utf-8', errors='replace')}")

    def _send_command(self, *args: str) -> Any:
        with self._lock:
            self._connect()
            if self._socket is None:
                raise RedisError("consumidor Redis esta fechado")
            try:
                return self._raw_command(*args)
            except (OSError, RedisError) as exc:
                sock = self._socket
                self._socket = None
                self._buffer = bytearray()
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                raise RedisError(f"falha de comunicacao com o Redis: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Interfaces do protocolo QueueConsumer
    # ------------------------------------------------------------------ #
    def poll(self, timeout: float = 1.0) -> Optional[QueueMessage]:
        if self._closed:
            return None
        block_seconds = max(1, int(max(0.0, timeout)))
        try:
            reply = self._send_command("BRPOP", self.queue_name, str(block_seconds))
        except RedisError as exc:
            logger.warning("Erro ao executar BRPOP em %s: %s", self.queue_name, exc)
            return None
        if reply is None:
            return None
        if not isinstance(reply, list) or len(reply) < 2:
            logger.warning("Resposta inesperada do BRPOP em %s: %r", self.queue_name, reply)
            return None
        value = reply[1]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if not isinstance(value, str):
            logger.warning("Payload de %s nao e texto: %r", self.queue_name, value)
            return None
        message = decode_job_payload(value, self.queue_name)
        if message is None:
            logger.warning("Payload invalido descartado de %s", self.queue_name)
        return message

    def ack(self, message: QueueMessage) -> None:
        logger.debug("BRPOP ack implicito para job %s", message.id)

    def nack(self, message: QueueMessage, reason: str) -> None:
        logger.warning("Job %s nao processado (%s, fila %s)", message.id, reason, message.queue)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            sock = self._socket
            self._socket = None
            self._buffer = bytearray()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()


__all__ = [
    "InMemoryQueueConsumer",
    "QueueConsumer",
    "RedisError",
    "RedisQueueConsumer",
    "decode_job_payload",
    "enqueue_payload",
]
