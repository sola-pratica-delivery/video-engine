"""Testes do notificador de lifecycle para o ``delivery-core``.

Cobertura: contrato de transicao de sucesso/falha (payload, headers e token
bearer), retentativas exponenciais (HTTP 5xx e falhas de rede), escalonamento
de erros definitivos 4xx e comportamento sem token.
"""

from __future__ import annotations

import io
import json
import urllib.error

import video_engine.worker.notifier as nmod
from video_engine.worker.models import WorkerConfig
from video_engine.worker.notifier import DeliveryCoreNotifier


class _FakeResponse:
    def __init__(self, data: bytes = b'{"ok": true}'):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.data


def _http_error(code: int, message: str = "erro"):
    return urllib.error.HTTPError(
        "http://core.test/uploads/x/transition", code, message, {}, io.BytesIO(b"")
    )


def _build_handler(responses):
    """Retorna um handler de urlopen e a lista de requests capturados.

    ``responses`` e uma lista de callables, respostas prontas ou excecoes; ao
    esgotar, o ultimo item e repetido.
    """
    captured = []

    def handler(request, timeout=None):
        captured.append(request)
        behavior = responses[min(len(responses) - 1, len(captured) - 1)]
        if isinstance(behavior, Exception):
            raise behavior
        if callable(behavior):
            return behavior(request)
        return behavior

    return captured, handler


def _notifier(config: WorkerConfig, **kwargs):
    return DeliveryCoreNotifier(config, **kwargs)


# --------------------------------------------------------------------------- #
# Sucesso
# --------------------------------------------------------------------------- #
def test_notify_success_payload_url_and_headers(monkeypatch):
    captured, handler = _build_handler([_FakeResponse()])
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    cfg = WorkerConfig(
        delivery_core_url="http://core.test:3000",
        api_token="token-123",
        success_status="AUTO_QA",
    )
    ok = _notifier(cfg).notify_success(
        "upload-abc", "/storage/abc_processed.mp4", {"duration": 42.5}
    )
    assert ok is True
    assert len(captured) == 1
    req = captured[0]
    assert req.get_method() == "POST"
    assert req.get_full_url() == "http://core.test:3000/uploads/upload-abc/transition"
    assert req.get_header("Authorization") == "Bearer token-123"
    # urllib.Request normaliza a chave via capitalize() -> "Content-type".
    assert req.headers["Content-type"] == "application/json"

    body = json.loads(req.data.decode("utf-8"))
    assert body["to"] == "AUTO_QA"
    assert body["reason"].startswith("Audiovisual processing completed")
    assert body["metadata"]["processedVideoPath"] == "/storage/abc_processed.mp4"
    assert body["metadata"]["duration"] == 42.5


def test_notify_success_respects_configured_status(monkeypatch):
    captured, handler = _build_handler([_FakeResponse()])
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    cfg = WorkerConfig(success_status="VALIDATING_QA", delivery_core_url="http://core.test")
    ok = _notifier(cfg).notify_success("u1", "/p/_processed.mp4", {})
    assert ok is True
    body = json.loads(captured[0].data.decode("utf-8"))
    assert body["to"] == "VALIDATING_QA"


def test_notify_success_without_token_omits_auth_header(monkeypatch):
    captured, handler = _build_handler([_FakeResponse()])
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    ok = _notifier(WorkerConfig(delivery_core_url="http://core.test")).notify_success("u1", "/p.mp4", {})
    assert ok is True
    assert captured[0].get_header("Authorization") is None


# --------------------------------------------------------------------------- #
# Falha
# --------------------------------------------------------------------------- #
def test_notify_failure_payload_contract(monkeypatch):
    captured, handler = _build_handler([_FakeResponse()])
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    ok = _notifier(WorkerConfig(delivery_core_url="http://core.test")).notify_failure(
        "upload-abc",
        "NO_AUDIO_STREAM",
        "Entrada sem stream de audio",
        details={"errorType": "ValueError"},
    )
    assert ok is True
    body = json.loads(captured[0].data.decode("utf-8"))
    assert body["to"] == "FAILED"
    assert body["reason"] == "Video processing error: Entrada sem stream de audio"
    assert body["metadata"]["errorCode"] == "NO_AUDIO_STREAM"
    assert body["metadata"]["errorType"] == "ValueError"
    assert body["metadata"]["errorDetails"] == "Entrada sem stream de audio"


def test_notify_failure_merges_custom_details(monkeypatch):
    captured, handler = _build_handler([_FakeResponse()])
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    _notifier(WorkerConfig(delivery_core_url="http://core.test")).notify_failure(
        "u1", "MEDIA_PROCESSING_ERROR", "boom", details={"ffprobeExitCode": 1}
    )
    body = json.loads(captured[0].data.decode("utf-8"))
    assert body["metadata"]["ffprobeExitCode"] == 1
    assert body["metadata"]["errorType"] == "WorkerError"


# --------------------------------------------------------------------------- #
# Retentativas
# --------------------------------------------------------------------------- #
def test_http_500_retries_then_fails(monkeypatch):
    responses = [_http_error(500), _http_error(500), _http_error(503)]
    captured, handler = _build_handler(responses)
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    sleeps = []
    monkeypatch.setattr(nmod.time, "sleep", sleeps.append)

    ok = _notifier(WorkerConfig(delivery_core_url="http://core.test", max_retries=3)).notify_success(
        "u1", "/p.mp4", {}
    )
    assert ok is False
    assert len(captured) == 3
    assert len(sleeps) == 2
    assert sleeps[0] < sleeps[1]  # backoff exponencial crescente


def test_http_500_succeeds_on_second_attempt(monkeypatch):
    responses = [_http_error(500), _FakeResponse()]
    captured, handler = _build_handler(responses)
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    monkeypatch.setattr(nmod.time, "sleep", lambda _: None)

    ok = _notifier(WorkerConfig(delivery_core_url="http://core.test", max_retries=3)).notify_success(
        "u1", "/p.mp4", {}
    )
    assert ok is True
    assert len(captured) == 2


def test_urlerror_retries_then_fails(monkeypatch):
    responses = [
        urllib.error.URLError("conexao recusada"),
        urllib.error.URLError("conexao recusada"),
        urllib.error.URLError("conexao recusada"),
    ]
    captured, handler = _build_handler(responses)
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    monkeypatch.setattr(nmod.time, "sleep", lambda _: None)

    ok = _notifier(WorkerConfig(delivery_core_url="http://core.test", max_retries=3)).notify_failure(
        "u1", "E", "msg"
    )
    assert ok is False
    assert len(captured) == 3


def test_http_400_is_not_retried(monkeypatch):
    captured, handler = _build_handler([_http_error(400, "bad request")])
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))

    ok = _notifier(WorkerConfig(delivery_core_url="http://core.test", max_retries=3)).notify_success(
        "u1", "/p.mp4", {}
    )
    assert ok is False
    assert len(captured) == 1


def test_retries_count_overrides_config(monkeypatch):
    responses = [_http_error(500), _http_error(500), _FakeResponse()]
    captured, handler = _build_handler(responses)
    monkeypatch.setattr(nmod.urllib.request, "urlopen", lambda request, timeout=None: handler(request, timeout))
    monkeypatch.setattr(nmod.time, "sleep", lambda _: None)

    # max_retries=2 no construtor supera o config (3) e nao chega ao sucesso.
    ok = _notifier(WorkerConfig(delivery_core_url="http://core.test", max_retries=3), max_retries=2).notify_success(
        "u1", "/p.mp4", {}
    )
    assert ok is False
    assert len(captured) == 2
