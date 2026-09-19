"""Notificacao de transicao de lifecycle para o ``delivery-core``.

Cliente HTTP minimalista baseado em ``urllib.request`` (biblioteca padrao,
resiliente e sem dependencias pesadas) com retentativas exponenciais para
erros transitórios (HTTP 5xx, timeouts e falhas de rede).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from video_engine.worker.models import WorkerConfig

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP = frozenset({500, 502, 503, 504})
_BACKOFF_SECONDS = 0.5


class DeliveryCoreNotifier:
    """Notifica sucesso/falha de um job para a API de lifecycle do core.

    Args:
        config: Configuracao do worker (URL do core, token, endpoints).
        max_retries: Numero de tentativas (padrao: ``config.max_retries``).

    Raises:
        urllib.error.URLError: se a URL do core for invalida (config).
    """

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.config = config or WorkerConfig()
        self.max_retries = self.config.max_retries if max_retries is None else max_retries

    def _transition_url(self, upload_id: str) -> str:
        base = self.config.delivery_core_url.rstrip("/")
        path = self.config.upload_endpoint_pattern.format(upload_id=upload_id)
        return f"{base}{path}"

    def notify_success(
        self,
        upload_id: str,
        processed_path: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Notifica transicao para ``success_status`` com as metricas do job."""
        body = dict(metadata)
        body["processedVideoPath"] = processed_path
        payload = {
            "to": self.config.success_status,
            "reason": "Audiovisual processing completed (VAD, surgical cuts, EBU R128 loudness)",
            "metadata": body,
        }
        return self._notify_with_retries(upload_id, payload)

    def notify_failure(
        self,
        upload_id: str,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Notifica transicao para ``FAILED`` com detalhes de diagnostico."""
        details = dict(details or {})
        metadata = {
            "errorCode": error_code,
            "errorType": details.pop("errorType", "WorkerError"),
            "errorDetails": error_message,
        }
        metadata.update(details)
        payload = {
            "to": "FAILED",
            "reason": f"Video processing error: {error_message}",
            "metadata": metadata,
        }
        return self._notify_with_retries(upload_id, payload)

    def _notify_with_retries(self, upload_id: str, payload: Dict[str, Any]) -> bool:
        url = self._transition_url(upload_id)
        for attempt in range(1, self.max_retries + 1):
            try:
                self._do_request(url, payload)
                logger.info("Transicao notificada para upload %s (tentativa %d)", upload_id, attempt)
                return True
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_HTTP and attempt < self.max_retries:
                    logger.warning(
                        "HTTP %d ao notificar %s (tentativa %d/%d)",
                        exc.code,
                        upload_id,
                        attempt,
                        self.max_retries,
                    )
                    self._backoff(attempt)
                    continue
                logger.error(
                    "Falha definitiva ao notificar %s: HTTP %d", upload_id, exc.code
                )
                return False
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    logger.warning(
                        "Erro de rede ao notificar %s (tentativa %d/%d): %s",
                        upload_id,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    self._backoff(attempt)
                    continue
                logger.error("Falha definitiva ao notificar %s: %s", upload_id, exc)
                return False
        return False

    def _do_request(self, url: str, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as resp:
            resp.read()

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(_BACKOFF_SECONDS * (2 ** (attempt - 1)))


__all__ = ["DeliveryCoreNotifier"]
