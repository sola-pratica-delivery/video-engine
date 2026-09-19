"""Analisador semantico de Dynamic Zoom via Google AI Studio (Issue #19).

Implementa ``SemanticZoomAnalyzer``, que consome a transcricao fonetica
(``TranscriptionResult`` do Faster-Whisper) e consulta o Google AI Studio
(Gemini) com structured output para identificar momentos ideais de punch-in
zoom (argumentos centrais, alertas, ganchos e punchlines).

O protocolo HTTP e injetavel (``http_client``) para permitir testes 100%
deterministicos e offline. Qualquer falha de chave, rede, rate limit ou
payload malformado levanta ``SemanticZoomError`` para que o chamador
``DynamicZoomProcessor`` execute fallback gracioso para a heuristica temporal.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional

import httpx
from pydantic import ValidationError

from video_engine.captions.models import TranscriptionResult
from video_engine.video.models import (
    GeminiZoomConfig,
    SemanticZoomInterval,
    SemanticZoomResponse,
)

logger = logging.getLogger(__name__)


class SemanticZoomError(Exception):
    """Falha na decisão semantica via Google AI Studio (para fallback).

    Cobrindo: ausência de chave, falha de rede/timeout, HTTP 429/5xx,
    payload malformado ou resposta que nao adere ao schema.
    """


class SemanticZoomAnalyzer:
    """Consome transcricao fonetica e consulta o Gemini para mapear cortes com zoom."""

    def __init__(
        self,
        config: Optional[GeminiZoomConfig] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        self.config = config or GeminiZoomConfig()
        self._client = http_client

    # ------------------------------------------------------------------ #
    # Infraestrutura HTTP
    # ------------------------------------------------------------------ #
    def _get_client(self) -> Any:
        if self._client is None:
            timeout = httpx.Timeout(self.config.timeout_s)
            self._client = httpx.Client(base_url=self.config.base_url, timeout=timeout)
        return self._client

    def _resolve_api_key(self) -> str:
        key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            from video_engine.env import load_env

            load_env()
            key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SemanticZoomError(
                "GEMINI_API_KEY ausente. Configure a chave em um arquivo .env, "
                "no ambiente ou em GeminiZoomConfig.api_key, ou use decision_mode='heuristic'."
            )
        return key

    # ------------------------------------------------------------------ #
    # Prompt e payload
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_prompt(transcription: TranscriptionResult, total_duration_ms: int) -> str:
        word_lines = []
        for word in transcription.words:
            start_s = word.start_ms / 1000.0
            end_s = word.end_ms / 1000.0
            word_lines.append(f"{start_s:.2f}-{end_s:.2f}s: {word.word}")
        words_block = "\n".join(word_lines) if word_lines else "(sem palavras)"
        return (
            "Voce e um editor de videos para Shorts/Reels/YouTube Shorts.\n"
            "Analise a transcricao fonetica abaixo (duracao total "
            f"{total_duration_ms} ms) e identifique trechos temporais ideais "
            "para punch-in zoom (1.15x): argumentos centrais, alertas, "
            "ganchos e punchlines.\n\n"
            "TRANSCRICAO (timestamps em segundos):\n"
            f"{words_block}\n\n"
            "Responda apenas em JSON aderente ao schema: "
            "{\"zoom_intervals\": [{\"start_ms\": int, \"end_ms\": int, "
            "\"reason\": string, \"confidence\": float}]}. "
            "Escolha apenas os trechos de maior impacto, sem repeticao excessiva."
        )

    @staticmethod
    def _build_request_body(prompt: str) -> dict:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "zoom_intervals": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "start_ms": {"type": "INTEGER"},
                                    "end_ms": {"type": "INTEGER"},
                                    "reason": {"type": "STRING"},
                                    "confidence": {"type": "NUMBER"},
                                },
                                "required": ["start_ms", "end_ms", "reason"],
                            },
                        }
                    },
                    "required": ["zoom_intervals"],
                },
            },
        }

    # ------------------------------------------------------------------ #
    # Parsing da resposta
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(response: Any) -> List[SemanticZoomInterval]:
        try:
            raw = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise SemanticZoomError(f"Resposta JSON invalida do Google AI Studio: {exc}") from exc

        candidates = raw.get("candidates") or []
        if not candidates:
            raise SemanticZoomError("Resposta do Google AI Studio sem candidates.")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text:
            raise SemanticZoomError("Resposta do Google AI Studio sem texto.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SemanticZoomError(f"JSON truncado/invalido na resposta do Gemini: {exc}") from exc

        try:
            parsed = SemanticZoomResponse.model_validate(payload)
        except ValidationError as exc:
            raise SemanticZoomError(
                f"Resposta nao adere ao schema SemanticZoomResponse: {exc}"
            ) from exc
        return list(parsed.zoom_intervals)

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        transcription: TranscriptionResult,
        total_duration_ms: int,
    ) -> List[SemanticZoomInterval]:
        """Extrai intervalos de enfase via Google AI Studio com structured JSON.

        Raises:
            SemanticZoomError: chave ausente, falha de rede/timeout,
                HTTP 429/5xx ou resposta invalida/malformada. O chamador
                deve capturar e executar fallback para a heuristica.
        """
        if not transcription or not transcription.words:
            return []

        api_key = self._resolve_api_key()
        prompt = self._build_prompt(transcription, total_duration_ms)
        body = self._build_request_body(prompt)
        url = f"{self.config.base_url}/models/{self.config.model}:generateContent?key={api_key}"
        client = self._get_client()

        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = client.post(url, json=body)
            except (httpx.RequestError, OSError) as exc:
                if attempt < attempts - 1:
                    delay = 0.1 * (attempt + 1)
                    logger.warning(
                        "Falha de rede no Google AI Studio, tentativa %d/%d: %s",
                        attempt + 1,
                        attempts,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                raise SemanticZoomError(
                    f"Falha de rede/timeout no Google AI Studio: {exc}"
                ) from exc

            if response.status_code in (429, 500, 502, 503, 504):
                raise SemanticZoomError(
                    f"Google AI Studio indisponivel ou em rate limit (HTTP {response.status_code})."
                )
            if response.status_code != 200:
                raise SemanticZoomError(
                    f"Resposta inesperada do Google AI Studio (HTTP {response.status_code})."
                )
            return self._parse_response(response)

        raise SemanticZoomError("Nao foi possivel concluir a decisão semantica.")  # pragma: no cover


__all__ = ["SemanticZoomAnalyzer", "SemanticZoomError"]
