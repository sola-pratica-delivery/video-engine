"""Analisador semantico de ganchos via Google AI Studio (Issue #15).

Implementa ``SemanticHookAnalyzer``, que consome a transcricao fonetica
(``TranscriptionResult`` do Faster-Whisper) e consulta o Google AI Studio
(Gemini, ``gemini-2.5-flash`` por padrao) com structured JSON output para
identificar momentos de gancho viral, quebras de padrao e raciocinios
autocontidos elegiveis a cortes verticais de 25s a 58s.

O protocolo HTTP e injetavel (``http_client``) para testes 100%
deterministicos e offline. Qualquer falha de chave, rede, rate limit ou
payload malformado levanta ``SemanticHookError`` para que o chamador
``HookDetector`` execute fallback gracioso para a heuristica offline.
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
from video_engine.shorts.models import (
    HookDetectorConfig,
    SemanticHookInterval,
    SemanticHookResponse,
)

logger = logging.getLogger(__name__)


class SemanticHookError(Exception):
    """Falha na decisao semantica via Google AI Studio (para fallback).

    Cobre: ausencia de chave, falha de rede/timeout, HTTP 429/5xx, payload
    malformado ou resposta que nao adere ao schema de ganchos.
    """


class SemanticHookAnalyzer:
    """Consulta o Gemini com structured output para mapear ganchos virais."""

    def __init__(
        self,
        config: Optional[HookDetectorConfig] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        self.config = config or HookDetectorConfig()
        self._client = http_client

    # ------------------------------------------------------------------ #
    # Infraestrutura HTTP
    # ------------------------------------------------------------------ #
    def _get_client(self) -> Any:
        if self._client is None:
            timeout = httpx.Timeout(self.config.gemini_timeout_s)
            self._client = httpx.Client(base_url=self.config.gemini_base_url, timeout=timeout)
        return self._client

    def _resolve_api_key(self) -> str:
        key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            from video_engine.env import load_env

            load_env()
            key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SemanticHookError(
                "GEMINI_API_KEY ausente. Configure a chave em um arquivo .env, "
                "no ambiente ou em HookDetectorConfig.gemini_api_key, ou use "
                "decision_mode='heuristic'."
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
            "Voce e um editor de Shorts/Reels/YouTube Shorts especializado em viralizacao.\n"
            "Analise a transcricao fonetica abaixo (duracao total "
            f"{total_duration_ms} ms) e identifique os 1 a 3 trechos mais virais "
            "para cortes verticais: introducoes fortes, quebras de expectativa, "
            "frases provocativas, perguntas retoricas e revelacoes.\n\n"
            "Cada corte deve ser AUTOCONTIDO (inicio, meio e fim, sem corte no "
            "meio de uma frase) e com duracao estrita entre 25000 e 58000 ms.\n\n"
            "TRANSCRICAO (timestamps em segundos):\n"
            f"{words_block}\n\n"
            "Responda apenas em JSON aderente ao schema: {\"hook_candidates\": "
            "[{\"start_ms\": int, \"end_ms\": int, \"hook_text\": string, "
            "\"summary\": string, \"reason\": string, \"confidence\": number}]}. "
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
                        "hook_candidates": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "start_ms": {"type": "INTEGER"},
                                    "end_ms": {"type": "INTEGER"},
                                    "hook_text": {"type": "STRING"},
                                    "summary": {"type": "STRING"},
                                    "reason": {"type": "STRING"},
                                    "confidence": {"type": "NUMBER"},
                                },
                                "required": ["start_ms", "end_ms"],
                            },
                        }
                    },
                    "required": ["hook_candidates"],
                },
            },
        }

    # ------------------------------------------------------------------ #
    # Parsing da resposta
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(response: Any) -> List[SemanticHookInterval]:
        try:
            raw = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise SemanticHookError(f"Resposta JSON invalida do Google AI Studio: {exc}") from exc

        candidates = raw.get("candidates") or []
        if not candidates:
            raise SemanticHookError("Resposta do Google AI Studio sem candidates.")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text:
            raise SemanticHookError("Resposta do Google AI Studio sem texto.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SemanticHookError(f"JSON truncado/invalido na resposta do Gemini: {exc}") from exc

        try:
            parsed = SemanticHookResponse.model_validate(payload)
        except ValidationError as exc:
            raise SemanticHookError(
                f"Resposta nao adere ao schema SemanticHookResponse: {exc}"
            ) from exc
        return list(parsed.hook_candidates)

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        transcription: TranscriptionResult,
        total_duration_ms: int,
    ) -> List[SemanticHookInterval]:
        """Extrai ganchos virais via Google AI Studio com structured JSON.

        Raises:
            SemanticHookError: chave ausente, falha de rede/timeout, HTTP 429/5xx
                ou resposta invalida/malformada. O chamador deve capturar e
                executar fallback para a heuristica.
        """
        if not transcription or not transcription.words:
            return []

        api_key = self._resolve_api_key()
        prompt = self._build_prompt(transcription, total_duration_ms)
        body = self._build_request_body(prompt)
        url = f"{self.config.gemini_base_url}/models/{self.config.gemini_model}:generateContent?key={api_key}"
        client = self._get_client()

        attempts = self.config.gemini_max_retries + 1
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
                raise SemanticHookError(
                    f"Falha de rede/timeout no Google AI Studio: {exc}"
                ) from exc

            if response.status_code in (429, 500, 502, 503, 504):
                raise SemanticHookError(
                    f"Google AI Studio indisponivel ou em rate limit (HTTP {response.status_code})."
                )
            if response.status_code != 200:
                raise SemanticHookError(
                    f"Resposta inesperada do Google AI Studio (HTTP {response.status_code})."
                )
            return self._parse_response(response)

        raise SemanticHookError("Nao foi possivel concluir a decisao semantica.")  # pragma: no cover


__all__ = ["SemanticHookAnalyzer", "SemanticHookError"]
