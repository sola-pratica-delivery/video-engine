"""Sintetizador de headlines de alto CTR para YouTube via Google Gemini (Issue #22).

Consome a transcricao fonetica completa do Faster-Whisper e/ou o titulo do video,
instruindo o modelo Gemini 2.5 Flash a gerar chamadas curtas, impactantes e
sem pontuacao (2 a 4 palavras em caixa alta). Inclui fallback deterministico
completo em caso de ausencia de API key, erro de rede, HTTP 429/5xx ou timeout.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional, Union

import httpx

from video_engine.captions.models import TranscriptionResult
from video_engine.thumbnail.models import (
    GeminiHeadlineConfig,
)

logger = logging.getLogger(__name__)


class HeadlineSynthesisError(Exception):
    """Falha na chamada a API do Gemini ou validacao do schema da headline."""


class GeminiHeadlineSynthesizer:
    """Sintetiza headlines de alto CTR para capas de video utilizando Google Gemini."""

    def __init__(
        self,
        config: Optional[GeminiHeadlineConfig] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config or GeminiHeadlineConfig()
        self._client = http_client

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            env_timeout = os.environ.get("GEMINI_TIMEOUT_S")
            timeout_val = float(env_timeout) if env_timeout else self.config.timeout_s
            timeout = httpx.Timeout(timeout_val)
            self._client = httpx.Client(base_url=self.config.base_url, timeout=timeout)
        return self._client

    def _resolve_api_key(self) -> str:
        key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            try:
                from video_engine.env import load_env

                load_env()
                key = os.environ.get("GEMINI_API_KEY")
            except Exception:
                pass
        if not key:
            raise HeadlineSynthesisError(
                "GEMINI_API_KEY ausente. Configure a chave no ambiente ou em GeminiHeadlineConfig."
            )
        return key

    @staticmethod
    def sanitize_headline(text: str, min_words: int = 2, max_words: int = 4) -> Optional[str]:
        """Normaliza o texto para caixa alta, remove pontuacao e valida entre min e max palavras."""
        if not text:
            return None
        # Remove pontuacoes, mantendo apenas letras, numeros e espacos
        cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        words = [w.upper() for w in cleaned.split() if w.strip()]
        if len(words) < min_words:
            return None
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words)

    def fallback_headline(
        self,
        title: Optional[str] = None,
        transcription: Optional[Union[str, TranscriptionResult, Any]] = None,
    ) -> str:
        """Deriva uma headline deterministica segura a partir do titulo ou transcricao."""
        if title:
            sanitized = self.sanitize_headline(title, self.config.min_words, self.config.max_words)
            if sanitized:
                return sanitized

        raw_text: Optional[str] = None
        if isinstance(transcription, TranscriptionResult):
            raw_text = transcription.text
        elif isinstance(transcription, str):
            raw_text = transcription
        elif transcription is not None and hasattr(transcription, "text"):
            raw_text = str(transcription.text)

        if raw_text:
            sanitized = self.sanitize_headline(raw_text, self.config.min_words, self.config.max_words)
            if sanitized:
                return sanitized

        return self.config.fallback_headline

    def synthesize(
        self,
        title: Optional[str] = None,
        transcription: Optional[Union[str, TranscriptionResult, Any]] = None,
    ) -> str:
        """Consulta o Gemini para sintetizar a headline, aplicando fallback automatico."""
        try:
            return self._do_synthesize(title=title, transcription=transcription)
        except Exception as exc:
            logger.warning(
                "Falha ao sintetizar headline com Gemini (%s); acionando fallback deterministico.",
                exc,
            )
            return self.fallback_headline(title=title, transcription=transcription)

    def _do_synthesize(
        self,
        title: Optional[str] = None,
        transcription: Optional[Union[str, TranscriptionResult, Any]] = None,
    ) -> str:
        api_key = self._resolve_api_key()
        client = self._get_client()

        raw_transcription: str = ""
        if isinstance(transcription, TranscriptionResult):
            raw_transcription = transcription.text or ""
        elif isinstance(transcription, str):
            raw_transcription = transcription
        elif transcription is not None and hasattr(transcription, "text"):
            raw_transcription = str(transcription.text or "")

        prompt = self._build_prompt(title=title, transcription=raw_transcription)
        body = self._build_request_body(prompt)

        base_url = str(self.config.base_url).rstrip("/")
        url = f"{base_url}/v1beta/models/{self.config.model}:generateContent?key={api_key}"
        response = client.post(url, json=body)

        if response.status_code != 200:
            raise HeadlineSynthesisError(
                f"Gemini API retornou HTTP {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        raw_headline = self._extract_headline_from_response(data)
        sanitized = self.sanitize_headline(raw_headline, self.config.min_words, self.config.max_words)
        if not sanitized:
            raise HeadlineSynthesisError(
                f"Headline retornada pelo Gemini nao atendeu aos requisitos (2-4 palavras): '{raw_headline}'"
            )
        return sanitized

    @staticmethod
    def _extract_headline_from_response(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            raise HeadlineSynthesisError("Resposta do Gemini sem candidatos")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or "text" not in parts[0]:
            raise HeadlineSynthesisError("Resposta do Gemini sem parts de texto")

        raw_text = parts[0]["text"].strip()
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "headline" in parsed:
                return str(parsed["headline"]).strip()
        except json.JSONDecodeError as exc:
            raise HeadlineSynthesisError(f"Resposta do Gemini nao e JSON valido: {exc}") from exc

        raise HeadlineSynthesisError("JSON do Gemini nao contem o campo 'headline'")


    @staticmethod
    def _build_prompt(title: Optional[str], transcription: str) -> str:
        context_parts = []
        if title:
            context_parts.append(f"TITULO DO VIDEO: {title}")
        if transcription:
            context_parts.append(f"TRANSCRICAO: {transcription[:3000]}")
        context_block = "\n".join(context_parts) if context_parts else "(sem contexto adicional)"

        return (
            "Voce e um estrategista especialista em packaging e CTR (Click-Through Rate) para o YouTube.\n"
            "Sua missao e sintetizar uma headline irresistivel e de alto impacto para a capa deste video.\n\n"
            f"{context_block}\n\n"
            "REGRAS ESTRITAS:\n"
            "1. A headline DEVE ter exatamente de 2 a 4 palavras.\n"

            "2. DEVE estar em CAIXA ALTA (ex.: 'O SEGREDO REVELADO', 'NUNCA FACA ISSO', 'ELES ENGANARAM VOCE').\n"
            "3. NAO use pontuacao (sem pontos, sem virgulas, sem exclamacao, sem interrogacao).\n"
            "4. Crie curiosidade extrema, urgencia ou quebra de crenca comum.\n"
            "5. Responda estritamente em JSON aderente ao schema: {\"headline\": \"SUA FRASE AQUI\"}."
        )

    def _build_request_body(self, prompt: str) -> dict:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "headline": {"type": "STRING"},
                    },
                    "required": ["headline"],
                },
            },
        }


__all__ = [
    "GeminiHeadlineSynthesizer",
    "HeadlineSynthesisError",
]
