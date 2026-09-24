"""Testes unitarios do sintetizador de headlines via Google Gemini (Issue #22).

Cobre o ciclo TDD: sanitizacao estrita (2 a 4 palavras em caixa alta, sem pontuacao),
fallback deterministico e chamadas HTTP com mock do Gemini 2.5 Flash.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.thumbnail.headline_synthesizer import (
    GeminiHeadlineSynthesizer,
)
from video_engine.thumbnail.models import (
    GeminiHeadlineConfig,
)


class TestHeadlineSanitization:
    """Valida a sanitizacao de 2 a 4 palavras em caixa alta sem pontuacao."""

    def test_sanitize_valid_two_to_four_words(self) -> None:
        assert GeminiHeadlineSynthesizer.sanitize_headline("O Segredo Revelado") == "O SEGREDO REVELADO"
        assert GeminiHeadlineSynthesizer.sanitize_headline("nunca faca isso") == "NUNCA FACA ISSO"
        assert GeminiHeadlineSynthesizer.sanitize_headline("ELES ENGANARAM VOCE HOJE") == "ELES ENGANARAM VOCE HOJE"

    def test_sanitize_strips_punctuation_and_symbols(self) -> None:
        assert GeminiHeadlineSynthesizer.sanitize_headline('"O SEGREDO REVELADO!"') == "O SEGREDO REVELADO"
        assert GeminiHeadlineSynthesizer.sanitize_headline("CUIDADO: PARE AGORA...") == "CUIDADO PARE AGORA"
        assert GeminiHeadlineSynthesizer.sanitize_headline("VOCE SABIA DISSO???") == "VOCE SABIA DISSO"

    def test_sanitize_truncates_excess_words(self) -> None:
        # Frases longas devem ser recortadas com seguranca para 4 palavras
        result = GeminiHeadlineSynthesizer.sanitize_headline("ISSO VAI MUDAR SUA VIDA COMPLETAMENTE HOJE")
        assert result == "ISSO VAI MUDAR SUA"
        assert len(result.split()) == 4

    def test_sanitize_rejects_single_word_or_empty(self) -> None:
        assert GeminiHeadlineSynthesizer.sanitize_headline("INCRIVEL") is None
        assert GeminiHeadlineSynthesizer.sanitize_headline("") is None
        assert GeminiHeadlineSynthesizer.sanitize_headline("   ") is None
        assert GeminiHeadlineSynthesizer.sanitize_headline("!!! ???") is None


class TestDeterministicFallback:
    """Valida o fallback seguro derivado do titulo ou da transcricao."""

    def test_fallback_from_title(self) -> None:
        synthesizer = GeminiHeadlineSynthesizer()
        headline = synthesizer.fallback_headline(title="Como Criar Thumbnails Incriveis no Youtube")
        assert headline == "COMO CRIAR THUMBNAILS INCRIVEIS"
        assert 2 <= len(headline.split()) <= 4

    def test_fallback_from_short_title(self) -> None:
        synthesizer = GeminiHeadlineSynthesizer()
        headline = synthesizer.fallback_headline(title="Novo Comeco")
        assert headline == "NOVO COMECO"

    def test_fallback_from_transcription_when_no_title(self) -> None:
        synthesizer = GeminiHeadlineSynthesizer()
        transcription = TranscriptionResult(
            text="hoje nos vamos aprender tudo sobre marketing digital",
            words=[
                WordTimestamp(word="hoje", start_ms=0, end_ms=500, probability=1.0),
                WordTimestamp(word="nos", start_ms=500, end_ms=1000, probability=1.0),
            ],
            duration_ms=5000,
        )
        headline = synthesizer.fallback_headline(title=None, transcription=transcription)
        assert headline == "HOJE NOS VAMOS APRENDER"
        assert 2 <= len(headline.split()) <= 4

    def test_fallback_to_default_when_no_title_and_no_transcription(self) -> None:
        synthesizer = GeminiHeadlineSynthesizer()
        assert synthesizer.fallback_headline(title=None, transcription=None) == "ASSISTA AGORA"
        assert synthesizer.fallback_headline(title="", transcription="") == "ASSISTA AGORA"


class TestGeminiApiSynthesis:
    """Valida a consulta ao Google Gemini com structured output e tratamento de falhas."""

    def _make_gemini_response(self, headline_text: str) -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps({"headline": headline_text})}
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }

    def test_synthesize_success_with_valid_gemini_response(self) -> None:
        fake_response = self._make_gemini_response("O SEGREDO REVELADO")

        def handler(request: httpx.Request) -> httpx.Response:
            assert "gemini-2.5-flash:generateContent" in str(request.url)
            assert "key=fake-key" in str(request.url)
            body = json.loads(request.content.decode("utf-8"))
            assert "contents" in body
            return httpx.Response(200, json=fake_response)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        synthesizer = GeminiHeadlineSynthesizer(
            config=GeminiHeadlineConfig(api_key="fake-key"),
            http_client=client,
        )

        result = synthesizer.synthesize(
            title="Video Revelador",
            transcription="Descubra o segredo que ninguem te conta sobre sucesso rapido",
        )
        assert result == "O SEGREDO REVELADO"

    def test_synthesize_fallback_when_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("video_engine.env.load_env"):
            synthesizer = GeminiHeadlineSynthesizer(
                config=GeminiHeadlineConfig(api_key=None)
            )
            result = synthesizer.synthesize(title="Guia Definitivo do Sucesso")
            assert result == "GUIA DEFINITIVO DO SUCESSO"


    def test_synthesize_fallback_on_http_500(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        synthesizer = GeminiHeadlineSynthesizer(
            config=GeminiHeadlineConfig(api_key="fake-key"),
            http_client=client,
        )
        result = synthesizer.synthesize(title="Resolvendo Problemas Complexos")
        assert result == "RESOLVENDO PROBLEMAS COMPLEXOS"

    def test_synthesize_fallback_on_http_429_rate_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Resource Exhausted")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        synthesizer = GeminiHeadlineSynthesizer(
            config=GeminiHeadlineConfig(api_key="fake-key"),
            http_client=client,
        )
        result = synthesizer.synthesize(title="Estrategia Infalivel")
        assert result == "ESTRATEGIA INFALIVEL"

    def test_synthesize_fallback_on_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Read timed out")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        synthesizer = GeminiHeadlineSynthesizer(
            config=GeminiHeadlineConfig(api_key="fake-key"),
            http_client=client,
        )
        result = synthesizer.synthesize(title="Aprenda Tudo Agora")
        assert result == "APRENDA TUDO AGORA"

    def test_synthesize_fallback_on_malformed_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "not a json"}]}}]})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        synthesizer = GeminiHeadlineSynthesizer(
            config=GeminiHeadlineConfig(api_key="fake-key"),
            http_client=client,
        )
        result = synthesizer.synthesize(title="Ultima Oportunidade Unica")
        assert result == "ULTIMA OPORTUNIDADE UNICA"
