"""Testes da integração semântica de Dynamic Zoom (Issue #19).

Cobre:
- Modelos de dados (GeminiZoomConfig, SemanticZoomInterval, SemanticZoomResponse)
  e extensoes de DynamicZoomConfig (decision_mode default GEMINI + subconfig)
- SemanticZoomAnalyzer:
  - Sucesso com structured output JSON (httpx.MockTransport)
  - Endpoint/models/{model}:generateContent com key e schema no body
  - Retry em erro transitorio de rede
  - Falhas: chave ausente, HTTP 429/5xx/inesperado, timeout/erro de rede,
    JSON truncado, resposta fora do schema
  - Transcricao sem palavras -> retorno vazio sem rede
- DynamicZoomProcessor:
  - Uso do plano semantico quando decision_mode=GEMINI e ha transcricao
  - Guardrails de duracao (zoom entre min e max; expansao/capping de intervalos)
  - Alinhamento de janelas a pausas de fala (tolerancia = min_shot)
  - Videos curtos (< min) mantem plano normal unico
  - Fallback gracioso para a heuristica temporal em erro/lista vazia
  - analyzer NAO e acionado em modo heuristic, sem transcricao ou sem palavras
- Integracao real com FFmpeg via apply_zoom encaminhando a transcricao
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import httpx
import pytest

from video_engine.audio.models import TimeInterval
from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.video.models import (
    DecisionMode,
    DynamicZoomConfig,
    GeminiZoomConfig,
    SemanticZoomInterval,
    SemanticZoomResponse,
    ZoomMode,
)
from video_engine.video.semantic_analyzer import (
    SemanticZoomAnalyzer,
    SemanticZoomError,
)
from video_engine.video.zoom import DynamicZoomProcessor

FFMPEG = shutil.which("ffmpeg")


def make_transcription(total_ms: int = 60000, step_ms: int = 1000) -> TranscriptionResult:
    """Transcricao sintetica com palavras a cada ``step_ms`` milissegundos."""
    words: List[WordTimestamp] = []
    start = 0
    while start < total_ms:
        end = min(total_ms, start + step_ms)
        words.append(
            WordTimestamp(word=f"palavra{start}", start_ms=start, end_ms=end, probability=0.9)
        )
        start = end
    return TranscriptionResult(
        text=" ".join(w.word for w in words),
        language="pt",
        duration_ms=total_ms,
        words=words,
    )


def gemini_json(intervals: list) -> dict:
    payload = json.dumps({"zoom_intervals": intervals})
    return {"candidates": [{"content": {"parts": [{"text": payload}]}}]}


class FakeAnalyzer:
    """Stub determinístico para testar o processador sem rede."""

    def __init__(
        self,
        intervals: Optional[List[SemanticZoomInterval]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self.intervals = list(intervals or [])
        self.error = error
        self.calls: List[tuple] = []

    def analyze(self, transcription, total_duration_ms):
        self.calls.append((transcription, total_duration_ms))
        if self.error is not None:
            raise self.error
        return list(self.intervals)


class BoomAnalyzer(FakeAnalyzer):
    """Analyzer que falha se for acionado (para provar que nao e usado)."""

    def analyze(self, transcription, total_duration_ms):
        raise AssertionError("semantic_analyzer.nao.deveria.ser.acionado")


def heuristic_reference():
    return DynamicZoomProcessor(
        config=DynamicZoomConfig(decision_mode=DecisionMode.HEURISTIC)
    ).plan_zoom_shots(total_duration_ms=60000)


# --------------------------------------------------------------------------- #
# 1. Modelos e Validacoes
# --------------------------------------------------------------------------- #
def test_gemini_zoom_config_defaults():
    cfg = GeminiZoomConfig()
    assert cfg.model == "gemini-2.5-flash"
    assert cfg.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert cfg.timeout_s == 10.0
    assert cfg.max_retries == 1
    assert cfg.api_key is None


def test_gemini_zoom_config_rejects_extra_fields():
    with pytest.raises(Exception):
        GeminiZoomConfig(unknown_field=True)


def test_dynamic_zoom_config_decision_mode_default_gemini():
    cfg = DynamicZoomConfig()
    assert cfg.decision_mode == DecisionMode.GEMINI
    assert isinstance(cfg.gemini, GeminiZoomConfig)
    assert cfg.gemini.model == "gemini-2.5-flash"


def test_dynamic_zoom_config_accepts_heuristic_mode():
    cfg = DynamicZoomConfig(decision_mode=DecisionMode.HEURISTIC)
    assert cfg.decision_mode == DecisionMode.HEURISTIC
    with pytest.raises(Exception):
        DynamicZoomConfig(decision_mode="desconhecido")


def test_semantic_zoom_interval_validation():
    with pytest.raises(Exception):
        SemanticZoomInterval(start_ms=-1, end_ms=100, reason="x")
    with pytest.raises(Exception):
        SemanticZoomInterval(start_ms=0, end_ms=100, reason="x", confidence=1.5)
    with pytest.raises(Exception):
        SemanticZoomInterval(start_ms=0, end_ms=100)
    interval = SemanticZoomInterval(start_ms=1000, end_ms=5000, reason="alerta", confidence=0.8)
    assert interval.confidence == 0.8


def test_semantic_zoom_response_parse():
    response = SemanticZoomResponse.model_validate(
        {
            "zoom_intervals": [
                {"start_ms": 1000, "end_ms": 5000, "reason": "punchline", "confidence": 0.9}
            ]
        }
    )
    assert len(response.zoom_intervals) == 1
    assert response.zoom_intervals[0].reason == "punchline"


def test_semantic_zoom_response_accepts_empty():
    response = SemanticZoomResponse.model_validate({})
    assert response.zoom_intervals == []


# --------------------------------------------------------------------------- #
# 2. SemanticZoomAnalyzer - Sucesso
# --------------------------------------------------------------------------- #
def test_measure_analyzer_success_from_structured_output():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=gemini_json([{"start_ms": 26000, "end_ms": 34000, "reason": "gancho"}]),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="chave-teste", max_retries=0),
        http_client=client,
    )
    intervals = analyzer.analyze(make_transcription(), 60000)
    assert len(intervals) == 1
    assert intervals[0].start_ms == 26000
    assert intervals[0].end_ms == 34000
    assert intervals[0].reason == "gancho"


def test_analyzer_posts_key_url_and_json_schema():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=gemini_json([]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(
            api_key="key-secret",
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            max_retries=0,
        ),
        http_client=client,
    )
    analyzer.analyze(make_transcription(total_ms=30000), 30000)

    assert captured["url"].startswith(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert "key=key-secret" in captured["url"]
    body = json.loads(captured["body"])
    assert "zoom_intervals" in body["generationConfig"]["responseSchema"]["properties"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert isinstance(body["contents"][0]["parts"][0]["text"], str)


def test_analyzer_env_key_and_retry_after_network_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-ambiente")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("caiu")
        return httpx.Response(
            200,
            json=gemini_json([{"start_ms": 1000, "end_ms": 9000, "reason": "x"}]),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key=None, max_retries=1),
        http_client=client,
    )
    intervals = analyzer.analyze(make_transcription(), 60000)
    assert calls["n"] == 2
    assert len(intervals) == 1
    assert intervals[0].confidence == 1.0


def test_analyzer_returns_empty_without_words():
    transcript = TranscriptionResult(text="", language="pt", duration_ms=60000, words=[])
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key=None, max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    assert analyzer.analyze(transcript, 60000) == []


# --------------------------------------------------------------------------- #
# 3. SemanticZoomAnalyzer - Falhas
# --------------------------------------------------------------------------- #
def test_analyzer_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key=None, max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    with pytest.raises(SemanticZoomError):
        analyzer.analyze(make_transcription(), 60000)


def test_analyzer_raises_on_rate_limit_429():
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="k", max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429))),
    )
    with pytest.raises(SemanticZoomError, match="rate limit"):
        analyzer.analyze(make_transcription(), 60000)


def test_analyzer_raises_on_server_error_500():
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="k", max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    with pytest.raises(SemanticZoomError, match="indisponivel"):
        analyzer.analyze(make_transcription(), 60000)


def test_analyzer_raises_on_unexpected_status():
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="k", max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400))),
    )
    with pytest.raises(SemanticZoomError, match="400"):
        analyzer.analyze(make_transcription(), 60000)


def test_analyzer_raises_on_network_error_without_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sem conexao")

    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="k", max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SemanticZoomError, match="rede/timeout"):
        analyzer.analyze(make_transcription(), 60000)


def test_analyzer_raises_on_truncated_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"zoom_intervals": ['}]}}]},
        )

    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="k", max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SemanticZoomError, match="JSON"):
        analyzer.analyze(make_transcription(), 60000)


def test_analyzer_raises_on_schema_violation():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"zoom_intervals": [{}]}'}]}}]},
        )

    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="k", max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SemanticZoomError, match="schema"):
        analyzer.analyze(make_transcription(), 60000)


def test_analyzer_raises_on_empty_text_response():
    analyzer = SemanticZoomAnalyzer(
        config=GeminiZoomConfig(api_key="k", max_retries=0),
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"candidates": []}))
        ),
    )
    with pytest.raises(SemanticZoomError, match="candidates"):
        analyzer.analyze(make_transcription(), 60000)


# --------------------------------------------------------------------------- #
# 4. DynamicZoomProcessor - Plano Semantico e Guardrails
# --------------------------------------------------------------------------- #
def test_plan_uses_semantic_shots_when_gemini():
    analyzer = FakeAnalyzer(
        intervals=[SemanticZoomInterval(start_ms=26000, end_ms=34000, reason="gancho", confidence=0.9)]
    )
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=analyzer)
    shots = processor.plan_zoom_shots(total_duration_ms=60000, transcription=make_transcription())

    assert len(analyzer.calls) == 1
    zooms = [s for s in shots if s.mode == ZoomMode.ZOOM]
    assert len(zooms) == 1
    assert zooms[0].start_ms == 26000
    assert zooms[0].end_ms == 34000
    assert zooms[0].scale == 1.15
    for shot in shots:
        assert 0 <= shot.start_ms <= shot.end_ms <= 60000
    normal_starts = [s.start_ms for s in shots if s.mode == ZoomMode.NORMAL]
    assert 0 in normal_starts


def test_plan_semantic_expands_short_interval_to_min_and_caps_long():
    analyzer = FakeAnalyzer(
        intervals=[
            SemanticZoomInterval(start_ms=30000, end_ms=31000, reason="curto"),
            SemanticZoomInterval(start_ms=15000, end_ms=60000, reason="longo"),
        ]
    )
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=analyzer)
    shots = processor.plan_zoom_shots(total_duration_ms=60000, transcription=make_transcription())
    zooms = [s for s in shots if s.mode == ZoomMode.ZOOM]
    assert zooms
    for zoom in zooms:
        assert 8000 <= zoom.duration_ms <= 15000
    for shot in shots[:-1]:
        assert shot.duration_ms <= 15000


def test_plan_semantic_aligns_window_to_pause():
    pauses = [TimeInterval(start_ms=40500, end_ms=42000)]
    analyzer = FakeAnalyzer(
        intervals=[SemanticZoomInterval(start_ms=26000, end_ms=34000, reason="alerta")]
    )
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=analyzer)
    shots = processor.plan_zoom_shots(
        total_duration_ms=60000,
        pause_intervals=pauses,
        transcription=make_transcription(),
    )
    zooms = [s for s in shots if s.mode == ZoomMode.ZOOM]
    assert zooms
    assert zooms[0].end_ms == 40500


def test_plan_semantic_short_video_stays_normal_no_call():
    spy = BoomAnalyzer()
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=spy)
    shots = processor.plan_zoom_shots(total_duration_ms=5000, transcription=make_transcription(total_ms=5000))
    assert len(shots) == 1
    assert shots[0].mode == ZoomMode.NORMAL
    assert shots[0].start_ms == 0
    assert shots[0].end_ms == 5000
    assert spy.calls == []


# --------------------------------------------------------------------------- #
# 5. Fallback para Heuristica
# --------------------------------------------------------------------------- #
def test_plan_falls_back_to_heuristic_on_semantic_error():
    analyzer = FakeAnalyzer(error=SemanticZoomError("falhou"))
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=analyzer)
    shots = processor.plan_zoom_shots(total_duration_ms=60000, transcription=make_transcription())
    assert len(analyzer.calls) == 1
    assert shots == heuristic_reference()


def test_plan_falls_back_to_heuristic_on_unexpected_exception():
    analyzer = FakeAnalyzer(error=RuntimeError("boom inesperado"))
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=analyzer)
    shots = processor.plan_zoom_shots(total_duration_ms=60000, transcription=make_transcription())
    assert shots == heuristic_reference()


def test_plan_falls_back_to_heuristic_on_empty_intervals():
    analyzer = FakeAnalyzer(intervals=[])
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=analyzer)
    shots = processor.plan_zoom_shots(total_duration_ms=60000, transcription=make_transcription())
    assert len(analyzer.calls) == 1
    assert shots == heuristic_reference()


# --------------------------------------------------------------------------- #
# 6. Analyzer NAO deve ser acionado fora do modo GEMINI
# --------------------------------------------------------------------------- #
def test_plan_heuristic_mode_does_not_call_analyzer():
    spy = BoomAnalyzer()
    processor = DynamicZoomProcessor(
        config=DynamicZoomConfig(decision_mode=DecisionMode.HEURISTIC),
        semantic_analyzer=spy,
    )
    shots = processor.plan_zoom_shots(total_duration_ms=60000, transcription=make_transcription())
    assert spy.calls == []
    assert shots == heuristic_reference()


def test_plan_without_transcription_does_not_call_analyzer():
    spy = BoomAnalyzer()
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=spy)
    shots = processor.plan_zoom_shots(total_duration_ms=60000, pause_intervals=[])
    assert spy.calls == []
    assert shots == heuristic_reference()  # mesmo com decision_mode GEMINI, sem transcricao


def test_plan_empty_words_does_not_call_analyzer():
    empty = TranscriptionResult(text="", language="pt", duration_ms=60000, words=[])
    spy = BoomAnalyzer()
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=spy)
    shots = processor.plan_zoom_shots(total_duration_ms=60000, transcription=empty)
    assert spy.calls == []
    assert shots == heuristic_reference()


# --------------------------------------------------------------------------- #
# 7. Integracao com FFmpeg
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not FFMPEG, reason="FFmpeg nao disponivel")
def test_apply_zoom_forwards_transcription(tmp_path):
    """Valida que apply_zoom encaminha a transcricao ao analyzer e gera o corte com zoom."""
    input_video = tmp_path / "input_semantic.mp4"
    out_video = tmp_path / "output_semantic.mp4"
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=20:size=320x240:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=20",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        str(input_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    analyzer = FakeAnalyzer(
        intervals=[SemanticZoomInterval(start_ms=10000, end_ms=18000, reason="punchline")]
    )
    processor = DynamicZoomProcessor(config=DynamicZoomConfig(), semantic_analyzer=analyzer)
    transcription = make_transcription(total_ms=20000, step_ms=1000)
    result = processor.apply_zoom(
        input_video=input_video,
        output_video=out_video,
        transcription=transcription,
    )

    assert Path(result.output_path).is_file()
    assert result.zoom_shots_count >= 1
    assert len(analyzer.calls) == 1
    assert analyzer.calls[0][0] is transcription
    assert analyzer.calls[0][1] == 20000
    zooms = [s for s in result.shots if s.mode == ZoomMode.ZOOM]
    assert all(8000 <= s.duration_ms <= 15000 for s in zooms)
