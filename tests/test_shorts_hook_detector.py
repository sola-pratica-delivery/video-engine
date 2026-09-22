"""Testes da deteccao semantica de ganchos virais e cortes 9:16 (Issue #15).

Cobre:
- Modelos Pydantic: defaults, restricao de extras, limites numericos (25s-58s)
  e serializacao.
- EnergyAnalyzer: picos de energia em audio sintetico, arquivo WAV, reamostragem,
  stereo->mono, casos de borda (array vazio, janela sem analyse).
- SemanticHookAnalyzer: sucesso com structured output (httpx.MockTransport),
  retry em erro de rede, falhas (429/5xx/timeout/JSON truncado/schema).
- HeuristicHookDetector: interrogacao retorica, quebra adversativa, palavra de
  alto impacto, frase provocativa, boost de energia e dedupe por oracao.
- HookDetector: duracao estrita, snapping a oracoes (sem cortar palavras),
  NMS temporal (IoU < 0.1), 1 a 3 cortes, fallback heuristico e casos de borda
  (video curto, transcricao vazia, sem audio, audio corrompido).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
import pytest
import soundfile as sf

from video_engine.captions.models import CaptionSegment, TranscriptionResult, WordTimestamp
from video_engine.shorts.energy_analyzer import EnergyAnalysisError, EnergyAnalyzer
from video_engine.shorts.heuristic_hook_detector import HeuristicHookDetector
from video_engine.shorts.hook_detector import (
    HookDetector,
    suppress_overlaps,
    temporal_iou,
)
from video_engine.shorts.models import (
    DetectionMode,
    EnergyAnalysisResult,
    EnergyProfile,
    HookDetectionResult,
    HookDetectorConfig,
    SemanticHookInterval,
    SemanticHookResponse,
    ShortCandidateCut,
)
from video_engine.shorts.semantic_hook_analyzer import (
    SemanticHookAnalyzer,
    SemanticHookError,
)

SR = 16000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_transcription(
    total_ms: int = 90000,
    step_ms: int = 500,
    per_seg: int = 8,
    text: Optional[List[str]] = None,
) -> TranscriptionResult:
    """Transcricao sintetica com palavras e oracoes (CaptionSegment) regulares."""
    words: List[WordTimestamp] = []
    start = 0
    i = 0
    while start < total_ms:
        end = min(total_ms, start + step_ms)
        word = text[i] if text and i < len(text) else f"palavra{i}"
        words.append(
            WordTimestamp(word=word, start_ms=start, end_ms=end, probability=0.9)
        )
        start = end
        i += 1

    segments: List[CaptionSegment] = []
    for s in range(0, len(words), per_seg):
        part = words[s : s + per_seg]
        segments.append(
            CaptionSegment(
                id=len(segments),
                text=" ".join(w.word for w in part),
                start_ms=part[0].start_ms,
                end_ms=part[-1].end_ms,
                words=part,
            )
        )
    return TranscriptionResult(
        text=" ".join(w.word for w in words),
        language="pt",
        duration_ms=total_ms,
        segments=segments,
        words=words,
    )


def hook_texts(n: int) -> List[str]:
    """Oracao rica em ganchos (adversativo + impacto + interrogacao)."""
    phrases = ["mas", "o", "segredo", "para", "viralizar", "e", "simples", "?"]
    return [phrases[i % len(phrases)] for i in range(n)]


def make_audio(duration_s: int = 90, loud_start_s: int = 30, loud_end_s: int = 40) -> np.ndarray:
    rng = np.random.RandomState(42)
    audio = 0.01 * rng.randn(duration_s * SR).astype(np.float32)
    audio[loud_start_s * SR : loud_end_s * SR] = (
        0.7 * rng.randn((loud_end_s - loud_start_s) * SR).astype(np.float32)
    )
    return audio


def gemini_json(intervals: List[dict]) -> dict:
    payload = json.dumps({"hook_candidates": intervals})
    return {"candidates": [{"content": {"parts": [{"text": payload}]}}]}


def make_cut(ms_start: int, ms_end: int, score: float, cid: str = "") -> ShortCandidateCut:
    return ShortCandidateCut(
        id=cid,
        start_ms=ms_start,
        end_ms=ms_end,
        duration_ms=ms_end - ms_start,
        hook_text="hook",
        summary="sum",
        reason="r",
        semantic_score=score,
        energy_score=0.0,
        virality_score=score,
    )


def mock_semantic_success(intervals: List[dict]):
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=gemini_json(intervals))))


# --------------------------------------------------------------------------- #
# 1. Modelos Pydantic
# --------------------------------------------------------------------------- #
def test_hook_detector_config_defaults():
    cfg = HookDetectorConfig()
    assert cfg.min_duration_ms == 25000
    assert cfg.max_duration_ms == 58000
    assert cfg.target_cuts == 3
    assert cfg.min_confidence == 0.6
    assert cfg.decision_mode is DetectionMode.HYBRID
    assert cfg.semantic_weight == 0.7
    assert cfg.energy_weight == 0.3
    assert cfg.snap_to_sentence_boundaries is True
    assert cfg.max_overlap_iou == 0.1
    assert cfg.gemini_model == "gemini-2.5-flash"
    assert cfg.gemini_timeout_s == 15.0
    assert cfg.gemini_max_retries == 1


def test_hook_detector_config_rejects_extra_fields():
    with pytest.raises(Exception):
        HookDetectorConfig(unknown_campo=True)


def test_hook_detector_config_enforces_bounds():
    with pytest.raises(Exception):
        HookDetectorConfig(min_duration_ms=5000)  # ge=10000
    with pytest.raises(Exception):
        HookDetectorConfig(max_duration_ms=65000)  # le=60000
    with pytest.raises(Exception):
        HookDetectorConfig(target_cuts=0)
    with pytest.raises(Exception):
        HookDetectorConfig(min_confidence=1.5)
    with pytest.raises(Exception):
        HookDetectorConfig(max_overlap_iou=0.9)
    with pytest.raises(Exception):
        HookDetectorConfig(decision_mode="desconhecido")


def test_detection_mode_values():
    assert DetectionMode.HYBRID.value == "hybrid"
    assert DetectionMode.SEMANTIC_ONLY.value == "semantic"
    assert DetectionMode.HEURISTIC.value == "heuristic"


def test_short_candidate_cut_duration_bounds():
    with pytest.raises(Exception):
        ShortCandidateCut(
            id="cut_01", start_ms=0, end_ms=10000, duration_ms=10000,
            hook_text="h", summary="s", reason="r",
            semantic_score=0.9, virality_score=0.9,
        )  # duration < 25000
    with pytest.raises(Exception):
        ShortCandidateCut(
            id="cut_01", start_ms=0, end_ms=65000, duration_ms=65000,
            hook_text="h", summary="s", reason="r",
            semantic_score=0.9, virality_score=0.9,
        )  # duration > 58000
    with pytest.raises(Exception):
        ShortCandidateCut(
            id="cut_01", start_ms=0, end_ms=30000, duration_ms=30000,
            hook_text="h", summary="s", reason="r",
            semantic_score=2.0, virality_score=0.9,
        )  # semantic_score fora de [0,1]


def test_short_candidate_cut_serialization_roundtrip():
    cut = ShortCandidateCut(
        id="cut_01", start_ms=1000, end_ms=30000, duration_ms=29000,
        hook_text="Gancho", summary="Resumo", reason="Razao",
        semantic_score=0.8, energy_score=0.4, virality_score=0.7,
        energy_peak_ms=15000,
        words=[WordTimestamp(word="oi", start_ms=1000, end_ms=1200, probability=0.9)],
    )
    raw = cut.model_dump_json()
    restored = ShortCandidateCut.model_validate_json(raw)
    assert restored == cut
    assert restored.words[0].word == "oi"


def test_hook_detection_result_model():
    result = HookDetectionResult(cuts=[], total_cuts_found=0, strategy_used="hybrid")
    assert result.cuts == []
    with pytest.raises(Exception):
        HookDetectionResult(cuts=[], total_cuts_found=-1, strategy_used="hybrid")


def test_semantic_hook_interval_defaults():
    interval = SemanticHookInterval(start_ms=0, end_ms=1000)
    assert interval.confidence == 1.0
    assert interval.hook_text == ""


def test_semantic_hook_response_parse():
    response = SemanticHookResponse.model_validate(
        {"hook_candidates": [{"start_ms": 1000, "end_ms": 5000, "confidence": 0.9}]}
    )
    assert response.hook_candidates[0].confidence == 0.9


def test_energy_profile_is_same_model():
    assert EnergyProfile is EnergyAnalysisResult


# --------------------------------------------------------------------------- #
# 2. EnergyAnalyzer
# --------------------------------------------------------------------------- #
def test_energy_peak_identification_on_synthetic_audio():
    profile = EnergyAnalyzer().analyze_audio(make_audio(), SR)
    assert profile.sample_rate == SR
    assert profile.duration_ms == 90000
    assert len(profile.energies) > 0
    assert all(0.0 <= e <= 1.0 for e in profile.energies)
    assert profile.peak_energy == pytest.approx(1.0)
    assert 30000 <= profile.peak_ms <= 40000
    loud_peak = max(profile.energies[int(30000 / 250):int(40000 / 250)])
    quiet_peak = max(profile.energies[int(0):int(10000 / 250)])
    assert loud_peak > quiet_peak


def test_energy_analysis_from_wav_file(tmp_path):
    wav = tmp_path / "audio.wav"
    sf.write(wav, make_audio(), SR)
    profile = EnergyAnalyzer().analyze_audio(wav, SR)
    assert profile.duration_ms == 90000
    assert 30000 <= profile.peak_ms <= 40000
    assert profile.max_window_energy(30000, 40000) > 0.5


def test_energy_window_energy_quiet_vs_loud():
    analyzer = EnergyAnalyzer()
    profile = analyzer.analyze_audio(make_audio(), SR)
    loud = profile.window_energy(30000, 40000)
    quiet = profile.window_energy(1000, 9000)
    assert loud > quiet
    assert analyzer.compute_window_energy(30000, 40000) == loud


def test_energy_out_of_range_interval_returns_zero():
    profile = EnergyAnalyzer().analyze_audio(make_audio(), SR)
    assert profile.window_energy(90000, 95000) == 0.0
    assert profile.max_window_energy(90000, 95000) == 0.0
    assert profile.strongest_peak(85000, 95000) is None


def test_compute_window_energy_requires_analysis():
    with pytest.raises(RuntimeError):
        EnergyAnalyzer().compute_window_energy(0, 100)


def test_energy_empty_array_raises():
    with pytest.raises(EnergyAnalysisError):
        EnergyAnalyzer().analyze_audio(np.array([], dtype=np.float32), SR)


def test_energy_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        EnergyAnalyzer().analyze_audio(tmp_path / "nao_existe.wav", SR)


def test_energy_resample_preserves_peak_locally(tmp_path):
    wav = tmp_path / "audio_8k.wav"
    sf.write(wav, make_audio()[::2], 8000)  # 90s de audio amostrados a 8kHz
    profile = EnergyAnalyzer().analyze_audio(wav, 16000)  # reamostra para 16k
    assert profile.duration_ms == 90000
    assert 30000 <= profile.peak_ms <= 40000


def test_energy_stereo_uses_mean_of_channels():
    mono = make_audio()
    stereo = np.stack([np.zeros_like(mono), mono], axis=1)
    profile = EnergyAnalyzer().analyze_audio(stereo, SR)
    assert profile.duration_ms == 90000
    assert 30000 <= profile.peak_ms <= 40000


def test_energy_invalid_window_ms():
    with pytest.raises(ValueError):
        EnergyAnalyzer(window_ms=0)


# --------------------------------------------------------------------------- #
# 3. SemanticHookAnalyzer
# --------------------------------------------------------------------------- #
def test_semantic_analyzer_success_from_structured_output():
    client = mock_semantic_success([{"start_ms": 30000, "end_ms": 38000, "reason": "gancho"}])
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="chave-teste", gemini_max_retries=0),
        http_client=client,
    )
    intervals = analyzer.analyze(make_transcription(), 90000)
    assert len(intervals) == 1
    assert intervals[0].start_ms == 30000
    assert intervals[0].end_ms == 38000
    assert intervals[0].confidence == 1.0


def test_semantic_analyzer_posts_key_url_and_json_schema():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=gemini_json([]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="key-secret", gemini_max_retries=0),
        http_client=client,
    )
    analyzer.analyze(make_transcription(total_ms=30000), 30000)

    assert "models/gemini-2.5-flash:generateContent" in captured["url"]
    assert "key=key-secret" in captured["url"]
    body = json.loads(captured["body"])
    assert "hook_candidates" in body["generationConfig"]["responseSchema"]["properties"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert isinstance(body["contents"][0]["parts"][0]["text"], str)


def test_semantic_analyzer_env_key_and_retry_after_network_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-ambiente")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("caiu")
        return httpx.Response(200, json=gemini_json([{"start_ms": 1000, "end_ms": 9000, "reason": "x"}]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key=None, gemini_max_retries=1),
        http_client=client,
    )
    intervals = analyzer.analyze(make_transcription(), 90000)
    assert calls["n"] == 2
    assert len(intervals) == 1


def test_semantic_analyzer_returns_empty_without_words():
    transcript = TranscriptionResult(text="", language="pt", duration_ms=90000, words=[])
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key=None, gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    assert analyzer.analyze(transcript, 90000) == []


def test_semantic_analyzer_raises_without_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key=None, gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    )
    with pytest.raises(SemanticHookError, match="GEMINI_API_KEY"):
        analyzer.analyze(make_transcription(), 90000)


def test_semantic_analyzer_raises_on_rate_limit():
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429))),
    )
    with pytest.raises(SemanticHookError, match="rate limit"):
        analyzer.analyze(make_transcription(), 90000)


def test_semantic_analyzer_raises_on_server_error():
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    with pytest.raises(SemanticHookError, match="indisponivel"):
        analyzer.analyze(make_transcription(), 90000)


def test_semantic_analyzer_raises_on_unexpected_status():
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400))),
    )
    with pytest.raises(SemanticHookError, match="400"):
        analyzer.analyze(make_transcription(), 90000)


def test_semantic_analyzer_raises_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sem conexao")

    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SemanticHookError, match="rede/timeout"):
        analyzer.analyze(make_transcription(), 90000)


def test_semantic_analyzer_raises_on_truncated_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"hook_candidates": ['}]}}]},
        )

    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SemanticHookError, match="JSON"):
        analyzer.analyze(make_transcription(), 90000)


def test_semantic_analyzer_raises_on_schema_violation():
    bad = '{"hook_candidates": [{"start_ms": -5}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": bad}]}}]})

    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SemanticHookError, match="schema"):
        analyzer.analyze(make_transcription(), 90000)


def test_semantic_analyzer_raises_on_empty_candidates():
    analyzer = SemanticHookAnalyzer(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"candidates": []}))
        ),
    )
    with pytest.raises(SemanticHookError, match="candidates"):
        analyzer.analyze(make_transcription(), 90000)


# --------------------------------------------------------------------------- #
# 4. HeuristicHookDetector
# --------------------------------------------------------------------------- #
def test_heuristic_detects_rhetorical_question():
    words = [
        WordTimestamp(word="voce", start_ms=0, end_ms=500, probability=1.0),
        WordTimestamp(word="quer", start_ms=500, end_ms=1000, probability=1.0),
        WordTimestamp(word="saber?", start_ms=1000, end_ms=1500, probability=1.0),
    ]
    transcript = TranscriptionResult(text="voce quer saber?", language="pt", duration_ms=1500, words=words)
    detector = HeuristicHookDetector()
    hits = detector.detect(transcript)
    assert len(hits) >= 1
    assert "interrogacao" in hits[0].reason


def test_heuristic_detects_adversative_break():
    transcript = make_transcription(total_ms=50000, step_ms=500, per_seg=1, text=["mas", "isto", "muda", "tudo"])
    hits = HeuristicHookDetector().detect(transcript)
    assert any("adversativa" in h.reason for h in hits)


def test_heuristic_detects_impact_word():
    transcript = make_transcription(total_ms=50000, step_ms=500, per_seg=1, text=["o", "segredo", "e", "simples"])
    hits = HeuristicHookDetector().detect(transcript)
    assert any("alto impacto" in h.reason for h in hits)


def test_heuristic_detects_provocative_phrase():
    transcript = make_transcription(
        total_ms=50000, step_ms=500, per_seg=1,
        text=["ninguem", "te", "conta", "isso"],
    )
    hits = HeuristicHookDetector().detect(transcript)
    assert any("provocativa" in h.reason for h in hits)


def test_heuristic_empty_transcription_returns_empty():
    assert HeuristicHookDetector().detect(None) == []
    empty = TranscriptionResult(text="", language="pt", duration_ms=1000, words=[])
    assert HeuristicHookDetector().detect(empty) == []


def test_heuristic_no_hooks_returns_empty():
    transcript = make_transcription(total_ms=50000, step_ms=500, per_seg=1, text=["tudo", "normal", "aqui", "ok"])
    assert HeuristicHookDetector().detect(transcript) == []


def test_heuristic_confidence_with_energy_boost():
    neutral = ["neutro"] * 180
    neutral[70] = "mas"  # oracao 32s-36s, dentro da regiao de alto volume
    profile = EnergyAnalyzer().analyze_audio(make_audio(), SR)
    transcript = make_transcription(total_ms=90000, step_ms=500, per_seg=8, text=neutral)
    with_energy = HeuristicHookDetector(energy_profile=profile).detect(transcript)
    without = HeuristicHookDetector().detect(transcript)
    assert with_energy and without
    assert with_energy[0].confidence > without[0].confidence


def test_heuristic_dedup_per_segment():
    transcript = make_transcription(total_ms=90000, step_ms=500, per_seg=8, text=hook_texts(190))
    hits = HeuristicHookDetector().detect(transcript)
    starts = [h.start_ms for h in hits]
    assert len(starts) == len(set(starts))


def test_heuristic_candidates_cover_full_sentences():
    transcript = make_transcription(total_ms=90000, step_ms=500, per_seg=8, text=hook_texts(190))
    segments = transcript.segments
    for hit in HeuristicHookDetector().detect(transcript):
        seg_starts = [s.start_ms for s in segments]
        seg_ends = [s.end_ms for s in segments]
        assert hit.start_ms in seg_starts
        assert hit.end_ms in seg_ends


# --------------------------------------------------------------------------- #
# 5. Duracao, snapping e NMS
# --------------------------------------------------------------------------- #
def test_temporal_iou_cases():
    assert temporal_iou(0, 10000, 20000, 30000) == 0.0
    assert temporal_iou(0, 10000, 0, 10000) == 1.0
    assert temporal_iou(0, 10000, 5000, 15000) == pytest.approx(5000 / 15000)


def test_suppress_overlaps_keeps_top_ranked_disjoint():
    candidates = [
        make_cut(0, 30000, 0.9),
        make_cut(5000, 35000, 0.8),  # IoU > 0.1 com o primeiro
        make_cut(40000, 70000, 0.7),
    ]
    kept = suppress_overlaps(candidates, max_overlap_iou=0.1, max_cuts=3)
    assert [c.id for c in kept] == ["cut_01", "cut_02"]
    assert kept[0].virality_score == 0.9
    assert all(temporal_iou(kept[0].start_ms, kept[0].end_ms, k.start_ms, k.end_ms) <= 0.1 for k in kept[1:])
    assert kept[1].id == "cut_02"


def test_suppress_overlaps_respects_max_cuts():
    candidates = [
        make_cut(0, 30000, 0.9),
        make_cut(35000, 65000, 0.8),
        make_cut(70000, 98000, 0.7),
    ]
    kept = suppress_overlaps(candidates, max_overlap_iou=0.1, max_cuts=2)
    assert len(kept) <= 2


def test_fit_duration_expands_short_candidate():
    detector = HookDetector()
    b_starts = list(range(0, 100000, 1000))
    b_ends = list(range(1000, 100001, 1000))
    result = detector._fit_duration(0, 1000, b_starts, b_ends, 100000)
    assert result == (0, 25000)


def test_fit_duration_trims_long_candidate():
    detector = HookDetector()
    b_starts = list(range(0, 100000, 1000))
    b_ends = list(range(1000, 100001, 1000))
    result = detector._fit_duration(0, 90000, b_starts, b_ends, 100000)
    assert result is not None
    start, end = result
    assert 25000 <= end - start <= 58000


def test_fit_duration_discards_when_material_too_short():
    detector = HookDetector()
    b_starts = list(range(0, 20000, 1000))
    b_ends = list(range(1000, 20001, 1000))
    assert detector._fit_duration(0, 1000, b_starts, b_ends, 20000) is None


def test_snap_never_slices_words():
    transcript = make_transcription(total_ms=90000, step_ms=500, per_seg=8)
    words = sorted(transcript.words, key=lambda w: w.start_ms)
    segments = transcript.segments
    detector = HookDetector()
    b_starts, b_ends = detector._boundaries(words, segments)
    interval = SemanticHookInterval(start_ms=5100, end_ms=60000, confidence=0.95)
    cut = detector._build_cut(
        interval, words, segments, b_starts, b_ends, 90000, None
    )
    assert cut is not None
    assert cut.start_ms == 4000  # inicio da oracao que contem o inicio
    assert 25000 <= cut.duration_ms <= 58000
    assert all(w.start_ms >= cut.start_ms and w.end_ms <= cut.end_ms for w in cut.words)
    assert cut.words[0].start_ms == cut.start_ms
    assert cut.words[-1].end_ms == cut.end_ms
    assert cut.energy_score == 0.5  # sem audio -> neutro


def test_build_cut_neutral_energy_without_audio_and_min_confidence_filter():
    transcript = make_transcription(total_ms=90000, step_ms=500, per_seg=8)
    words = sorted(transcript.words, key=lambda w: w.start_ms)
    detector = HookDetector(config=HookDetectorConfig(min_confidence=0.99))
    b_starts, b_ends = detector._boundaries(words, transcript.segments)
    interval = SemanticHookInterval(start_ms=5100, end_ms=60000, confidence=0.5)  # virality < 0.99
    cut = detector._build_cut(
        interval, words, transcript.segments, b_starts, b_ends, 90000, None
    )
    assert cut is None


# --------------------------------------------------------------------------- #
# 6. HookDetector - casos de borda
# --------------------------------------------------------------------------- #
def test_detect_short_video_returns_empty_without_network():
    boom = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(999)))
    detector = HookDetector(http_client=boom)
    result = detector.detect_cuts(make_transcription(total_ms=20000), audio_source=make_audio())
    assert result.cuts == []
    assert result.total_cuts_found == 0


def test_detect_empty_transcription_returns_empty():
    empty = TranscriptionResult(text="", language="pt", duration_ms=90000, words=[])
    detector = HookDetector()
    result = detector.detect_cuts(empty)
    assert result.cuts == []
    assert result.total_cuts_found == 0
    assert result.strategy_used == "heuristic_fallback"


# --------------------------------------------------------------------------- #
# 7. HookDetector - estrategias e integracao
# --------------------------------------------------------------------------- #
def test_detect_hybrid_success_with_mock_semantic():
    payload = [
        {
            "start_ms": 30000,
            "end_ms": 38000,
            "hook_text": "Mas o segredo.",
            "summary": "s",
            "reason": "quebra",
            "confidence": 0.9,
        }
    ]
    client = mock_semantic_success(payload)
    detector = HookDetector(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=client,
    )
    result = detector.detect_cuts(make_transcription(), audio_source=make_audio())
    assert result.strategy_used == "hybrid"
    assert result.audio_energy_analyzed is True
    assert len(result.cuts) >= 1
    for cut in result.cuts:
        assert 25000 <= cut.duration_ms <= 58000
        assert cut.energy_peak_ms is not None


def test_detect_semantic_only_strategy():
    client = mock_semantic_success([{"start_ms": 30000, "end_ms": 38000, "reason": "gancho", "confidence": 0.9}])
    detector = HookDetector(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0, decision_mode=DetectionMode.SEMANTIC_ONLY),
        http_client=client,
    )
    result = detector.detect_cuts(make_transcription(), audio_source=make_audio())
    assert result.strategy_used == "semantic"
    assert result.cuts


def test_detect_falls_back_to_heuristic_without_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    detector = HookDetector()
    result = detector.detect_cuts(
        make_transcription(text=hook_texts(190)),
        audio_source=make_audio(),
    )
    assert result.strategy_used == "heuristic_fallback"
    assert result.audio_energy_analyzed is True
    assert 1 <= result.total_cuts_found <= 3
    for cut in result.cuts:
        assert 25000 <= cut.duration_ms <= 58000


def test_detect_heuristic_mode_is_offline():
    booster = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(999)))
    detector = HookDetector(
        config=HookDetectorConfig(decision_mode=DetectionMode.HEURISTIC),
        http_client=booster,
    )
    result = detector.detect_cuts(
        make_transcription(text=hook_texts(190)),
        audio_source=make_audio(),
    )
    assert result.strategy_used == "heuristic_fallback"
    assert result.cuts


def test_detect_no_audio_neutral_energy():
    client = mock_semantic_success([{"start_ms": 30000, "end_ms": 38000, "reason": "gancho", "confidence": 0.9}])
    detector = HookDetector(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=client,
    )
    result = detector.detect_cuts(make_transcription())
    assert result.audio_energy_analyzed is False
    assert result.cuts
    assert all(cut.energy_score == 0.5 for cut in result.cuts)
    assert all(cut.energy_peak_ms is None for cut in result.cuts)


def test_detect_corrupt_audio_falls_back_to_neutral():
    tmp = Path(__file__).parent / "nao_e_audio.wav"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("isto nao e audio valido")
        detector = HookDetector(
            config=HookDetectorConfig(decision_mode=DetectionMode.HEURISTIC)
        )
        result = detector.detect_cuts(
            make_transcription(text=hook_texts(190)),
            audio_source=tmp,
        )
        assert result.audio_energy_analyzed is False
        assert result.cuts
        assert all(cut.energy_score == 0.5 for cut in result.cuts)
    finally:
        if tmp.exists():
            tmp.unlink()


def test_detect_cuts_between_1_and_3_and_non_overlapping():
    client = mock_semantic_success(
        [
            {"start_ms": 5000, "end_ms": 15000, "reason": "a", "confidence": 0.9},
            {"start_ms": 35000, "end_ms": 45000, "reason": "b", "confidence": 0.8},
            {"start_ms": 65000, "end_ms": 75000, "reason": "c", "confidence": 0.7},
        ]
    )
    detector = HookDetector(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=client,
    )
    result = detector.detect_cuts(make_transcription(), audio_source=make_audio())
    assert 1 <= result.total_cuts_found <= 3
    for i, cut in enumerate(result.cuts):
        assert cut.id == f"cut_{i + 1:02d}"
        assert 25000 <= cut.duration_ms <= 58000
        for other in result.cuts[i + 1:]:
            assert temporal_iou(cut.start_ms, cut.end_ms, other.start_ms, other.end_ms) <= 0.1


def test_detect_preserves_original_word_timestamps():
    client = mock_semantic_success([{"start_ms": 30000, "end_ms": 38000, "reason": "gancho", "confidence": 0.9}])
    detector = HookDetector(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=client,
    )
    original = make_transcription()
    result = detector.detect_cuts(original)
    assert result.cuts
    for cut in result.cuts:
        words_in_video = {id(w) for w in original.words}
        assert all(id(w) in words_in_video for w in cut.words)
        if cut.words:
            first_word_id = id(cut.words[0])
            original_word = next(w for w in original.words if id(w) == first_word_id)
            assert cut.words[0].start_ms == original_word.start_ms


def test_detect_explicit_total_duration_overrides_transcription():
    client = mock_semantic_success([{"start_ms": 30000, "end_ms": 38000, "reason": "gancho", "confidence": 0.9}])
    detector = HookDetector(
        config=HookDetectorConfig(gemini_api_key="k", gemini_max_retries=0),
        http_client=client,
    )
    base = make_transcription(total_ms=60000)
    flawed = TranscriptionResult(
        text=base.text,
        language="pt",
        duration_ms=1000,  # duracao incorreta na transcricao
        segments=base.segments,
        words=base.words,
    )
    result = detector.detect_cuts(flawed, total_duration_ms=90000)
    assert result.cuts
    assert result.total_cuts_found >= 1
