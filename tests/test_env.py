"""Testes do utilitário load_env (carregamento de arquivo .env)."""

from __future__ import annotations

import os
from pathlib import Path

from video_engine.env import load_env
from video_engine.video.models import GeminiZoomConfig
from video_engine.video.semantic_analyzer import SemanticZoomAnalyzer


def test_load_env_basic(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEST_VAR_ABC", raising=False)
    monkeypatch.delenv("TEST_VAR_XYZ", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comentario\n"
        "TEST_VAR_ABC=valor_123\n"
        "\n"
        "TEST_VAR_XYZ=\"valor com aspas\"\n"
        "TEST_VAR_SINGLE='aspas simples'\n",
        encoding="utf-8",
    )

    assert load_env(env_file) is True
    assert os.environ.get("TEST_VAR_ABC") == "valor_123"
    assert os.environ.get("TEST_VAR_XYZ") == "valor com aspas"
    assert os.environ.get("TEST_VAR_SINGLE") == "aspas simples"


def test_load_env_does_not_overwrite_existing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXISTING_KEY", "original_value")

    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=new_value\n", encoding="utf-8")

    assert load_env(env_file) is True
    assert os.environ.get("EXISTING_KEY") == "original_value"


def test_load_env_nonexistent_returns_false():
    assert load_env(Path("nonexistent_path_to_env_file_123.env")) is False


def test_semantic_analyzer_resolves_key_from_env_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=test_api_key_from_file\n", encoding="utf-8")

    # Mudar cwd para tmp_path para que load_env sem argumentos encontre o .env
    monkeypatch.chdir(tmp_path)

    analyzer = SemanticZoomAnalyzer(config=GeminiZoomConfig())
    resolved = analyzer._resolve_api_key()
    assert resolved == "test_api_key_from_file"
