"""Utilitario leve para carregamento de variaveis a partir de arquivo .env.

Nao requer dependencias externas adicionais e preserva variaveis ja existentes
no os.environ.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def load_env(file_path: Optional[Union[Path, str]] = None) -> bool:
    """Carrega variaveis de um arquivo .env para os.environ sem sobrescrever valores existentes.

    Se ``file_path`` nao for informado, busca por ``.env`` no diretorio de trabalho
    atual (cwd) e nos diretorios pais da raiz do projeto.
    """
    path: Optional[Path] = None
    if file_path:
        p = Path(file_path)
        if p.is_file():
            path = p
    else:
        cwd_candidate = Path.cwd() / ".env"
        if cwd_candidate.is_file():
            path = cwd_candidate
        else:
            for parent in Path(__file__).resolve().parents:
                candidate = parent / ".env"
                if candidate.is_file():
                    path = candidate
                    break

    if not path or not path.is_file():
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (
                        val.startswith("'") and val.endswith("'")
                    ):
                        val = val[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = val
        return True
    except Exception:
        return False


__all__ = ["load_env"]
