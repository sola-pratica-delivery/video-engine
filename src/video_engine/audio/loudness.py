"""Motor de normalizacao de loudness EBU R128 (2 passes) via FFmpeg ``loudnorm``.

Arquitetura (Spec Issue #4 / SDD):
    1. Passe 1 (Medicao): ``loudnorm=...:print_format=json`` extrai as metricas
       reais do audio de entrada (``input_i``, ``input_tp``, ``input_lra``,
       ``input_thresh``, ``target_offset``) em JSON no stderr do ffmpeg.
    2. Passe 2 (Normalizacao): os valores medidos sao realimentados com
       ``linear=true`` (ganho transparente sem pumping) e um ``alimiter`` com
       teto em -1.0 dBTP garante o dominio do true peak em conteudo de picos
       esparsos (o ``loudnorm`` cai em modo dinamico e o ``alimiter`` atua como
       limitador de teto suave).
    3. QA (Validacao): re-medicao da saida e conformidade contra -14.0 LUFS
       (+/- 0.5) e TP <= -1.0 dBTP (folga de 0.05 dB para codecs lossy).

Videos (MP4/MKV) tem o stream de video copiado com ``-c:v copy`` (0 re-encode).
Arquivos exclusivamente de audio (WAV/MP3) recebem codec adequado ao conteneir.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from video_engine.audio.loudness_models import LoudnessConfig, LoudnessResult, LoudnormStats
from video_engine.editing.media_probe import MediaProbe

_DBTP_SLACK = 0.05  # Folga numerica do true peak para codecs lossy (CA2, <= -0.95).
# Folga adicional aplicada ao teto de processamento: codecs lossy (AAC/Opus)
# acrescentam ~0.3 dB de overshoot ao conteiner final, entao o processamento usa
# um teto mais baixo para que a saida lossy ainda atenda a CA2 (TP <= -0.95).
_LOSSY_TP_HEADROOM_DB = 0.3


def _extract_json_block(text: str, key: str = "input_i") -> Optional[str]:
    """Extrai o bloco JSON balanceado que contem a ultima ocorrencia de ``key``.

    O ``loudnorm`` imprime o JSON no stderr cercado por logs do ffmpeg; o bloco
    e localizado de trás para frente a partir de ``key`` e delimitado por chaves
    balanceadas respeitando strings e escapes.
    """
    idx = text.rfind(f'"{key}"')
    if idx == -1:
        return None
    start = text.rfind("{", 0, idx)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def loudness_compliant(measured: LoudnormStats, config: LoudnessConfig) -> bool:
    """Verifica conformidade EBU R128: -14 LUFS (+/- tolerance) e TP <= alvo."""
    lufs_ok = abs(measured.input_i - config.target_i) <= config.tolerance_lufs
    dbtp_ok = measured.input_tp <= config.target_tp + _DBTP_SLACK
    return lufs_ok and dbtp_ok


class LoudnessNormalizer:
    """Motor de normalizacao de sonoridade de 2 passes via FFmpeg loudnorm."""

    def __init__(
        self,
        config: Optional[LoudnessConfig] = None,
        ffmpeg_path: Optional[str] = None,
    ) -> None:
        self.config = config or LoudnessConfig()
        self.ffmpeg_path = ffmpeg_path

    def _resolve_ffmpeg(self) -> str:
        binary = self.ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise RuntimeError("ffmpeg nao encontrado no PATH")
        return binary

    def _run_ffmpeg(self, cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True)

    @staticmethod
    def _parse_stats(payload: dict) -> LoudnormStats:
        """Converte o JSON do loudnorm em :class:`LoudnormStats`.

        Raises:
            ValueError: se algum valor for nao finito (``-inf``/``inf``), caso
                de audios mais curtos que a janela EBU R128 (~400ms).
            RuntimeError: se faltarem chaves essenciais da medicao.
        """

        def _finite(value: object, name: str) -> float:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Valor invalido para {name} no JSON do loudnorm") from exc
            if not math.isfinite(number):
                raise ValueError("Audio muito curto para medicao EBU R128 (menos de ~400ms)")
            return number

        missing = [k for k in ("input_i", "input_tp", "input_lra", "input_thresh") if k not in payload]
        if missing:
            raise RuntimeError(f"Dados de medicao incompletos: faltam {', '.join(missing)}")
        return LoudnormStats(
            input_i=_finite(payload["input_i"], "input_i"),
            input_tp=_finite(payload["input_tp"], "input_tp"),
            input_lra=_finite(payload["input_lra"], "input_lra"),
            input_thresh=_finite(payload["input_thresh"], "input_thresh"),
            target_offset=_finite(payload.get("target_offset", 0.0), "target_offset"),
        )

    def measure(self, input_path: Union[str, Path]) -> LoudnormStats:
        """Executa o 1o passe de analise no arquivo e retorna as metricas EBU R128.

        Raises:
            FileNotFoundError: se ``input_path`` nao existir.
            RuntimeError: se ffmpeg nao for encontrado, falhar ou o JSON nao
                for produzido.
            ValueError: se o audio for curto demais para a janela EBU R128.
        """
        src = Path(input_path)
        if not src.is_file():
            raise FileNotFoundError(f"Arquivo de midia nao encontrado: {src}")
        binary = self._resolve_ffmpeg()

        cfg = self.config
        af = (
            f"loudnorm=I={cfg.target_i}:TP={cfg.target_tp}:LRA={cfg.target_lra}"
            ":print_format=json"
        )
        cmd = [binary, "-hide_banner", "-nostats", "-i", str(src), "-af", af, "-f", "null", os.devnull]
        proc = self._run_ffmpeg(cmd)
        if proc.returncode != 0:
            detail = _stderr_tail(proc)
            raise RuntimeError(f"ffmpeg falhou ao medir loudness de {src}: {detail}")

        block = _extract_json_block(proc.stderr or "")
        if block is None:
            detail = _stderr_tail(proc)
            raise RuntimeError(f"Dados EBU R128 nao encontrados na saida do ffmpeg: {detail}")
        try:
            payload = json.loads(block)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSON do loudnorm malformado para {src}: {exc}") from exc
        return self._parse_stats(payload)

    def _apply_filter(self, measured: LoudnormStats) -> str:
        cfg = self.config
        process_tp = cfg.target_tp - _LOSSY_TP_HEADROOM_DB
        limit = 10 ** (process_tp / 20.0)
        loudnorm = (
            f"loudnorm=I={cfg.target_i}:TP={process_tp}:LRA={cfg.target_lra}"
            f":measured_I={measured.input_i:.2f}:measured_TP={measured.input_tp:.2f}"
            f":measured_LRA={measured.input_lra:.2f}:measured_thresh={measured.input_thresh:.2f}"
            f":offset={measured.target_offset:.2f}:linear=true"
        )
        alimiter = (
            "alimiter=level_in=1.0:level_out=1.0"
            f":limit={limit}:attack=5:release=50:level=disabled:asc=true"
        )
        return f"{loudnorm},{alimiter}"

    def normalize_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> LoudnessResult:
        """Executa a normalizacao completa de 2 passes + QA.

        Raises:
            ValueError: se a entrada estiver vazia ou nao tiver stream de audio.
            FileNotFoundError: se ``input_path`` nao existir.
            RuntimeError: se ffmpeg nao for encontrado ou falhar.
        """
        src = Path(input_path)
        out = Path(output_path)
        if not src.is_file():
            raise FileNotFoundError(f"Arquivo de midia nao encontrado: {src}")
        if src.stat().st_size == 0:
            raise ValueError("Entrada sem stream de áudio válida")

        info = MediaProbe().probe(src)
        if not info.has_audio:
            raise ValueError("Entrada sem stream de áudio válida")
        has_video = info.has_video

        measured = self.measure(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        binary = self._resolve_ffmpeg()

        af = self._apply_filter(measured)
        cmd: List[str] = [binary, "-hide_banner", "-nostats", "-y", "-i", str(src), "-af", af]
        if has_video:
            cmd += ["-c:v", "copy"]
        if out.suffix.lower() == ".wav":
            cmd += ["-c:a", "pcm_s16le"]
        else:
            cmd += ["-c:a", self.config.audio_codec, "-b:a", self.config.audio_bitrate]
        cmd.append(str(out))

        proc = self._run_ffmpeg(cmd)
        if proc.returncode != 0:
            detail = _stderr_tail(proc)
            raise RuntimeError(f"ffmpeg falhou ao normalizar {src} -> {out}: {detail}")

        measured_output = self.measure(out)
        return LoudnessResult(
            output_path=str(out),
            measured_input=measured,
            measured_output=measured_output,
            target_i=self.config.target_i,
            target_tp=self.config.target_tp,
            is_compliant=loudness_compliant(measured_output, self.config),
            has_video=has_video,
        )


def _stderr_tail(proc: subprocess.CompletedProcess) -> str:
    lines = (proc.stderr or "").strip().splitlines()
    return "\n".join(lines[-8:]) if lines else "sem stderr"


__all__ = ["LoudnessNormalizer", "loudness_compliant"]
