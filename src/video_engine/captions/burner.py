"""Queima de legendas animadas ASS (hardsub) em cortes de video (Issue #9).

Implementa ``CaptionBurner``, que renderiza as legendas estilizadas de um
corte/clip via FFmpeg (``-vf ass=...``) com:

- Tratamento rigoroso de caminhos no Windows (forward slashes e escape de
  dois-pontos ``C\\:/caminho/arquivo.ass``).
- Reindexacao temporal do intervalo do corte (``cut_start_ms``/``cut_end_ms``).
- Preservacao direta do audio (``-c:a copy``).
- Resolucao adaptativa lida por ``MediaProbe`` (PlayResX/PlayResY do corte).

O video longo principal permanece intacto (clean feed): a queima e invocada
apenas para artefatos de corte/clip, de forma desacoplada do pipeline longo.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

from video_engine.captions.ass_generator import AssSubtitleGenerator, TranscriptionInput
from video_engine.captions.ass_models import CutCaptionBurnResult, SubtitleStyleConfig
from video_engine.editing.media_probe import MediaProbe


class CaptionBurnError(Exception):
    """Falha na queima de legendas hardsub em um corte.

    Cobrindo, por exemplo, ausencia de ffmpeg/libass ou falha na execucao
    do filtro ``ass``.
    """


def _filter_ass_path(path: Union[str, Path]) -> str:
    """Sanitiza o caminho ASS para o filtro ``-vf ass=...`` (Windows-safe).

    Converte barras invertidas em forward slashes, escapa dois-pontos de
    drive (``C\\:``) e emoldura o caminho com aspas simples, mantendo
    espacos e unicodes como um unico argumento no grafo de filtros.
    """
    raw = str(path).replace("\\", "/").replace(":", "\\:")
    if "'" in raw:
        raw = raw.replace("'", "\u201d")
    return f"ass='{raw}'"


def _count_ass_cues(ass_text: str) -> int:
    """Conta eventos ``Dialogue`` de um arquivo ASS."""
    return sum(1 for line in ass_text.splitlines() if line.startswith("Dialogue:"))


def _count_ass_words(ass_text: str) -> int:
    """Conta palavras visiveis (sem overrides) nos eventos de um ASS."""
    total = 0
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        fields = line.split(",", 9)
        if len(fields) < 10:
            continue
        clean = fields[9].replace("{", " ").replace("}", " ")
        total += len(clean.split())
    return total


class CaptionBurner:
    """Queima as legendas animadas geradas em um arquivo de corte/clip."""

    def __init__(
        self,
        generator: Optional[AssSubtitleGenerator] = None,
        probe: Optional[MediaProbe] = None,
        ffmpeg_path: Optional[str] = None,
    ) -> None:
        self.generator = generator or AssSubtitleGenerator()
        self.probe = probe or MediaProbe()
        self.ffmpeg_path = ffmpeg_path

    def _resolve_ffmpeg(self) -> str:
        binary = self.ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise CaptionBurnError(
                "ffmpeg nao encontrado no PATH. "
                "Instale o ffmpeg com suporte a libass (filtro 'ass')."
            )
        return binary

    def burn_cut_subtitles(
        self,
        cut_video_path: Union[str, Path],
        output_path: Union[str, Path],
        transcription: Union[TranscriptionInput, str, Path],
        cut_start_ms: int = 0,
        cut_end_ms: Optional[int] = None,
        style_config: Optional[SubtitleStyleConfig] = None,
    ) -> CutCaptionBurnResult:
        """Queima as legendas estilizadas no video do corte.

        Se ``transcription`` for um caminho ``str``/``Path``, ele e tratado
        como arquivo ASS pre-existente e a queima usa o arquivo diretamente.
        Caso contrario, a transcricao completa do video longo pode ser
        fornecida junto do intervalo do corte (``cut_start_ms``/``cut_end_ms``),
        fazendo a reindexacao automatica dos timestamps para ``t = 0``.

        Raises:
            FileNotFoundError: video do corte ou arquivo ASS inexistente.
            CaptionBurnError: corte sem stream de video, ffmpeg/libass ausente
                ou falha na execucao do ffmpeg.
        """
        src = Path(cut_video_path)
        if not src.is_file():
            raise FileNotFoundError(f"Arquivo de corte nao encontrado: {src}")

        ffmpeg = self._resolve_ffmpeg()
        info = self.probe.probe(src)
        if not info.has_video:
            raise CaptionBurnError(f"Arquivo de corte sem stream de video: {src}")

        width = info.video_width or self.generator.style.play_res_x
        height = info.video_height or self.generator.style.play_res_y

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(transcription, (str, Path)):
            ass_path = Path(transcription)
            if not ass_path.is_file():
                raise FileNotFoundError(f"Arquivo de legendas ASS nao encontrado: {ass_path}")
            ass_text = ass_path.read_text(encoding="utf-8")
            cues_count = _count_ass_cues(ass_text)
            total_words_count = _count_ass_words(ass_text)
        else:
            generator = (
                AssSubtitleGenerator(style_config=style_config)
                if style_config is not None
                else self.generator
            )
            ass_path = out.with_suffix(".ass")
            cues = generator.build_cues(
                transcription,
                start_time_ms=cut_start_ms,
                end_time_ms=cut_end_ms,
            )
            generator.write_ass_file(
                transcription,
                ass_path,
                start_time_ms=cut_start_ms,
                end_time_ms=cut_end_ms,
                video_width=width,
                video_height=height,
            )
            cues_count = len(cues)
            total_words_count = sum(len(cue.words) for cue in cues)

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vf",
            _filter_ass_path(ass_path),
            "-map",
            "0:v:0",
        ]
        if info.has_audio:
            cmd += ["-map", "0:a:0", "-c:a", "copy"]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            raise CaptionBurnError(
                f"Nao foi possivel executar o ffmpeg ({ffmpeg}): {exc}"
            ) from exc

        if proc.returncode != 0:
            stderr = proc.stderr or ""
            tail = stderr.strip().splitlines()[-8:]
            detail = "\n".join(tail) if tail else "sem stderr"
            if "ass" in stderr.lower():
                raise CaptionBurnError(
                    "ffmpeg nao possui suporte a libass (filtro 'ass'). "
                    f"Reinstale o ffmpeg com libass.\n{detail}"
                )
            raise CaptionBurnError(f"ffmpeg falhou ao queimar legendas: {detail}")

        return CutCaptionBurnResult(
            output_path=str(out),
            ass_path=str(ass_path),
            cues_count=cues_count,
            total_words_count=total_words_count,
            duration_sec=info.duration_ms / 1000.0,
        )


__all__ = ["CaptionBurnError", "CaptionBurner"]
