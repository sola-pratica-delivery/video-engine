"""Camada de midia FFmpeg: emendas cirurgicas com micro-crossfade.

O grafo ``-filter_complex`` espelha exatamente a camada de sinal pura
(``video_engine.audio.splicer.splice_audio_array``): apenas os pontos de emenda
recebem fades (fade-out no final do segmento ``i``, fade-in no inicio do
segmento ``i+1``) e o stream/dados de audio nao sao sobrepostos, preservando a
duracao total (soma das duracoes) e a sincronizacao A/V frame-a-frame.

Curvas FFmpeg equivalentes via ``afade``:
    LINEAR       -> ``tri`` (rampa linear)
    EQUAL_POWER  -> ``qsin`` (fade-in, ``sin(pi/2*x)`) e ``iqsin`` (fade-out,
                    inversa exata do qsin, monotona descrecente).

O script e gravado em arquivo e passado via ``-filter_complex_script`` pois o
construtor de linha de comando do Windows limita comandos a ~8191 caracteres,
e a emenda de "dezenas ou centenas de cortes" estoura esse limite.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from video_engine.audio.models import TimeInterval
from video_engine.audio.vad_utils import merge_intervals
from video_engine.editing.media_probe import MediaProbe
from video_engine.editing.models import FadeCurve, SplicerConfig, SpliceResult

_CURVE_TO_FFMPEG = {
    FadeCurve.LINEAR: {"in": "tri", "out": "tri"},
    FadeCurve.EQUAL_POWER: {"in": "qsin", "out": "iqsin"},
}


def _sec(ms: float) -> str:
    return f"{ms / 1000.0:.6f}"


class MediaSplicer:
    """Recorta e emenda um arquivo de midia ao longo de intervalos (ms)."""

    def __init__(
        self,
        config: Optional[SplicerConfig] = None,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ) -> None:
        self.config = config or SplicerConfig()
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def _resolve_ffmpeg(self) -> str:
        binary = self.ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise RuntimeError("ffmpeg nao encontrado no PATH")
        return binary

    # ------------------------------------------------------------------ #
    # Grafo de filtros
    # ------------------------------------------------------------------ #
    def build_filter_complex_script(
        self,
        segments: Sequence[TimeInterval],
        has_video: bool,
    ) -> str:
        """Gera o conteudo do ``-filter_complex_script`` para ``segments``.

        Fades apenas nas emendas: o primeiro segmento recebe apenas fade-out,
        segmentos intermediarios recebem ambos e o ultimo apenas fade-in
        (mesmo contrato da camada de sinal ``splice_audio_array``).
        """
        if not segments:
            raise ValueError("Segments list cannot be empty")
        merged = [s for s in merge_intervals(segments) if s.end_ms > s.start_ms]
        if not merged:
            raise ValueError("Segments list cannot be empty")
        n = len(merged)
        crossfade_ms = self.config.crossfade_ms
        curves = _CURVE_TO_FFMPEG[self.config.curve]

        lines: List[str] = []
        for i, seg in enumerate(merged):
            start_s = _sec(float(seg.start_ms))
            end_s = _sec(float(seg.end_ms))
            dur_ms = float(seg.end_ms - seg.start_ms)
            fade_ms = min(crossfade_ms, dur_ms / 2.0)

            audio_chain = f"[0:a]atrim=start={start_s}:end={end_s},asetpts=PTS-STARTPTS"
            if i < n - 1 and fade_ms > 0:
                audio_chain += (
                    f",afade=t=out:st={_sec(dur_ms - fade_ms)}"
                    f":d={_sec(fade_ms)}:c={curves['out']}"
                )
            if i > 0 and fade_ms > 0:
                audio_chain += (
                    f",afade=t=in:st=0.000000:d={_sec(fade_ms)}:c={curves['in']}"
                )
            audio_chain += f"[a{i}]"
            lines.append(audio_chain)

            if has_video:
                lines.append(f"[0:v]trim=start={start_s}:end={end_s},setpts=PTS-STARTPTS[v{i}]")

        if has_video:
            labels = "".join(f"[v{i}][a{i}]" for i in range(n))
            lines.append(f"{labels}concat=n={n}:v=1:a=1[outv][outa]")
        else:
            labels = "".join(f"[a{i}]" for i in range(n))
            lines.append(f"{labels}concat=n={n}:v=0:a=1[outa]")
        return ";\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # Execucao
    # ------------------------------------------------------------------ #
    def _run_ffmpeg(self, cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True)

    def splice_file(
        self,
        input_path,
        output_path,
        segments: Sequence[TimeInterval],
    ) -> SpliceResult:
        """Emenda ``segments`` de ``input_path`` gravando em ``output_path``.

        Raises:
            ValueError: se ``segments`` estiver vazio ou a entrada nao tiver
                stream de audio.
            FileNotFoundError: se ``input_path`` nao existir.
            RuntimeError: se ffmpeg nao for encontrado ou falhar na execucao.
        """
        if not segments:
            raise ValueError("Segments list cannot be empty")
        src = Path(input_path)
        if not src.is_file():
            raise FileNotFoundError(f"Arquivo de midia nao encontrado: {src}")
        ffmpeg_bin = self._resolve_ffmpeg()

        probe = MediaProbe(ffprobe_path=self.ffprobe_path).probe(src)
        if not probe.has_audio:
            raise ValueError(f"Entrada sem stream de audio: {src}")

        config = self.config
        script = self.build_filter_complex_script(segments, has_video=probe.has_video)

        with tempfile.TemporaryDirectory(prefix="video_engine_splice_") as tmp_dir:
            script_path = Path(tmp_dir) / "filter_complex.txt"
            script_path.write_text(script, encoding="utf-8")

            cmd: List[str] = [
                ffmpeg_bin,
                "-hide_banner",
                "-y",
                "-i",
                str(src),
                "-filter_complex_script",
                str(script_path),
            ]
            if probe.has_video:
                cmd += [
                    "-map",
                    "[outv]",
                    "-map",
                    "[outa]",
                    "-c:v",
                    config.video_codec,
                    "-preset",
                    config.preset,
                    "-crf",
                    str(config.crf),
                    "-c:a",
                    config.audio_codec,
                    "-b:a",
                    config.audio_bitrate,
                ]
            else:
                cmd += [
                    "-map",
                    "[outa]",
                    "-c:a",
                    config.audio_codec,
                    "-b:a",
                    config.audio_bitrate,
                ]
            cmd.append(str(Path(output_path)))

            proc = self._run_ffmpeg(cmd)
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()[-8:]
                detail = "\n".join(tail) if tail else "sem stderr"
                raise RuntimeError(
                    f"ffmpeg falhou ao emendar {src} -> {output_path}: {detail}"
                )

        merged = [s for s in merge_intervals(segments) if s.end_ms > s.start_ms]
        out_info = MediaProbe(ffprobe_path=self.ffprobe_path).probe(output_path)
        return SpliceResult(
            output_path=str(Path(output_path)),
            total_duration_ms=out_info.duration_ms,
            num_segments=len(merged),
            num_junctions=len(merged) - 1 if len(merged) > 1 else 0,
            audio_sample_rate=out_info.audio_sample_rate or probe.audio_sample_rate or 16000,
            has_video=out_info.has_video,
        )


__all__ = ["MediaSplicer"]
