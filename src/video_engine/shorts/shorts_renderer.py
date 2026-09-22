"""Renderizacao final de Shorts verticais 9:16 prontos para publicacao (Issue #17).

Consolida a extracao do intervalo do corte (Issue #15), o reenquadramento
vertical 9:16 com face tracking (Issue #16) e a queima de legendas animadas
na safe area do YouTube Shorts (Issue #9), codificando o resultado em
H.264/AAC 1080x1920 (``-movflags +faststart``) e exportando o pacote de
metadados pronto para upload.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from video_engine.captions.ass_generator import AssSubtitleGenerator
from video_engine.captions.ass_models import KaraokeHighlightMode, SubtitleStyleConfig
from video_engine.captions.burner import CaptionBurner
from video_engine.captions.models import TranscriptionResult
from video_engine.editing.media_probe import MediaProbe
from video_engine.shorts.models import ShortCandidateCut
from video_engine.shorts.reframer_models import ReframerConfig
from video_engine.shorts.renderer_models import (
    ShortsPublishPackage,
    ShortsRendererConfig,
    ShortsRenderResult,
)
from video_engine.shorts.vertical_reframer import VerticalReframer

logger = logging.getLogger(__name__)


class ShortsRenderError(Exception):
    """Falha na renderizacao de um Short vertical 9:16.

    Cobrindo, por exemplo, origem inexistente, janela invalida, ausencia de
    ffmpeg ou falha de execucao do FFmpeg na etapa final de encoding.
    """


class ShortsRenderer:
    """Motor de renderizacao de Shorts 9:16 com legendas seguras e metadados."""

    def __init__(
        self,
        config: Optional[ShortsRendererConfig] = None,
        reframer: Optional[VerticalReframer] = None,
        caption_burner: Optional[CaptionBurner] = None,
        ffmpeg_path: Optional[str] = None,
    ) -> None:
        self.config = config or ShortsRendererConfig()
        if reframer is None:
            reframer = VerticalReframer(
                ReframerConfig(
                    target_width=self.config.target_width,
                    target_height=self.config.target_height,
                )
            )
        self.reframer = reframer
        self.caption_burner = caption_burner or CaptionBurner(ffmpeg_path=ffmpeg_path)
        self.ffmpeg_path = ffmpeg_path
        self.probe = MediaProbe()

    # ------------------------------------------------------------------ #
    # Safe area & legenda
    # ------------------------------------------------------------------ #
    def build_safe_style_config(self) -> SubtitleStyleConfig:
        """Configura estilo tipografico com margens seguras no terco medio."""
        cfg = self.config
        return SubtitleStyleConfig(
            font_name=cfg.font_name,
            font_size=cfg.font_size,
            highlight_color=cfg.highlight_color,
            outline_width=cfg.outline_width,
            alignment=2,
            margin_v=cfg.margin_v,
            margin_l=cfg.margin_l,
            margin_r=cfg.margin_r,
            highlight_mode=KaraokeHighlightMode.WORD_HIGHLIGHT,
            play_res_x=cfg.target_width,
            play_res_y=cfg.target_height,
        )

    # ------------------------------------------------------------------ #
    # Metadados prontos para publicacao
    # ------------------------------------------------------------------ #
    def generate_metadata_package(
        self,
        cut: ShortCandidateCut,
        video_path: Union[str, Path],
        metadata_path: Union[str, Path],
        video_title: Optional[str] = None,
    ) -> ShortsPublishPackage:
        """Gera titulo, descricao e hashtags prontos para publicacao.

        Persiste o JSON ``{metadata_path}`` com a serializacao integral do
        contrato ``ShortsPublishPackage``.
        """
        base = cut.hook_text or cut.summary or video_title or "Corte de alto impacto"
        title = self._truncate_at_word(base, self.config.max_title_chars)
        hashtags = self._normalize_hashtags(self.config.default_hashtags)

        parts = [title]
        if cut.summary:
            parts.append(cut.summary)
        if cut.reason:
            parts.append(cut.reason)
        parts.append(" ".join(hashtags))

        duration_sec = 0.0
        try:
            info = self.probe.probe(video_path)
            duration_sec = info.duration_ms / 1000.0
        except Exception:
            logger.warning("Nao foi possivel ler a duracao de %s", video_path)

        package = ShortsPublishPackage(
            id=cut.id,
            title=title,
            description="\n\n".join(parts),
            hashtags=hashtags,
            video_path=str(video_path),
            metadata_path=str(metadata_path),
            duration_sec=round(duration_sec, 3),
            start_ms=cut.start_ms,
            end_ms=cut.end_ms,
            resolution=f"{self.config.target_width}x{self.config.target_height}",
            video_codec="h264",
            audio_codec="aac",
            virality_score=cut.virality_score,
            hook_text=cut.hook_text,
        )

        metadata = Path(metadata_path)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(package.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return package

    # ------------------------------------------------------------------ #
    # Renderizacao
    # ------------------------------------------------------------------ #
    def render_cut(
        self,
        source_video: Union[str, Path],
        cut: ShortCandidateCut,
        output_dir: Union[str, Path],
        transcription: Optional[TranscriptionResult] = None,
        video_title: Optional[str] = None,
    ) -> ShortsPublishPackage:
        """Renderiza um unico corte 9:16 com legendas seguras e metadados.

        Artefatos finais (``{id}.mp4`` e ``{id}_metadata.json``) sao gravados
        em ``output_dir``; os intermediarios (vertical 9:16 e video queimado)
        vivem em um ``tempfile.TemporaryDirectory`` descartado ao final.
        """
        src = Path(source_video)
        out_dir = Path(output_dir)

        if not src.is_file():
            raise ShortsRenderError(f"Video de origem nao encontrado: {src}")
        info = self.probe.probe(src)
        if not info.has_video:
            raise ShortsRenderError(f"Arquivo sem stream de video: {src}")

        start_ms, end_ms = self._clamp_window(cut, info.duration_ms)
        if start_ms != cut.start_ms or end_ms != cut.end_ms:
            logger.warning(
                "Janela do corte %s ajustada para [%d, %d]ms "
                "(duracao do video de origem: %dms)",
                cut.id,
                start_ms,
                end_ms,
                info.duration_ms,
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = self._safe_basename(cut.id)
        out_video = out_dir / f"{base_name}.mp4"
        metadata_json = out_dir / f"{base_name}_metadata.json"

        with tempfile.TemporaryDirectory(prefix="shorts_render_") as work:
            work_dir = Path(work)

            vertical = work_dir / "vertical.mp4"
            self.reframer.reframe(
                src,
                vertical,
                cut_start_ms=start_ms,
                cut_end_ms=end_ms,
            )

            style = self.build_safe_style_config()
            burn_input = self._resolve_burn_input(cut, transcription)
            video_to_encode = vertical

            if self._has_captions(burn_input, start_ms, end_ms, style):
                burned = work_dir / "burned.mp4"
                self.caption_burner.burn_cut_subtitles(
                    vertical,
                    burned,
                    burn_input,
                    cut_start_ms=start_ms,
                    cut_end_ms=end_ms,
                    style_config=style,
                )
                video_to_encode = burned
            else:
                logger.warning(
                    "Corte %s sem palavras no trecho [%d, %d]ms: "
                    "renderizado em clean feed (sem legendas)",
                    cut.id,
                    start_ms,
                    end_ms,
                )

            final_info = self.probe.probe(video_to_encode)
            self._final_encode(video_to_encode, out_video, has_audio=final_info.has_audio)

        return self.generate_metadata_package(
            cut,
            out_video,
            metadata_json,
            video_title=video_title,
        )

    def render_batch(
        self,
        source_video: Union[str, Path],
        cuts: Sequence[ShortCandidateCut],
        output_dir: Union[str, Path],
        transcription: Optional[TranscriptionResult] = None,
        video_title: Optional[str] = None,
    ) -> ShortsRenderResult:
        """Renderiza lote de 1 a 3 cortes gerando pacotes completos."""
        cut_list = list(cuts)
        if not cut_list:
            raise ShortsRenderError(
                "Lote de renderizacao vazio: informe ao menos um corte"
            )
        packages = [
            self.render_cut(
                source_video,
                cut,
                output_dir,
                transcription=transcription,
                video_title=video_title,
            )
            for cut in cut_list
        ]
        return ShortsRenderResult(
            packages=packages,
            total_rendered=len(packages),
            output_dir=str(Path(output_dir)),
        )

    # ------------------------------------------------------------------ #
    # Encoding final otimizado
    # ------------------------------------------------------------------ #
    def _final_encode_cmd(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        has_audio: bool,
    ) -> List[str]:
        """Monta o comando FFmpeg final (H.264/AAC, faststart) pronto para teste."""
        cfg = self.config
        cmd = [
            self._resolve_ffmpeg(),
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            cfg.video_codec,
            "-preset",
            cfg.preset,
            "-crf",
            str(cfg.crf),
            "-pix_fmt",
            cfg.pix_fmt,
        ]
        if has_audio:
            cmd += [
                "-c:a",
                cfg.audio_codec,
                "-b:a",
                cfg.audio_bitrate,
                "-ar",
                "48000",
            ]
        if self.config.faststart:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(output_path))
        return cmd

    def _final_encode(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        has_audio: bool,
    ) -> None:
        """Executa o encoding final fazendo upload de erros em ``ShortsRenderError``."""
        cmd = self._final_encode_cmd(input_path, output_path, has_audio)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            raise ShortsRenderError(
                f"Nao foi possivel executar o ffmpeg ({cmd[0]}): {exc}"
            ) from exc
        if proc.returncode != 0:
            stderr = proc.stderr or ""
            tail = stderr.strip().splitlines()[-8:]
            detail = "\n".join(tail) if tail else "sem stderr"
            raise ShortsRenderError(
                f"ffmpeg falhou ao renderizar {output_path} ({proc.returncode}):\n{detail}"
            )

    # ------------------------------------------------------------------ #
    # Utilitarios
    # ------------------------------------------------------------------ #
    def _clamp_window(self, cut: ShortCandidateCut, source_duration_ms: int) -> Tuple[int, int]:
        """Ajusta/valida a janela ``[start, end]`` do corte contra a origem."""
        start = max(0, cut.start_ms)
        end = min(cut.end_ms, source_duration_ms)
        if end <= start or start >= source_duration_ms:
            raise ShortsRenderError(
                f"Janela do corte {cut.id} vazia ou invalida: "
                f"start={cut.start_ms}ms, end={cut.end_ms}ms "
                f"(duracao do video: {source_duration_ms}ms)"
            )
        return start, end

    def _resolve_burn_input(
        self,
        cut: ShortCandidateCut,
        transcription: Optional[TranscriptionResult],
    ):
        """Define o insumo da queima de legendas (transcricao ou palavras do corte)."""
        if transcription is not None:
            return transcription
        if cut.words:
            return list(cut.words)
        return None

    def _has_captions(
        self,
        burn_input,
        start_ms: int,
        end_ms: int,
        style: SubtitleStyleConfig,
    ) -> bool:
        """True se existirem palavras reindexadas dentro do trecho do corte."""
        if burn_input is None:
            return False
        generator = AssSubtitleGenerator(style_config=style)
        cues = generator.build_cues(
            burn_input,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        return any(cue.words for cue in cues)

    def _resolve_ffmpeg(self) -> str:
        binary = self.ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise ShortsRenderError("ffmpeg nao encontrado no PATH")
        return binary

    @staticmethod
    def _truncate_at_word(value: str, max_chars: int) -> str:
        """Trunca o texto nas palavras completas respeitando ``max_chars``."""
        text = " ".join(str(value or "").split())
        if not text:
            return text
        if len(text) <= max_chars:
            return text
        words = text.split(" ")
        kept: List[str] = []
        for word in words:
            candidate = " ".join(kept + [word])
            if len(candidate) > max_chars:
                break
            kept.append(word)
        return " ".join(kept)

    @staticmethod
    def _normalize_hashtags(tags: Sequence[str]) -> List[str]:
        """Normaliza hashtags garantindo prefixo ``#`` e removendo duplicatas."""
        seen = set()
        normalized: List[str] = []
        for tag in tags:
            token = str(tag).strip()
            if not token or token == "#":
                continue
            if not token.startswith("#"):
                token = f"#{token}"
            token = token.replace(" ", "_")
            if token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        return normalized

    @staticmethod
    def _safe_basename(cut_id: str) -> str:
        """Sanitiza o identificador do corte para uso como nome de arquivo."""
        cleaned = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in str(cut_id)
        )
        return cleaned or "short"


__all__ = ["ShortsRenderError", "ShortsRenderer"]
