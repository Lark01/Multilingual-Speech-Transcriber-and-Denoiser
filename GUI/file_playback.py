"""Load audio files (MP3, WAV, …), denoise offline, play through speakers."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from scipy.signal import resample_poly

from audio_engine import AudioEngine
from model import MIN_SAMPLES, SAMPLE_RATE as MODEL_SR, DenoiserModel

logger = logging.getLogger(__name__)

# ~2 s at 16 kHz — balances progress updates vs model overhead
_DENOISE_CHUNK_SAMPLES = 32_000

# Refuse absurdly long files to avoid exhausting RAM (45 minutes @ 16 kHz mono float32)
_MAX_SAMPLES_16K = MODEL_SR * 60 * 45


def _resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    return resample_poly(x.astype(np.float32, copy=False), up, down).astype(np.float32, copy=False)


def _load_via_ffmpeg(path: Path, ffmpeg_exe: str, intermediate_sr: int = 48_000) -> tuple[np.ndarray, int]:
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(intermediate_sr),
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: {err}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if raw.size == 0:
        raise RuntimeError("ffmpeg produced no audio samples")
    peak = float(np.max(np.abs(raw))) + 1e-12
    if peak > 1.0:
        raw = raw / peak
    logger.info(
        "[file] Loaded via ffmpeg: %s @ %s Hz, %s samples", path.name, intermediate_sr, len(raw)
    )
    return raw, intermediate_sr


def _load_via_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if x.shape[1] > 1:
        x = x.mean(axis=1, dtype=np.float32)
    else:
        x = x[:, 0]
    if x.size == 0:
        raise ValueError("Empty audio file")
    peak = float(np.max(np.abs(x))) + 1e-12
    if peak > 1.0:
        x = x / peak
    logger.info("[file] Loaded via soundfile: %s @ %s Hz, %s samples", path.name, sr, len(x))
    return x.astype(np.float32, copy=False), int(sr)


def _ffmpeg_candidates() -> list[str]:
    """Prefer system ffmpeg, then imageio-ffmpeg's bundled binary (no PATH / TorchCodec needed)."""
    found: list[str] = []
    w = shutil.which("ffmpeg")
    if w:
        found.append(w)
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and bundled not in found:
            found.append(bundled)
    except Exception as e:
        logger.debug("[file] imageio_ffmpeg not available: %s", e)
    return found


def _load_via_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    import torchaudio

    try:
        wav, sr = torchaudio.load(str(path), normalize=True)
    except TypeError:
        wav, sr = torchaudio.load(str(path))
    if wav.numel() == 0:
        raise ValueError("Empty audio file")
    sr = int(sr)
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    x = wav.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) + 1e-12
    if peak > 1.0:
        x = x / peak
    logger.info("[file] Loaded via torchaudio: %s @ %s Hz, %s samples", path.name, sr, len(x))
    return x, sr


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """
    Load file to mono float32 in [-1, 1], native sample rate.

    Order (first success wins):
      1. soundfile — WAV / FLAC / OGG when libsndfile supports them
      2. ffmpeg — system PATH, then ``imageio-ffmpeg``'s bundled ffmpeg (MP3/M4A, etc.)
      3. torchaudio — last resort (often needs TorchCodec + matching FFmpeg DLLs on Windows)
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))

    errors: list[str] = []
    ext = path.suffix.lower()

    if ext in (".wav", ".flac", ".ogg"):
        try:
            return _load_via_soundfile(path)
        except Exception as e:
            errors.append(f"soundfile: {e}")
            logger.debug("[file] soundfile load failed: %s", e)

    cands = _ffmpeg_candidates()
    if not cands:
        errors.append("No ffmpeg binary found — pip install imageio-ffmpeg (recommended).")

    system_ffmpeg = shutil.which("ffmpeg")
    for ffmpeg_exe in cands:
        label = (
            "ffmpeg (PATH)"
            if system_ffmpeg and ffmpeg_exe == system_ffmpeg
            else "ffmpeg (imageio-ffmpeg)"
        )
        try:
            out = _load_via_ffmpeg(path, ffmpeg_exe)
            logger.info("[file] Decoder: %s — %s", label, ffmpeg_exe)
            return out
        except Exception as e:
            errors.append(f"{label}: {e}")
            logger.debug("[file] %s failed: %s", label, e)

    try:
        return _load_via_torchaudio(path)
    except Exception as e:
        msg = str(e)
        if "libtorchcodec" in msg or "TorchCodec" in msg.lower():
            msg = (
                "TorchCodec DLL load failed (common on Windows with PyTorch/torchcodec mismatch). "
                "MP3/M4A should decode via imageio-ffmpeg if installed."
            )
        elif len(msg) > 450:
            msg = msg[:450] + "…"
        errors.append(f"torchaudio: {msg}")
        logger.debug("[file] torchaudio load failed: %s", e)

    raise RuntimeError(
        "Could not decode this file. Try:\n"
        "  • pip install imageio-ffmpeg  (ships a private ffmpeg — used automatically), or\n"
        "  • Install ffmpeg and add it to your PATH, or\n"
        "  • Use WAV/FLAC (via soundfile).\n"
        "TorchCodec/torchaudio on Windows is fragile; bundled ffmpeg is preferred.\n\n"
        + "\n".join(errors)
    )


def pick_output_sample_rate(output_device_index: int | None) -> int:
    try:
        if output_device_index is None:
            d = sd.query_devices(kind="output")
        else:
            d = sd.query_devices(int(output_device_index))
        sr = float(d.get("default_samplerate") or 0)
        if 8000 <= sr <= 384_000:
            return int(round(sr))
    except Exception:
        logger.exception("[file] Could not query output default sample rate")
    return 48_000


class DenoiseFileThread(QThread):
    """Background: decode → denoise at 16 kHz → resample → play; then signal done."""

    progress = pyqtSignal(int)  # 0–100 (denoise phase; stays at 99 while playing)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(
        self,
        model: DenoiserModel,
        engine: AudioEngine,
        file_path: Path,
        strength: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._engine = engine
        self._path = file_path
        self._strength = float(max(0.0, min(1.0, strength)))
        self._play_pos = 0
        self._play_buf: np.ndarray | None = None
        self._stream: sd.OutputStream | None = None

    def stop_playback(self) -> None:
        self.requestInterruption()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("[file] Error stopping playback stream")
            self._stream = None

    def run(self) -> None:
        try:
            if not self._model.is_ready:
                self.failed.emit("Model is not loaded.")
                return

            self.status.emit("Loading file…")
            audio, sr_in = load_audio_mono(self._path)
            if self.isInterruptionRequested():
                return

            self.status.emit("Resampling to model rate…")
            x16 = _resample_mono(audio, sr_in, MODEL_SR)
            del audio

            if len(x16) > _MAX_SAMPLES_16K:
                self.failed.emit(
                    f"File is too long (max {_MAX_SAMPLES_16K // MODEL_SR // 60} minutes at 16 kHz)."
                )
                return

            _in_idx, out_idx = self._engine.get_device_indices()
            out_sr = pick_output_sample_rate(out_idx)

            self.status.emit("Denoising…")
            n = len(x16)
            parts: list[np.ndarray] = []
            for start in range(0, n, _DENOISE_CHUNK_SAMPLES):
                if self.isInterruptionRequested():
                    self.status.emit("Cancelled.")
                    return
                end = min(start + _DENOISE_CHUNK_SAMPLES, n)
                chunk = np.asarray(x16[start:end], dtype=np.float32)
                orig_len = len(chunk)
                if orig_len < MIN_SAMPLES:
                    chunk = np.pad(chunk, (0, MIN_SAMPLES - orig_len))
                clean = self._model.denoise(chunk)
                clean = clean[:orig_len]
                mix = self._strength * clean + (1.0 - self._strength) * x16[start:end]
                parts.append(np.clip(mix, -1.0, 1.0).astype(np.float32, copy=False))
                pct = int(85 * end / max(n, 1))
                self.progress.emit(min(85, pct))

            if self.isInterruptionRequested():
                return

            full_16 = np.concatenate(parts, axis=0)
            del parts, x16

            self.status.emit("Preparing playback…")
            play_buf = _resample_mono(full_16, MODEL_SR, out_sr)
            play_buf = np.clip(play_buf, -1.0, 1.0).astype(np.float32, copy=False)
            del full_16

            if self.isInterruptionRequested():
                return

            self._play_buf = play_buf
            self._play_pos = 0
            total = len(play_buf)

            def callback(outdata: np.ndarray, frames: int, _time, _status) -> None:
                pos = self._play_pos
                buf = self._play_buf
                if buf is None or pos >= len(buf):
                    outdata.fill(0.0)
                    raise sd.CallbackStop
                total = len(buf)
                take = min(frames, total - pos)
                outdata[:take, 0] = buf[pos : pos + take]
                if take < frames:
                    outdata[take:, 0] = 0.0
                self._play_pos = pos + take
                if self._play_pos >= total:
                    raise sd.CallbackStop

            self.progress.emit(95)
            self.status.emit("Playing…")

            block = min(2048, max(256, total // 100 or 256))
            self._stream = sd.OutputStream(
                device=out_idx,
                samplerate=out_sr,
                channels=1,
                dtype="float32",
                blocksize=block,
                callback=callback,
            )
            self._stream.start()
            if not self.isInterruptionRequested():
                wait_fn = getattr(self._stream, "wait", None)
                if callable(wait_fn):
                    try:
                        wait_fn()
                    except Exception:
                        logger.exception("[file] stream.wait() failed; polling until end of buffer")
                # Some PortAudio builds have no working wait(); .active may be missing too — poll _play_pos.
                deadline = time.perf_counter() + float(total) / float(out_sr) + 60.0
                while self._play_pos < total and not self.isInterruptionRequested():
                    if time.perf_counter() > deadline:
                        logger.error("[file] Playback timed out before end of buffer")
                        break
                    time.sleep(0.02)

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    logger.exception("[file] Error closing stream")
                self._stream = None

            self._play_buf = None
            self.progress.emit(100)
            self.status.emit("Done.")
            if not self.isInterruptionRequested():
                self.finished_ok.emit()
        except Exception as e:
            logger.exception("[file] Denoise/play failed")
            self.failed.emit(str(e))
        finally:
            self.stop_playback()
