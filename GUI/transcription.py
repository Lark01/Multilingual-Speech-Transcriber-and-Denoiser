"""Real-time and offline transcription using TranscriberModel."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from scipy.signal import resample_poly

from audio_util import SILENCE_RMS, segment_rms
from file_playback import _resample_mono, load_audio_mono
from model import MIN_SAMPLES, SAMPLE_RATE as MODEL_SR

if TYPE_CHECKING:
    from transcriber_model import TranscriberModel

logger = logging.getLogger(__name__)

LIVE_QUEUE_MAX = 2
OFFLINE_CHUNK_SEC = 30
_MAX_SAMPLES_16K = MODEL_SR * 60 * 45


def merge_overlapping_text(previous: str, new: str) -> str:
    """Return only the new words when sliding windows repeat prior text."""
    new = new.strip()
    if not new:
        return ""
    if not previous:
        return new
    if new in previous:
        return ""
    prev_words = previous.split()
    new_words = new.split()
    max_k = min(len(prev_words), len(new_words))
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == new_words[:k]:
            tail = " ".join(new_words[k:])
            return tail.strip()
    return new


class RealtimeTranscription:
    """Sliding-window live transcription + background infer thread."""

    def __init__(
        self,
        get_model: Callable[[], TranscriberModel | None],
        segment_sec: float = 2.0,
        hop_sec: float = 1.0,
    ) -> None:
        self._get_model = get_model
        self._segment_sec = max(1.0, float(segment_sec))
        self._hop_sec = max(0.5, min(float(hop_sec), self._segment_sec))
        self._segment_samples = int(MODEL_SR * self._segment_sec)
        self._hop_samples = int(MODEL_SR * self._hop_sec)

        self.active = False
        self.paused = False
        self.transcript_lines: list[str] = []
        self.last_text = ""
        self.latency_ms = 0.0
        self.on_new_text: Callable[[str], None] | None = None
        self.on_status: Callable[[str], None] | None = None
        self.last_skip_reason: str = ""
        self.segments_processed = 0

        self._lock = threading.Lock()
        self._model_unready_logged = False
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._infer_queue: deque[np.ndarray] = deque()
        self._infer_running = False
        self._infer_thread: threading.Thread | None = None
        self._full_transcript = ""

    def set_segment_sec(self, sec: float) -> None:
        with self._lock:
            self._segment_sec = max(1.0, float(sec))
            self._segment_samples = int(MODEL_SR * self._segment_sec)
            self._hop_sec = min(self._hop_sec, self._segment_sec)
            self._hop_samples = int(MODEL_SR * self._hop_sec)

    def set_hop_sec(self, sec: float) -> None:
        with self._lock:
            self._hop_sec = max(0.5, min(float(sec), self._segment_sec))
            self._hop_samples = int(MODEL_SR * self._hop_sec)

    def start_worker(self) -> None:
        if self._infer_thread is not None and self._infer_thread.is_alive():
            return
        self._infer_running = True
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

    def stop_worker(self) -> None:
        self._infer_running = False
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2.0)
            self._infer_thread = None

    def feed_16k(self, chunk: np.ndarray) -> None:
        if not self.active or self.paused:
            return
        if chunk is None or len(chunk) == 0:
            return
        audio = np.asarray(chunk, dtype=np.float32).ravel()
        with self._lock:
            self._buffer.append(audio)
            combined = (
                np.concatenate(self._buffer)
                if len(self._buffer) > 1
                else np.asarray(self._buffer[0], dtype=np.float32)
            )
            self._buffer = [combined]
            self._buffer_samples = len(combined)

            seg_n = self._segment_samples
            hop_n = self._hop_samples
            while self._buffer_samples >= seg_n:
                segment = np.asarray(combined[:seg_n], dtype=np.float32).copy()
                combined = np.asarray(combined[hop_n:], dtype=np.float32)
                self._buffer = [combined] if len(combined) else []
                self._buffer_samples = len(combined)
                while len(self._infer_queue) >= LIVE_QUEUE_MAX:
                    self._infer_queue.popleft()
                self._infer_queue.append(segment)

    def clear(self) -> None:
        with self._lock:
            self.transcript_lines.clear()
            self.last_text = ""
            self._full_transcript = ""
            self._buffer = []
            self._buffer_samples = 0
            self._infer_queue.clear()

    def _infer_worker(self) -> None:
        while self._infer_running:
            segment: np.ndarray | None = None
            with self._lock:
                if self._infer_queue:
                    segment = self._infer_queue.popleft()
            if segment is None:
                time.sleep(0.01)
                continue
            rms = segment_rms(segment)
            if rms < SILENCE_RMS:
                self.last_skip_reason = f"segment too quiet (rms={rms:.5f})"
                logger.debug("[transcription] skip silence rms=%.5f", rms)
                continue
            try:
                model = self._get_model()
            except Exception:
                logger.exception("[transcription] model load failed")
                self.last_skip_reason = "model load failed"
                continue
            if model is None or not model.is_ready:
                if not self._model_unready_logged:
                    self._model_unready_logged = True
                    err = model.load_error if model else "not loaded"
                    logger.warning("[transcription] Whisper not ready: %s", err)
                    self.last_skip_reason = err or "Whisper not ready"
                continue
            self._model_unready_logged = False
            try:
                self._emit_status("Transcribing…")
                text = model.transcribe(segment, sample_rate=MODEL_SR, live=True)
                self.latency_ms = float(model.last_inference_ms)
                self.segments_processed += 1
            except Exception:
                logger.exception("[transcription] Live infer failed")
                self.last_skip_reason = "inference error (see log)"
                continue
            delta = merge_overlapping_text(self._full_transcript, text)
            if delta:
                self.last_text = delta
                self._full_transcript = (self._full_transcript + " " + delta).strip()
                self.transcript_lines.append(delta)
                if len(self.transcript_lines) > 200:
                    self.transcript_lines = self.transcript_lines[-200:]
                self.last_skip_reason = ""
                logger.debug("[transcription] live +%r", delta[:80])
                if self.on_new_text:
                    try:
                        self.on_new_text(delta)
                    except Exception:
                        logger.exception("[transcription] on_new_text callback failed")
            else:
                self.last_skip_reason = "no new words in segment"
                self._emit_status("Listening…")

    def _emit_status(self, msg: str) -> None:
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception:
                logger.exception("[transcription] on_status callback failed")


class TranscribeFileThread(QThread):
    """Background: decode file → optional denoise → transcribe in chunks."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    segment = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal(str)

    def __init__(
        self,
        get_model: Callable[[], TranscriberModel | None],
        file_path: Path,
        denoise_first: bool = False,
        denoise_model: object | None = None,
        denoise_strength: float = 1.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._get_model = get_model
        self._path = file_path
        self._denoise_first = denoise_first
        self._denoise_model = denoise_model
        self._denoise_strength = float(max(0.0, min(1.0, denoise_strength)))

    def run(self) -> None:
        try:
            model = self._get_model()
            if model is None or not model.is_ready:
                self.failed.emit(
                    model.load_error if model and model.load_error else "Transcriber not loaded."
                )
                return

            self.status.emit("Loading file…")
            audio, sr_in = load_audio_mono(self._path)
            if self.isInterruptionRequested():
                return

            self.status.emit("Resampling…")
            x16 = _resample_mono(audio, sr_in, MODEL_SR)
            del audio

            if len(x16) > _MAX_SAMPLES_16K:
                self.failed.emit(
                    f"File is too long (max {_MAX_SAMPLES_16K // MODEL_SR // 60} minutes at 16 kHz)."
                )
                return

            if self._denoise_first and self._denoise_model is not None:
                dm = self._denoise_model
                if not getattr(dm, "is_ready", False):
                    self.failed.emit("Denoiser not loaded — cannot denoise before transcribe.")
                    return
                self.status.emit("Denoising…")
                n = len(x16)
                parts: list[np.ndarray] = []
                chunk_n = 32_000
                for start in range(0, n, chunk_n):
                    if self.isInterruptionRequested():
                        return
                    end = min(start + chunk_n, n)
                    chunk = np.asarray(x16[start:end], dtype=np.float32)
                    orig_len = len(chunk)
                    if orig_len < MIN_SAMPLES:
                        chunk = np.pad(chunk, (0, MIN_SAMPLES - orig_len))
                    clean = dm.denoise(chunk)
                    clean = clean[:orig_len]
                    mix = self._denoise_strength * clean + (1.0 - self._denoise_strength) * x16[
                        start:end
                    ]
                    parts.append(np.clip(mix, -1.0, 1.0).astype(np.float32, copy=False))
                    self.progress.emit(int(40 * end / max(n, 1)))
                x16 = np.concatenate(parts, axis=0)
                del parts

            chunk_samples = int(MODEL_SR * OFFLINE_CHUNK_SEC)
            n = len(x16)
            lines: list[str] = []
            self.status.emit("Transcribing…")
            base_pct = 45 if self._denoise_first else 5
            span = 90 - base_pct

            for start in range(0, n, chunk_samples):
                if self.isInterruptionRequested():
                    self.status.emit("Cancelled.")
                    return
                end = min(start + chunk_samples, n)
                segment = x16[start:end]
                if segment_rms(segment) >= SILENCE_RMS:
                    text = model.transcribe(segment, sample_rate=MODEL_SR, live=False)
                    if text:
                        lines.append(text)
                        self.segment.emit(text)
                pct = base_pct + int(span * end / max(n, 1))
                self.progress.emit(min(99, pct))

            full = " ".join(lines).strip()
            self.progress.emit(100)
            self.status.emit("Done.")
            if not self.isInterruptionRequested():
                self.finished_ok.emit(full)
        except Exception as e:
            logger.exception("[transcription] File transcribe failed")
            self.failed.emit(str(e))
