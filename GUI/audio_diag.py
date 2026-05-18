"""
Real-time audio pipeline diagnostics — writes to a log file for stutter debugging.

Enable from main via enable_audio_diag(). Read logs/clearvoice_audio_debug.log while
reproducing glitches; look for infer_q_full, clean_dropped, dry_wait, and PortAudio flags.
"""

from __future__ import annotations

import logging
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_DIAG_LOGGER_NAME = "clearvoice.audio_diag"
_SUMMARY_INTERVAL_S = 1.0

_diag: "AudioPipelineDiag | None" = None


def get_audio_diag() -> "AudioPipelineDiag | None":
    return _diag


def enable_audio_diag(log_path: Path | None = None) -> AudioPipelineDiag:
    """Attach rotating file handler and return the global diag instance."""
    global _diag
    if log_path is None:
        log_path = Path(__file__).resolve().parent / "logs" / "clearvoice_audio_debug.log"
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    diag_logger = logging.getLogger(_DIAG_LOGGER_NAME)
    diag_logger.setLevel(logging.DEBUG)
    diag_logger.propagate = False
    diag_logger.handlers.clear()

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    diag_logger.addHandler(handler)

    _diag = AudioPipelineDiag(diag_logger, log_path)
    _diag.log_session_start()
    return _diag


class AudioPipelineDiag:
    """Thread-safe counters + rate-limited summaries for the duplex denoise path."""

    def __init__(self, logger: logging.Logger, log_path: Path) -> None:
        self._log = logger
        self._path = log_path
        self._lock = threading.Lock()
        self._last_summary = time.monotonic()
        self._callback_count = 0
        self._paths: dict[str, int] = {}
        self._events: dict[str, int] = {}
        self._infer_ms_sum = 0.0
        self._infer_ms_max = 0.0
        self._infer_ms_n = 0
        self._infer_q_max = 0
        self._clean_q_max = 0
        self._pending_max = 0
        self._callback_us_max = 0.0
        self._pa_flags: dict[str, int] = {}

    @property
    def log_file(self) -> Path:
        return self._path

    def _bump(self, bucket: dict[str, int], key: str, n: int = 1) -> None:
        bucket[key] = bucket.get(key, 0) + n

    def log_session_start(self) -> None:
        self._log.info("=== ClearVoice audio diagnostics === log=%s", self._path)

    def log_line(self, level: int, msg: str, *args: Any) -> None:
        self._log.log(level, msg, *args)

    def event(self, name: str, detail: str = "") -> None:
        """Immediate log for rare / important events."""
        with self._lock:
            self._bump(self._events, name)
        if detail:
            self._log.warning("%s | %s", name, detail)
        else:
            self._log.warning("%s", name)

    def record_callback(
        self,
        *,
        path: str,
        frames: int,
        sr: int,
        infer_q: int,
        clean_q: int,
        pending_dry: int,
        infer_ms: float,
        callback_us: float,
        pa_status: str,
        strength: float,
    ) -> None:
        with self._lock:
            self._callback_count += 1
            self._bump(self._paths, path)
            self._infer_q_max = max(self._infer_q_max, infer_q)
            self._clean_q_max = max(self._clean_q_max, clean_q)
            self._pending_max = max(self._pending_max, pending_dry)
            self._callback_us_max = max(self._callback_us_max, callback_us)
            if infer_ms > 0:
                self._infer_ms_sum += infer_ms
                self._infer_ms_max = max(self._infer_ms_max, infer_ms)
                self._infer_ms_n += 1
            if pa_status:
                self._bump(self._pa_flags, pa_status)
            now = time.monotonic()
            if now - self._last_summary < _SUMMARY_INTERVAL_S:
                return
            self._emit_summary(now, frames, sr, strength)

    def _emit_summary(self, now: float, frames: int, sr: int, strength: float) -> None:
        block_ms = 1000.0 * frames / sr if sr > 0 else 0.0
        infer_avg = (
            self._infer_ms_sum / self._infer_ms_n if self._infer_ms_n else 0.0
        )
        paths = ",".join(f"{k}={v}" for k, v in sorted(self._paths.items()))
        events = ",".join(f"{k}={v}" for k, v in sorted(self._events.items())) or "-"
        pa = ",".join(f"{k}={v}" for k, v in sorted(self._pa_flags.items())) or "-"
        self._log.info(
            "[summary 1s] callbacks=%d block_ms=%.1f sr=%d strength=%.2f "
            "paths={%s} infer_q_max=%d clean_q_max=%d pending_max=%d "
            "infer_ms_avg=%.1f infer_ms_max=%.1f (budget~%.1f ms) callback_us_max=%.0f "
            "events={%s} pa_flags={%s}",
            self._callback_count,
            block_ms,
            sr,
            strength,
            paths or "-",
            self._infer_q_max,
            self._clean_q_max,
            self._pending_max,
            infer_avg,
            self._infer_ms_max,
            block_ms,
            self._callback_us_max,
            events,
            pa,
        )
        self._callback_count = 0
        self._paths.clear()
        self._infer_ms_sum = 0.0
        self._infer_ms_max = 0.0
        self._infer_ms_n = 0
        self._infer_q_max = 0
        self._clean_q_max = 0
        self._pending_max = 0
        self._callback_us_max = 0.0
        self._pa_flags.clear()
        self._last_summary = now

    def record_infer_done(
        self,
        *,
        infer_ms: float,
        infer_q_after: int,
        clean_q_after: int,
        clean_dropped: bool,
        native_samples: int,
        sr: int,
    ) -> None:
        with self._lock:
            if clean_dropped:
                self._bump(self._events, "clean_dropped")
        if clean_dropped:
            self._log.warning(
                "clean_dropped | infer_ms=%.1f infer_q=%d clean_q=%d native_n=%d sr=%d",
                infer_ms,
                infer_q_after,
                clean_q_after,
                native_samples,
                sr,
            )
        elif infer_ms > 0 and sr > 0:
            block_ms = 1000.0 * native_samples / sr
            if infer_ms > block_ms * 1.05:
                self._log.warning(
                    "infer_slower_than_block | infer_ms=%.1f block_ms=%.1f "
                    "(worker cannot keep realtime)",
                    infer_ms,
                    block_ms,
                )
