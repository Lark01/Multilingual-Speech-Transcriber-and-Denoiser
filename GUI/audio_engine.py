"""Real-time audio: split capture/playback streams, denoise at 16 kHz in a worker thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import math

if TYPE_CHECKING:
    from transcription import RealtimeTranscription

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from audio_diag import get_audio_diag
from model import CHUNK_CONTEXT_SAMPLES_16K, DenoiserModel, SAMPLE_RATE as MODEL_SAMPLE_RATE
from audio_util import SILENCE_RMS as _TX_SILENCE_RMS, segment_rms

logger = logging.getLogger(__name__)

# Small PortAudio I/O blocks keep callbacks lightweight (~20 ms).
IO_BLOCK_MS = 20
# Denoise batch size — model is trained at 16 kHz; 160 ms ≈ 2560 samples there.
INFER_CHUNK_MS = 160
# Drop stale mic blocks when the capture queue grows (never skip denoise on new audio).
_CAPTURE_QUEUE_TRIM = 24
# Output waits until the playback queue has enough blocks (~400 ms) for the first infer pass.
_PLAYBACK_PREROLL_BLOCKS = max(6, 400 // IO_BLOCK_MS)
_QUEUE_MAXSIZE = 64


def _hostapi_short_tag(api_full_name: str) -> str:
    """Short label for combo boxes (PortAudio host API name)."""
    n = (api_full_name or "").lower()
    if "wasapi" in n:
        return "WASAPI"
    if "wdm-ks" in n or "wdm ks" in n:
        return "WDM-KS"
    if "directsound" in n:
        return "DirectSound"
    if "mme" in n or "mapper" in n:
        return "MME"
    if "asio" in n:
        return "ASIO"
    return (api_full_name or "API")[:12]


class AudioEngine:
    """Thread-safe audio I/O with optional denoising."""

    def __init__(self, model: DenoiserModel) -> None:
        self.model = model
        self._lock = threading.Lock()
        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None

        # Denoise on/off (stream may still run — bypass passes input through).
        self._denoise_active = True
        self._strength = 1.0
        self._input_device: int | None = None
        self._output_device: int | None = None
        self._stream_sample_rate = MODEL_SAMPLE_RATE

        self._input_level = 0.0
        self._output_level = 0.0
        self._latency_ms = 0.0
        self._error_message: str | None = None

        self._input_wave: deque[float] = deque(maxlen=MODEL_SAMPLE_RATE)
        self._output_wave: deque[float] = deque(maxlen=MODEL_SAMPLE_RATE)
        self._resize_wave_buffers(MODEL_SAMPLE_RATE)

        # Split-stream pipeline: capture callback → _capture_q → worker → _playback_q → out callback.
        self._capture_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._playback_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._proc_thread: threading.Thread | None = None
        self._proc_running = False
        # 16 kHz tail from previous input prepended before denoise (STFT boundary context).
        self._ctx_in_16k = np.zeros(CHUNK_CONTEXT_SAMPLES_16K, dtype=np.float32)
        # Raised-cosine splice vs previous callback output (removes clicks on mute/denoise edges).
        self._out_xfade_tail: np.ndarray | None = None
        self._playback_ready = False

        self.transcription: RealtimeTranscription | None = None
        self._transcribe_denoised = True

    @property
    def transcribe_denoised(self) -> bool:
        with self._lock:
            return self._transcribe_denoised

    @transcribe_denoised.setter
    def transcribe_denoised(self, value: bool) -> None:
        with self._lock:
            self._transcribe_denoised = bool(value)

    @staticmethod
    def _io_block_samples(sr: int) -> int:
        return max(1, int(sr * IO_BLOCK_MS // 1000))

    @staticmethod
    def _infer_chunk_samples(sr: int) -> int:
        return max(1, int(sr * INFER_CHUNK_MS // 1000))

    def _drain_audio_queues(self) -> None:
        for q in (self._capture_q, self._playback_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    @staticmethod
    def _pa_status_str(status: sd.CallbackFlags) -> str:
        if not status:
            return ""
        parts: list[str] = []
        for name in (
            "input_overflow",
            "output_underflow",
            "input_underflow",
            "output_overflow",
            "priming_output",
        ):
            if getattr(status, name, False):
                parts.append(name)
        return "|".join(parts) if parts else str(status)

    def _flush_denoise_pipeline(self) -> None:
        """
        Discard queued capture/playback audio and overlap context so stopping bypass or
        the stream cannot play delayed denoise from old audio.
        """
        diag = get_audio_diag()
        if diag is not None:
            diag.event(
                "pipeline_flush",
                f"capture_q~{self._capture_q.qsize()} playback_q~{self._playback_q.qsize()}",
            )
        self._drain_audio_queues()
        self._ctx_in_16k.fill(0.0)

    def _resize_wave_buffers(self, maxlen: int) -> None:
        m = max(1024, int(maxlen))
        self._input_wave = deque(self._input_wave, maxlen=m)
        self._output_wave = deque(self._output_wave, maxlen=m)

    def _apply_output_block_crossfade(self, out: np.ndarray, sr: int) -> np.ndarray:
        """Blend start of ``out`` with end of previous block to avoid step discontinuities."""
        out = np.array(out, dtype=np.float32, copy=True)
        n = int(out.shape[0])
        if n <= 0:
            return out
        h = min(max(32, sr // 250), 160, n)
        prev = self._out_xfade_tail
        if prev is not None and prev.shape[0] == h:
            t = np.linspace(0.0, 1.0, h, dtype=np.float32)
            w = 0.5 - 0.5 * np.cos(np.pi * t)
            out[:h] = (1.0 - w) * prev + w * out[:h]
        self._out_xfade_tail = np.asarray(out[-h:], dtype=np.float32).copy()
        return out

    def _denoise_16k_overlap_add(self, audio_16k: np.ndarray) -> np.ndarray:
        """
        Prepend rolling input context, denoise the longer segment, return only the
        tail aligned with ``audio_16k`` so STFT boundaries match continuous audio.
        """
        ol = CHUNK_CONTEXT_SAMPLES_16K
        n = int(len(audio_16k))
        if n <= 0:
            return audio_16k.astype(np.float32, copy=False)
        ctx = self._ctx_in_16k
        combined = np.concatenate([ctx, np.asarray(audio_16k, dtype=np.float32)])
        den_full = self.model.denoise(combined)
        if len(den_full) < ol + n:
            den_full = np.pad(np.asarray(den_full, dtype=np.float32), (0, ol + n - len(den_full)))
        out = np.array(den_full[ol : ol + n], dtype=np.float32, copy=True)
        ctx[:] = np.asarray(combined[-ol:], dtype=np.float32)
        return out

    # --- processing worker (split-stream pipeline) ---

    def _resample_native_to_16k(self, audio_native: np.ndarray, sr: int) -> np.ndarray:
        if sr == MODEL_SAMPLE_RATE:
            return np.asarray(audio_native, dtype=np.float32)
        g = math.gcd(sr, MODEL_SAMPLE_RATE)
        up, down = MODEL_SAMPLE_RATE // g, sr // g
        return resample_poly(audio_native, up, down).astype(np.float32, copy=False)

    def _resample_16k_to_native(self, audio_16k: np.ndarray, sr: int, n: int) -> np.ndarray:
        if sr == MODEL_SAMPLE_RATE:
            clean = np.asarray(audio_16k, dtype=np.float32)
        else:
            g = math.gcd(MODEL_SAMPLE_RATE, sr)
            up, down = sr // g, MODEL_SAMPLE_RATE // g
            clean = resample_poly(audio_16k, up, down).astype(np.float32, copy=False)
        if len(clean) > n:
            return clean[:n]
        if len(clean) < n:
            return np.pad(clean, (0, n - len(clean)))
        return clean

    @staticmethod
    def _pick_transcription_source(
        original: np.ndarray,
        out: np.ndarray,
        denoise_on: bool,
        model_ok: bool,
        use_denoised: bool,
    ) -> np.ndarray:
        """Prefer capture input for ASR; denoised audio is often too quiet for Whisper."""
        if not (denoise_on and model_ok and use_denoised):
            return original
        r_in = segment_rms(original)
        r_out = segment_rms(out)
        if r_out < max(_TX_SILENCE_RMS * 4.0, r_in * 0.2):
            return original
        return out

    def _denoise_native_block(self, audio_native: np.ndarray, sr: int) -> np.ndarray:
        """Denoise one native-rate segment; length matches ``audio_native``."""
        audio_16k = self._resample_native_to_16k(audio_native, sr)
        clean_16k = self._denoise_16k_overlap_add(audio_16k)
        return self._resample_16k_to_native(clean_16k, sr, len(audio_native))

    def _trim_capture_queue(self, keep: int = 12) -> int:
        """Drop oldest mic blocks so denoise tracks recent audio, not seconds-old queue."""
        dropped = 0
        while self._capture_q.qsize() > keep:
            try:
                self._capture_q.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    def _push_playback_blocks(self, audio: np.ndarray, sr: int) -> None:
        """Slice processed audio into IO-sized blocks for the output callback."""
        io_n = self._io_block_samples(sr)
        for start in range(0, len(audio), io_n):
            blk = np.asarray(audio[start : start + io_n], dtype=np.float32)
            if len(blk) < io_n:
                break
            try:
                self._playback_q.put(blk, timeout=0.10)
            except queue.Full:
                try:
                    self._playback_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._playback_q.put_nowait(blk)
                except queue.Full:
                    break

    def _proc_loop(self) -> None:
        """
        Accumulate small capture blocks, denoise INFER_CHUNK_MS segments, enqueue
        IO_BLOCK_MS playback pieces. Matches the split-stream DeepFilterNet pattern.
        """
        accum = np.empty(0, dtype=np.float32)

        while self._proc_running:
            try:
                chunk = self._capture_q.get(timeout=0.05)
            except queue.Empty:
                continue

            accum = np.concatenate((accum, chunk))

            with self._lock:
                sr = int(self._stream_sample_rate)

            infer_n = self._infer_chunk_samples(sr)

            while len(accum) >= infer_n:
                dropped = self._trim_capture_queue(_CAPTURE_QUEUE_TRIM)
                if dropped:
                    diag = get_audio_diag()
                    if diag is not None:
                        diag.event("capture_trim", f"dropped={dropped} q={self._capture_q.qsize()}")

                # Worker fell behind: discard oldest buffered audio (avoid playing stale dry).
                max_accum = infer_n * 3
                if len(accum) > max_accum:
                    accum = accum[-max_accum:]
                    diag = get_audio_diag()
                    if diag is not None:
                        diag.event("accum_trim", f"len={len(accum)}")

                original = accum[:infer_n].copy()
                accum = accum[infer_n:]

                with self._lock:
                    denoise_on = self._denoise_active
                    strength = self._strength
                    model_ok = self.model.is_ready

                path = "passthrough"
                if denoise_on and model_ok:
                    try:
                        enhanced = self._denoise_native_block(original, sr)
                        with self._lock:
                            self._latency_ms = float(self.model.last_inference_ms)
                        out = original * (1.0 - strength) + enhanced * strength
                        path = "denoise"
                        diag = get_audio_diag()
                        if diag is not None:
                            diag.record_infer_done(
                                infer_ms=float(self.model.last_inference_ms),
                                infer_q_after=self._capture_q.qsize(),
                                clean_q_after=self._playback_q.qsize(),
                                clean_dropped=False,
                                native_samples=len(original),
                                sr=sr,
                            )
                    except Exception:
                        logger.exception("[proc] Denoise failed in worker thread")
                        diag = get_audio_diag()
                        if diag is not None:
                            diag.event("infer_exception", str(original.shape))
                        with self._lock:
                            self._error_message = "Denoising error — check log."
                        out = original
                        path = "infer_error"
                else:
                    out = original
                    if not denoise_on:
                        with self._lock:
                            self._latency_ms = 0.0

                out = np.clip(out, -1.0, 1.0)
                self._push_playback_blocks(out, sr)

                tx = self.transcription
                if tx is not None and tx.active and not tx.paused:
                    with self._lock:
                        use_denoised = self._transcribe_denoised
                    src = self._pick_transcription_source(
                        original, out, denoise_on, model_ok, use_denoised
                    )
                    try:
                        audio_16k = self._resample_native_to_16k(src, sr)
                        tx.feed_16k(audio_16k)
                    except Exception:
                        logger.exception("[proc] Transcription feed failed")

                diag = get_audio_diag()
                if diag is not None and path != "denoise":
                    diag.record_callback(
                        path=path,
                        frames=infer_n,
                        sr=sr,
                        infer_q=self._capture_q.qsize(),
                        clean_q=self._playback_q.qsize(),
                        pending_dry=0,
                        infer_ms=0.0,
                        callback_us=0.0,
                        pa_status="",
                        strength=strength,
                    )

    # --- thread-safe properties ---

    @property
    def chunk_ms(self) -> int:
        """Processing chunk size in milliseconds (denoise batch, not I/O block)."""
        return int(INFER_CHUNK_MS)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._denoise_active

    @active.setter
    def active(self, value: bool) -> None:
        with self._lock:
            was = self._denoise_active
            self._denoise_active = bool(value)
        if was and not self._denoise_active:
            self._flush_denoise_pipeline()

    @property
    def strength(self) -> float:
        with self._lock:
            return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        with self._lock:
            self._strength = float(max(0.0, min(1.0, value)))

    @property
    def input_level(self) -> float:
        with self._lock:
            return self._input_level

    @property
    def output_level(self) -> float:
        with self._lock:
            return self._output_level

    @property
    def latency_ms(self) -> float:
        with self._lock:
            return self._latency_ms

    @property
    def error_message(self) -> str | None:
        with self._lock:
            return self._error_message

    @error_message.setter
    def error_message(self, msg: str | None) -> None:
        with self._lock:
            self._error_message = msg

    @property
    def stream_sample_rate(self) -> int:
        with self._lock:
            return self._stream_sample_rate

    @property
    def streaming(self) -> bool:
        return self._in_stream is not None

    def clear_error(self) -> None:
        with self._lock:
            self._error_message = None

    def get_wave_snapshot(self) -> tuple[list[float], list[float]]:
        """Copy current ring buffers for GUI thread (approx. 1 s at stream rate)."""
        with self._lock:
            return list(self._input_wave), list(self._output_wave)

    # --- devices ---

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        devices = sd.query_devices()
        defaults = sd.default.device
        default_in = defaults[0] if defaults is not None else None
        default_out = defaults[1] if defaults is not None else None
        hostapis = sd.query_hostapis()
        out: list[dict[str, Any]] = []
        for i, d in enumerate(devices):
            api_idx = int(d["hostapi"])
            api_full = str(hostapis[api_idx]["name"])
            out.append(
                {
                    "index": i,
                    "name": str(d["name"]),
                    "hostapi": api_idx,
                    "api": api_full,
                    "api_short": _hostapi_short_tag(api_full),
                    "is_input": int(d["max_input_channels"]) > 0,
                    "is_output": int(d["max_output_channels"]) > 0,
                    "is_default": (i == default_in or i == default_out),
                }
            )
        return out

    @staticmethod
    def find_vbcable_playback() -> int | None:
        """CABLE Input — the playback device apps send audio INTO.
        Set this as Windows default output so all audio flows through the cable."""
        for d in AudioEngine.list_devices():
            if d["is_output"] and ("CABLE Input" in d["name"] or "VB-Audio Virtual Cable" in d["name"]):
                return int(d["index"])
        return None

    # Keep old name as alias so nothing breaks.
    find_vbcable = find_vbcable_playback

    @staticmethod
    def find_real_output() -> int | None:
        """
        Return the index of a real hardware output device (speakers / headphones),
        explicitly skipping VB-Cable and other virtual devices.

        Needed when CABLE Input is set as the Windows default playback device —
        in that case sd.default.device[1] resolves to the cable, so we must pick
        the hardware output explicitly.

        Priority:
          1. The current Windows default output — if it is NOT a virtual device
          2. Any output device whose name suggests real hardware (Speakers, Headphones,
             Realtek, HD Audio, Analog, Digital, HDMI) that is not VB-Audio / CABLE
        """
        _VIRTUAL = ("cable", "vb-audio", "voicemeeter", "virtual")
        _REAL_HINTS = ("speaker", "headphone", "headset", "realtek", "hd audio",
                       "analog", "digital", "hdmi", "display audio", "intel")

        devices = AudioEngine.list_devices()
        devmap  = {d["index"]: d for d in devices}

        # Check Windows default output first.
        defaults = sd.default.device
        if defaults is not None and defaults[1] is not None:
            default_out = devmap.get(int(defaults[1]))
            if default_out and default_out["is_output"]:
                n = default_out["name"].lower()
                if not any(v in n for v in _VIRTUAL):
                    logger.info(
                        "[engine] real output (default): [%s] %s",
                        default_out["index"], default_out["name"],
                    )
                    return int(default_out["index"])

        # Search for hardware-sounding names.
        for hint in _REAL_HINTS:
            for d in devices:
                if not d["is_output"]:
                    continue
                n = d["name"].lower()
                if any(v in n for v in _VIRTUAL):
                    continue
                if hint in n:
                    logger.info(
                        "[engine] real output (by name): [%s] %s",
                        d["index"], d["name"],
                    )
                    return int(d["index"])

        # Last resort: first non-virtual output.
        for d in devices:
            if not d["is_output"]:
                continue
            if not any(v in d["name"].lower() for v in _VIRTUAL):
                logger.info(
                    "[engine] real output (fallback): [%s] %s",
                    d["index"], d["name"],
                )
                return int(d["index"])

        logger.warning("[engine] Could not find any real hardware output device.")
        return None

    @staticmethod
    def find_vbcable_capture() -> int | None:
        """CABLE Output — the recording device that captures what went into the cable.
        Use this as ClearVoice input to intercept all audio before the speakers."""
        for d in AudioEngine.list_devices():
            if d["is_input"] and "CABLE Output" in d["name"]:
                logger.info(
                    "[engine] VB-Cable capture found: [%s] %s (%s)",
                    d["index"], d["name"], d.get("api_short", ""),
                )
                return int(d["index"])
        return None

    @staticmethod
    def find_loopback_input() -> int | None:
        """
        Find the best capture device to intercept all system audio.

        Priority:
          1. CABLE Output  (VB-Cable recording end — true interception, no doubling)
          2. Stereo Mix / Wave Out Mix / What U Hear  (copy-capture, original still plays)
          3. Any "loopback" named device
        Returns the device index, or None when nothing is found.
        """
        # 1. VB-Cable Output — proper intercept
        vbc = AudioEngine.find_vbcable_capture()
        if vbc is not None:
            return vbc

        # 2. Stereo Mix and friends — captures a copy; original still plays alongside
        _PRIORITY = ["stereo mix", "wave out mix", "what u hear", "loopback", " mix"]
        devices = AudioEngine.list_devices()
        for keyword in _PRIORITY:
            for d in devices:
                if d["is_input"] and keyword in d["name"].lower():
                    logger.info(
                        "[engine] loopback input found: [%s] %s (%s) "
                        "[NOTE: original audio still plays — use VB-Cable for true intercept]",
                        d["index"], d["name"], d.get("api_short", ""),
                    )
                    return int(d["index"])

        logger.warning(
            "[engine] No capture device found. "
            "Install VB-Cable from https://vb-audio.com/Cable/ for full audio intercept, "
            "or enable Stereo Mix in Windows Sound settings."
        )
        return None

    def _pick_stream_sample_rate(self) -> int:
        """Prefer input device native rate; fall back to 16 kHz."""
        try:
            idx = self._input_device
            if idx is None:
                di = sd.query_devices(kind="input")
            else:
                di = sd.query_devices(idx)
            sr = float(di["default_samplerate"])
            if sr <= 0 or sr > 384000:
                return MODEL_SAMPLE_RATE
            return int(round(sr))
        except Exception:
            logger.exception("Could not query input default_samplerate")
            return MODEL_SAMPLE_RATE

    @staticmethod
    def _query_dev(idx: int | None, kind: str) -> dict[str, Any] | None:
        if idx is None:
            try:
                return sd.query_devices(kind=kind)
            except Exception:
                return None
        try:
            return sd.query_devices(idx)
        except Exception:
            return None

    @staticmethod
    def _paren_inside(name: str) -> str:
        """Lowercased text inside the last '( ... )' suffix, e.g. driver / bus label."""
        s = str(name).strip()
        if "(" not in s or ")" not in s:
            return ""
        return s[s.rfind("(") + 1 : s.rfind(")")].strip().lower()

    @staticmethod
    def _before_paren(name: str) -> str:
        s = str(name).strip()
        i = s.find("(")
        return s[:i].strip().lower() if i != -1 else s.lower()

    @staticmethod
    def _is_loopback_device(name: str) -> bool:
        """True for Stereo Mix / Wave Out Mix / loopback-style capture devices.
        These must never be remapped to WASAPI — WASAPI treats them as render
        loopback endpoints (AUDCLNT_E_WRONG_ENDPOINT_TYPE) and cannot open them
        as normal capture devices in a duplex stream."""
        n = name.lower()
        return any(k in n for k in ("stereo mix", "wave out mix", "what u hear", "loopback", " mix"))

    @staticmethod
    def _vendorish_similar(a: str, b: str) -> bool:
        """True if two parenthetical driver strings likely refer to the same hardware."""
        if not a or not b:
            return False
        if a == b:
            return True
        if len(a) >= 8 and (a in b or b in a):
            return True
        if len(b) >= 8 and (b in a or a in b):
            return True
        # token overlap (e.g. "realtek" "high" "definition")
        ta = {t for t in a.replace(",", " ").split() if len(t) > 2}
        tb = {t for t in b.replace(",", " ").split() if len(t) > 2}
        if not ta or not tb:
            return False
        inter = ta & tb
        return len(inter) >= 2 or ("realtek" in ta and "realtek" in tb) or ("vb-audio" in ta and "vb-audio" in tb)

    @staticmethod
    def _remap_duplex_for_same_hostapi(
        input_device: int | None, output_device: int | None
    ) -> tuple[int | None, int | None, str | None]:
        """
        PortAudio duplex on Windows often fails (PaErrorCode -9993) when input and
        output use different host APIs. Try alternate device indices on the other
        API: exact name first, then same driver text in parentheses (WDM-KS vs
        DirectSound often differs outside the parens).
        """
        di = AudioEngine._query_dev(input_device, "input")
        do = AudioEngine._query_dev(output_device, "output")
        if di is None or do is None:
            return input_device, output_device, None
        hi = int(di["hostapi"])
        ho = int(do["hostapi"])
        if hi == ho:
            return input_device, output_device, None

        devices = sd.query_devices()
        out_name = str(do["name"])
        in_name = str(di["name"])
        out_paren = AudioEngine._paren_inside(out_name)
        in_paren = AudioEngine._paren_inside(in_name)
        out_core = AudioEngine._before_paren(out_name)
        in_core = AudioEngine._before_paren(in_name)
        _GENERIC_CORE = frozenset(
            {"speakers", "headphones", "headset earphone", "headset", "line out", "output"}
        )

        for i, d in enumerate(devices):
            if int(d["max_output_channels"]) <= 0:
                continue
            if int(d["hostapi"]) == hi and str(d["name"]) == out_name:
                msg = (
                    f"Duplex remap: output device index {i} ({out_name!r}) matches input host API "
                    f"(was output index {output_device})."
                )
                return input_device, int(i), msg

        # Never remap a loopback/mix device to WASAPI — WASAPI rejects them with
        # AUDCLNT_E_WRONG_ENDPOINT_TYPE.  Move the OUTPUT to the input's API instead.
        if not AudioEngine._is_loopback_device(in_name):
            for i, d in enumerate(devices):
                if int(d["max_input_channels"]) <= 0:
                    continue
                if int(d["hostapi"]) == ho and str(d["name"]) == in_name:
                    msg = (
                        f"Duplex remap: input device index {i} ({in_name!r}) matches output host API "
                        f"(was input index {input_device})."
                    )
                    return int(i), output_device, msg

        # Fuzzy: same vendor / driver text in parentheses (KS vs DS / MME names differ).
        if out_paren and len(out_paren) >= 6:
            best_i: int | None = None
            best_dn: str = ""
            best_score = -1
            for i, d in enumerate(devices):
                if int(d["max_output_channels"]) <= 0 or int(d["hostapi"]) != hi:
                    continue
                dn = str(d["name"])
                pin = AudioEngine._paren_inside(dn)
                if not AudioEngine._vendorish_similar(out_paren, pin):
                    continue
                cin = AudioEngine._before_paren(dn)
                if cin != out_core and (out_core in _GENERIC_CORE or cin in _GENERIC_CORE):
                    pass
                elif cin != out_core and out_core not in _GENERIC_CORE:
                    continue
                score = 20
                if cin == out_core:
                    score += 30
                if dn.strip().lower() == out_name.strip().lower():
                    score += 50
                if score > best_score:
                    best_score, best_i, best_dn = score, int(i), dn
            if best_i is not None:
                msg = (
                    f"Duplex remap: output index {best_i} ({best_dn!r}) on input’s host API "
                    f"(matched driver label; was output index {output_device})."
                )
                return input_device, best_i, msg

        # Only fuzzy-remap the input if it is NOT a loopback device.
        if not AudioEngine._is_loopback_device(in_name) and in_paren and len(in_paren) >= 6:
            best_i: int | None = None
            best_dn: str = ""
            best_score = -1
            for i, d in enumerate(devices):
                if int(d["max_input_channels"]) <= 0 or int(d["hostapi"]) != ho:
                    continue
                dn = str(d["name"])
                pin = AudioEngine._paren_inside(dn)
                if not AudioEngine._vendorish_similar(in_paren, pin):
                    continue
                cin = AudioEngine._before_paren(dn)
                if cin != in_core and (in_core in _GENERIC_CORE or cin in _GENERIC_CORE):
                    pass
                elif cin != in_core and in_core not in _GENERIC_CORE:
                    continue
                score = 20
                if cin == in_core:
                    score += 30
                if dn.strip().lower() == in_name.strip().lower():
                    score += 50
                if score > best_score:
                    best_score, best_i, best_dn = score, int(i), dn
            if best_i is not None:
                msg = (
                    f"Duplex remap: input index {best_i} ({best_dn!r}) on output’s host API "
                    f"(matched driver label; was input index {input_device})."
                )
                return best_i, output_device, msg

        try:
            hi_name = str(sd.query_hostapis(hi)["name"])
            ho_name = str(sd.query_hostapis(ho)["name"])
        except Exception:
            hi_name, ho_name = str(hi), str(ho)

        hint = (
            f"Incompatible audio backends: input is on “{hi_name}”, output on “{ho_name}”. "
            "Pick another pair, or choose two list entries that use the same driver line "
            "(e.g. both WASAPI, both MME, or both “… (Realtek …)” under the same API column)."
        )
        return input_device, output_device, hint

    @staticmethod
    def _is_duplex_hostapi_error(exc: BaseException) -> bool:
        s = str(exc).lower()
        return "-9993" in s or "illegal combination" in s

    @staticmethod
    def _duplex_same_hostapi(input_device: int | None, output_device: int | None) -> bool:
        di = AudioEngine._query_dev(input_device, "input")
        do = AudioEngine._query_dev(output_device, "output")
        if di is None or do is None:
            return True
        return int(di["hostapi"]) == int(do["hostapi"])

    @staticmethod
    def _find_compatible_output(
        input_device: int | None, preferred_output: int | None
    ) -> tuple[int | None, str | None]:
        """
        Pick an output device index on the same host API as ``input_device``.
        Order: preferred if already compatible → VB-Cable on that API → system
        default output on that API → first output on that API.
        """
        di = AudioEngine._query_dev(input_device, "input")
        if di is None:
            return preferred_output, None
        hi = int(di["hostapi"])
        defaults = sd.default.device

        pref: int | None = preferred_output
        if pref is None and defaults is not None and defaults[1] is not None:
            pref = int(defaults[1])

        if pref is not None:
            try:
                if int(sd.query_devices(pref)["hostapi"]) == hi:
                    return pref, None
            except Exception:
                logger.exception("Could not query preferred output device %s", pref)

        _VIRTUAL = ("cable", "vb-audio", "voicemeeter", "virtual")
        in_is_vbcable = any(v in str(di["name"]).lower() for v in _VIRTUAL)

        devices = sd.query_devices()
        # When input is VB-Cable, never route output back to any virtual device.
        for i, d in enumerate(devices):
            if int(d["hostapi"]) != hi or int(d["max_output_channels"]) <= 0:
                continue
            n = str(d["name"])
            if in_is_vbcable and any(v in n.lower() for v in _VIRTUAL):
                continue
            if not in_is_vbcable and ("CABLE Input" in n or "VB-Audio" in n):
                return int(i), (
                    f"[duplex] output -> [{i}] {n!r} (VB-Audio on same API as input)"
                )

        if defaults is not None and defaults[1] is not None:
            doi = int(defaults[1])
            try:
                doi_dev = sd.query_devices(doi)
                doi_name = str(doi_dev["name"])
                if not (in_is_vbcable and any(v in doi_name.lower() for v in _VIRTUAL)):
                    if int(doi_dev["hostapi"]) == hi:
                        return doi, (
                            f"[duplex] output -> [{doi}] {doi_name!r} "
                            "(system default on same API as input)"
                        )
            except Exception:
                logger.exception("Could not query default output for duplex fix")

        for i, d in enumerate(devices):
            if int(d["hostapi"]) != hi or int(d["max_output_channels"]) <= 0:
                continue
            n = str(d["name"])
            if in_is_vbcable and any(v in n.lower() for v in _VIRTUAL):
                continue
            return int(i), (
                f"[duplex] output -> [{i}] {n!r} "
                "(first non-virtual playback device on input's API)"
            )

        return preferred_output, None

    def get_device_indices(self) -> tuple[int | None, int | None]:
        """Effective input/output indices last used or selected for the stream."""
        return (self._input_device, self._output_device)

    def sync_resolved_devices_to_settings(self, settings: Any) -> bool:
        """If the stream is running, persist effective indices when they differ (e.g. API fix)."""
        if not self.streaming:
            return False
        ri, ro = self.get_device_indices()
        changed = False
        if ri != settings.input_device_index:
            settings.input_device_index = ri
            changed = True
        if ro != settings.output_device_index:
            settings.output_device_index = ro
            changed = True
        if changed:
            settings.save()
        return changed

    def start(self, input_device: int | None, output_device: int | None) -> None:
        self.stop()
        self._input_device = input_device
        self._output_device = output_device
        self.clear_error()
        duplex_note: str | None = None

        # When input is a VB-Cable device, skip name-based remapping entirely —
        # the remap would otherwise find CABLE Input (same API, "VB-Audio" name)
        # and route denoised audio back into the cable instead of the speakers.
        _VIRTUAL = ("cable", "vb-audio", "voicemeeter", "virtual")
        in_dev_info = self._query_dev(self._input_device, "input")
        in_is_vbcable = in_dev_info is not None and any(
            v in str(in_dev_info["name"]).lower() for v in _VIRTUAL
        )

        if in_is_vbcable:
            # Force output to real hardware, ignoring whatever was passed in.
            real_out = self.find_real_output()
            if real_out is not None:
                logger.info(
                    "[start] VB-Cable input detected — forcing output to real speakers [%s] %s",
                    real_out, sd.query_devices(real_out)["name"],
                )
                self._output_device = real_out
        else:
            in_dev, out_dev, duplex_note = self._remap_duplex_for_same_hostapi(
                self._input_device, self._output_device
            )
            if duplex_note and (in_dev != self._input_device or out_dev != self._output_device):
                logger.info("%s", duplex_note)
            self._input_device, self._output_device = in_dev, out_dev

            if not self._duplex_same_hostapi(self._input_device, self._output_device):
                out_fixed, fix_msg = self._find_compatible_output(
                    self._input_device, self._output_device
                )
                if fix_msg:
                    logger.info("%s", fix_msg)
                self._output_device = out_fixed

        if not self._duplex_same_hostapi(self._input_device, self._output_device):
            logger.warning(
                "[start] Input and output use different host APIs — OK with split streams, "
                "but device latency may differ."
            )

        sr_primary = self._pick_stream_sample_rate()
        # Prefer 16 kHz (model native rate) to avoid double resampling and improve quality.
        # Fall back to device native / 44100 for hardware that rejects 16 kHz.
        rates: list[int] = []
        for _sr in (MODEL_SAMPLE_RATE, sr_primary, 44100, 48000):
            if _sr not in rates:
                rates.append(_sr)

        last_err: Exception | None = None
        for sr in rates:
            io_block = self._io_block_samples(sr)
            in_stream: sd.InputStream | None = None
            out_stream: sd.OutputStream | None = None
            try:
                in_stream = sd.InputStream(
                    device=self._input_device,
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    blocksize=io_block,
                    callback=self._in_callback,
                    latency="high",
                )
                out_stream = sd.OutputStream(
                    device=self._output_device,
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    blocksize=io_block,
                    callback=self._out_callback,
                    latency="high",
                )
                with self._lock:
                    self._stream_sample_rate = sr
                    self._resize_wave_buffers(sr)
                self._start_proc_thread()
                in_stream.start()
                out_stream.start()
                self._in_stream = in_stream
                self._out_stream = out_stream
                infer_n = self._infer_chunk_samples(sr)
                logger.info(
                    "Audio streams started — input=%s output=%s sr=%s io_block=%s infer_chunk=%s",
                    self._input_device,
                    self._output_device,
                    sr,
                    io_block,
                    infer_n,
                )
                diag = get_audio_diag()
                if diag is not None:
                    io_ms = 1000.0 * io_block / sr if sr else 0.0
                    infer_ms = 1000.0 * infer_n / sr if sr else 0.0
                    diag.log_line(
                        logging.INFO,
                        "stream_started in=%s out=%s sr=%d io_block_samples=%d io_ms=%.1f "
                        "infer_chunk_samples=%d infer_ms=%.1f queue_max=%d ctx_samples=%d device=%s",
                        self._input_device,
                        self._output_device,
                        sr,
                        io_block,
                        io_ms,
                        infer_n,
                        infer_ms,
                        _QUEUE_MAXSIZE,
                        CHUNK_CONTEXT_SAMPLES_16K,
                        getattr(self.model, "inference_device", "?"),
                    )
                return
            except Exception as e:
                last_err = e
                logger.warning("Stream open failed at sr=%s: %s", sr, e)
                self._proc_running = False
                if self._proc_thread is not None:
                    self._proc_thread.join(timeout=1.0)
                    self._proc_thread = None
                for stream in (in_stream, out_stream):
                    if stream is not None:
                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            logger.exception("Error closing failed stream object")
                self._in_stream = None
                self._out_stream = None

        msg = "Could not open audio streams at native or fallback sample rates."
        if last_err is not None:
            msg = f"{msg} ({last_err})"
        if duplex_note and "Incompatible" in duplex_note:
            msg = f"{duplex_note}\n\n{msg}"
        self.error_message = msg
        logger.error("%s", msg)

    def _start_proc_thread(self) -> None:
        self._stop_proc_thread()
        self._drain_audio_queues()
        self._ctx_in_16k.fill(0.0)
        self._playback_ready = False
        self._proc_running = True
        t = threading.Thread(target=self._proc_loop, daemon=True, name="clearvoice-proc")
        t.start()
        self._proc_thread = t

    def _stop_proc_thread(self) -> None:
        self._proc_running = False
        if self._proc_thread is not None and self._proc_thread.is_alive():
            self._proc_thread.join(timeout=2.0)
        self._proc_thread = None

    def stop(self) -> None:
        had_streams = self._in_stream is not None or self._out_stream is not None
        for attr in ("_in_stream", "_out_stream"):
            stream = getattr(self, attr, None)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    logger.exception("Error stopping audio stream")
                setattr(self, attr, None)
        self._stop_proc_thread()
        self._flush_denoise_pipeline()
        if had_streams:
            logger.info("Audio streams stopped")
            diag = get_audio_diag()
            if diag is not None:
                diag.event("stream_stopped")
        self._out_xfade_tail = None
        self._playback_ready = False

    def restart(self) -> None:
        self.start(self._input_device, self._output_device)

    def _in_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Capture microphone audio and enqueue for the processing thread."""
        if status:
            logger.debug("[audio] input callback status: %s", status)
        try:
            if indata.shape[1] < 1:
                return
            audio = indata[:, 0].copy()
            self._input_level = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
            with self._lock:
                self._input_wave.extend(audio.tolist())
            if self._capture_q.full():
                try:
                    self._capture_q.get_nowait()
                except queue.Empty:
                    pass
                diag = get_audio_diag()
                if diag is not None:
                    diag.event("capture_q_drop", f"capture_q={self._capture_q.qsize()}")
            try:
                self._capture_q.put_nowait(audio)
            except queue.Full:
                pass
        except Exception:
            logger.exception("Fatal error in input callback")
            diag = get_audio_diag()
            if diag is not None:
                diag.event("in_callback_exception")

    def _out_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Feed processed audio to the output device."""
        t_cb0 = time.perf_counter()
        pa_s = self._pa_status_str(status)
        if status:
            logger.debug("[audio] output callback status: %s", status)

        path = "silence"
        try:
            with self._lock:
                sr = int(self._stream_sample_rate)
                strength = self._strength
                infer_ms = float(self.model.last_inference_ms)

            q_depth = self._playback_q.qsize()
            if not self._playback_ready:
                if q_depth >= _PLAYBACK_PREROLL_BLOCKS:
                    self._playback_ready = True
                else:
                    outdata.fill(0.0)
                    path = "preroll"
                    out = np.zeros(int(frames), dtype=np.float32)
                    out = self._apply_output_block_crossfade(out, sr)
                    outdata[:, 0] = out
                    with self._lock:
                        self._output_wave.extend(out.tolist())
                    diag = get_audio_diag()
                    if diag is not None:
                        diag.record_callback(
                            path=path,
                            frames=int(frames),
                            sr=sr,
                            infer_q=self._capture_q.qsize(),
                            clean_q=q_depth,
                            pending_dry=0,
                            infer_ms=infer_ms,
                            callback_us=(time.perf_counter() - t_cb0) * 1e6,
                            pa_status=pa_s,
                            strength=strength,
                        )
                    return

            try:
                blk = self._playback_q.get_nowait()
                n = min(len(blk), frames)
                outdata[:n, 0] = blk[:n]
                if n < frames:
                    outdata[n:, 0] = 0.0
                path = "play"
            except queue.Empty:
                outdata.fill(0.0)
                path = "underrun"

            out = np.array(outdata[:, 0], dtype=np.float32, copy=True)
            out = self._apply_output_block_crossfade(out, sr)
            outdata[:, 0] = out
            self._output_level = float(np.sqrt(np.mean(np.square(out)) + 1e-12))

            with self._lock:
                self._output_wave.extend(out.tolist())

            diag = get_audio_diag()
            if diag is not None:
                diag.record_callback(
                    path=path,
                    frames=int(frames),
                    sr=sr,
                    infer_q=self._capture_q.qsize(),
                    clean_q=self._playback_q.qsize(),
                    pending_dry=0,
                    infer_ms=infer_ms,
                    callback_us=(time.perf_counter() - t_cb0) * 1e6,
                    pa_status=pa_s,
                    strength=strength,
                )
        except Exception:
            logger.exception("Fatal error in output callback")
            diag = get_audio_diag()
            if diag is not None:
                diag.event("out_callback_exception")
            with self._lock:
                self._error_message = "Audio callback error — check log."
            try:
                outdata.fill(0.0)
            except Exception:
                pass
