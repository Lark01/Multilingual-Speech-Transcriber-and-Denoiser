"""
AudioEngine — captures microphone audio, accumulates it into segments,
and transcribes each segment using TranscriberModel in a background thread.
"""
import numpy as np
import sounddevice as sd
import threading
import time
from collections import deque
from model import TranscriberModel

SAMPLE_RATE     = 16_000   # Whisper requires 16 kHz
CHUNK_MS        = 100      # sounddevice block size (ms) — low latency capture
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_MS / 1000)

# How many seconds of audio to send to Whisper per inference call.
# Shorter = lower latency but more fragmented sentences.
SEGMENT_SEC     = 5
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SEC

# Silence threshold — chunks below this RMS are skipped
SILENCE_RMS     = 0.005


class AudioEngine:
    def __init__(self, model: TranscriberModel):
        self.model  = model
        self.active = True          # when False the mic is muted / paused

        self._stream   = None
        self._lock     = threading.Lock()

        # rolling buffer for incoming audio
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0

        # worker thread for inference
        self._infer_queue: deque = deque()
        self._infer_thread  = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_running = False

        # stats / results exposed to the GUI
        self.input_level   = 0.0
        self.latency_ms    = 0.0
        self.last_text     = ""
        self.transcript_lines: list[str] = []
        self.on_new_text   = None   # optional callback(text: str)

        # device index — set before start()
        self.input_device  = None   # None = system default mic

    # ------------------------------------------------------------------
    def list_devices(self):
        """Return list of (index, name) for all INPUT audio devices."""
        devices = sd.query_devices()
        return [(i, d["name"]) for i, d in enumerate(devices)
                if d["max_input_channels"] > 0]

    # ------------------------------------------------------------------
    def start(self):
        if self._stream is not None:
            return
        self._infer_running = True
        self._infer_thread.start()
        self._stream = sd.InputStream(
            samplerate  = SAMPLE_RATE,
            blocksize   = CHUNK_SAMPLES,
            dtype       = "float32",
            channels    = 1,
            device      = self.input_device,
            callback    = self._callback,
            latency     = "low",
        )
        self._stream.start()
        print(f"[audio] stream started — input={self.input_device}")

    def stop(self):
        self._infer_running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[audio] stream stopped")

    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice on every audio chunk (audio thread)."""
        if status:
            print(f"[audio] {status}")

        audio = indata[:, 0].copy()   # mono, float32
        self.input_level = float(np.abs(audio).mean())

        if not self.active:
            return

        with self._lock:
            self._buffer.append(audio)
            self._buffer_samples += len(audio)

            if self._buffer_samples >= SEGMENT_SAMPLES:
                segment = np.concatenate(self._buffer)
                self._buffer = []
                self._buffer_samples = 0
                self._infer_queue.append(segment)

    # ------------------------------------------------------------------
    def _infer_worker(self):
        """Background thread: drains the queue and runs Whisper."""
        while self._infer_running:
            if not self._infer_queue:
                time.sleep(0.02)
                continue

            segment = self._infer_queue.popleft()

            # skip near-silent segments
            if float(np.abs(segment).mean()) < SILENCE_RMS:
                continue

            t0   = time.perf_counter()
            text = self.model.transcribe(segment, sample_rate=SAMPLE_RATE)
            self.latency_ms = (time.perf_counter() - t0) * 1000

            if text:
                self.last_text = text
                self.transcript_lines.append(text)
                if len(self.transcript_lines) > 200:          # cap history
                    self.transcript_lines = self.transcript_lines[-200:]
                if self.on_new_text:
                    self.on_new_text(text)
