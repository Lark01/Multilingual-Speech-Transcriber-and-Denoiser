"""Shared audio helpers (no imports from engine/playback to avoid cycles)."""

from __future__ import annotations

import numpy as np

SILENCE_RMS = 0.0008


def segment_rms(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float32).ravel()
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))
