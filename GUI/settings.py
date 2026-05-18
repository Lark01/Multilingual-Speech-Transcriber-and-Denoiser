"""Persistent JSON settings next to the application directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).resolve().parent
_SETTINGS_PATH = _APP_DIR / "settings.json"


def _default_inference_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


_DEFAULTS: dict[str, Any] = {
    "weights_path": str(_APP_DIR / "weights" / "denoiser_epoch_15.pt"),
    "denoise_active": True,
    "input_device_index": None,
    "output_device_index": None,
    "strength": 1.0,
    "inference_device": _default_inference_device(),
    "window_x": None,
    "window_y": None,
    "start_minimized": False,
    "launch_on_startup": False,
    "hide_vbcable_warning": False,
    "whisper_weights_path": str(_APP_DIR / "whisper-lora-weights"),
    "transcribe_active": False,
    "transcription_language": None,
    "transcription_segment_sec": 2.0,
    "transcription_hop_sec": 1.0,
    "transcribe_denoised_audio": False,
    "file_transcribe_denoise_first": False,
    "transcription_inference_device": "auto",
}


class Settings:
    """Load/save ClearVoice settings."""

    weights_path: str = _DEFAULTS["weights_path"]
    denoise_active: bool = True
    input_device_index: int | None = None
    output_device_index: int | None = None
    strength: float = 1.0
    inference_device: str = _default_inference_device()
    window_x: int | None = None
    window_y: int | None = None
    start_minimized: bool = False
    launch_on_startup: bool = False
    hide_vbcable_warning: bool = False
    whisper_weights_path: str = _DEFAULTS["whisper_weights_path"]
    transcribe_active: bool = False
    transcription_language: str | None = None
    transcription_segment_sec: float = 2.0
    transcription_hop_sec: float = 1.0
    transcribe_denoised_audio: bool = False
    file_transcribe_denoise_first: bool = False
    transcription_inference_device: str = "auto"

    @classmethod
    def load(cls) -> Settings:
        s = cls()
        if not _SETTINGS_PATH.is_file():
            logger.info("No settings file at %s — using defaults", _SETTINGS_PATH)
            return s
        try:
            raw = _SETTINGS_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            logger.exception("Failed to read settings from %s", _SETTINGS_PATH)
            return s
        for key in _DEFAULTS:
            if key not in data:
                continue
            val = data[key]
            try:
                if key in ("input_device_index", "output_device_index", "window_x", "window_y"):
                    setattr(s, key, val if val is None else int(val))
                elif key == "strength":
                    setattr(s, key, float(val))
                elif key in (
                    "denoise_active",
                    "start_minimized",
                    "launch_on_startup",
                    "hide_vbcable_warning",
                    "transcribe_active",
                    "transcribe_denoised_audio",
                    "file_transcribe_denoise_first",
                ):
                    setattr(s, key, bool(val))
                elif key in ("transcription_segment_sec", "transcription_hop_sec"):
                    setattr(s, key, float(val))
                elif key == "transcription_language":
                    setattr(s, key, None if val is None or val == "" else str(val))
                elif key == "whisper_weights_path":
                    setattr(s, key, str(val))
                elif key == "transcription_inference_device":
                    v = str(val).lower()
                    setattr(s, key, v if v in ("auto", "cpu", "cuda") else "auto")
                elif key == "inference_device":
                    v = str(val).lower()
                    setattr(s, key, v if v in ("cpu", "cuda") else _default_inference_device())
                elif key == "weights_path":
                    setattr(s, key, str(val))
            except (TypeError, ValueError) as e:
                logger.warning("Invalid settings value for %s: %r (%s)", key, val, e)
        return s

    def save(self) -> None:
        data = {
            "weights_path": self.weights_path,
            "denoise_active": self.denoise_active,
            "input_device_index": self.input_device_index,
            "output_device_index": self.output_device_index,
            "strength": self.strength,
            "inference_device": self.inference_device,
            "window_x": self.window_x,
            "window_y": self.window_y,
            "start_minimized": self.start_minimized,
            "launch_on_startup": self.launch_on_startup,
            "hide_vbcable_warning": self.hide_vbcable_warning,
            "whisper_weights_path": self.whisper_weights_path,
            "transcribe_active": self.transcribe_active,
            "transcription_language": self.transcription_language,
            "transcription_segment_sec": self.transcription_segment_sec,
            "transcription_hop_sec": self.transcription_hop_sec,
            "transcribe_denoised_audio": self.transcribe_denoised_audio,
            "file_transcribe_denoise_first": self.file_transcribe_denoise_first,
            "transcription_inference_device": self.transcription_inference_device,
        }
        try:
            _SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("Saved settings to %s", _SETTINGS_PATH)
        except Exception:
            logger.exception("Failed to save settings to %s", _SETTINGS_PATH)


def resolve_transcription_device(denoiser_device: str, setting: str = "auto") -> str:
    """Use CPU for Whisper when denoiser uses CUDA to avoid GPU contention (unless forced)."""
    s = (setting or "auto").strip().lower()
    if s in ("cpu", "cuda"):
        return s
    d = (denoiser_device or "cpu").strip().lower()
    return "cpu" if d == "cuda" else d
