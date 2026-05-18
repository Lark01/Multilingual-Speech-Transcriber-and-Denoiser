"""
TranscriberModel — Whisper large-v3-turbo + LoRA adapter for local transcription.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from model import resolve_inference_device

logger = logging.getLogger(__name__)

BASE_MODEL = "openai/whisper-large-v3-turbo"

_ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
)


def validate_whisper_adapter_dir(adapter_dir: Path) -> str | None:
    """
    Return None if the folder looks loadable; otherwise a human-readable error.
    """
    adapter_dir = adapter_dir.expanduser().resolve()
    if not adapter_dir.is_dir():
        return f"Whisper LoRA folder not found: {adapter_dir}"
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.is_file():
        return (
            f"Missing adapter_config.json in {adapter_dir}. "
            "Point whisper_weights_path at your LoRA adapter folder."
        )
    if not any((adapter_dir / name).is_file() for name in _ADAPTER_WEIGHT_NAMES):
        return (
            f"No adapter weights in {adapter_dir}. "
            f"Expected one of: {', '.join(_ADAPTER_WEIGHT_NAMES)}"
        )
    return None


def resolve_whisper_weights_path(
    cli_path: str | None,
    settings_path: str,
    app_dir: Path,
) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    p = Path(settings_path).expanduser()
    if p.is_dir():
        return p.resolve()
    default = app_dir / "whisper-lora-weights"
    if default.is_dir():
        return default.resolve()
    return p.resolve()


class TranscriberHolder:
    """Lazy-load Whisper + LoRA on first transcription use (thread-safe)."""

    def __init__(
        self,
        adapter_path: Path,
        device: str,
        language: str | None,
    ) -> None:
        self._adapter_path = adapter_path
        self._device = device
        self._language = language
        self._model: TranscriberModel | None = None
        self._lock = threading.Lock()

    @property
    def loaded_model(self) -> TranscriberModel | None:
        with self._lock:
            return self._model

    @property
    def is_ready(self) -> bool:
        m = self.loaded_model
        return m is not None and m.is_ready

    def get(self) -> TranscriberModel:
        with self._lock:
            if self._model is None:
                logger.info("[transcriber] Loading from %s …", self._adapter_path)
                self._model = TranscriberModel(
                    self._adapter_path,
                    device=self._device,
                    language=self._language,
                )
            return self._model


class LoadTranscriberThread(QThread):
    """Load Whisper + LoRA off the GUI thread (first run may download ~1.6 GB)."""

    status = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, holder: TranscriberHolder, parent=None) -> None:
        super().__init__(parent)
        self._holder = holder

    def run(self) -> None:
        err = validate_whisper_adapter_dir(self._holder._adapter_path)
        if err:
            self.failed.emit(err)
            return
        try:
            self.status.emit(
                "Loading Whisper (first run downloads ~1.6 GB from Hugging Face — keep app open)…"
            )
            model = self._holder.get()
            if not model.is_ready:
                self.failed.emit(model.load_error or "Transcriber failed to load.")
                return
            self.status.emit(f"Whisper ready ({model.inference_device})")
            self.finished_ok.emit(model)
        except Exception as e:
            logger.exception("[transcriber] Background load failed")
            self.failed.emit(str(e))


class TranscriberModel:
    """Loads Whisper base + PEFT LoRA; exposes transcribe(audio, sr)."""

    def __init__(
        self,
        adapter_path: str | Path,
        device: str = "cpu",
        language: str | None = None,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.language = language or None
        self.load_error: str | None = None
        self.last_inference_ms: float = 0.0
        self.processor: WhisperProcessor | None = None
        self._model: PeftModel | None = None

        err = validate_whisper_adapter_dir(self.adapter_path)
        if err:
            self.load_error = err
            self.device = torch.device("cpu")
            logger.error("[transcriber] %s", err)
            return

        req = (device or "cpu").strip().lower()
        if req not in ("cpu", "cuda"):
            req = "cpu"
        eff = resolve_inference_device(req)
        self.device = torch.device(eff)

        try:
            self._load()
        except Exception as e:
            self.load_error = str(e)
            self.processor = None
            self._model = None
            logger.exception("[transcriber] Failed to load from %s", self.adapter_path)
            if eff == "cuda":
                logger.info("[transcriber] Retrying on CPU after CUDA load failure")
                self.device = torch.device("cpu")
                try:
                    self._load()
                    self.load_error = None
                except Exception as e2:
                    self.load_error = str(e2)
                    self.processor = None
                    self._model = None
                    logger.exception("[transcriber] CPU fallback failed")

    def _load(self) -> None:
        logger.info("[transcriber] loading base %s on %s …", BASE_MODEL, self.device)
        self.processor = WhisperProcessor.from_pretrained(BASE_MODEL)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
        )
        logger.info("[transcriber] applying LoRA from %s …", self.adapter_path)
        self._model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        self._model = self._model.to(self.device)
        self._model.eval()
        self.load_error = None
        logger.info("[transcriber] ready on %s", self.device)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self.processor is not None

    def transcribe(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        *,
        live: bool = False,
    ) -> str:
        if not self.is_ready:
            raise RuntimeError(self.load_error or "Transcriber model is not loaded")
        if len(audio_chunk) == 0:
            return ""

        t0 = time.perf_counter()
        inputs = self.processor(
            audio_chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)

        max_tokens = 48 if live else 128
        gen_kwargs: dict = {
            "input_features": input_features,
            "max_new_tokens": max_tokens,
            "num_beams": 1,
            "do_sample": False,
        }
        if self.language:
            gen_kwargs["language"] = self.language
            gen_kwargs["task"] = "transcribe"

        with torch.inference_mode():
            try:
                predicted_ids = self._model.generate(**gen_kwargs)
            except TypeError:
                gen_fallback = {
                    "input_features": input_features,
                    "max_new_tokens": max_tokens,
                    "num_beams": 1,
                }
                forced_ids = self.processor.get_decoder_prompt_ids(
                    language=self.language,
                    task="transcribe",
                )
                gen_fallback["forced_decoder_ids"] = forced_ids
                predicted_ids = self._model.generate(**gen_fallback)

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return text[0].strip() if text else ""

    @property
    def inference_device(self) -> str:
        return str(self.device)
