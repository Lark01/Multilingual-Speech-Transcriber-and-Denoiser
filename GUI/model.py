"""
model.py — SpectrogramDenoiser
Architecture: STFT → U-Net mask → ISTFT
Verified against denoiser_epoch_15.pt weights.
"""

from __future__ import annotations

import collections
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── STFT / audio constants ────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

# Minimum audio samples so the spectrogram survives 3× max-pool2d.
MIN_SAMPLES = WIN_LENGTH + 7 * HOP_LENGTH  # = 1408

# Realtime chunked denoise: prepend this many past samples at 16 kHz so STFT has
# context at chunk edges (overlap-add style). ~128 ms; covers several hop frames.
CHUNK_CONTEXT_SAMPLES_16K = max(2048, WIN_LENGTH + 8 * HOP_LENGTH)


def resolve_inference_device(requested: str) -> str:
    """
    Return ``'cuda'`` or ``'cpu'`` that PyTorch can actually use.
    Logs clearly when GPU was requested but this install cannot use CUDA.
    """
    req = (requested or "cuda").strip().lower()
    if req not in ("cpu", "cuda"):
        req = "cuda"
    if req == "cpu":
        return "cpu"

    cuda_built = getattr(torch.version, "cuda", None)
    avail = torch.cuda.is_available()
    try:
        n_dev = torch.cuda.device_count() if avail else 0
    except Exception:
        n_dev = 0

    logger.info(
        "[model] torch %s | torch.version.cuda=%r | cuda.is_available()=%s | device_count=%s",
        torch.__version__,
        cuda_built,
        avail,
        n_dev,
    )

    if avail and n_dev > 0:
        try:
            logger.info("[model] Using CUDA: %s", torch.cuda.get_device_name(0))
        except Exception:
            logger.info("[model] Using CUDA device 0")
        return "cuda"

    if cuda_built is None:
        logger.error(
            "[model] GPU inference was requested but this PyTorch wheel is CPU-only "
            "(torch.version.cuda is None). Install a CUDA-enabled build, for example:\n"
            "  pip uninstall -y torch torchaudio\n"
            "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126\n"
            "Choose the CUDA version from https://pytorch.org/get-started/locally/ that matches your GPU driver."
        )
        if shutil.which("nvidia-smi"):
            logger.error(
                "[model] 'nvidia-smi' is on PATH (Windows sees a GPU) but PyTorch has no CUDA support — "
                "you almost certainly installed the default CPU-only 'torch' from pip."
            )
    else:
        logger.error(
            "[model] PyTorch includes CUDA %s but torch.cuda.is_available() is False. "
            "Update the NVIDIA driver, reboot, or install a PyTorch wheel whose CUDA version matches your driver.",
            cuda_built,
        )
    return "cpu"


def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv2d + BN + ReLU — used for encoder, bottleneck, and decoder."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _UNet(nn.Module):
    """
    3-level U-Net that operates on spectrogram magnitude.

    Input  : (B, 1, F, T)   where F = N_FFT/2+1 = 257
    Output : (B, 1, F, T)   soft mask in [0, 1]
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = _enc_block(1, 32)
        self.enc2 = _enc_block(32, 64)
        self.enc3 = _enc_block(64, 128)
        self.bottleneck = _enc_block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _enc_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _enc_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _enc_block(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _align(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh = skip.shape[2] - up.shape[2]
        dw = skip.shape[3] - up.shape[3]
        if dh > 0 or dw > 0:
            up = F.pad(up, (0, max(dw, 0), 0, max(dh, 0)))
        return up[:, :, : skip.shape[2], : skip.shape[3]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self._align(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._align(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._align(self.up1(d2), e1), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class _SpectrogramDenoiser(nn.Module):
    """Wraps _UNet with STFT pre-processing and ISTFT post-processing."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH))
        self.unet = _UNet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, samples)  float32, normalised to [-1, 1], 16 kHz mono
        out : (B, samples)  float32, denoised, same length as input
        """
        orig_len = x.shape[-1]
        if orig_len < MIN_SAMPLES:
            x = F.pad(x, (0, MIN_SAMPLES - orig_len))

        stft = torch.stft(
            x,
            N_FFT,
            HOP_LENGTH,
            WIN_LENGTH,
            self.window,
            return_complex=True,
        )
        mag = stft.abs().unsqueeze(1)
        phase = stft.angle()
        mask = self.unet(mag)
        clean_stft = torch.polar((mag * mask).squeeze(1), phase)
        out = torch.istft(
            clean_stft,
            N_FFT,
            HOP_LENGTH,
            WIN_LENGTH,
            self.window,
            length=x.shape[-1],
        )
        return out[..., :orig_len]


class DenoiserModel:
    """
    Public interface used by AudioEngine.

    If loading fails, ``load_error`` is set and ``is_ready`` is False.
    """

    def __init__(self, weights_path: str | Path, device: str = "cuda") -> None:
        self.weights_path = Path(weights_path)
        self.model_name = self.weights_path.stem
        self.load_error: str | None = None
        self._net: _SpectrogramDenoiser | None = None
        self.last_inference_ms: float = 0.0

        req = (device or "cuda").strip().lower()
        if req not in ("cpu", "cuda"):
            req = "cuda"
        eff = resolve_inference_device(req)
        self.device = torch.device(eff)

        try:
            self._net = self._load()
        except Exception as e:
            self.load_error = str(e)
            logger.exception("[model] Failed to load weights from %s", self.weights_path)

    def _load(self) -> _SpectrogramDenoiser:
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Weights not found: {self.weights_path}\n"
                f"Put denoiser_epoch_15.pt in the weights/ folder."
            )

        logger.info("[model] loading '%s' on %s", self.weights_path, self.device)

        try:
            state = torch.load(
                str(self.weights_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            state = torch.load(str(self.weights_path), map_location=self.device)

        if isinstance(state, nn.Module):
            state = state.state_dict()

        for key in ("model", "state_dict", "net"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break

        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = collections.OrderedDict(
                (k.replace("module.", "", 1), v) for k, v in state.items()
            )

        net = _SpectrogramDenoiser().to(self.device)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            logger.warning("[model] missing keys : %s", missing)
        if unexpected:
            logger.warning("[model] unexpected keys: %s", unexpected)
        net.eval()

        n_params = sum(p.numel() for p in net.parameters())
        logger.info("[model] ready — %s parameters", f"{n_params:,}")
        self.load_error = None
        return net

    @property
    def is_ready(self) -> bool:
        return self._net is not None

    def reload(self, new_path: str | Path | None = None) -> None:
        """Hot-reload weights, optionally from a new path."""
        if new_path is not None:
            self.weights_path = Path(new_path)
            self.model_name = self.weights_path.stem
        try:
            self._net = self._load()
        except Exception as e:
            self.load_error = str(e)
            self._net = None
            logger.exception("[model] reload failed")

    def denoise(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        audio_chunk : (samples,) float32 mono @ 16 kHz, [-1, 1]
        """
        if self._net is None:
            raise RuntimeError("Model is not loaded")

        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(audio_chunk, dtype=np.float32)).to(self.device)
            x = x.unsqueeze(0)
            out = self._net(x)
            result = out.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return result

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def inference_device(self) -> str:
        return str(self.device)

    def set_inference_device(self, device: str) -> None:
        """Move loaded weights to CPU or CUDA without reloading from disk."""
        req = (device or "cpu").strip().lower()
        if req not in ("cpu", "cuda"):
            req = "cpu"
        eff = resolve_inference_device(req)
        d = torch.device(eff)
        self.device = d
        if self._net is None:
            return
        try:
            self._net = self._net.to(d)
            self.load_error = None
        except Exception:
            logger.exception("[model] Failed to move network to %s", d)
            self.load_error = f"Failed to move model to {d}"
