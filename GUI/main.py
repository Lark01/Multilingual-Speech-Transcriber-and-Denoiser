"""
ClearVoice — real-time audio denoiser overlay (Windows).

Run from project root:  python main.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from audio_diag import enable_audio_diag
from audio_engine import AudioEngine
from model import DenoiserModel
from overlay import CLEARVOICE_APP_STYLESHEET, OverlayWindow
from settings import Settings, resolve_transcription_device
from tray import TrayIcon
from transcriber_model import TranscriberHolder, resolve_whisper_weights_path
from transcription import RealtimeTranscription

APP_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def _configure_logging(debug_audio_log: bool = True) -> Path | None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not debug_audio_log:
        return None
    diag = enable_audio_diag(APP_DIR / "logs" / "clearvoice_audio_debug.log")
    logger.info("Audio diagnostics log: %s", diag.log_file)
    return diag.log_file


def apply_launch_on_startup(settings: Settings) -> None:
    if sys.platform != "win32":
        logger.info("launch_on_startup is only implemented on Windows")
        return
    import winreg

    run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "ClearVoice"
    script = str(Path(__file__).resolve())
    value = f'"{sys.executable}" "{script}"'
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_SET_VALUE) as rk:
            if settings.launch_on_startup:
                winreg.SetValueEx(rk, app_name, 0, winreg.REG_SZ, value)
                logger.info("Registered ClearVoice in Windows startup")
            else:
                try:
                    winreg.DeleteValue(rk, app_name)
                    logger.info("Removed ClearVoice from Windows startup")
                except FileNotFoundError:
                    pass
    except Exception:
        logger.exception("Failed to update Windows Run registry")


def _resolve_weights_path(cli_weights: str | None, settings: Settings) -> Path:
    if cli_weights:
        return Path(cli_weights).expanduser().resolve()
    p = Path(settings.weights_path).expanduser()
    if p.is_file():
        return p.resolve()
    candidate = APP_DIR / "weights" / "denoiser_epoch_15.pt"
    if candidate.is_file():
        settings.weights_path = str(candidate.resolve())
        settings.save()
        return candidate.resolve()
    beside_main = APP_DIR / "denoiser_epoch_15.pt"
    if beside_main.is_file():
        settings.weights_path = str(beside_main.resolve())
        settings.save()
        return beside_main.resolve()
    parent = APP_DIR.parent / "denoiser_epoch_15.pt"
    if parent.is_file():
        settings.weights_path = str(parent.resolve())
        settings.save()
        return parent.resolve()
    wdir = APP_DIR / "weights"
    for pattern in ("*.pt", "*.pth"):
        for f in sorted(wdir.glob(pattern)):
            settings.weights_path = str(f.resolve())
            settings.save()
            return f.resolve()
    return Path(settings.weights_path)


def _prompt_weights_file(settings: Settings) -> Path | None:
    QMessageBox.information(
        None,
        "ClearVoice",
        "No model weights found. Please select a .pt or .pth file.",
    )
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select denoiser weights",
        str(APP_DIR / "weights"),
        "PyTorch (*.pt *.pth);;All files (*.*)",
    )
    if not path:
        return None
    p = Path(path).resolve()
    settings.weights_path = str(p)
    settings.save()
    return p


def maybe_warn_vbcable(settings: Settings) -> None:
    if settings.hide_vbcable_warning:
        return
    mb = QMessageBox()
    mb.setWindowTitle("VB-Audio Virtual Cable")
    mb.setIcon(QMessageBox.Icon.Information)
    mb.setTextFormat(Qt.TextFormat.RichText)
    mb.setText(
        "VB-Cable was not detected. For the best setup, route ClearVoice output "
        "to VB-Cable and use it as a microphone in other apps."
    )
    mb.setInformativeText(
        '<a href="https://vb-audio.com/Cable/">https://vb-audio.com/Cable/</a>'
    )
    from PyQt6.QtWidgets import QCheckBox

    cb = QCheckBox("Don't show again")
    mb.setCheckBox(cb)
    mb.setStandardButtons(QMessageBox.StandardButton.Ok)
    mb.exec()
    if cb.isChecked():
        settings.hide_vbcable_warning = True
        settings.save()


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearVoice real-time denoiser")
    parser.add_argument("--weights", default=None, help="path to .pt weights")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="inference device (default: CUDA when available; use --device cpu to force CPU)",
    )
    parser.add_argument("--minimized", action="store_true", help="start hidden to tray")
    parser.add_argument(
        "--no-audio-debug-log",
        action="store_true",
        help="disable logs/clearvoice_audio_debug.log stutter diagnostics",
    )
    parser.add_argument(
        "--whisper-weights",
        default=None,
        help="path to whisper LoRA adapter folder (adapter_config.json + weights)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="force transcription language (e.g. en, ar); auto-detect if omitted",
    )
    args = parser.parse_args()

    _configure_logging(debug_audio_log=not args.no_audio_debug_log)

    settings = Settings.load()
    if args.device:
        settings.inference_device = args.device
        settings.save()
        logger.info("[startup] inference_device=%s (from --device)", settings.inference_device)
    elif settings.inference_device == "cpu" and torch.cuda.is_available():
        settings.inference_device = "cuda"
        settings.save()
        logger.info(
            "[startup] Migrated inference_device CPU→CUDA (saved setting was CPU; GPU is available)."
        )
    elif settings.inference_device == "cpu":
        logger.info(
            "[startup] inference_device=cpu (saved in settings.json). "
            "CUDA unavailable in this Python — same interpreter as `where python` / IDE run config."
        )

    if settings.inference_device == "cuda" and not torch.cuda.is_available():
        settings.inference_device = "cpu"
        settings.save()
        logger.warning("[startup] CUDA not available — using CPU for inference.")
    if args.minimized:
        settings.start_minimized = True

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(CLEARVOICE_APP_STYLESHEET)

    weights_path = _resolve_weights_path(args.weights, settings)
    if not weights_path.is_file():
        picked = _prompt_weights_file(settings)
        if picked is not None:
            weights_path = picked

    if not weights_path.is_file():
        logger.warning("Weights file missing — starting in error state until user selects weights")

    model = DenoiserModel(weights_path, device=settings.inference_device)
    engine = AudioEngine(model)
    engine.active = settings.denoise_active
    engine.strength = float(settings.strength)

    whisper_path = resolve_whisper_weights_path(
        args.whisper_weights,
        settings.whisper_weights_path,
        APP_DIR,
    )
    if args.whisper_weights:
        settings.whisper_weights_path = str(whisper_path)
        settings.save()
    lang = args.language if args.language else settings.transcription_language
    tx_device = resolve_transcription_device(
        settings.inference_device,
        settings.transcription_inference_device,
    )
    logger.info(
        "[startup] transcription device=%s (denoiser=%s, setting=%s)",
        tx_device,
        settings.inference_device,
        settings.transcription_inference_device,
    )
    transcriber_holder = TranscriberHolder(
        whisper_path,
        tx_device,
        lang,
    )
    transcription = RealtimeTranscription(
        transcriber_holder.get,
        segment_sec=settings.transcription_segment_sec,
        hop_sec=settings.transcription_hop_sec,
    )
    transcription.active = False
    engine.transcription = transcription
    engine.transcribe_denoised = settings.transcribe_denoised_audio

    if model.is_ready:
        vbc_capture = engine.find_vbcable_capture()   # CABLE Output (recording end)
        vbc_play    = engine.find_vbcable_playback()  # CABLE Input  (playback end)

        if vbc_capture is not None:
            # VB-Cable is installed.
            # Input  = CABLE Output  (captures everything routed to CABLE Input)
            # Output = real speakers — must be explicit because the Windows default
            #          output is now CABLE Input, so None would loop back into the cable.
            in_idx  = vbc_capture
            saved_out = settings.output_device_index
            if saved_out is not None and saved_out != vbc_play:
                out_idx = saved_out          # user pinned a real device previously
            else:
                out_idx = engine.find_real_output()   # auto-find hardware speakers
            logger.info(
                "[startup] VB-Cable mode: capture from CABLE Output [%s], "
                "output to speakers [%s]", in_idx, out_idx,
            )
        else:
            # No VB-Cable — fall back to Stereo Mix (copy-capture; original still plays).
            loopback_in = engine.find_loopback_input()
            in_idx  = loopback_in if loopback_in is not None else settings.input_device_index
            out_idx = settings.output_device_index
            if loopback_in is not None:
                logger.info(
                    "[startup] Stereo Mix mode (original audio also plays). "
                    "Install VB-Cable for true intercept."
                )

        engine.start(in_idx, out_idx)
        engine.sync_resolved_devices_to_settings(settings)
    else:
        logger.error("Model not loaded: %s", model.load_error)

    overlay = OverlayWindow(
        engine,
        settings,
        model,
        lambda: apply_launch_on_startup(settings),
        transcriber_holder,
        transcription,
    )
    if settings.transcribe_active:
        overlay.ensure_live_transcription_started()
    tray = TrayIcon(None, overlay, engine, settings, transcription)
    overlay.set_tray(tray)
    tray.setVisible(True)
    tray.notify_startup()

    apply_launch_on_startup(settings)

    if not settings.start_minimized and not args.minimized:
        overlay.show()
    else:
        overlay.hide()

    def _on_about_to_quit() -> None:
        try:
            transcription.stop_worker()
        except Exception:
            logger.exception("transcription.stop_worker on quit")
        try:
            engine.stop()
        except Exception:
            logger.exception("engine.stop on quit")
        try:
            settings.save()
        except Exception:
            logger.exception("settings.save on quit")

    app.aboutToQuit.connect(_on_about_to_quit)

    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
