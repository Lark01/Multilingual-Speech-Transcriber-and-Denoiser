"""
ClearScribe — reference mic-only transcription (see repo root main.py for unified ClearVoice).

Usage:  python main.py --weights path/to/whisper-lora-weights/
"""
import sys
import argparse
from PyQt6.QtWidgets import QApplication
from model import TranscriberModel
from audio_engine import AudioEngine
from overlay import OverlayWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", default="whisper-lora-weights/",
        help="path to your LoRA adapter folder (contains adapter_config.json)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="inference device"
    )
    parser.add_argument(
        "--language", default=None,
        help="force language (e.g. 'en', 'ar'). Auto-detect if omitted."
    )
    args = parser.parse_args()

    model  = TranscriberModel(args.weights, device=args.device, language=args.language)
    engine = AudioEngine(model)

    engine.start()

    app    = QApplication(sys.argv)
    window = OverlayWindow(engine)
    window.show()

    ret = app.exec()
    engine.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
