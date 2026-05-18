"""PyQt6 frameless overlay for ClearVoice."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QTextCursor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QApplication,
)

import torch

from audio_engine import AudioEngine
from file_playback import DenoiseFileThread
from model import DenoiserModel
from settings import Settings
from transcriber_model import LoadTranscriberThread, TranscriberHolder
from transcription import RealtimeTranscription, TranscribeFileThread

logger = logging.getLogger(__name__)

# Design tokens (also referenced in custom paint)
_CLR_BG = "#0e1015"
_CLR_CARD = "#151821"
_CLR_INNER = "#0c0e14"
_CLR_BORDER = "#252b38"
_CLR_MUTED = "#8b93a7"
_CLR_TEXT = "#e8ebf2"

CLEARVOICE_APP_STYLESHEET = f"""
QWidget {{
  background-color: {_CLR_BG};
  color: {_CLR_TEXT};
  font-family: "Segoe UI", "Segoe UI Variable", system-ui, sans-serif;
  font-size: 13px;
}}
QFrame#appShell {{
  background-color: {_CLR_BG};
  border: 1px solid {_CLR_BORDER};
  border-radius: 14px;
}}
QFrame#titleBar {{
  background-color: transparent;
  border: none;
}}
QFrame#divider {{
  background-color: {_CLR_BORDER};
  max-height: 1px;
  min-height: 1px;
  border: none;
}}
QFrame#card {{
  background-color: {_CLR_CARD};
  border: 1px solid {_CLR_BORDER};
  border-radius: 12px;
}}
QFrame#innerPanel {{
  background-color: {_CLR_INNER};
  border: 1px solid {_CLR_BORDER};
  border-radius: 10px;
}}
QTextEdit#transcriptBox {{
  background-color: {_CLR_INNER};
  color: #d0e8ff;
  border: 1px solid {_CLR_BORDER};
  border-radius: 10px;
  padding: 8px;
  font-family: Consolas, "Cascadia Mono", monospace;
  font-size: 13px;
  selection-background-color: #1e3a5f;
}}
QFrame#footerBar {{
  background-color: #12151c;
  border: none;
  border-top: 1px solid {_CLR_BORDER};
  border-bottom-left-radius: 14px;
  border-bottom-right-radius: 14px;
}}
QLabel#cardTitle {{
  color: {_CLR_MUTED};
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
}}
QLabel#hint {{
  color: {_CLR_MUTED};
  font-size: 12px;
  line-height: 1.35;
}}
QLabel#sectionLabel {{
  color: #a8b0c4;
  font-size: 11px;
  font-weight: 600;
}}
QLabel#tagline {{
  color: {_CLR_MUTED};
  font-size: 11px;
}}
QLabel#statMuted {{
  color: #6b7287;
  font-size: 11px;
}}
QLabel#deviceCap {{
  color: #a8b0c4;
  font-size: 12px;
}}
QFrame#statusChip {{
  border-radius: 12px;
  min-height: 28px;
}}
QLabel#statusDot {{ font-size: 10px; font-weight: 800; }}
QLabel#statusText {{ font-size: 11px; font-weight: 600; color: #e8ebf2; }}
QPushButton {{
  background-color: #1e2430;
  color: {_CLR_TEXT};
  border: 1px solid {_CLR_BORDER};
  border-radius: 9px;
  padding: 9px 16px;
  min-height: 22px;
  font-weight: 500;
}}
QPushButton:hover {{
  background-color: #252b38;
  border-color: #3d4658;
}}
QPushButton:pressed {{ background-color: #1a1f2a; }}
QPushButton:disabled {{ color: #5c6378; border-color: #1f2430; background: #141820; }}
QPushButton#accent {{
  background-color: #1a2230;
  color: #94a3b8;
  border: 1px solid {_CLR_BORDER};
  font-weight: 600;
}}
QPushButton#accent:checked {{
  background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #10b981, stop:1 #059669);
  color: #ecfdf5;
  border: 1px solid #047857;
}}
QPushButton#accent:hover {{
  border-color: #3d4f66;
  background-color: #222a38;
}}
QPushButton#accent:checked:hover {{
  background-color: #12d687;
  border-color: #34d399;
}}
QPushButton#ghost {{
  background-color: transparent;
  border: 1px solid {_CLR_BORDER};
  color: #c5cad8;
}}
QPushButton#ghost:hover {{ background-color: rgba(255,255,255,0.04); }}
QPushButton#iconOnly {{
  background-color: transparent;
  border: none;
  color: #7c8498;
  padding: 6px;
  min-width: 32px;
  min-height: 32px;
  border-radius: 8px;
}}
QPushButton#iconOnly:hover {{
  color: #e8ebf2;
  background-color: rgba(255,255,255,0.06);
}}
QComboBox {{
  background-color: {_CLR_INNER};
  border: 1px solid {_CLR_BORDER};
  border-radius: 8px;
  padding: 6px 10px;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QSlider::groove:horizontal {{
  height: 6px;
  background: #1e2430;
  border-radius: 3px;
  border: 1px solid {_CLR_BORDER};
}}
QSlider::handle:horizontal {{
  width: 18px;
  height: 18px;
  margin: -7px 0;
  background: #34d399;
  border-radius: 9px;
  border: 2px solid #0f172a;
}}
    QSlider::sub-page:horizontal {{
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #34d399);
  border-radius: 3px;
}}
QRadioButton {{ spacing: 10px; color: #c5cad8; }}
QRadioButton::indicator {{ width: 16px; height: 16px; }}
QCheckBox {{ color: #c5cad8; spacing: 10px; }}
QCheckBox::indicator {{
  width: 18px;
  height: 18px;
  border-radius: 4px;
  border: 1px solid {_CLR_BORDER};
  background: {_CLR_INNER};
}}
QCheckBox::indicator:checked {{
  background: #059669;
  border-color: #34d399;
}}
QScrollArea {{ border: none; background-color: {_CLR_BG}; }}
QScrollArea > QWidget > QWidget {{ background-color: {_CLR_BG}; }}
QScrollBar:vertical {{
  width: 10px;
  background: transparent;
  margin: 4px 2px 4px 0;
}}
QScrollBar::handle:vertical {{
  background: #343d52;
  border-radius: 5px;
  min-height: 36px;
}}
QScrollBar::handle:vertical:hover {{ background: #455068; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{
  border: 1px solid {_CLR_BORDER};
  border-radius: 8px;
  text-align: center;
  height: 24px;
  background-color: {_CLR_INNER};
  color: #a8b0c4;
  font-size: 11px;
  font-weight: 500;
}}
QProgressBar::chunk {{
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #34d399);
  border-radius: 6px;
}}
"""


def _make_card_section(title: str) -> tuple[QFrame, QVBoxLayout]:
    """Section card with uppercase eyebrow title and body layout."""
    card = QFrame()
    card.setObjectName("card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    head = QLabel(title.upper())
    head.setObjectName("cardTitle")
    hl = QHBoxLayout()
    hl.setContentsMargins(18, 16, 18, 0)
    hl.addWidget(head)
    hl.addStretch()
    outer.addLayout(hl)
    body = QVBoxLayout()
    body.setContentsMargins(18, 10, 18, 18)
    body.setSpacing(14)
    outer.addLayout(body)
    return card, body


def _linear_to_db(linear: float) -> float:
    return 20.0 * math.log10(max(linear, 1e-9))


class TitleBar(QWidget):
    """Draggable title bar with brand row and status chip."""

    def __init__(self, parent_window: QWidget) -> None:
        super().__init__(parent_window)
        self._win = parent_window
        self._drag_pos: QPoint | None = None
        self.setObjectName("titleBar")
        self.setFixedHeight(52)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 8, 12, 8)
        lay.setSpacing(14)

        brand = QVBoxLayout()
        brand.setSpacing(2)
        self.lbl_title = QLabel("ClearVoice")
        ft = QFont("Segoe UI")
        ft.setPointSize(15)
        ft.setBold(True)
        self.lbl_title.setFont(ft)
        self.lbl_tagline = QLabel("System audio denoiser")
        self.lbl_tagline.setObjectName("tagline")
        brand.addWidget(self.lbl_title)
        brand.addWidget(self.lbl_tagline)
        lay.addLayout(brand)

        lay.addStretch()

        self.status_chip = QFrame()
        self.status_chip.setObjectName("statusChip")
        self.status_chip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        chip_lay = QHBoxLayout(self.status_chip)
        chip_lay.setContentsMargins(12, 4, 12, 4)
        chip_lay.setSpacing(6)
        self.lbl_dot = QLabel("●")
        self.lbl_dot.setObjectName("statusDot")
        self.lbl_status = QLabel("Live")
        self.lbl_status.setObjectName("statusText")
        chip_lay.addWidget(self.lbl_dot)
        chip_lay.addWidget(self.lbl_status)
        lay.addWidget(self.status_chip, 0, Qt.AlignmentFlag.AlignVCenter)

        self.status_chip.setStyleSheet(
            "QFrame#statusChip { background-color: rgba(148,163,184,0.1); "
            "border: 1px solid rgba(100,116,139,0.35); border-radius: 12px; }"
        )

        sty = parent_window.style()
        self.btn_min = QPushButton()
        self.btn_min.setObjectName("iconOnly")
        self.btn_min.setIcon(sty.standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton))
        self.btn_min.setIconSize(QSize(18, 18))
        self.btn_min.setFixedSize(36, 36)
        self.btn_min.setToolTip("Minimize to tray")
        self.btn_min.clicked.connect(self._on_minimize)
        self.btn_close = QPushButton()
        self.btn_close.setObjectName("iconOnly")
        self.btn_close.setIcon(sty.standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        self.btn_close.setIconSize(QSize(18, 18))
        self.btn_close.setFixedSize(36, 36)
        self.btn_close.setToolTip("Quit")
        self.btn_close.clicked.connect(parent_window.close)
        lay.addWidget(self.btn_min, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.btn_close, 0, Qt.AlignmentFlag.AlignVCenter)

    def _on_minimize(self) -> None:
        w = self._win
        if hasattr(w, "minimize_to_tray"):
            getattr(w, "minimize_to_tray")()
        else:
            w.showMinimized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._win.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            self._win.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)


class WaveformWidget(QWidget):
    """Scrolling waveform from engine ring buffer."""

    def __init__(self, engine: AudioEngine, channel: str, color: QColor) -> None:
        super().__init__()
        self._engine = engine
        self._channel = channel
        self._color = color
        self.setMinimumHeight(44)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bg = QColor(_CLR_INNER)
        painter.fillRect(self.rect(), bg)
        w, h = self.width(), self.height()
        mid = h * 0.5
        painter.setPen(QPen(QColor("#252b38"), 1))
        painter.drawLine(0, int(mid), w, int(mid))
        snap_in, snap_out = self._engine.get_wave_snapshot()
        data = snap_in if self._channel == "in" else snap_out
        if len(data) < 2:
            return
        painter.setPen(QPen(self._color, 1.75))
        n = len(data)
        for i in range(1, n):
            x0 = (i - 1) * (w - 1) / (n - 1)
            x1 = i * (w - 1) / (n - 1)
            y0 = mid - data[i - 1] * (mid - 6)
            y1 = mid - data[i] * (mid - 6)
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))


class DbBarMeter(QWidget):
    """Horizontal level bar with dB readout."""

    def __init__(self, title: str, bar_color: QColor) -> None:
        super().__init__()
        self._title = title
        self._bar_color = bar_color
        self._db = -96.0
        self._fill01 = 0.0
        self.setMinimumHeight(30)

    def set_level_db(self, db: float, fill01: float) -> None:
        self._db = db
        self._fill01 = max(0.0, min(1.0, fill01))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.rect()
        painter.fillRect(r, QColor(_CLR_INNER))
        painter.setPen(QColor("#a8b0c4"))
        painter.drawText(0, 15, self._title)
        bar_rect = r.adjusted(0, 20, 0, -2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1e2430"))
        painter.drawRoundedRect(bar_rect, 4, 4)
        inner = bar_rect.adjusted(3, 3, -3, -3)
        fw = int(inner.width() * self._fill01)
        if fw > 0:
            painter.setBrush(self._bar_color)
            painter.drawRoundedRect(inner.x(), inner.y(), fw, inner.height(), 3, 3)
        painter.setPen(QColor("#8b93a7"))
        txt = f"{self._db:.0f} dB" if self._db > -90 else "— dB"
        painter.drawText(bar_rect.right() - 56, 15, txt)


class _TranscriptUiBridge(QObject):
    """Thread-safe bridge: inference worker → main-thread overlay updates."""

    new_segment = pyqtSignal(str)
    status = pyqtSignal(str)


class OverlayWindow(QWidget):
    """Main ClearVoice control surface."""

    def __init__(
        self,
        engine: AudioEngine,
        settings: Settings,
        model: DenoiserModel,
        apply_launch_on_startup: Callable[[], None],
        transcriber_holder: TranscriberHolder,
        transcription: RealtimeTranscription,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.settings = settings
        self.model = model
        self._transcriber_holder = transcriber_holder
        self.transcription = transcription
        self._apply_launch_on_startup = apply_launch_on_startup
        self._tray = None
        self._file_path: Path | None = None
        self._file_thread: DenoiseFileThread | None = None
        self._transcribe_thread: TranscribeFileThread | None = None
        self._load_transcriber_thread: LoadTranscriberThread | None = None
        self._pending_transcribe_after_load = False
        self._pending_file_transcribe_after_load = False
        self._transcript_displayed_count = 0
        self._tx_ui = _TranscriptUiBridge(self)
        self._tx_ui.new_segment.connect(self._append_transcript_text)
        self._tx_ui.status.connect(self._set_transcript_status)
        self.transcription.on_new_text = self._tx_ui.new_segment.emit
        self.transcription.on_status = self._tx_ui.status.emit

        self.setWindowTitle("ClearVoice")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.resize(920, 460)
        self.setMinimumSize(760, 320)
        self.setMaximumHeight(920)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(0)

        shell = QFrame()
        shell.setObjectName("appShell")
        shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shell_lay = QVBoxLayout(shell)
        shell_lay.setContentsMargins(0, 0, 0, 0)
        shell_lay.setSpacing(0)

        self.title_bar = TitleBar(self)
        shell_lay.addWidget(self.title_bar)

        div = QFrame()
        div.setObjectName("divider")
        div.setFixedHeight(1)
        shell_lay.addWidget(div)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        inner = QWidget()
        inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        scroll.setWidget(inner)
        body = QVBoxLayout(inner)
        body.setContentsMargins(16, 18, 16, 22)
        body.setSpacing(18)

        card_live, gl = _make_card_section("Monitor")
        row_vis = QHBoxLayout()
        row_vis.setSpacing(14)

        wf = QFrame()
        wf.setObjectName("innerPanel")
        wf.setMinimumWidth(280)
        wf.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        wfl = QVBoxLayout(wf)
        wfl.setContentsMargins(14, 12, 14, 12)
        wfl.setSpacing(10)
        lbl_wi = QLabel("Raw input")
        lbl_wi.setObjectName("sectionLabel")
        wfl.addWidget(lbl_wi)
        self.wave_in = WaveformWidget(engine, "in", QColor("#fb923c"))
        wfl.addWidget(self.wave_in)
        lbl_wo = QLabel("Denoised output")
        lbl_wo.setObjectName("sectionLabel")
        wfl.addWidget(lbl_wo)
        self.wave_out = WaveformWidget(engine, "out", QColor("#34d399"))
        wfl.addWidget(self.wave_out)
        row_vis.addWidget(wf, 3)

        meters = QFrame()
        meters.setObjectName("innerPanel")
        meters.setMinimumWidth(200)
        meters.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        ml = QVBoxLayout(meters)
        ml.setContentsMargins(14, 12, 14, 12)
        ml.setSpacing(12)
        self.meter_in = DbBarMeter("Input level", QColor("#fb923c"))
        self.meter_out = DbBarMeter("Output level", QColor("#34d399"))
        self.meter_red = DbBarMeter("Noise reduction (est.)", QColor("#38bdf8"))
        ml.addWidget(self.meter_in)
        ml.addWidget(self.meter_out)
        ml.addWidget(self.meter_red)
        row_vis.addWidget(meters, 2)
        gl.addLayout(row_vis)
        body.addWidget(card_live)

        card_denoise, gdl = _make_card_section("Processing")
        ctrl = QFrame()
        ctrl.setObjectName("innerPanel")
        ctrl.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.setSpacing(14)
        self.btn_power = QPushButton("Denoising on")
        self.btn_power.setCheckable(True)
        self.btn_power.setObjectName("accent")
        self.btn_power.setMinimumHeight(40)
        self.btn_power.setChecked(self.engine.active)
        self.btn_power.toggled.connect(self._on_power)
        cl.addWidget(self.btn_power)
        row_s = QHBoxLayout()
        row_s.setSpacing(16)
        sl = QLabel("Strength")
        sl.setObjectName("sectionLabel")
        row_s.addWidget(sl)
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(int(self.settings.strength * 100))
        self.slider_strength.valueChanged.connect(self._on_strength_changed)
        self.slider_strength.sliderReleased.connect(self._on_strength_released)
        row_s.addWidget(self.slider_strength, 1)
        self.lbl_strength = QLabel(f"{int(self.settings.strength * 100)}%")
        self.lbl_strength.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_strength.setMinimumWidth(44)
        self.lbl_strength.setObjectName("sectionLabel")
        row_s.addWidget(self.lbl_strength)
        cl.addLayout(row_s)
        gdl.addWidget(ctrl)
        body.addWidget(card_denoise)

        card_tx, gtx = _make_card_section("Transcription")
        hint_tx = QLabel(
            "Live captions: ~2 s audio windows, updated every ~1 s (+ Whisper speed). "
            "First run downloads the base model. LoRA folder needs adapter weights."
        )
        hint_tx.setObjectName("hint")
        hint_tx.setWordWrap(True)
        gtx.addWidget(hint_tx)
        row_tx = QHBoxLayout()
        row_tx.setSpacing(10)
        self.btn_transcribe = QPushButton("Transcription off")
        self.btn_transcribe.setCheckable(True)
        self.btn_transcribe.setObjectName("accent")
        self.btn_transcribe.setMinimumHeight(38)
        self.btn_transcribe.setChecked(self.settings.transcribe_active)
        self.btn_transcribe.toggled.connect(self._on_transcribe_toggle)
        row_tx.addWidget(self.btn_transcribe)
        self.btn_tx_clear = QPushButton("Clear")
        self.btn_tx_clear.setObjectName("ghost")
        self.btn_tx_clear.clicked.connect(self._on_transcript_clear)
        row_tx.addWidget(self.btn_tx_clear)
        self.btn_tx_copy = QPushButton("Copy")
        self.btn_tx_copy.setObjectName("ghost")
        self.btn_tx_copy.clicked.connect(self._on_transcript_copy)
        row_tx.addWidget(self.btn_tx_copy)
        self.btn_tx_save = QPushButton("Save…")
        self.btn_tx_save.setObjectName("ghost")
        self.btn_tx_save.clicked.connect(self._on_transcript_save)
        row_tx.addWidget(self.btn_tx_save)
        row_tx.addStretch()
        gtx.addLayout(row_tx)
        self.chk_tx_denoised = QCheckBox("Caption denoised / mixed audio (when denoise is on)")
        self.chk_tx_denoised.setChecked(self.settings.transcribe_denoised_audio)
        self.chk_tx_denoised.toggled.connect(self._on_tx_denoised_toggled)
        gtx.addWidget(self.chk_tx_denoised)
        row_tx_timing = QHBoxLayout()
        row_tx_timing.setSpacing(12)
        lbl_chunk = QLabel("Chunk (s)")
        lbl_chunk.setObjectName("sectionLabel")
        row_tx_timing.addWidget(lbl_chunk)
        self.spin_tx_chunk = QSpinBox()
        self.spin_tx_chunk.setRange(1, 8)
        self.spin_tx_chunk.setValue(max(1, int(round(self.settings.transcription_segment_sec))))
        self.spin_tx_chunk.setToolTip("Seconds of audio per Whisper call")
        self.spin_tx_chunk.valueChanged.connect(self._on_tx_timing_changed)
        row_tx_timing.addWidget(self.spin_tx_chunk)
        lbl_step = QLabel("Step (s)")
        lbl_step.setObjectName("sectionLabel")
        row_tx_timing.addWidget(lbl_step)
        self.spin_tx_step = QSpinBox()
        self.spin_tx_step.setRange(1, 8)
        self.spin_tx_step.setValue(max(1, int(round(self.settings.transcription_hop_sec))))
        self.spin_tx_step.setToolTip("Enqueue a new chunk every N seconds (lower = more live)")
        self.spin_tx_step.valueChanged.connect(self._on_tx_timing_changed)
        row_tx_timing.addWidget(self.spin_tx_step)
        row_tx_timing.addStretch()
        gtx.addLayout(row_tx_timing)
        gtx.addWidget(QLabel("Transcript"))
        self.txt_transcript = QTextEdit()
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setMinimumHeight(120)
        self.txt_transcript.setMaximumHeight(200)
        self.txt_transcript.setObjectName("transcriptBox")
        gtx.addWidget(self.txt_transcript)
        self.lbl_tx_status = QLabel("")
        self.lbl_tx_status.setObjectName("hint")
        gtx.addWidget(self.lbl_tx_status)
        row_tx_stats = QHBoxLayout()
        self.lbl_tx_lat = QLabel("— ms/seg")
        self.lbl_tx_lat.setObjectName("statMuted")
        self.lbl_tx_seg = QLabel("0 segments")
        self.lbl_tx_seg.setObjectName("statMuted")
        row_tx_stats.addWidget(self.lbl_tx_lat)
        row_tx_stats.addWidget(self.lbl_tx_seg)
        row_tx_stats.addStretch()
        gtx.addLayout(row_tx_stats)
        body.addWidget(card_tx)

        card_file, gfl = _make_card_section("Offline file")
        hint_file = QLabel(
            "Denoise a file and play it through your speakers. Live capture pauses during playback. "
            "MP3/M4A: install imageio-ffmpeg or add ffmpeg to PATH. WAV/FLAC use soundfile."
        )
        hint_file.setObjectName("hint")
        hint_file.setWordWrap(True)
        gfl.addWidget(hint_file)
        row_file = QHBoxLayout()
        row_file.setSpacing(10)
        self.btn_file_open = QPushButton("Open…")
        self.btn_file_open.setObjectName("ghost")
        self.btn_file_open.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.btn_file_open.setIconSize(QSize(18, 18))
        self.btn_file_open.clicked.connect(self._on_file_open)
        row_file.addWidget(self.btn_file_open)
        self.btn_file_play = QPushButton("Play denoised")
        self.btn_file_play.setObjectName("accent")
        self.btn_file_play.setMinimumHeight(38)
        self.btn_file_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_file_play.setIconSize(QSize(18, 18))
        self.btn_file_play.clicked.connect(self._on_file_play)
        row_file.addWidget(self.btn_file_play)
        self.btn_file_stop = QPushButton("Stop")
        self.btn_file_stop.setObjectName("ghost")
        self.btn_file_stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.btn_file_stop.setIconSize(QSize(18, 18))
        self.btn_file_stop.clicked.connect(self._on_file_stop)
        self.btn_file_stop.setEnabled(False)
        row_file.addWidget(self.btn_file_stop)
        self.btn_file_transcribe = QPushButton("Transcribe file")
        self.btn_file_transcribe.setObjectName("accent")
        self.btn_file_transcribe.setMinimumHeight(38)
        self.btn_file_transcribe.clicked.connect(self._on_file_transcribe)
        row_file.addWidget(self.btn_file_transcribe)
        row_file.addStretch()
        gfl.addLayout(row_file)
        self.chk_file_denoise_first = QCheckBox("Denoise before transcribe (offline)")
        self.chk_file_denoise_first.setChecked(self.settings.file_transcribe_denoise_first)
        self.chk_file_denoise_first.toggled.connect(self._on_file_denoise_first_toggled)
        gfl.addWidget(self.chk_file_denoise_first)
        self.lbl_file_path = QLabel("No file selected")
        self.lbl_file_path.setObjectName("hint")
        gfl.addWidget(self.lbl_file_path)
        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 100)
        self.file_progress.setValue(0)
        self.file_progress.setTextVisible(True)
        self.file_progress.setFormat("%p%")
        gfl.addWidget(self.file_progress)
        self.lbl_file_status = QLabel("")
        self.lbl_file_status.setObjectName("hint")
        gfl.addWidget(self.lbl_file_status)
        body.addWidget(card_file)

        card_dev, gdev = _make_card_section("Audio routing & model")
        row_mid = QHBoxLayout()
        row_mid.setSpacing(14)

        # ── DEVICE SELECTION UI ── commented out until further notice ──────────
        # dev = QFrame()
        # dev.setObjectName("surface")
        # dev.setMinimumWidth(220)
        # dl = QVBoxLayout(dev)
        # dl.setContentsMargins(8, 6, 8, 6)
        # dl.setSpacing(4)
        # dl.addWidget(QLabel("Input device"))
        # self.combo_in = QComboBox()
        # dl.addWidget(self.combo_in)
        # hint_in = QLabel(
        #     "Mic ≠ PC playback. Use Stereo Mix / loopback or a virtual mixer if you need desktop audio."
        # )
        # hint_in.setObjectName("hint")
        # hint_in.setWordWrap(True)
        # hint_in.setMaximumHeight(40)
        # dl.addWidget(hint_in)
        # dl.addWidget(QLabel("Output device"))
        # self.combo_out = QComboBox()
        # dl.addWidget(self.combo_out)
        # row_dev = QHBoxLayout()
        # self.btn_apply_dev = QPushButton("Apply & Restart")
        # self.btn_apply_dev.setObjectName("accent")
        # self.btn_apply_dev.clicked.connect(self._apply_devices)
        # self.btn_refresh_dev = QPushButton("Refresh")
        # self.btn_refresh_dev.clicked.connect(self._populate_devices)
        # row_dev.addWidget(self.btn_apply_dev)
        # row_dev.addWidget(self.btn_refresh_dev)
        # dl.addLayout(row_dev)
        # row_mid.addWidget(dev, 1)
        # ── END DEVICE SELECTION UI ─────────────────────────────────────────────

        dev_info = QFrame()
        dev_info.setObjectName("innerPanel")
        dev_info.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dil = QVBoxLayout(dev_info)
        dil.setContentsMargins(14, 12, 14, 12)
        dil.setSpacing(8)
        cap = QLabel("Capture")
        cap.setObjectName("sectionLabel")
        dil.addWidget(cap)
        self.lbl_capture_device = QLabel("—")
        self.lbl_capture_device.setObjectName("deviceCap")
        self.lbl_capture_device.setWordWrap(True)
        dil.addWidget(self.lbl_capture_device)
        outl = QLabel("Playback output")
        outl.setObjectName("sectionLabel")
        dil.addWidget(outl)
        self.lbl_output_device = QLabel("—")
        self.lbl_output_device.setObjectName("hint")
        self.lbl_output_device.setWordWrap(True)
        dil.addWidget(self.lbl_output_device)
        row_mid.addWidget(dev_info, 1)

        modf = QFrame()
        modf.setObjectName("innerPanel")
        modf.setMinimumWidth(220)
        modf.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        ml2 = QVBoxLayout(modf)
        ml2.setContentsMargins(14, 12, 14, 12)
        ml2.setSpacing(10)
        mw = QLabel("Model weights")
        mw.setObjectName("sectionLabel")
        ml2.addWidget(mw)
        self.lbl_model = QLabel(self.model.model_name if self.model.is_ready else "Not loaded")
        self.lbl_model.setWordWrap(True)
        ml2.addWidget(self.lbl_model)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setObjectName("ghost")
        self.btn_browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogStart))
        self.btn_browse.setIconSize(QSize(18, 18))
        self.btn_browse.clicked.connect(self._browse_weights)
        ml2.addWidget(self.btn_browse)
        inf = QLabel("Inference")
        inf.setObjectName("sectionLabel")
        ml2.addWidget(inf)
        row_inf = QHBoxLayout()
        row_inf.setSpacing(14)
        self.radio_cpu = QRadioButton("CPU")
        self.radio_cuda = QRadioButton("GPU · CUDA")
        cuda_ok = torch.cuda.is_available()
        self.radio_cuda.setEnabled(cuda_ok)
        self.radio_cpu.setEnabled(not cuda_ok)
        if cuda_ok:
            self.radio_cpu.setToolTip(
                "GPU is used by default. For CPU-only inference run: python main.py --device cpu"
            )
        grp = QButtonGroup(self)
        grp.addButton(self.radio_cpu)
        grp.addButton(self.radio_cuda)
        self.radio_cpu.blockSignals(True)
        self.radio_cuda.blockSignals(True)
        if self.settings.inference_device == "cuda" and cuda_ok:
            self.radio_cuda.setChecked(True)
        else:
            self.radio_cpu.setChecked(True)
        self.radio_cpu.blockSignals(False)
        self.radio_cuda.blockSignals(False)
        self.radio_cpu.toggled.connect(self._on_inference_changed)
        self.radio_cuda.toggled.connect(self._on_inference_changed)
        row_inf.addWidget(self.radio_cuda)
        row_inf.addWidget(self.radio_cpu)
        row_inf.addStretch()
        ml2.addLayout(row_inf)
        row_mid.addWidget(modf, 1)
        gdev.addLayout(row_mid)
        body.addWidget(card_dev)

        shell_lay.addWidget(scroll, 1)

        footer = QFrame()
        footer.setObjectName("footerBar")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(18, 12, 18, 14)
        fl.setSpacing(14)
        self.chk_startup = QCheckBox("Launch on Windows startup")
        self.chk_startup.setChecked(self.settings.launch_on_startup)
        self.chk_startup.toggled.connect(self._on_startup_toggled)
        fl.addWidget(self.chk_startup)
        fl.addStretch()
        self.lbl_stats = QLabel("—")
        self.lbl_stats.setObjectName("statMuted")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        fl.addWidget(self.lbl_stats, 1)
        shell_lay.addWidget(footer)

        outer.addWidget(shell)

        self._refresh_device_labels()

        self._timer_wave = QTimer(self)
        self._timer_wave.timeout.connect(self._tick_waveforms)
        self._timer_wave.start(33)

        self._timer_meters = QTimer(self)
        self._timer_meters.timeout.connect(self._tick_meters)
        self._timer_meters.start(100)

        self._timer_stats = QTimer(self)
        self._timer_stats.timeout.connect(self._tick_stats)
        self._timer_stats.start(500)

        self._sync_power_ui()
        self._sync_model_error_ui()
        self._sync_transcribe_ui()
        self._update_file_play_button()
        self._update_file_transcribe_button()

    def ensure_live_transcription_started(self) -> None:
        """Load Whisper if needed and enable live transcription (startup / tray)."""
        if self._transcriber_holder.is_ready:
            self._activate_live_transcription()
            return
        if self._load_transcriber_thread is not None and self._load_transcriber_thread.isRunning():
            self._pending_transcribe_after_load = True
            return
        self._pending_transcribe_after_load = True
        self.btn_transcribe.blockSignals(True)
        self.btn_transcribe.setChecked(True)
        self.btn_transcribe.blockSignals(False)
        self.settings.transcribe_active = True
        self.btn_transcribe.setEnabled(False)
        self._start_load_transcriber(for_file=False)

    def set_tray(self, tray) -> None:
        self._tray = tray

    def minimize_to_tray(self) -> None:
        p = self.pos()
        self.settings.window_x = int(p.x())
        self.settings.window_y = int(p.y())
        self.settings.save()
        self.hide()
        if self._tray is not None:
            self._tray.setVisible(True)

    def _file_playback_busy(self) -> bool:
        return self._file_thread is not None and self._file_thread.isRunning()

    def _file_transcribe_busy(self) -> bool:
        return self._transcribe_thread is not None and self._transcribe_thread.isRunning()

    def _file_job_busy(self) -> bool:
        return self._file_playback_busy() or self._file_transcribe_busy()

    def _update_file_play_button(self) -> None:
        ok = (
            self.model.is_ready
            and self._file_path is not None
            and self._file_path.is_file()
            and not self._file_job_busy()
        )
        self.btn_file_play.setEnabled(ok)

    def _update_file_transcribe_button(self) -> None:
        ok = (
            self._file_path is not None
            and self._file_path.is_file()
            and not self._file_job_busy()
        )
        self.btn_file_transcribe.setEnabled(ok)

    def _set_file_ui_busy(self, busy: bool) -> None:
        self.btn_file_open.setEnabled(not busy)
        self.btn_file_stop.setEnabled(busy)
        self.slider_strength.setEnabled(not busy)
        self.btn_power.setEnabled(not busy)
        self.btn_browse.setEnabled(not busy)
        self.radio_cpu.setEnabled(not busy and not torch.cuda.is_available())
        self.radio_cuda.setEnabled(not busy and torch.cuda.is_available())
        self.chk_startup.setEnabled(not busy)
        self.btn_transcribe.setEnabled(not busy)
        self.chk_tx_denoised.setEnabled(not busy)
        self.btn_file_transcribe.setEnabled(not busy)
        self.chk_file_denoise_first.setEnabled(not busy)
        if busy:
            self.transcription.paused = True
        elif self.settings.transcribe_active:
            self.transcription.paused = False
        self._update_file_play_button()
        self._update_file_transcribe_button()

    def _on_file_open(self) -> None:
        if self._file_job_busy():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open audio file",
            str(self._file_path.parent) if self._file_path else "",
            "Audio (*.mp3 *.wav *.flac *.m4a *.ogg *.wma);;All files (*.*)",
        )
        if not path:
            return
        self._file_path = Path(path)
        self.lbl_file_path.setText(str(self._file_path))
        self._update_file_play_button()
        self._update_file_transcribe_button()

    def _on_file_play(self) -> None:
        if self._file_job_busy():
            return
        if self._file_path is None or not self._file_path.is_file():
            QMessageBox.warning(self, "ClearVoice", "Choose an audio file first.")
            return
        if not self.model.is_ready:
            QMessageBox.warning(self, "ClearVoice", "Load model weights before playing a file.")
            return
        self._set_file_ui_busy(True)
        self.file_progress.setValue(0)
        self.lbl_file_status.setText("Stopping live stream…")
        self.engine.stop()
        self.lbl_file_status.setText("")
        self._file_thread = DenoiseFileThread(
            self.model,
            self.engine,
            self._file_path,
            self.engine.strength,
            parent=self,
        )
        self._file_thread.progress.connect(self.file_progress.setValue)
        self._file_thread.status.connect(self.lbl_file_status.setText)
        self._file_thread.failed.connect(self._on_file_failed)
        self._file_thread.finished.connect(self._on_file_thread_finished)
        self._file_thread.start()

    def _on_file_stop(self) -> None:
        if self._transcribe_thread is not None and self._transcribe_thread.isRunning():
            self.lbl_file_status.setText("Stopping…")
            self._transcribe_thread.requestInterruption()
            self._transcribe_thread.wait(20_000)
            return
        if self._file_thread is not None and self._file_thread.isRunning():
            self.lbl_file_status.setText("Stopping…")
            self._file_thread.stop_playback()
            self._file_thread.wait(20_000)

    def _on_file_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "File playback", msg)

    def _on_file_thread_finished(self) -> None:
        self._set_file_ui_busy(False)
        self.file_progress.setValue(0)
        self.lbl_file_status.setText("")
        try:
            self.engine.restart()
        except Exception:
            logger.exception("Could not restart live audio after file playback")
        self._refresh_device_labels()
        self._file_thread = None

    def _on_file_transcribe(self) -> None:
        if self._file_job_busy():
            return
        if self._file_path is None or not self._file_path.is_file():
            QMessageBox.warning(self, "ClearVoice", "Choose an audio file first.")
            return
        if self._transcriber_holder.is_ready:
            self._start_file_transcribe_job()
            return
        self._pending_file_transcribe_after_load = True
        self._set_file_ui_busy(True)
        self.file_progress.setValue(0)
        self._start_load_transcriber(for_file=True)

    def _start_file_transcribe_job(self) -> None:
        if self._file_path is None or not self._file_path.is_file():
            return
        self._set_file_ui_busy(True)
        self.file_progress.setValue(0)
        self.lbl_file_status.setText("Stopping live stream…")
        self.engine.stop()
        self.lbl_file_status.setText("")
        self.txt_transcript.clear()
        self._transcript_displayed_count = 0
        self._transcribe_thread = TranscribeFileThread(
            self._transcriber_holder.get,
            self._file_path,
            denoise_first=self.settings.file_transcribe_denoise_first,
            denoise_model=self.model if self.settings.file_transcribe_denoise_first else None,
            denoise_strength=self.engine.strength,
            parent=self,
        )
        self._transcribe_thread.progress.connect(self.file_progress.setValue)
        self._transcribe_thread.status.connect(self.lbl_file_status.setText)
        self._transcribe_thread.segment.connect(self._append_transcript_text)
        self._transcribe_thread.failed.connect(self._on_transcribe_failed)
        self._transcribe_thread.finished.connect(self._on_transcribe_thread_finished)
        self._transcribe_thread.finished_ok.connect(self._on_transcribe_finished_ok)
        self._transcribe_thread.start()

    def _on_transcribe_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "Transcription", msg)

    def _on_transcribe_finished_ok(self, full_text: str) -> None:
        if full_text and not self.txt_transcript.toPlainText().strip():
            self._append_transcript_text(full_text)

    def _on_transcribe_thread_finished(self) -> None:
        self._set_file_ui_busy(False)
        self.file_progress.setValue(0)
        self.lbl_file_status.setText("")
        try:
            self.engine.restart()
        except Exception:
            logger.exception("Could not restart live audio after file transcription")
        self._refresh_device_labels()
        self._transcribe_thread = None
        self._sync_transcribe_ui()

    def _on_transcribe_toggle(self, checked: bool) -> None:
        if self._load_transcriber_thread is not None and self._load_transcriber_thread.isRunning():
            self.btn_transcribe.blockSignals(True)
            self.btn_transcribe.setChecked(not checked)
            self.btn_transcribe.blockSignals(False)
            return
        self.settings.transcribe_active = bool(checked)
        self.settings.save()
        if checked:
            if self._transcriber_holder.is_ready:
                self._activate_live_transcription()
            else:
                self._pending_transcribe_after_load = True
                self.btn_transcribe.setEnabled(False)
                self._start_load_transcriber(for_file=False)
            return
        self.transcription.active = False
        self.transcription.paused = False
        self._sync_transcribe_ui()

    def _start_load_transcriber(self, *, for_file: bool) -> None:
        if self._load_transcriber_thread is not None and self._load_transcriber_thread.isRunning():
            return
        self.lbl_tx_status.setText(
            "Loading Whisper (first run ~1.6 GB download — do not close the app)…"
        )
        self.lbl_tx_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
        if for_file:
            self.lbl_file_status.setText(self.lbl_tx_status.text())
        self._load_transcriber_thread = LoadTranscriberThread(
            self._transcriber_holder,
            parent=self,
        )
        self._load_transcriber_thread.status.connect(self._on_load_transcriber_status)
        self._load_transcriber_thread.finished_ok.connect(self._on_load_transcriber_ok)
        self._load_transcriber_thread.failed.connect(self._on_load_transcriber_failed)
        self._load_transcriber_thread.finished.connect(self._on_load_transcriber_finished)
        self._load_transcriber_thread.start()

    def _on_load_transcriber_status(self, msg: str) -> None:
        self.lbl_tx_status.setText(msg)
        if self._pending_file_transcribe_after_load:
            self.lbl_file_status.setText(msg)

    def _on_load_transcriber_ok(self, _model: object) -> None:
        if self._pending_transcribe_after_load:
            self._pending_transcribe_after_load = False
            self._activate_live_transcription()
        if self._pending_file_transcribe_after_load:
            self._pending_file_transcribe_after_load = False
            self._start_file_transcribe_job()

    def _on_load_transcriber_failed(self, msg: str) -> None:
        was_file = self._pending_file_transcribe_after_load
        self._pending_transcribe_after_load = False
        self._pending_file_transcribe_after_load = False
        self.btn_transcribe.blockSignals(True)
        self.btn_transcribe.setChecked(False)
        self.btn_transcribe.blockSignals(False)
        self.settings.transcribe_active = False
        self.settings.save()
        self.transcription.active = False
        if was_file:
            self._set_file_ui_busy(False)
            self.lbl_file_status.setText("")
        QMessageBox.warning(self, "Transcription", msg)
        self._sync_transcribe_ui()

    def _on_load_transcriber_finished(self) -> None:
        self.btn_transcribe.setEnabled(not self._file_job_busy())
        self._load_transcriber_thread = None

    def _activate_live_transcription(self) -> None:
        self.transcription.start_worker()
        self.transcription.active = True
        self.transcription.paused = False
        self._sync_transcribe_ui()

    def _on_tx_denoised_toggled(self, checked: bool) -> None:
        self.settings.transcribe_denoised_audio = bool(checked)
        self.engine.transcribe_denoised = bool(checked)
        self.settings.save()

    def _on_tx_timing_changed(self, _val: int = 0) -> None:
        chunk = float(self.spin_tx_chunk.value())
        step = float(min(self.spin_tx_step.value(), chunk))
        if self.spin_tx_step.value() > chunk:
            self.spin_tx_step.blockSignals(True)
            self.spin_tx_step.setValue(int(chunk))
            self.spin_tx_step.blockSignals(False)
        self.transcription.set_segment_sec(chunk)
        self.transcription.set_hop_sec(step)
        self.settings.transcription_segment_sec = chunk
        self.settings.transcription_hop_sec = step
        self.settings.save()

    def _on_file_denoise_first_toggled(self, checked: bool) -> None:
        self.settings.file_transcribe_denoise_first = bool(checked)
        self.settings.save()

    def _on_transcript_clear(self) -> None:
        self.transcription.clear()
        self.txt_transcript.clear()
        self._transcript_displayed_count = 0

    def _on_transcript_copy(self) -> None:
        text = self.txt_transcript.toPlainText()
        if text:
            QApplication.clipboard().setText(text)

    def _on_transcript_save(self) -> None:
        text = self.txt_transcript.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Transcription", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save transcript",
            "transcript.txt",
            "Text (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "Transcription", f"Could not save file:\n{e}")

    def _set_transcript_status(self, msg: str) -> None:
        self.lbl_tx_status.setText(msg)

    def _append_transcript_text(self, text: str) -> None:
        if not text:
            return
        self.txt_transcript.append(text.strip())
        cursor = self.txt_transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.txt_transcript.setTextCursor(cursor)
        self.txt_transcript.ensureCursorVisible()
        self._transcript_displayed_count = len(self.transcription.transcript_lines)

    def _sync_transcript_display(self) -> None:
        """Fallback: mirror transcript_lines if a signal was missed."""
        lines = self.transcription.transcript_lines
        n = len(lines)
        if n <= self._transcript_displayed_count:
            return
        for line in lines[self._transcript_displayed_count :]:
            self._append_transcript_text(line)
        self._transcript_displayed_count = n

    def _sync_transcribe_ui(self) -> None:
        on = self.settings.transcribe_active and self.transcription.active
        self.btn_transcribe.blockSignals(True)
        self.btn_transcribe.setChecked(on)
        self.btn_transcribe.setText("Transcription on" if on else "Transcription off")
        self.btn_transcribe.blockSignals(False)
        tm = self._transcriber_holder.loaded_model
        if on and tm is not None and tm.is_ready:
            self.lbl_tx_status.setText(f"Whisper ready ({tm.inference_device})")
            self.lbl_tx_status.setStyleSheet("color: #6ee7b7; font-size: 12px;")
        elif tm is not None and tm.load_error:
            self.lbl_tx_status.setText(tm.load_error)
            self.lbl_tx_status.setStyleSheet("color: #fca5a5; font-size: 12px;")
        else:
            self.lbl_tx_status.setText("Transcription paused" if not on else "")
            self.lbl_tx_status.setStyleSheet("")
        if self.transcription.latency_ms > 0:
            self.lbl_tx_lat.setText(f"{self.transcription.latency_ms:.0f} ms/seg")
        n = len(self.transcription.transcript_lines)
        self.lbl_tx_seg.setText(f"{n} segments")
        if on and tm is not None and tm.is_ready and n == 0:
            skip = self.transcription.last_skip_reason
            if skip:
                self.lbl_tx_status.setText(f"Listening… ({skip})")
            elif not self.transcription.active:
                self.lbl_tx_status.setText("Waiting for audio…")

    def _populate_devices(self) -> None:
        # ── DEVICE SELECTION UI commented out ── just refresh the info labels ──
        self._refresh_device_labels()

    def _refresh_device_labels(self) -> None:
        """Update the read-only capture/output info labels from active engine indices."""
        in_idx, out_idx = self.engine.get_device_indices()
        devs = {d["index"]: d for d in self.engine.list_devices()}

        def _dev_label(idx: int | None, kind: str) -> str:
            if idx is None:
                return f"Default {kind}"
            d = devs.get(idx)
            if d is None:
                return f"#{idx}"
            api = d.get("api_short") or ""
            return f"{d['name']}  [{api}]" if api else d["name"]

        in_label = _dev_label(in_idx, "input")

        # Check whether the capture device is actually a loopback/mix device.
        if in_idx is not None:
            d = devs.get(in_idx)
            name = d["name"] if d else ""
            is_loopback = AudioEngine._is_loopback_device(name)
        else:
            is_loopback = False

        in_d = devs.get(in_idx) if in_idx is not None else None
        dev_name = in_d["name"] if in_d else ""
        is_vbcable_capture = "CABLE Output" in dev_name

        if not self.engine.streaming:
            cap_text = "⚠  No audio stream open — check logs."
            self.lbl_capture_device.setStyleSheet(
                "color: #fca5a5; font-size: 12px; line-height: 1.35; font-weight: 500;"
            )
        elif is_vbcable_capture:
            cap_text = (
                f"✓  {in_label}\n"
                "VB-Cable mode — true intercept.  All apps routed to "
                "CABLE Input will be denoised.  Set CABLE Input as your "
                "Windows default playback device."
            )
            self.lbl_capture_device.setStyleSheet(
                "color: #6ee7b7; font-size: 12px; line-height: 1.35; font-weight: 500;"
            )
        elif is_loopback:
            cap_text = (
                f"⚡  {in_label}\n"
                "Stereo Mix mode — ClearVoice hears everything, but the "
                "original audio ALSO plays through the speakers at the same "
                "time.  For true replacement, install VB-Cable "
                "(vb-audio.com/Cable) and set CABLE Input as your default "
                "Windows output."
            )
            self.lbl_capture_device.setStyleSheet(
                "color: #fcd34d; font-size: 12px; line-height: 1.35; font-weight: 500;"
            )
        else:
            cap_text = (
                f"⚠  {in_label}\n"
                "Not a system-audio source.  Enable Stereo Mix (Recording → "
                "Show Disabled Devices) or install VB-Cable."
            )
            self.lbl_capture_device.setStyleSheet(
                "color: #fca5a5; font-size: 12px; line-height: 1.35; font-weight: 500;"
            )

        self.lbl_capture_device.setText(cap_text)
        self.lbl_output_device.setText(_dev_label(out_idx, "output"))
        self.lbl_output_device.setStyleSheet("color: #c7d2fe; font-size: 12px; font-weight: 500;")

    # ── DEVICE SELECTION UI ── commented out until further notice ──────────────
    # @staticmethod
    # def _select_combo_by_data(combo, index_val):
    #     ...
    # def _apply_devices(self):
    #     ...
    # ── END DEVICE SELECTION UI ─────────────────────────────────────────────────

    def _browse_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model weights",
            str(Path(self.settings.weights_path).parent),
            "PyTorch (*.pt *.pth);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        self.settings.weights_path = str(p.resolve())
        self.settings.save()
        self.model.reload(p)
        self.lbl_model.setText(self.model.model_name if self.model.is_ready else "Not loaded")
        self._sync_model_error_ui()
        if self.model.is_ready and not self.engine.streaming:
            self.engine.start(self.settings.input_device_index, self.settings.output_device_index)
        elif self.model.is_ready:
            self.engine.restart()
        self.engine.sync_resolved_devices_to_settings(self.settings)
        self._refresh_device_labels()
        self._update_file_play_button()

    def _on_power(self, checked: bool) -> None:
        self.engine.active = checked
        self.settings.denoise_active = checked
        self.settings.save()
        self._sync_power_ui()

    def _sync_power_ui(self) -> None:
        on = self.engine.active
        self.btn_power.blockSignals(True)
        self.btn_power.setChecked(on)
        self.btn_power.blockSignals(False)
        self.btn_power.setText("Denoising on" if on else "Denoising off")
        chip = self.title_bar.status_chip
        err = self.engine.error_message or (self.model.load_error if not self.model.is_ready else None)
        if err:
            chip.setStyleSheet(
                "QFrame#statusChip { background-color: rgba(248,113,113,0.12); "
                "border: 1px solid rgba(248,113,113,0.45); border-radius: 12px; }"
            )
            self.title_bar.lbl_dot.setStyleSheet("color: #f87171; font-size: 10px; font-weight: 800;")
            self.title_bar.lbl_status.setText("Error")
            self.title_bar.lbl_status.setStyleSheet("color: #fecaca; font-size: 11px; font-weight: 600;")
        elif on:
            chip.setStyleSheet(
                "QFrame#statusChip { background-color: rgba(52,211,153,0.14); "
                "border: 1px solid rgba(52,211,153,0.45); border-radius: 12px; }"
            )
            self.title_bar.lbl_dot.setStyleSheet("color: #34d399; font-size: 10px; font-weight: 800;")
            self.title_bar.lbl_status.setText("Live")
            self.title_bar.lbl_status.setStyleSheet("color: #d1fae5; font-size: 11px; font-weight: 600;")
        else:
            chip.setStyleSheet(
                "QFrame#statusChip { background-color: rgba(148,163,184,0.1); "
                "border: 1px solid rgba(100,116,139,0.35); border-radius: 12px; }"
            )
            self.title_bar.lbl_dot.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 800;")
            self.title_bar.lbl_status.setText("Bypass")
            self.title_bar.lbl_status.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 600;")

    def _sync_model_error_ui(self) -> None:
        if self.model.load_error and not self.model.is_ready:
            self.lbl_model.setText(f"Error: {self.model.load_error}")
            self.lbl_model.setStyleSheet("color: #fca5a5; font-weight: 500;")
        elif self.model.is_ready:
            self.lbl_model.setText(self.model.model_name)
            self.lbl_model.setStyleSheet("color: #e8ebf2; font-weight: 500;")
        self._sync_power_ui()
        self._update_file_play_button()

    def _on_strength_changed(self, val: int) -> None:
        self.engine.strength = val / 100.0
        self.lbl_strength.setText(f"{val}%")

    def _on_strength_released(self) -> None:
        self.settings.strength = self.engine.strength
        self.settings.save()

    def _on_inference_changed(self, _checked: bool = False) -> None:
        dev = "cuda" if self.radio_cuda.isChecked() and torch.cuda.is_available() else "cpu"
        self.settings.inference_device = dev
        self.settings.save()
        self.model.set_inference_device(dev)

    def _on_startup_toggled(self, checked: bool) -> None:
        self.settings.launch_on_startup = bool(checked)
        self.settings.save()
        try:
            self._apply_launch_on_startup()
        except Exception:
            logger.exception("apply_launch_on_startup failed")

    def _tick_waveforms(self) -> None:
        self.wave_in.update()
        self.wave_out.update()

    def _tick_meters(self) -> None:
        lin_in = max(self.engine.input_level, 1e-9)
        lin_out = max(self.engine.output_level, 1e-9)
        db_in = _linear_to_db(lin_in)
        db_out = _linear_to_db(lin_out)
        self.meter_in.set_level_db(db_in, min(1.0, lin_in * 8.0))
        self.meter_out.set_level_db(db_out, min(1.0, lin_out * 8.0))
        ratio = (lin_in + 1e-9) / (lin_out + 1e-9)
        red_db = max(0.0, min(48.0, 20.0 * math.log10(ratio)))
        self.meter_red.set_level_db(red_db, min(1.0, red_db / 36.0))

    def _tick_stats(self) -> None:
        import psutil

        cpu = psutil.cpu_percent(interval=None)
        lat = self.engine.latency_ms
        sr = self.engine.stream_sample_rate
        suffix = ""
        if self._file_playback_busy():
            suffix = "  |  File denoise / playback"
        elif self._file_transcribe_busy():
            suffix = "  |  File transcription"
        self.lbl_stats.setText(
            f"Latency: {lat:.1f} ms   |   CPU: {cpu:.0f}%   |   Stream: {sr} Hz   |   Model: 16000 Hz{suffix}"
        )
        self._sync_power_ui()
        self._sync_transcribe_ui()
        self._sync_transcript_display()
        self._refresh_device_labels()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.settings.window_x is not None and self.settings.window_y is not None:
            self.move(self.settings.window_x, self.settings.window_y)

    def moveEvent(self, event) -> None:
        super().moveEvent(event)

    def closeEvent(self, event) -> None:
        if self._load_transcriber_thread is not None and self._load_transcriber_thread.isRunning():
            self._load_transcriber_thread.wait(60_000)
        if self._file_thread is not None and self._file_thread.isRunning():
            self._file_thread.stop_playback()
            self._file_thread.wait(15_000)
        if self._transcribe_thread is not None and self._transcribe_thread.isRunning():
            self._transcribe_thread.requestInterruption()
            self._transcribe_thread.wait(20_000)
        self.transcription.stop_worker()
        p = self.pos()
        self.settings.window_x = int(p.x())
        self.settings.window_y = int(p.y())
        self.settings.save()
        event.accept()
        QApplication.instance().quit()
