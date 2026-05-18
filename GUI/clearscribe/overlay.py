"""
OverlayWindow — frameless always-on-top PyQt6 overlay for ClearScribe.
Mirrors the look and feel of the ClearVoice denoiser overlay.
"""
import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFrame,
    QTextEdit, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor


_STYLE = """
QWidget {
    background: #1e1e1e;
    color: #e0e0e0;
    font-family: Segoe UI;
    font-size: 12px;
}
QLabel  { color: #e0e0e0; }
QPushButton {
    background: #2d2d2d;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 5px 10px;
    color: #e0e0e0;
}
QPushButton:hover   { background: #3a3a3a; }
QPushButton:checked { background: #2a5e8c; border-color: #3a8ec8; }
QComboBox {
    background: #2d2d2d;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 6px;
}
QComboBox::drop-down { border: none; }
QTextEdit {
    background: #121212;
    border: 1px solid #333;
    border-radius: 6px;
    color: #d0e8ff;
    font-family: Consolas, Courier New, monospace;
    font-size: 12px;
    padding: 4px;
}
"""


class OverlayWindow(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine    = engine
        self._drag_pos = None

        self._build_ui()
        self._populate_devices()
        self._setup_timer()

        # hook engine callback so new text arrives immediately
        self.engine.on_new_text = self._on_new_text

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("ClearScribe")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint  |
            Qt.WindowType.Tool
        )
        self.setFixedWidth(380)
        self.setStyleSheet(_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        # ---- title bar ----
        titlebar = QHBoxLayout()
        lbl_title = QLabel("🎤 ClearScribe")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.lbl_status = QLabel("● Listening")
        self.lbl_status.setStyleSheet("color: #5a9e6f; font-size: 11px;")
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(22, 22)
        btn_close.setStyleSheet(
            "background:transparent; border:none; color:#888; font-size:14px;"
        )
        btn_close.clicked.connect(self.close)
        titlebar.addWidget(lbl_title)
        titlebar.addStretch()
        titlebar.addWidget(self.lbl_status)
        titlebar.addSpacing(8)
        titlebar.addWidget(btn_close)
        layout.addLayout(titlebar)

        layout.addWidget(self._hline())

        # ---- toggle ----
        row1 = QHBoxLayout()
        self.btn_toggle = QPushButton("Transcription  ON")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.clicked.connect(self._on_toggle)
        row1.addWidget(self.btn_toggle)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(60)
        self.btn_clear.clicked.connect(self._on_clear)
        row1.addWidget(self.btn_clear)
        layout.addLayout(row1)

        layout.addWidget(self._hline())

        # ---- device selector ----
        layout.addWidget(QLabel("Input device (microphone)"))
        self.combo_in = QComboBox()
        layout.addWidget(self.combo_in)

        btn_apply = QPushButton("Apply device & restart")
        btn_apply.clicked.connect(self._apply_device)
        layout.addWidget(btn_apply)

        layout.addWidget(self._hline())

        # ---- transcript box ----
        layout.addWidget(QLabel("Transcript"))
        self.txt_transcript = QTextEdit()
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setFixedHeight(180)
        self.txt_transcript.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.txt_transcript)

        layout.addWidget(self._hline())

        # ---- input meter ----
        meter_row = QHBoxLayout()
        meter_row.addWidget(QLabel("Input"))
        self.bar_in = QLabel()
        self.bar_in.setFixedSize(140, 6)
        self.bar_in.setStyleSheet("background:#3a3a3a; border-radius:3px;")
        self.lbl_in = QLabel("—")
        self.lbl_in.setFixedWidth(48)
        self.lbl_in.setAlignment(Qt.AlignmentFlag.AlignRight)
        meter_row.addStretch()
        meter_row.addWidget(self.bar_in)
        meter_row.addWidget(self.lbl_in)
        layout.addLayout(meter_row)

        layout.addWidget(self._hline())

        # ---- stats ----
        stats = QHBoxLayout()
        self.lbl_lat = QLabel("— ms")
        self.lbl_seg = QLabel("— segments")
        for w in [self.lbl_lat, self.lbl_seg]:
            w.setStyleSheet("color:#888; font-size:11px;")
            stats.addWidget(w)
        stats.addStretch()
        layout.addLayout(stats)

    # ------------------------------------------------------------------
    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333;")
        return line

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_stats)
        self.timer.start(150)

    def _populate_devices(self):
        for idx, name in self.engine.list_devices():
            self.combo_in.addItem(name, idx)

    # ------------------------------------------------------------------
    def _on_toggle(self, checked: bool):
        self.engine.active = checked
        self.btn_toggle.setText(
            "Transcription  ON" if checked else "Transcription  OFF"
        )
        self.lbl_status.setText("● Listening" if checked else "● Paused")
        self.lbl_status.setStyleSheet(
            "color:#5a9e6f;" if checked else "color:#888;"
        )

    def _on_clear(self):
        self.engine.transcript_lines.clear()
        self.engine.last_text = ""
        self.txt_transcript.clear()

    def _apply_device(self):
        self.engine.stop()
        self.engine.input_device = self.combo_in.currentData()
        self.engine.start()

    def _on_new_text(self, text: str):
        """Called from the inference thread — post to main thread via QTimer."""
        QTimer.singleShot(0, lambda: self._append_text(text))

    def _append_text(self, text: str):
        self.txt_transcript.append(text)
        # auto-scroll to bottom
        cursor = self.txt_transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.txt_transcript.setTextCursor(cursor)

    def _refresh_stats(self):
        # input level meter
        val = self.engine.input_level
        pct = min(int(val * 600), 140)
        ratio = pct / 140
        self.bar_in.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #5a8cae, stop:{ratio:.2f} #5a8cae, "
            f"stop:{min(ratio+0.01,1):.2f} #3a3a3a, stop:1 #3a3a3a);"
            "border-radius:3px;"
        )
        db = f"{20*math.log10(val+1e-9):.0f} dB" if val > 0 else "— dB"
        self.lbl_in.setText(db)

        # stats bar
        if self.engine.latency_ms > 0:
            self.lbl_lat.setText(f"{self.engine.latency_ms:.0f} ms/seg")
        self.lbl_seg.setText(f"{len(self.engine.transcript_lines)} segments")

    # ------------------------------------------------------------------
    # draggable frameless window
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() == Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None
