"""System tray integration for ClearVoice."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from audio_engine import AudioEngine
from settings import Settings
from transcription import RealtimeTranscription

logger = logging.getLogger(__name__)


def _circle_icon(active: bool) -> QIcon:
    pm = QPixmap(22, 22)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = QColor("#4caf7d") if active else QColor("#888888")
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(2, 2, 18, 18)
    painter.end()
    return QIcon(pm)


class TrayIcon(QSystemTrayIcon):
    def __init__(
        self,
        parent: QWidget | None,
        overlay: QWidget,
        engine: AudioEngine,
        settings: Settings,
        transcription: RealtimeTranscription,
    ) -> None:
        super().__init__(parent)
        self._overlay = overlay
        self._engine = engine
        self._settings = settings
        self._transcription = transcription
        self.setIcon(_circle_icon(engine.active))
        self.setToolTip("ClearVoice")

        menu = QMenu()
        act_show = QAction("Show", self)
        act_show.triggered.connect(self._show_overlay)
        menu.addAction(act_show)
        act_toggle = QAction("Toggle denoising", self)
        act_toggle.triggered.connect(self._toggle_denoise)
        menu.addAction(act_toggle)
        act_tx = QAction("Toggle transcription", self)
        act_tx.triggered.connect(self._toggle_transcription)
        menu.addAction(act_tx)
        menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self._exit_app)
        menu.addAction(act_exit)
        self.setContextMenu(menu)

        self.activated.connect(self._on_activated)

        self._timer = None
        try:
            from PyQt6.QtCore import QTimer

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh_icon)
            self._timer.start(400)
        except Exception:
            logger.exception("Tray icon refresh timer setup failed")

    def _refresh_icon(self) -> None:
        self.setIcon(_circle_icon(self._engine.active))

    def _show_overlay(self) -> None:
        self._overlay.show()
        self._overlay.raise_()
        self._overlay.activateWindow()

    def _toggle_transcription(self) -> None:
        if hasattr(self._overlay, "btn_transcribe"):
            self._overlay.btn_transcribe.toggle()

    def _toggle_denoise(self) -> None:
        self._engine.active = not self._engine.active
        self._settings.denoise_active = self._engine.active
        self._settings.save()
        if hasattr(self._overlay, "_sync_power_ui"):
            getattr(self._overlay, "_sync_power_ui")()
        self.setIcon(_circle_icon(self._engine.active))

    def _exit_app(self) -> None:
        from PyQt6.QtWidgets import QApplication

        QApplication.instance().quit()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self._overlay.isVisible():
                self._overlay.hide()
            else:
                self._show_overlay()

    def notify_startup(self) -> None:
        try:
            self.showMessage("ClearVoice", "ClearVoice is running", QSystemTrayIcon.MessageIcon.Information, 2500)
        except Exception:
            logger.exception("Tray startup message failed")
