from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget


MIN_REGION_SIZE = 8


@dataclass(frozen=True, slots=True)
class RegionSelection:
    screen_name: str
    rect: QRect


def normalize_selection(start: QPoint, end: QPoint, minimum: int = MIN_REGION_SIZE) -> QRect | None:
    left = min(start.x(), end.x())
    top = min(start.y(), end.y())
    right = max(start.x(), end.x())
    bottom = max(start.y(), end.y())
    rect = QRect(left, top, right - left, bottom - top)
    if rect.width() < minimum or rect.height() < minimum:
        return None
    return rect


def screen_at_cursor():
    app = QGuiApplication.instance()
    if app is None:
        return None
    return app.screenAt(QCursor.pos()) or app.primaryScreen()


def qimage_to_bgr(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    data = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes())
    rgba = data.reshape((height, image.bytesPerLine()))[:, : width * 4].reshape((height, width, 4))
    import cv2

    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR).copy()


def grab_region(screen, rect: QRect) -> np.ndarray:
    geometry = screen.geometry()
    pixmap = screen.grabWindow(0, rect.x() - geometry.x(), rect.y() - geometry.y(), rect.width(), rect.height())
    if pixmap.isNull():
        raise RuntimeError("无法截取选定区域")
    frame = qimage_to_bgr(pixmap.toImage())
    if frame.size == 0:
        raise RuntimeError("选定区域截图为空")
    return frame


class RegionSelector(QWidget):
    selected = Signal(object)
    cancelled = Signal()

    def __init__(self, screen, parent=None) -> None:
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
        super().__init__(None, flags)
        self._screen = screen
        self._start: QPoint | None = None
        self._current: QPoint | None = None
        self._finished = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(screen.geometry())
        self._hint = QLabel("拖拽选择区域 · Esc 取消", self)
        self._hint.setStyleSheet("color: white; background: rgba(0, 0, 0, 150); padding: 6px 10px;")
        self._hint.adjustSize()
        self._hint.move(16, 16)

    def begin(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self.grabMouse()
        self.grabKeyboard()

    def _finish_cancel(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.releaseMouse()
        self.releaseKeyboard()
        self.hide()
        self.cancelled.emit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._finish_cancel()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self._current = self._start
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._start is not None:
            self._current = event.position().toPoint()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._start is not None:
            self._current = event.position().toPoint()
            rect = normalize_selection(self._start, self._current)
            if rect is None:
                self._hint.setText("区域太小，请重新拖拽 · Esc 取消")
                self._hint.adjustSize()
                self._start = None
                self._current = None
                self.update()
            else:
                self._finished = True
                self.releaseMouse()
                self.releaseKeyboard()
                self.hide()
                global_rect = QRect(rect)
                global_rect.moveTopLeft(self.geometry().topLeft() + rect.topLeft())
                self.selected.emit(RegionSelection(self._screen.name(), global_rect))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 105))
        if self._start is not None and self._current is not None:
            rect = normalize_selection(self._start, self._current, minimum=0)
            if rect is not None:
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                painter.fillRect(rect, Qt.GlobalColor.transparent)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                painter.setPen(QPen(QColor(80, 180, 255), 2))
                painter.drawRect(rect)
                self._hint.setText(f"{rect.width()} × {rect.height()} · 松开鼠标完成")
                self._hint.adjustSize()
        painter.end()

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._finished:
            self._finish_cancel()
        event.accept()
