"""Lightweight icon helpers for the OmniViz toolbar.

Two sources, no new asset files:

* :func:`standard_icon` wraps Qt's built-in ``QStyle.StandardPixmap`` set so
  common actions (import, reload, save) get familiar glyphs for free.
* :func:`badge_icon` paints a small rounded "badge" with a short label
  (``X`` / ``Y`` / ``Z`` / ``ISO`` / ``FLIP``) via ``QPainter`` — used for the
  camera-view actions that have no sensible standard glyph.
"""

from __future__ import annotations

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import QApplication, QStyle

_BADGE_ACCENT = "#2f81f7"


def standard_icon(name: str) -> QIcon:
    """Return a built-in Qt standard icon by ``QStyle.StandardPixmap`` name.

    Falls back to an empty :class:`QIcon` if the name is unknown or no style is
    available (e.g. under some headless configurations).
    """
    app = QApplication.instance()
    style = app.style() if app is not None else None
    if style is None:
        return QIcon()
    pixmap = getattr(QStyle.StandardPixmap, name, None)
    if pixmap is None:
        return QIcon()
    return style.standardIcon(pixmap)


def badge_icon(
    text: str,
    *,
    size: int = 22,
    bg: str = _BADGE_ACCENT,
    fg: str = "#ffffff",
) -> QIcon:
    """Paint ``text`` into a rounded-rectangle badge and return it as an icon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(bg))
    rect = QRectF(1, 1, size - 2, size - 2)
    radius = size * 0.28
    painter.drawRoundedRect(rect, radius, radius)

    # Shrink the font for longer labels (ISO / FLIP) so they still fit.
    point = 10.0 if len(text) <= 1 else max(5.0, 9.0 - (len(text) - 1) * 1.4)
    font = QFont()
    font.setBold(True)
    font.setPointSizeF(point)
    painter.setFont(font)
    painter.setPen(QColor(fg))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return QIcon(pixmap)


def swatch_icon(color: QColor, *, size: int = 16) -> QIcon:
    """Return a small rounded colour swatch icon for a colour-picker button."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QColor("#00000040"))
    painter.setBrush(color)
    painter.drawRoundedRect(QRectF(0.5, 0.5, size - 1, size - 1), 4, 4)
    painter.end()
    return QIcon(pixmap)
