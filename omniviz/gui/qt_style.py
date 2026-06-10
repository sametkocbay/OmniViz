"""Cohesive dark/light Qt stylesheet (QSS) for the OmniViz single window.

A single accent colour (the OmniViz blue, also used by the old CTk theme) ties
the whole UI together. :func:`stylesheet` returns a ready-to-apply QSS string
for either ``"dark"`` or ``"light"`` mode; :func:`apply_theme` installs it on a
running ``QApplication`` so the appearance can be switched live.

This module is purely cosmetic — it touches no plotter / IO logic and adds no
new asset dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qtpy.QtWidgets import QApplication

log = logging.getLogger(__name__)

#: Shared accent (matches ``omniviz.gui.theme.ACCENT`` / the old CTk blue).
ACCENT = "#2f81f7"
ACCENT_HOVER = "#4b94f9"
ACCENT_PRESSED = "#1f6aa5"


@dataclass(frozen=True)
class _Palette:
    """The handful of colours every widget rule is derived from."""

    window: str
    base: str  # input / list backgrounds
    panel: str  # dock / group-box backgrounds
    border: str
    text: str
    text_muted: str
    selection: str
    hover: str
    accent: str = ACCENT
    accent_hover: str = ACCENT_HOVER
    accent_pressed: str = ACCENT_PRESSED


_DARK = _Palette(
    window="#1b1f24",
    base="#22272e",
    panel="#1e242b",
    border="#313a44",
    text="#e6edf3",
    text_muted="#8b97a3",
    selection="#2f81f7",
    hover="#2c333c",
)

_LIGHT = _Palette(
    window="#f3f5f8",
    base="#ffffff",
    panel="#eaeef3",
    border="#cdd5df",
    text="#1b1f24",
    text_muted="#5c6773",
    selection="#2f81f7",
    hover="#e2e8f0",
)


def _palette(mode: str) -> _Palette:
    return _LIGHT if mode.lower() == "light" else _DARK


def viz_background(mode: str = "dark") -> str:
    """Background colour for the embedded 3D view, matched to the theme."""
    return _palette(mode).panel


def stylesheet(mode: str = "dark") -> str:
    """Return the full QSS string for the requested ``mode``."""
    p = _palette(mode)
    # Text on the accent is always near-white for contrast.
    on_accent = "#ffffff"
    return f"""
    /* ---- base ------------------------------------------------------- */
    QWidget {{
        background-color: {p.window};
        color: {p.text};
        font-size: 13px;
    }}
    QMainWindow, QMainWindow > QWidget {{
        background-color: {p.window};
    }}
    QToolTip {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.border};
        padding: 4px 6px;
        border-radius: 4px;
    }}

    /* ---- dock widgets ---------------------------------------------- */
    QDockWidget {{
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
        color: {p.text};
        font-weight: 600;
    }}
    QDockWidget::title {{
        background-color: {p.panel};
        padding: 7px 10px;
        border-bottom: 2px solid {p.accent};
        text-align: left;
    }}

    /* ---- group boxes / section headers ----------------------------- */
    QGroupBox {{
        background-color: {p.panel};
        border: 1px solid {p.border};
        border-radius: 8px;
        margin-top: 14px;
        padding: 10px 8px 8px 8px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
        color: {p.text_muted};
    }}

    /* ---- tabs ------------------------------------------------------- */
    QTabWidget::pane {{
        border: 1px solid {p.border};
        border-radius: 6px;
        top: -1px;
        background-color: {p.panel};
    }}
    QTabBar::tab {{
        background: transparent;
        color: {p.text_muted};
        padding: 7px 12px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}
    QTabBar::tab:hover {{
        color: {p.text};
        background: {p.hover};
    }}
    QTabBar::tab:selected {{
        color: {p.text};
        background: {p.panel};
        border-bottom: 2px solid {p.accent};
    }}

    /* ---- buttons ---------------------------------------------------- */
    QPushButton {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        padding: 6px 12px;
    }}
    QPushButton:hover {{
        background-color: {p.hover};
        border-color: {p.accent};
    }}
    QPushButton:pressed {{
        background-color: {p.accent_pressed};
        color: {on_accent};
    }}
    QPushButton:disabled {{
        color: {p.text_muted};
        background-color: {p.panel};
    }}
    QPushButton[class="primary"] {{
        background-color: {p.accent};
        color: {on_accent};
        border: 1px solid {p.accent};
        font-weight: 600;
    }}
    QPushButton[class="primary"]:hover {{
        background-color: {p.accent_hover};
    }}
    QPushButton[class="primary"]:pressed {{
        background-color: {p.accent_pressed};
    }}

    /* ---- toolbar ---------------------------------------------------- */
    QToolBar {{
        background-color: {p.panel};
        border: none;
        border-bottom: 1px solid {p.border};
        padding: 4px;
        spacing: 4px;
    }}
    QToolBar::separator {{
        background: {p.border};
        width: 1px;
        margin: 4px 6px;
    }}
    QToolButton {{
        background: transparent;
        color: {p.text};
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 5px 7px;
    }}
    QToolButton:hover {{
        background-color: {p.hover};
        border-color: {p.border};
    }}
    QToolButton:pressed, QToolButton:checked {{
        background-color: {p.accent};
        color: {on_accent};
    }}

    /* ---- menus ------------------------------------------------------ */
    QMenuBar {{
        background-color: {p.panel};
        color: {p.text};
        border-bottom: 1px solid {p.border};
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 5px 10px;
    }}
    QMenuBar::item:selected {{
        background-color: {p.hover};
        border-radius: 4px;
    }}
    QMenu {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.border};
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 12px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {p.accent};
        color: {on_accent};
    }}
    QMenu::separator {{
        height: 1px;
        background: {p.border};
        margin: 4px 8px;
    }}

    /* ---- lists / trees --------------------------------------------- */
    QListWidget, QTreeWidget, QTreeView {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        outline: 0;
        alternate-background-color: {p.panel};
    }}
    QListWidget::item, QTreeWidget::item {{
        padding: 4px 6px;
        border-radius: 4px;
    }}
    QListWidget::item:hover, QTreeWidget::item:hover {{
        background-color: {p.hover};
    }}
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {p.accent};
        color: {on_accent};
    }}
    QHeaderView::section {{
        background-color: {p.panel};
        color: {p.text_muted};
        padding: 5px 8px;
        border: none;
        border-bottom: 1px solid {p.border};
        font-weight: 600;
    }}

    /* ---- inputs ----------------------------------------------------- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 6px;
        padding: 5px 8px;
        selection-background-color: {p.accent};
        selection-color: {on_accent};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {p.accent};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 18px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.border};
        selection-background-color: {p.accent};
        selection-color: {on_accent};
        outline: 0;
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background: {p.panel};
        border: none;
        width: 16px;
    }}

    /* ---- checkboxes ------------------------------------------------- */
    QCheckBox {{
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {p.border};
        border-radius: 4px;
        background: {p.base};
    }}
    QCheckBox::indicator:hover {{
        border-color: {p.accent};
    }}
    QCheckBox::indicator:checked {{
        background: {p.accent};
        border-color: {p.accent};
    }}

    /* ---- sliders ---------------------------------------------------- */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p.border};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {p.accent};
        width: 14px;
        height: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{
        background: {p.accent};
        border-radius: 2px;
    }}

    /* ---- scrollbars ------------------------------------------------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 11px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border};
        min-height: 28px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p.text_muted};
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 11px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {p.border};
        min-width: 28px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {p.text_muted};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        height: 0; width: 0;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
    }}

    /* ---- status bar ------------------------------------------------- */
    QStatusBar {{
        background-color: {p.panel};
        color: {p.text_muted};
        border-top: 1px solid {p.border};
    }}
    QStatusBar QLabel {{
        background: transparent;
        color: {p.text_muted};
    }}
    """


def apply_theme(app: QApplication, mode: str = "dark") -> None:
    """Apply the QSS for ``mode`` to ``app`` (live-switchable)."""
    app.setStyleSheet(stylesheet(mode))
    # Stash the active mode so callers (and widgets) can query / toggle it.
    app.setProperty("omniviz_theme", mode.lower())
    log.debug("Applied %s theme", mode)
