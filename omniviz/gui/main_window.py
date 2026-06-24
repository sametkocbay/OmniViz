"""Single-window PySide6 + pyvistaqt GUI for OmniViz.

A `QMainWindow` with one persistent embedded 3D scene (a `pyvistaqt.QtInteractor`
wrapped in :class:`omniviz.plotter.UnifiedPlotter`). Data-source panels (left
dock) build typed `ViewItem`s that are applied to the live scene immediately;
the scene-tree dock (right) lets the user toggle visibility, opacity and color
of each queued item and remove it — all updating the embedded view live.

This is a Qt port of the behavior in the old Tk ``app.py``; the parsing logic
still lives entirely in ``omniviz.io`` via the item ``apply()`` methods.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# pyvistaqt/qtpy must see the chosen Qt binding before they import it.
os.environ.setdefault("QT_API", "pyside6")

from qtpy.QtCore import QSize, Qt, QThread, Signal  # noqa: E402
from qtpy.QtGui import (  # noqa: E402
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolBox,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniviz import __version__  # noqa: E402
from omniviz.assets import LOGO_PATH  # noqa: E402
from omniviz.gui.qt_icons import badge_icon, standard_icon, swatch_icon  # noqa: E402
from omniviz.gui.qt_panels import (  # noqa: E402
    BoundaryPanel,
    CariddiCurrentDensityPanel,
    CariddiMeshPanel,
    Hdf5RestartPanel,
    PatranMeshPanel,
    PointCloudPanel,
    ProfilePanel,
    VectorFieldPanel,
    VtkMeshPanel,
    WirePanel,
    categorize_files_qt,
)
from omniviz.gui.qt_style import apply_theme, viz_background  # noqa: E402
from omniviz.gui.theme import COLORMAPS  # noqa: E402
from omniviz.models import ProfileItem, ViewItem  # noqa: E402
from omniviz.plotter import UnifiedPlotter, min_distance_between  # noqa: E402

log = logging.getLogger(__name__)


def _project_root() -> Path:
    """Return the project root (parent of the ``omniviz`` package)."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Scene-tree row widget
# --------------------------------------------------------------------------- #


#: Item ``kind`` strings that colour by a scalar/magnitude and therefore expose
#: the inline colormap + scalar-bar control.
_SCALAR_KINDS = {"VECTOR FIELD", "CARIDDI J", "JOREK HDF5"}

#: Per-kind badge tint for the scene-tree rows.
_KIND_COLORS: dict[str, str] = {
    "POINT CLOUD": "#e0533d",
    "BOUNDARY": "#16a3a3",
    "VTK": "#3d7be0",
    "PATRAN": "#9b59b6",
    "VECTOR FIELD": "#d4366b",
    "WIRE": "#e08a2f",
    "CARIDDI MESH": "#8e44ad",
    "CARIDDI J": "#c0392b",
    "JOREK HDF5": "#2f81f7",
    "PROFILE": "#27ae60",
}


def _kind_badge(kind: str) -> QLabel:
    """A small coloured pill showing the item kind."""
    label = QLabel(kind)
    tint = _KIND_COLORS.get(kind, "#5c6773")
    label.setStyleSheet(
        f"background-color: {tint}; color: white; font-weight: 600; "
        "font-size: 10px; padding: 2px 7px; border-radius: 7px;"
    )
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class _SceneRow(QWidget):
    """Inline controls for one queued item.

    Top line: kind badge + summary + eye toggle + colour swatch + remove.
    Second line: an opacity slider and (for scalar-coloured items) a colormap
    picker and scalar-bar toggle.
    """

    def __init__(
        self,
        item: ViewItem,
        on_update: Callable[[ViewItem], None],
        on_remove: Callable[[ViewItem], None],
        on_scalar_change: Callable[[ViewItem], None] | None = None,
        on_select: Callable[[ViewItem, Qt.KeyboardModifiers], None] | None = None,
    ) -> None:
        super().__init__()
        self._item = item
        self._on_update = on_update
        self._on_remove = on_remove
        self._on_scalar_change = on_scalar_change
        self._on_select = on_select
        self._color = QColor(getattr(item, "color", "white") or "white")
        self._visible_state = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 5, 6, 5)
        outer.setSpacing(4)

        # -- line 1: badge + summary + actions -------------------------------
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(_kind_badge(item.kind))

        summary = item.summary()
        self._text = QLabel(item.label or summary)
        self._text.setToolTip(summary)
        self._text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top.addWidget(self._text, stretch=1)

        self._eye = QToolButton()
        self._eye.setCheckable(True)
        self._eye.setChecked(True)
        self._eye.setAutoRaise(True)
        self._eye.setToolTip("Toggle visibility")
        self._eye.toggled.connect(self._on_eye_toggled)
        self._refresh_eye()
        top.addWidget(self._eye)

        self._color_btn = QToolButton()
        self._color_btn.setAutoRaise(True)
        self._color_btn.setToolTip("Pick colour")
        self._color_btn.clicked.connect(self._pick_color)
        self._refresh_color_btn()
        top.addWidget(self._color_btn)

        remove = QToolButton()
        remove.setText("✕")
        remove.setAutoRaise(True)
        remove.setToolTip("Remove from scene")
        remove.clicked.connect(lambda: self._on_remove(self._item))
        top.addWidget(remove)
        outer.addLayout(top)

        # -- line 2: opacity (+ optional colormap / scalar bar) --------------
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        op_label = QLabel("Opacity")
        op_label.setStyleSheet("color: palette(mid);")
        bottom.addWidget(op_label)

        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(int(float(getattr(item, "opacity", 1.0) or 1.0) * 100))
        self._opacity.setToolTip("Opacity")
        self._opacity.valueChanged.connect(lambda _v: self._on_update(self._item))
        bottom.addWidget(self._opacity, stretch=1)

        self._cmap: QComboBox | None = None
        self._bar_btn: QToolButton | None = None
        if item.kind in _SCALAR_KINDS:
            self._cmap = QComboBox()
            self._cmap.addItems(list(COLORMAPS))
            current = str(getattr(item, "colormap", "viridis") or "viridis")
            if current in COLORMAPS:
                self._cmap.setCurrentText(current)
            self._cmap.setToolTip("Colormap")
            self._cmap.setMinimumWidth(96)
            self._cmap.currentTextChanged.connect(self._emit_scalar)
            bottom.addWidget(self._cmap)

            self._bar_btn = QToolButton()
            self._bar_btn.setText("Bar")
            self._bar_btn.setCheckable(True)
            self._bar_btn.setChecked(True)
            self._bar_btn.setToolTip("Toggle scalar (colour) bar")
            self._bar_btn.toggled.connect(self._emit_scalar)
            bottom.addWidget(self._bar_btn)

        outer.addLayout(bottom)

    @property
    def item(self) -> ViewItem:
        return self._item

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Clicks on the eye/colour/remove child buttons are consumed by them;
        # a press that reaches the row itself means "select this item".
        if self._on_select is not None:
            self._on_select(self._item, event.modifiers())
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        """Visually mark the row as part of the current selection."""
        self.setStyleSheet("background: palette(highlight);" if selected else "")

    # -- visuals -------------------------------------------------------------

    def _refresh_eye(self) -> None:
        self._eye.setText("👁" if self._eye.isChecked() else "🚫")

    def _refresh_color_btn(self) -> None:
        self._color_btn.setIcon(swatch_icon(self._color))

    def _on_eye_toggled(self, _checked: bool) -> None:
        self._visible_state = self._eye.isChecked()
        self._refresh_eye()
        self._on_update(self._item)

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Pick colour")
        if chosen.isValid():
            self._color = chosen
            self._refresh_color_btn()
            self._on_update(self._item)

    def _emit_scalar(self, *_a: object) -> None:
        if self._on_scalar_change is not None:
            self._on_scalar_change(self._item)

    # -- read current control state -----------------------------------------

    @property
    def visible(self) -> bool:
        return self._eye.isChecked()

    @property
    def opacity(self) -> float:
        return self._opacity.value() / 100.0

    @property
    def color_rgb(self) -> tuple[float, float, float]:
        return (self._color.redF(), self._color.greenF(), self._color.blueF())

    @property
    def colormap(self) -> str | None:
        return self._cmap.currentText() if self._cmap is not None else None

    @property
    def scalar_bar(self) -> bool:
        return self._bar_btn.isChecked() if self._bar_btn is not None else False


@dataclass
class _ClipPlaneItem(ViewItem):
    """A scene-tree entry standing in for the interactive clip plane.

    It owns no actors of its own; enabling/disabling clipping is handled by the
    window. Present so the clip shows up in the scene tree and can be removed.
    """

    label: str = "Clip plane"
    kind: str = "CLIP PLANE"

    def summary(self) -> str:
        return "interactive clip plane — drag the handle in the view"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:  # noqa: D401
        # Clipping is driven by the window, not by an actor.
        return None


def _pil_to_qpixmap(image) -> QPixmap:
    """Convert a PIL image to a QPixmap (copying so the buffer can be freed)."""
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, 4 * rgba.width, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class _RenderWorker(QThread):
    """Render the scene to a high-quality image off the main thread.

    Builds a *separate* off-screen ``pv.Plotter`` entirely inside this thread
    (so no VTK object is shared with the live QtInteractor), re-applies the
    queued items, matches the camera, and emits a PIL image. This keeps the
    main window responsive while a publication-quality frame is produced.
    """

    done = Signal(object)  # PIL.Image
    failed = Signal(object)  # Exception

    def __init__(
        self,
        items: list[ViewItem],
        data_dir: Path,
        camera,
        background: str,
        size: tuple[int, int] = (1920, 1440),
    ) -> None:
        super().__init__()
        self._items = items
        self._data_dir = data_dir
        self._camera = camera
        self._background = background
        self._size = size

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            import numpy as np
            import pyvista as pv
            from PIL import Image

            pl = pv.Plotter(off_screen=True, window_size=list(self._size), lighting="light_kit")
            pl.set_background(self._background)
            up = UnifiedPlotter(plotter=pl)
            for item in self._items:
                if isinstance(item, ProfileItem):
                    continue
                try:
                    item.apply(up, self._data_dir)
                except Exception:  # noqa: BLE001
                    log.debug("Render: applying %s failed", item.label, exc_info=True)
            for enable in ("enable_anti_aliasing", "enable_shadows"):
                try:
                    getattr(pl, enable)()
                except Exception:  # noqa: BLE001
                    log.debug("Render: %s unavailable", enable, exc_info=True)
            if self._camera is not None:
                try:
                    pl.camera_position = self._camera
                except Exception:  # noqa: BLE001
                    log.debug("Render: camera restore failed", exc_info=True)
            arr = pl.screenshot(return_img=True)
            pl.close()
            self.done.emit(Image.fromarray(np.asarray(arr)))
        except Exception as exc:  # noqa: BLE001
            log.exception("High-quality render failed")
            self.failed.emit(exc)


class _RenderPreviewDialog(QDialog):
    """Non-modal preview of a rendered frame, with a Save button."""

    def __init__(self, parent: QWidget, image, default_dir: Path) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rendered preview")
        self.resize(960, 760)
        self._image = image
        self._default_dir = default_dir

        layout = QVBoxLayout(self)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(
            _pil_to_qpixmap(image).scaled(
                920,
                680,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        save = QPushButton("Save…")
        save.setProperty("class", "primary")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _save(self) -> None:
        default = str(self._default_dir / f"omniviz_render_{datetime.now():%Y%m%d_%H%M%S}.png")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save rendered figure",
            default,
            "PNG image (*.png);;JPEG image (*.jpg);;TIFF image (*.tif);;All files (*)",
        )
        if not path:
            return
        try:
            if path.lower().endswith((".jpg", ".jpeg")):
                self._image.convert("RGB").save(path, quality=95)
            else:
                self._image.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    """The OmniViz single window."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__()
        self.data_dir = data_dir or _project_root() / "data"
        self.setWindowTitle(f"OmniViz {__version__}")
        self.resize(1440, 900)
        self.setMinimumSize(960, 640)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks
        )
        self._theme_mode = "dark"

        self._items: list[ViewItem] = []
        self._min_dist_line = None  # actor for the latest min-distance marker
        self._categories = categorize_files_qt(self.data_dir)
        self._last_import_dir = self.data_dir if self.data_dir.exists() else Path.home()

        if LOGO_PATH.is_file():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        # -- central 3D scene
        self.interactor = self._make_interactor()
        if self.interactor is not None:
            self.setCentralWidget(self.interactor)
        else:
            placeholder = QLabel("3D view unavailable (no OpenGL context).")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setCentralWidget(placeholder)
        self.plotter = UnifiedPlotter(
            background=viz_background(self._theme_mode), plotter=self.interactor
        )
        self._axes_on = False
        self._bounds_on = False

        # -- clip-plane state (per-actor clipping that preserves each colour)
        self._clip_item: _ClipPlaneItem | None = None
        self._clip_sources: list[tuple[object, object]] = []  # (actor, original dataset)
        self._clip_widget = None
        self._clip_params: tuple | None = None  # (normal, origin), persisted across rebuilds
        self._clip_invert = False
        self._clip_handle_visible = True

        # -- background render state (kept referenced so threads/dialogs survive)
        self._render_threads: list[_RenderWorker] = []
        self._preview_dialogs: list[_RenderPreviewDialog] = []

        # -- 2D profile canvas (lazy)
        self._profile_canvas = None
        self._profile_ax = None
        self._profile_dock: QDockWidget | None = None

        self._build_left_dock()
        self._build_right_dock()
        self._build_status_bar()
        self._build_menus_and_toolbar()
        self._bind_shortcuts()

    # ----------------------------------------------------------------- setup

    def _make_interactor(self):
        """Construct the embedded QtInteractor, degrading gracefully without GL."""
        try:
            from pyvistaqt import QtInteractor

            return QtInteractor(self)
        except Exception:  # noqa: BLE001 - missing GL context under offscreen, etc.
            log.warning("QtInteractor unavailable; running without the 3D view", exc_info=True)
            return None

    def _build_header(self) -> QWidget:
        """A polished header strip: logo + title + subtitle."""
        header = QFrame()
        header.setObjectName("OmniVizHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        if LOGO_PATH.is_file():
            logo = QLabel()
            pix = QPixmap(str(LOGO_PATH))
            if not pix.isNull():
                logo.setPixmap(pix.scaledToHeight(40, Qt.TransformationMode.SmoothTransformation))
            layout.addWidget(logo)

        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel("OmniViz")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        subtitle = QLabel(f"CARIDDI / JOREK visualization · v{__version__}")
        subtitle.setStyleSheet("font-size: 11px; color: palette(mid);")
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text)
        layout.addStretch(1)
        return header

    def _build_left_dock(self) -> None:
        dock = QDockWidget("Data sources", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._build_header())

        # An accordion: sections stack vertically and only one is open at a
        # time — clicking a section header expands it and collapses the others.
        self._tabs = QToolBox()
        col.addWidget(self._tabs, stretch=1)

        self._panels: dict[str, object] = {}
        cats = self._categories
        specs: list[tuple[str, type, str | None]] = [
            ("Point Cloud", PointCloudPanel, "point_cloud"),
            ("Boundary", BoundaryPanel, "boundary"),
            ("VTK", VtkMeshPanel, "vtk"),
            ("Patran", PatranMeshPanel, "patran"),
            ("Vector", VectorFieldPanel, "vector_field"),
            ("Wire", WirePanel, None),
            ("CARIDDI Mesh", CariddiMeshPanel, "dat"),
            ("CARIDDI J", CariddiCurrentDensityPanel, "dat"),
            ("JOREK H5", Hdf5RestartPanel, "hdf5"),
            ("Profile", ProfilePanel, "dat"),
        ]
        for name, cls, key in specs:
            files = cats.get(key, []) if key else []
            panel = cls(on_add=self._add_item, files=files)
            self._panels[name] = panel
            # Wrap each panel so tall option forms scroll cleanly.
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(panel)
            self._tabs.addItem(scroll, name)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._left_dock = dock

    def _build_right_dock(self) -> None:
        dock = QDockWidget("Scene tree", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # -- header row
        header = QLabel("SCENE ITEMS")
        header.setStyleSheet("font-weight: 700; font-size: 11px; color: palette(mid);")
        layout.addWidget(header)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setUniformRowHeights(False)
        self._tree.setIndentation(0)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._tree.itemSelectionChanged.connect(self._sync_row_selection)
        layout.addWidget(self._tree, stretch=1)

        # -- empty-state placeholder (overlaps the tree area)
        self._empty_label = QLabel("No items yet —\nadd from the panels on the left.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet("color: palette(mid); font-style: italic;")
        layout.addWidget(self._empty_label)

        clear_btn = QPushButton("Clear all")
        clear_btn.setToolTip("Remove every item from the scene")
        clear_btn.clicked.connect(self._clear_items)
        layout.addWidget(clear_btn)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._right_dock = dock
        self.resizeDocks([self._left_dock, self._right_dock], [340, 320], Qt.Orientation.Horizontal)
        self._refresh_empty_state()

    def _refresh_empty_state(self) -> None:
        """Show the placeholder only when no items are queued."""
        empty = self._tree.topLevelItemCount() == 0
        self._empty_label.setVisible(empty)
        self._tree.setVisible(not empty)

    def _build_status_bar(self) -> None:
        self._log_label = QLabel("")
        self.statusBar().addPermanentWidget(self._log_label)
        self._update_status()

    def _build_menus_and_toolbar(self) -> None:
        menubar = self.menuBar()
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        # -- Views menu + toolbar (text-under-badge icons)
        view_menu = menubar.addMenu("&Views")
        for label, short, name, key, tip in (
            ("View X", "X", "x", "x", "Look down +X (Y-Z plane)"),
            ("View Y", "Y", "y", "y", "Look down +Y (X-Z plane)"),
            ("View Z", "Z", "z", "z", "Look down +Z (X-Y plane)"),
            ("Isometric", "ISO", "iso", "i", "Isometric view"),
            ("Flip", "FLIP", "flip", "f", "Flip 180° to the far side"),
        ):
            act = QAction(badge_icon(short), short, self)
            act.setShortcut(QKeySequence(key))
            act.setToolTip(f"{label}  ({key.upper()}) — {tip}")
            act.triggered.connect(lambda _checked=False, n=name: self._set_view(n))
            view_menu.addAction(act)
            toolbar.addAction(act)
        toolbar.addSeparator()

        # -- Toggles menu
        toggles = menubar.addMenu("&Toggles")
        self._axes_action = QAction(standard_icon("SP_FileDialogDetailedView"), "Axes", self)
        self._axes_action.setCheckable(True)
        self._axes_action.setToolTip("Show/hide the orientation axes")
        self._axes_action.triggered.connect(self._toggle_axes)
        toggles.addAction(self._axes_action)
        toolbar.addAction(self._axes_action)

        self._bounds_action = QAction(standard_icon("SP_FileDialogListView"), "Grid", self)
        self._bounds_action.setCheckable(True)
        self._bounds_action.setToolTip("Show/hide the bounds grid")
        self._bounds_action.triggered.connect(self._toggle_bounds)
        toggles.addAction(self._bounds_action)
        toolbar.addAction(self._bounds_action)

        bg_action = QAction(standard_icon("SP_DialogResetButton"), "Background…", self)
        bg_action.setToolTip("Pick the 3D background colour")
        bg_action.triggered.connect(self._pick_background)
        toggles.addAction(bg_action)

        self._clip_action = QAction(standard_icon("SP_DialogDiscardButton"), "Clip", self)
        self._clip_action.setCheckable(True)
        self._clip_action.setToolTip("Toggle an interactive clip plane")
        self._clip_action.triggered.connect(self._toggle_clip)
        toggles.addAction(self._clip_action)
        toolbar.addAction(self._clip_action)
        toolbar.addSeparator()

        # -- Tools menu
        tools = menubar.addMenu("Too&ls")
        self._min_dist_action = QAction(badge_icon("MIN"), "Min Dist", self)
        self._min_dist_action.setToolTip(
            "Minimum distance between two selected scene items (select two rows in the right panel)"
        )
        self._min_dist_action.triggered.connect(self._compute_min_distance)
        tools.addAction(self._min_dist_action)
        toolbar.addAction(self._min_dist_action)
        toolbar.addSeparator()

        # -- File menu
        file_menu = menubar.addMenu("&File")
        import_action = QAction(standard_icon("SP_DialogOpenButton"), "Import", self)
        import_action.setToolTip("Import a file into the data folder")
        import_action.triggered.connect(self._import_file)
        file_menu.addAction(import_action)
        toolbar.addAction(import_action)

        reload_action = QAction(standard_icon("SP_BrowserReload"), "Reload files", self)
        reload_action.setToolTip("Re-scan the data folder")
        reload_action.triggered.connect(self._reload_files)
        file_menu.addAction(reload_action)
        toolbar.addAction(reload_action)

        reload_scene_action = QAction("Reload scene (re-apply from disk)", self)
        reload_scene_action.setToolTip("Clear and rebuild every queued item from disk")
        reload_scene_action.triggered.connect(self._reload_scene)
        file_menu.addAction(reload_scene_action)

        screenshot_action = QAction(standard_icon("SP_DialogSaveButton"), "Export", self)
        screenshot_action.setToolTip("Capture a quick screenshot of the current view")
        screenshot_action.triggered.connect(self._export_screenshot)
        file_menu.addAction(screenshot_action)
        toolbar.addAction(screenshot_action)

        self._render_action = QAction(standard_icon("SP_MediaPlay"), "Render", self)
        self._render_action.setToolTip(
            "Render a high-quality shaded frame in the background, then preview & save"
        )
        self._render_action.triggered.connect(self._render_quality)
        file_menu.addAction(self._render_action)
        toolbar.addAction(self._render_action)

        # -- View menu (appearance) + toolbar spacer + theme toggle
        appearance = menubar.addMenu("&Appearance")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._dark_action = QAction("Dark", self, checkable=True)
        self._dark_action.setChecked(True)
        self._dark_action.triggered.connect(lambda: self._set_theme("dark"))
        self._light_action = QAction("Light", self, checkable=True)
        self._light_action.triggered.connect(lambda: self._set_theme("light"))
        for act in (self._dark_action, self._light_action):
            self._theme_group.addAction(act)
            appearance.addAction(act)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self._theme_action = QAction(standard_icon("SP_DesktopIcon"), "Theme", self)
        self._theme_action.setToolTip("Switch between Dark and Light appearance")
        self._theme_action.triggered.connect(self._cycle_theme)
        toolbar.addAction(self._theme_action)

    # ----------------------------------------------------------------- theme

    def _set_theme(self, mode: str) -> None:
        """Apply a theme live to the running application."""
        self._theme_mode = mode
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        self._dark_action.setChecked(mode == "dark")
        self._light_action.setChecked(mode == "light")
        # Keep the 3D view background in step with the theme.
        try:
            self.plotter.plotter.set_background(viz_background(mode))
            self.plotter.render()
        except Exception:  # noqa: BLE001
            log.debug("Background theme sync failed", exc_info=True)
        self._log(f"Theme: {mode}")

    def _cycle_theme(self) -> None:
        self._set_theme("light" if self._theme_mode == "dark" else "dark")

    def _bind_shortcuts(self) -> None:
        # Shortcuts are also on the view actions, but bind them explicitly so
        # they work regardless of menu/toolbar focus.
        for key, name in (("x", "x"), ("y", "y"), ("z", "z"), ("i", "iso"), ("f", "flip")):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda n=name: self._set_view(n))

    # ----------------------------------------------------------------- items

    def _add_item(self, item: ViewItem) -> None:
        self._items.append(item)
        self._add_tree_row(item)
        if isinstance(item, ProfileItem):
            self._render_profile(item)
            self._log(f"Added profile {item.label}")
        else:
            self._apply_item(item)
        self._update_status()

    def _apply_item(self, item: ViewItem) -> None:
        """Build and add an item's actor.

        ``item.apply()`` creates VTK actors on the live ``QtInteractor``; VTK is
        not thread-safe, so this runs synchronously on the Qt main thread. Heavy
        loads briefly block the UI — splitting parse (worker) from add (main
        thread) is a future optimization that needs ``apply()`` reworked.
        """
        self._log(f"Building {item.kind}: {item.label}…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            item.apply(self.plotter, self.data_dir)
        except Exception as exc:  # noqa: BLE001
            log.exception("Applying item '%s' failed", item.label)
            self._on_apply_failed(item, exc)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._on_apply_done(item)

    def _on_apply_done(self, item: ViewItem) -> None:
        self._apply_row_state(item)
        # Fold any newly-added geometry into an active clip plane.
        if self._clip_item is not None:
            self._rebuild_clip()
        self.plotter.render()
        self._log(f"Added {item.kind}: {item.label}")

    def _on_apply_failed(self, item: ViewItem, exc: Exception) -> None:
        QMessageBox.critical(self, "Add failed", f"Could not add {item.label}:\n{exc}")
        # Roll the failed item back out of the queue/tree.
        if item in self._items:
            self._items.remove(item)
        self._remove_tree_row(item)
        self._update_status()

    def _clear_items(self) -> None:
        # Drop the clip widget first so it doesn't dangle over a cleared scene.
        if self._clip_item is not None:
            self._teardown_clip()
            self._clip_item = None
            self._clip_action.setChecked(False)
        self.plotter.clear_items()
        self.plotter.render()
        self._items.clear()
        self._tree.clear()
        self._refresh_empty_state()
        if self._profile_ax is not None:
            self._profile_ax.clear()
            self._profile_canvas.draw_idle()
        self._update_status()

    def _remove_item(self, item: ViewItem) -> None:
        # The clip plane is special: removing its row tears the clip down and
        # restores the hidden originals.
        if item is self._clip_item:
            self._disable_clip()
            return
        self.plotter.remove_item(item.id)
        if item in self._items:
            self._items.remove(item)
        self._remove_tree_row(item)
        if isinstance(item, ProfileItem):
            self._rebuild_profile_plot()
        # Keep an active clip consistent with the changed actor set.
        if self._clip_item is not None:
            self._rebuild_clip()
        self._update_status()
        self.plotter.render()
        self._log(f"Removed {item.label}")

    # -- scene tree ----------------------------------------------------------

    def _add_tree_row(self, item: ViewItem) -> None:
        node = QTreeWidgetItem(self._tree)
        row = _SceneRow(
            item,
            self._on_row_changed,
            self._remove_item,
            on_scalar_change=self._on_scalar_changed,
            on_select=self._select_row,
        )
        node.setData(0, Qt.ItemDataRole.UserRole, item.id)
        self._tree.addTopLevelItem(node)
        self._tree.setItemWidget(node, 0, row)
        node.setSizeHint(0, row.sizeHint())
        self._refresh_empty_state()

    def _find_tree_node(self, item: ViewItem) -> tuple[QTreeWidgetItem | None, _SceneRow | None]:
        for i in range(self._tree.topLevelItemCount()):
            node = self._tree.topLevelItem(i)
            if node.data(0, Qt.ItemDataRole.UserRole) == item.id:
                widget = self._tree.itemWidget(node, 0)
                return node, widget if isinstance(widget, _SceneRow) else None
        return None, None

    def _select_row(self, item: ViewItem, modifiers: Qt.KeyboardModifiers) -> None:
        """Select the row for ``item`` (additive when Ctrl/Shift is held)."""
        node, _row = self._find_tree_node(item)
        if node is None:
            return
        additive = bool(
            modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        )
        if additive:
            node.setSelected(not node.isSelected())
        else:
            self._tree.clearSelection()
            node.setSelected(True)

    def _sync_row_selection(self) -> None:
        """Mirror the tree's selection state onto the custom row widgets."""
        for i in range(self._tree.topLevelItemCount()):
            node = self._tree.topLevelItem(i)
            widget = self._tree.itemWidget(node, 0)
            if isinstance(widget, _SceneRow):
                widget.set_selected(node.isSelected())

    def _selected_items(self) -> list[ViewItem]:
        """The `ViewItem`s whose rows are currently selected, in tree order."""
        ids = {node.data(0, Qt.ItemDataRole.UserRole) for node in self._tree.selectedItems()}
        return [it for it in self._items if it.id in ids]

    def _remove_tree_row(self, item: ViewItem) -> None:
        node, _row = self._find_tree_node(item)
        if node is not None:
            self._tree.takeTopLevelItem(self._tree.indexOfTopLevelItem(node))
        self._refresh_empty_state()

    def _on_row_changed(self, item: ViewItem) -> None:
        if isinstance(item, ProfileItem):
            return
        if item is self._clip_item:
            # The clip row's eye shows/hides the interactive plane handle while
            # the clip itself stays applied.
            _node, row = self._find_tree_node(item)
            if row is not None:
                self._set_clip_handle_visible(row.visible)
            return
        self._apply_row_state(item)

    def _apply_row_state(self, item: ViewItem) -> None:
        """Push the row's control state into the live actor.

        Scalar-coloured items (vector-by-magnitude, CARIDDI J, JOREK HDF5) are
        driven by a colormap, so we must NOT force a flat ``prop.color`` onto
        them — doing so would collapse the colormap to a single tint. Their
        colour is managed via the colormap control (``_on_scalar_changed``).
        """
        _node, row = self._find_tree_node(item)
        if row is None:
            return
        push_color = item.kind not in _SCALAR_KINDS
        self.plotter.update_item(
            item.id,
            color=row.color_rgb if push_color else None,
            opacity=row.opacity,
            visibility=row.visible,
        )

    def _on_scalar_changed(self, item: ViewItem) -> None:
        """Live-update colormap / scalar-bar for a scalar-coloured item.

        Wires directly to the embedded actors' mappers (the plotter exposes the
        actor handles via ``_actors``); no plotter.py logic is changed.
        """
        _node, row = self._find_tree_node(item)
        if row is None:
            return
        cmap = row.colormap
        show_bar = row.scalar_bar
        handles = self.plotter._actors.get(item.id, [])
        applied = False
        for handle in handles:
            mapper = getattr(handle, "mapper", None)
            if mapper is None:
                continue
            try:
                if cmap is not None:
                    # Rebuild the lookup table with the chosen colormap while
                    # preserving the existing scalar range.
                    import pyvista as pv

                    rng = None
                    lut = getattr(mapper, "lookup_table", None)
                    if lut is not None and hasattr(lut, "scalar_range"):
                        rng = lut.scalar_range
                    new_lut = pv.LookupTable(cmap=cmap)
                    if rng is not None:
                        new_lut.scalar_range = rng
                    mapper.lookup_table = new_lut
                    applied = True
            except Exception:  # noqa: BLE001
                log.debug("Colormap update failed for %s", item.id, exc_info=True)
        self._toggle_scalar_bar(item, show_bar)
        if applied:
            self.plotter.render()
        self._log(f"{item.label}: cmap={cmap}, scalar bar={'on' if show_bar else 'off'}")

    def _toggle_scalar_bar(self, item: ViewItem, show: bool) -> None:
        """Best-effort show/hide of the scalar bar for ``item``'s actor."""
        p = self.plotter.plotter
        title = item.label or item.kind
        try:
            if show:
                handles = self.plotter._actors.get(item.id, [])
                mapper = next(
                    (getattr(h, "mapper", None) for h in handles if getattr(h, "mapper", None)),
                    None,
                )
                if mapper is not None:
                    p.add_scalar_bar(title=title, mapper=mapper)
            else:
                p.remove_scalar_bar(title=title)
        except Exception:  # noqa: BLE001
            log.debug("Scalar-bar toggle failed for %s", item.id, exc_info=True)

    # ----------------------------------------------------------------- views

    def _set_view(self, name: str) -> None:
        self.plotter._set_view(name)

    def _toggle_axes(self) -> None:
        p = self.plotter.plotter
        try:
            if self._axes_action.isChecked():
                p.add_axes()
            else:
                p.hide_axes()
            self.plotter.render()
        except Exception:  # noqa: BLE001
            log.debug("Axes toggle failed", exc_info=True)

    def _toggle_bounds(self) -> None:
        p = self.plotter.plotter
        try:
            if self._bounds_action.isChecked():
                p.show_bounds(grid="front", location="outer", all_edges=True)
            else:
                p.remove_bounds_axes()
            self.plotter.render()
        except Exception:  # noqa: BLE001
            log.debug("Bounds toggle failed", exc_info=True)

    def _pick_background(self) -> None:
        chosen = QColorDialog.getColor(QColor("white"), self, "Background color")
        if chosen.isValid():
            try:
                self.plotter.plotter.set_background(chosen.name())
                self.plotter.render()
            except Exception:  # noqa: BLE001
                log.debug("Background change failed", exc_info=True)

    # ------------------------------------------------------------ min distance

    def _item_geometry(self, item: ViewItem):
        """Combine every actor dataset for ``item`` into one pyvista mesh."""
        import pyvista as pv

        parts = []
        for handle in self.plotter._actors.get(item.id, []):
            ds = self._actor_dataset(handle)
            if ds is None:
                continue
            mesh = pv.wrap(ds)
            if isinstance(mesh, pv.MultiBlock):
                mesh = mesh.combine()  # composite (e.g. boundary patches) -> one grid
            if mesh is not None and getattr(mesh, "n_points", 0) > 0:
                parts.append(mesh)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return pv.MultiBlock(parts).combine()

    def _compute_min_distance(self) -> None:
        """Report the minimum distance between the two selected scene items."""
        selected = self._selected_items()
        if len(selected) != 2:
            QMessageBox.information(
                self,
                "Minimum distance",
                "Select exactly two items in the scene panel "
                "(Ctrl/Shift-click to pick a second), then run Min Dist.",
            )
            return

        a, b = selected
        geom_a = self._item_geometry(a)
        geom_b = self._item_geometry(b)
        if geom_a is None or geom_b is None:
            QMessageBox.warning(
                self,
                "Minimum distance",
                "Could not read geometry for one of the selected items.",
            )
            return

        try:
            dist, pa, pb = min_distance_between(geom_a, geom_b)
        except Exception as exc:  # noqa: BLE001
            log.debug("Min-distance computation failed", exc_info=True)
            QMessageBox.warning(self, "Minimum distance", f"Computation failed: {exc}")
            return

        self._draw_min_distance_line(pa, pb)
        msg = f"{a.label or a.kind} ↔ {b.label or b.kind}:  {dist:.6g} m"
        self.statusBar().showMessage(msg, 15000)
        self._log(msg)
        QMessageBox.information(self, "Minimum distance", msg)

    def _draw_min_distance_line(self, pa, pb) -> None:
        """Draw (replacing any previous) a marker line between the closest points."""
        import pyvista as pv

        p = self.plotter.plotter
        try:
            if self._min_dist_line is not None:
                p.remove_actor(self._min_dist_line)
            self._min_dist_line = p.add_mesh(
                pv.Line(pa, pb),
                color="yellow",
                line_width=4,
                name="__min_distance__",
            )
            self.plotter.render()
        except Exception:  # noqa: BLE001
            log.debug("Min-distance line draw failed", exc_info=True)

    @staticmethod
    def _actor_dataset(handle):
        """Best-effort extraction of the dataset behind an actor (incl. glyphs)."""
        mapper = getattr(handle, "mapper", None)
        if mapper is None:
            return None
        ds = getattr(mapper, "dataset", None)
        if ds is not None:
            return ds
        for getter in ("GetInputAsDataSet", "GetInput"):
            try:
                got = getattr(mapper, getter)()
                if got is not None:
                    return got
            except Exception:  # noqa: BLE001
                continue
        return None

    @staticmethod
    def _set_actor_dataset(handle, dataset) -> None:
        """Swap the geometry behind an actor while keeping its colour/scalars."""
        mapper = getattr(handle, "mapper", None)
        if mapper is None:
            return
        try:
            mapper.dataset = dataset
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            mapper.SetInputData(dataset)
        except Exception:  # noqa: BLE001
            log.debug("set actor dataset failed", exc_info=True)

    def _toggle_clip(self) -> None:
        """Enable/disable the interactive clip plane from the toolbar toggle."""
        if self._clip_action.isChecked():
            self._enable_clip()
        else:
            self._disable_clip()

    def _enable_clip(self) -> None:
        """Turn on an interactive clip plane and register it in the scene tree."""
        if self._clip_item is not None:
            return
        self._clip_item = _ClipPlaneItem()
        self._items.append(self._clip_item)
        self._add_tree_row(self._clip_item)
        if not self._install_clip():
            # Nothing to clip — roll the row/item back out.
            self._remove_tree_row(self._clip_item)
            if self._clip_item in self._items:
                self._items.remove(self._clip_item)
            self._clip_item = None
            self._clip_action.setChecked(False)
            self._update_status()
            self._log("Clip: nothing to clip")
            return
        self._clip_action.setChecked(True)
        self._update_status()
        self._log("Clip enabled — drag the handle; the row 👁 hides it, ✕ turns it off")

    def _disable_clip(self) -> None:
        """Tear down the clip plane and restore the original (unclipped) actors."""
        self._teardown_clip()
        self._clip_params = None
        if self._clip_item is not None:
            self._remove_tree_row(self._clip_item)
            if self._clip_item in self._items:
                self._items.remove(self._clip_item)
            self._clip_item = None
        self._clip_action.setChecked(False)
        self._update_status()
        self.plotter.render()
        self._log("Clip disabled")

    def _teardown_clip(self) -> None:
        """Remove the plane widget and restore each actor's full geometry."""
        try:
            self.plotter.plotter.clear_plane_widgets()
        except Exception:  # noqa: BLE001
            log.debug("clear_plane_widgets failed", exc_info=True)
        self._clip_widget = None
        for handle, orig in self._clip_sources:
            self._set_actor_dataset(handle, orig)
        self._clip_sources = []

    def _install_clip(self) -> bool:
        """Attach ONE interactive clip plane that clips every data actor in place.

        Each actor keeps its own colour / colormap because we only swap the
        geometry behind it (to ``dataset.clip(...)``), never its mapper styling.
        """
        self._teardown_clip()
        sources: list[tuple[object, object]] = []
        bounds: tuple[float, ...] | None = None
        for handles in self.plotter._actors.values():
            for handle in handles:
                ds = self._actor_dataset(handle)
                if ds is None or not hasattr(ds, "clip"):
                    continue
                sources.append((handle, ds))
                b = getattr(ds, "bounds", None)
                if b is not None:
                    bounds = (
                        b
                        if bounds is None
                        else (
                            min(bounds[0], b[0]),
                            max(bounds[1], b[1]),
                            min(bounds[2], b[2]),
                            max(bounds[3], b[3]),
                            min(bounds[4], b[4]),
                            max(bounds[5], b[5]),
                        )
                    )
        if not sources:
            return False
        self._clip_sources = sources

        if self._clip_params is not None:
            normal, origin = self._clip_params
        else:
            normal = (1.0, 0.0, 0.0)
            origin = (
                (
                    (bounds[0] + bounds[1]) / 2.0,
                    (bounds[2] + bounds[3]) / 2.0,
                    (bounds[4] + bounds[5]) / 2.0,
                )
                if bounds is not None
                else (0.0, 0.0, 0.0)
            )
        try:
            self._clip_widget = self.plotter.plotter.add_plane_widget(
                self._clip_callback, normal=normal, origin=origin, bounds=bounds, implicit=True
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("add_plane_widget failed", exc_info=True)
            self._log(f"Clip unavailable: {exc}")
            return False
        self._clip_callback(normal, origin)
        self._set_clip_handle_visible(self._clip_handle_visible)
        return True

    def _clip_callback(self, normal, origin) -> None:
        """Plane-widget callback: clip every source actor by the current plane."""
        self._clip_params = (tuple(normal), tuple(origin))
        for handle, orig in self._clip_sources:
            try:
                clipped = orig.clip(normal=normal, origin=origin, invert=self._clip_invert)
                self._set_actor_dataset(handle, clipped)
            except Exception:  # noqa: BLE001
                log.debug("clip apply failed", exc_info=True)
        self.plotter.render()

    def _rebuild_clip(self) -> bool:
        """Re-collect actors and re-apply the clip after the scene set changes."""
        return self._install_clip()

    def _set_clip_handle_visible(self, show: bool) -> None:
        """Show/hide the interactive plane handle; the clip itself stays applied."""
        self._clip_handle_visible = show
        widget = self._clip_widget
        if widget is None:
            return
        try:
            widget.On() if show else widget.Off()
        except Exception:  # noqa: BLE001
            try:
                widget.SetEnabled(1 if show else 0)
            except Exception:  # noqa: BLE001
                log.debug("clip handle toggle failed", exc_info=True)
        self.plotter.render()

    # ----------------------------------------------------------------- profile

    def _ensure_profile_dock(self) -> None:
        if self._profile_dock is not None:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4, 3), tight_layout=True)
        self._profile_ax = fig.add_subplot(111)
        self._profile_canvas = FigureCanvasQTAgg(fig)

        dock = QDockWidget("Profiles (2D)", self)
        dock.setWidget(self._profile_canvas)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._profile_dock = dock

    def _render_profile(self, item: ProfileItem) -> None:
        self._ensure_profile_dock()
        try:
            df = item.load(self.data_dir)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Profile failed", f"Could not load {item.file}:\n{exc}")
            return
        self._profile_ax.plot(df["x"], df["value"], label=item.label or item.file)
        self._profile_ax.legend(fontsize=8)
        self._profile_canvas.draw_idle()
        self._profile_dock.raise_()

    def _rebuild_profile_plot(self) -> None:
        """Re-draw the 2D plot from the remaining queued profiles."""
        if self._profile_ax is None:
            return
        self._profile_ax.clear()
        for item in self._items:
            if isinstance(item, ProfileItem):
                try:
                    df = item.load(self.data_dir)
                    self._profile_ax.plot(df["x"], df["value"], label=item.label or item.file)
                except Exception:  # noqa: BLE001
                    log.debug("Profile reload failed for %s", item.file, exc_info=True)
        if self._profile_ax.has_data():
            self._profile_ax.legend(fontsize=8)
        self._profile_canvas.draw_idle()

    # ----------------------------------------------------------------- import

    _IMPORT_FILTER = (
        "All supported (*.dat *.txt *.vtk *.vtu *.msh *.out *.h5 *.hdf5);;"
        "Data (*.dat *.txt);;VTK/VTU (*.vtk *.vtu);;Patran (*.msh *.out);;"
        "HDF5 (*.h5 *.hdf5);;All files (*)"
    )

    def _import_file(self) -> None:
        """Copy a chosen file into the data dir (ported from app._import_file)."""
        src_str, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a file to import into the data folder",
            str(self._last_import_dir),
            self._IMPORT_FILTER,
        )
        if not src_str:
            return
        src = Path(src_str)
        if not src.is_file():
            QMessageBox.critical(self, "Import failed", f"Not a regular file:\n{src}")
            return

        self._last_import_dir = src.parent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        dst = self.data_dir / src.name

        try:
            if dst.exists() and dst.resolve() == src.resolve():
                QMessageBox.information(
                    self, "Already in data folder", f"{src.name} is already in {self.data_dir}."
                )
                return
        except OSError:
            pass

        if dst.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite?",
                f"{dst.name} already exists in {self.data_dir}.\n\nReplace it?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        self._reload_files()
        self._log(f"Imported {dst.name}")

    def _reload_files(self) -> None:
        """Re-read the data dir and refresh each panel's file list."""
        self._categories = categorize_files_qt(self.data_dir)
        cats = self._categories
        keymap = {
            "Point Cloud": "point_cloud",
            "Boundary": "boundary",
            "VTK": "vtk",
            "Patran": "patran",
            "Vector": "vector_field",
            "CARIDDI Mesh": "dat",
            "CARIDDI J": "dat",
            "JOREK H5": "hdf5",
            "Profile": "dat",
        }
        for name, panel in self._panels.items():
            key = keymap.get(name)
            if key is not None and hasattr(panel, "set_files"):
                panel.set_files(cats.get(key, []))
        self._update_status()
        self._log("Reloaded file lists")

    def _reload_scene(self) -> None:
        """Clear the live scene and re-apply every queued item from disk."""
        items = list(self._items)
        clip_was_active = self._clip_item is not None
        self._teardown_clip()
        self.plotter.clear_items()
        self.plotter.render()
        if self._profile_ax is not None:
            self._profile_ax.clear()
            self._profile_canvas.draw_idle()
        for item in items:
            if isinstance(item, _ClipPlaneItem):
                continue
            if isinstance(item, ProfileItem):
                self._render_profile(item)
            else:
                self._apply_item(item)
        if clip_was_active:
            self._rebuild_clip()
            self.plotter.render()
        self._log("Reloaded scene from disk")

    # ----------------------------------------------------------------- export

    def _export_screenshot(self) -> None:
        """Capture a high-res image from the embedded view and save it."""
        if self.interactor is None:
            QMessageBox.warning(self, "No view", "The 3D view is not available.")
            return
        try:
            img = self._capture_highres()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Capture failed", str(exc))
            return

        default = str(self.data_dir / f"omniviz_{datetime.now():%Y%m%d_%H%M%S}.png")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            default,
            "PNG image (*.png);;JPEG image (*.jpg);;TIFF image (*.tif);;All files (*)",
        )
        if not path:
            return

        caption, _ok = self._ask_caption()
        if caption:
            from omniviz.gui.photo import composite_label

            img = composite_label(img, caption, font_frac=0.045, color=(255, 255, 255))
        try:
            if path.lower().endswith((".jpg", ".jpeg")):
                img.convert("RGB").save(path, quality=95)
            else:
                img.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._log(f"Saved figure to {path}")

    def _capture_highres(self):
        """High-res capture: prefer UnifiedPlotter.capture_highres, else window bump."""
        try:
            return self.plotter.capture_highres(scale=3)
        except Exception:  # noqa: BLE001
            log.debug("capture_highres failed; falling back to screenshot", exc_info=True)
            from PIL import Image

            arr = self.plotter.plotter.screenshot(return_img=True, scale=3)
            import numpy as np

            return Image.fromarray(np.asarray(arr))

    def _ask_caption(self) -> tuple[str, bool]:
        from qtpy.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(self, "Caption", "Optional caption (blank for none):")
        return text, ok

    # ----------------------------------------------------------------- render

    def _render_quality(self) -> None:
        """Render a high-quality shaded frame in the background, then preview."""
        renderables = [
            it for it in self._items if not isinstance(it, (ProfileItem, _ClipPlaneItem))
        ]
        if not renderables:
            QMessageBox.information(self, "Nothing to render", "Add at least one 3D item first.")
            return

        camera = None
        if self.interactor is not None:
            try:
                camera = self.plotter.plotter.camera_position
            except Exception:  # noqa: BLE001
                camera = None

        worker = _RenderWorker(
            list(renderables),
            self.data_dir,
            camera,
            viz_background(self._theme_mode),
        )
        worker.done.connect(self._on_render_done)
        worker.failed.connect(self._on_render_failed)
        worker.finished.connect(lambda w=worker: self._render_threads.remove(w))
        self._render_threads.append(worker)
        self._render_action.setEnabled(False)
        self._log("Rendering high-quality frame in the background…")
        worker.start()

    def _on_render_done(self, image) -> None:
        self._render_action.setEnabled(True)
        self._log("Render ready.")
        dialog = _RenderPreviewDialog(self, image, self.data_dir)
        dialog.finished.connect(
            lambda _r, d=dialog: (
                self._preview_dialogs.remove(d) if d in self._preview_dialogs else None
            )
        )
        self._preview_dialogs.append(dialog)
        dialog.show()  # non-modal: the main window stays usable

    def _on_render_failed(self, exc: Exception) -> None:
        self._render_action.setEnabled(True)
        QMessageBox.critical(self, "Render failed", str(exc))

    # ----------------------------------------------------------------- status

    def _update_status(self) -> None:
        self.statusBar().showMessage(f"{len(self._items)} item(s) · data dir: {self.data_dir}")

    def _log(self, message: str) -> None:
        self._log_label.setText(message)
        log.info(message)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(data_dir: str | Path | None = None) -> None:
    """Create the QApplication, show the OmniViz window and run the event loop."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = QApplication.instance() or QApplication([])
    apply_theme(app, "dark")
    if LOGO_PATH.is_file():
        app.setWindowIcon(QIcon(str(LOGO_PATH)))

    path = Path(data_dir) if data_dir else None
    window = MainWindow(data_dir=path)
    window.show()
    app.exec()


if __name__ == "__main__":
    run()
