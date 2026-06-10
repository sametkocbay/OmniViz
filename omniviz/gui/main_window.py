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
from datetime import datetime
from pathlib import Path

# pyvistaqt/qtpy must see the chosen Qt binding before they import it.
os.environ.setdefault("QT_API", "pyside6")

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal  # noqa: E402
from qtpy.QtGui import QAction, QColor, QIcon, QKeySequence, QShortcut  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QColorDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from omniviz import __version__  # noqa: E402
from omniviz.assets import LOGO_PATH  # noqa: E402
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
from omniviz.models import ProfileItem, ViewItem  # noqa: E402
from omniviz.plotter import UnifiedPlotter  # noqa: E402

log = logging.getLogger(__name__)


def _project_root() -> Path:
    """Return the project root (parent of the ``omniviz`` package)."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Background worker — keep heavy item.apply() off the UI thread
# --------------------------------------------------------------------------- #


class _ApplySignals(QObject):
    done = Signal(object)  # ViewItem
    failed = Signal(object, object)  # ViewItem, Exception


class _ApplyWorker(QRunnable):
    """Run ``item.apply(plotter, data_dir)`` off the main thread.

    PyVista/VTK actor mutation is not thread-safe, so this is used only for the
    expensive parse + mesh-build that ``apply()`` performs internally; the
    actual ``render()`` is scheduled back on the main thread via the signal.
    """

    def __init__(self, item: ViewItem, plotter: UnifiedPlotter, data_dir: Path) -> None:
        super().__init__()
        self.item = item
        self.plotter = plotter
        self.data_dir = data_dir
        self.signals = _ApplySignals()

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        try:
            self.item.apply(self.plotter, self.data_dir)
            self.signals.done.emit(self.item)
        except Exception as exc:  # noqa: BLE001
            log.exception("Applying item '%s' failed", self.item.label)
            self.signals.failed.emit(self.item, exc)


# --------------------------------------------------------------------------- #
# Scene-tree row widget
# --------------------------------------------------------------------------- #


class _SceneRow(QWidget):
    """Inline controls for one queued item: visibility / opacity / color / remove."""

    def __init__(
        self,
        item: ViewItem,
        on_update: Callable[[ViewItem], None],
        on_remove: Callable[[ViewItem], None],
    ) -> None:
        super().__init__()
        self._item = item
        self._on_update = on_update
        self._on_remove = on_remove
        self._color = QColor(getattr(item, "color", "white") or "white")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._visible = QCheckBox()
        self._visible.setChecked(True)
        self._visible.setToolTip("Visibility")
        self._visible.stateChanged.connect(lambda _s: self._on_update(self._item))
        layout.addWidget(self._visible)

        text = QLabel(f"{item.kind}: {item.summary()}")
        text.setToolTip(item.summary())
        text.setWordWrap(False)
        layout.addWidget(text, stretch=1)

        self._opacity = QDoubleSpinBox()
        self._opacity.setRange(0.0, 1.0)
        self._opacity.setSingleStep(0.05)
        self._opacity.setValue(float(getattr(item, "opacity", 1.0) or 1.0))
        self._opacity.setToolTip("Opacity")
        self._opacity.valueChanged.connect(lambda _v: self._on_update(self._item))
        layout.addWidget(self._opacity)

        self._color_btn = QPushButton("color")
        self._color_btn.setToolTip("Color")
        self._color_btn.clicked.connect(self._pick_color)
        self._refresh_color_btn()
        layout.addWidget(self._color_btn)

        remove = QPushButton("✕")
        remove.setFixedWidth(28)
        remove.setToolTip("Remove")
        remove.clicked.connect(lambda: self._on_remove(self._item))
        layout.addWidget(remove)

    @property
    def item(self) -> ViewItem:
        return self._item

    def _refresh_color_btn(self) -> None:
        self._color_btn.setStyleSheet(f"background-color: {self._color.name()};")

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Pick color")
        if chosen.isValid():
            self._color = chosen
            self._refresh_color_btn()
            self._on_update(self._item)

    # -- read current control state -----------------------------------------

    @property
    def visible(self) -> bool:
        return self._visible.isChecked()

    @property
    def opacity(self) -> float:
        return self._opacity.value()

    @property
    def color_rgb(self) -> tuple[float, float, float]:
        return (self._color.redF(), self._color.greenF(), self._color.blueF())


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    """The OmniViz single window."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__()
        self.data_dir = data_dir or _project_root() / "data"
        self.setWindowTitle(f"OmniViz {__version__}")
        self.resize(1280, 820)

        self._items: list[ViewItem] = []
        self._categories = categorize_files_qt(self.data_dir)
        self._last_import_dir = self.data_dir if self.data_dir.exists() else Path.home()
        self._pool = QThreadPool.globalInstance()
        self._workers: list[_ApplyWorker] = []

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
        self.plotter = UnifiedPlotter(background="white", plotter=self.interactor)
        self._axes_on = False
        self._bounds_on = False

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

    def _build_left_dock(self) -> None:
        dock = QDockWidget("Data sources", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self._tabs = QTabWidget()

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
            self._tabs.addTab(panel, name)

        dock.setWidget(self._tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_right_dock(self) -> None:
        dock = QDockWidget("Scene tree", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        layout.addWidget(self._tree)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_items)
        layout.addWidget(clear_btn)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_status_bar(self) -> None:
        self._log_label = QLabel("")
        self.statusBar().addPermanentWidget(self._log_label)
        self._update_status()

    def _build_menus_and_toolbar(self) -> None:
        menubar = self.menuBar()
        toolbar = self.addToolBar("Main")

        # -- Views menu + toolbar
        view_menu = menubar.addMenu("&Views")
        for label, name, key in (
            ("View X", "x", "x"),
            ("View Y", "y", "y"),
            ("View Z", "z", "z"),
            ("Isometric", "iso", "i"),
            ("Flip", "flip", "f"),
        ):
            act = QAction(label, self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _checked=False, n=name: self._set_view(n))
            view_menu.addAction(act)
            toolbar.addAction(act)
        toolbar.addSeparator()

        # -- Toggles menu
        toggles = menubar.addMenu("&Toggles")
        self._axes_action = QAction("Axes", self, checkable=True)
        self._axes_action.triggered.connect(self._toggle_axes)
        toggles.addAction(self._axes_action)
        toolbar.addAction(self._axes_action)

        self._bounds_action = QAction("Bounds grid", self, checkable=True)
        self._bounds_action.triggered.connect(self._toggle_bounds)
        toggles.addAction(self._bounds_action)
        toolbar.addAction(self._bounds_action)

        bg_action = QAction("Background…", self)
        bg_action.triggered.connect(self._pick_background)
        toggles.addAction(bg_action)

        self._clip_action = QAction("Clip plane", self, checkable=True)
        self._clip_action.triggered.connect(self._toggle_clip)
        toggles.addAction(self._clip_action)
        toolbar.addAction(self._clip_action)

        # -- File menu
        file_menu = menubar.addMenu("&File")
        import_action = QAction("Import file…", self)
        import_action.triggered.connect(self._import_file)
        file_menu.addAction(import_action)
        toolbar.addAction(import_action)

        reload_action = QAction("Reload files", self)
        reload_action.triggered.connect(self._reload_files)
        file_menu.addAction(reload_action)

        reload_scene_action = QAction("Reload scene (re-apply from disk)", self)
        reload_scene_action.triggered.connect(self._reload_scene)
        file_menu.addAction(reload_scene_action)

        screenshot_action = QAction("Screenshot / Export…", self)
        screenshot_action.triggered.connect(self._export_screenshot)
        file_menu.addAction(screenshot_action)
        toolbar.addAction(screenshot_action)

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
            self._apply_item_async(item)
        self._update_status()

    def _apply_item_async(self, item: ViewItem) -> None:
        worker = _ApplyWorker(item, self.plotter, self.data_dir)
        worker.signals.done.connect(self._on_apply_done)
        worker.signals.failed.connect(self._on_apply_failed)
        self._workers.append(worker)
        self._log(f"Building {item.kind}: {item.label}…")
        self._pool.start(worker)

    def _on_apply_done(self, item: ViewItem) -> None:
        # render() touches VTK; this slot runs on the main thread (queued signal).
        self.plotter.render()
        self._apply_row_state(item)
        self._log(f"Added {item.kind}: {item.label}")
        self._reap_worker(item)

    def _on_apply_failed(self, item: ViewItem, exc: Exception) -> None:
        QMessageBox.critical(self, "Add failed", f"Could not add {item.label}:\n{exc}")
        # Roll the failed item back out of the queue/tree.
        if item in self._items:
            self._items.remove(item)
        self._remove_tree_row(item)
        self._update_status()
        self._reap_worker(item)

    def _reap_worker(self, item: ViewItem) -> None:
        self._workers = [w for w in self._workers if w.item is not item]

    def _clear_items(self) -> None:
        self.plotter.clear_items()
        self.plotter.render()
        self._items.clear()
        self._tree.clear()
        if self._profile_ax is not None:
            self._profile_ax.clear()
            self._profile_canvas.draw_idle()
        self._update_status()

    def _remove_item(self, item: ViewItem) -> None:
        self.plotter.remove_item(item.id)
        if item in self._items:
            self._items.remove(item)
        self._remove_tree_row(item)
        if isinstance(item, ProfileItem):
            self._rebuild_profile_plot()
        self._update_status()
        self._log(f"Removed {item.label}")

    # -- scene tree ----------------------------------------------------------

    def _add_tree_row(self, item: ViewItem) -> None:
        node = QTreeWidgetItem(self._tree)
        row = _SceneRow(item, self._on_row_changed, self._remove_item)
        node.setData(0, Qt.ItemDataRole.UserRole, item.id)
        self._tree.addTopLevelItem(node)
        self._tree.setItemWidget(node, 0, row)

    def _find_tree_node(self, item: ViewItem) -> tuple[QTreeWidgetItem | None, _SceneRow | None]:
        for i in range(self._tree.topLevelItemCount()):
            node = self._tree.topLevelItem(i)
            if node.data(0, Qt.ItemDataRole.UserRole) == item.id:
                widget = self._tree.itemWidget(node, 0)
                return node, widget if isinstance(widget, _SceneRow) else None
        return None, None

    def _remove_tree_row(self, item: ViewItem) -> None:
        node, _row = self._find_tree_node(item)
        if node is not None:
            self._tree.takeTopLevelItem(self._tree.indexOfTopLevelItem(node))

    def _on_row_changed(self, item: ViewItem) -> None:
        if isinstance(item, ProfileItem):
            return
        self._apply_row_state(item)

    def _apply_row_state(self, item: ViewItem) -> None:
        """Push the row's control state into the live actor."""
        _node, row = self._find_tree_node(item)
        if row is None:
            return
        self.plotter.update_item(
            item.id,
            color=row.color_rgb,
            opacity=row.opacity,
            visibility=row.visible,
        )

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

    def _toggle_clip(self) -> None:
        """Add/remove an interactive clip-plane widget on the whole scene.

        We use PyVista's `add_mesh_clip_plane` interactive widget against a
        merged copy of the live meshes — simpler and more discoverable than
        reapplying `UnifiedPlotter.set_clip_plane` to every item.
        """
        p = self.plotter.plotter
        if self._clip_action.isChecked():
            try:
                merged = p.renderer.actors  # noqa: F841 - presence check only
                # Re-add each tracked actor's dataset under an interactive clip.
                # Simplest robust path: enable a clip box widget on the union.
                import pyvista as pv

                blocks = []
                for handles in self.plotter._actors.values():
                    for h in handles:
                        mapper = getattr(h, "mapper", None)
                        ds = getattr(mapper, "dataset", None) if mapper is not None else None
                        if ds is not None:
                            blocks.append(ds)
                if not blocks:
                    self._log("Clip: nothing to clip")
                    self._clip_action.setChecked(False)
                    return
                union = pv.MultiBlock(blocks).combine()
                self._clip_actor = p.add_mesh_clip_plane(union, name="__omniviz_clip__")
                self._log("Clip plane enabled (drag the widget)")
                self.plotter.render()
            except Exception as exc:  # noqa: BLE001
                log.debug("Clip enable failed", exc_info=True)
                self._log(f"Clip unavailable: {exc}")
                self._clip_action.setChecked(False)
        else:
            try:
                p.clear_plane_widgets()
                if getattr(self, "_clip_actor", None) is not None:
                    p.remove_actor(self._clip_actor)
                    self._clip_actor = None
                self.plotter.render()
                self._log("Clip plane disabled")
            except Exception:  # noqa: BLE001
                log.debug("Clip disable failed", exc_info=True)

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
        self.plotter.clear_items()
        self.plotter.render()
        if self._profile_ax is not None:
            self._profile_ax.clear()
            self._profile_canvas.draw_idle()
        for item in items:
            if isinstance(item, ProfileItem):
                self._render_profile(item)
            else:
                self._apply_item_async(item)
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
    if LOGO_PATH.is_file():
        app.setWindowIcon(QIcon(str(LOGO_PATH)))

    path = Path(data_dir) if data_dir else None
    window = MainWindow(data_dir=path)
    window.show()
    app.exec()


if __name__ == "__main__":
    run()
