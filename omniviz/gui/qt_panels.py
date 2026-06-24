"""Qt (PySide6) port of the OmniViz input panels.

Each panel encapsulates a file picker (where relevant), the per-item options
form, and a factory method :meth:`build_item` that produces a typed
:class:`omniviz.models.ViewItem`. Panels publish queue additions through an
``on_add`` callback supplied by the parent window — the same contract as the
original Tk panels in ``panels.py``.

No parsing lives here; file binning reuses ``categorize_files`` / io helpers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

from qtpy.QtCore import QLocale, Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from omniviz.gui.theme import COLOR_PALETTE, COLORMAPS
from omniviz.models import (
    BoundaryItem,
    CariddiCurrentDensityItem,
    CariddiMeshItem,
    Hdf5RestartItem,
    PatranMeshItem,
    PointCloudItem,
    ProfileItem,
    VectorFieldItem,
    ViewItem,
    VtkMeshItem,
    WireItem,
)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=ViewItem)

OnAdd = Callable[[ViewItem], None]


# --------------------------------------------------------------------------- #
# File categorization (extends panels.categorize_files for the new types)
# --------------------------------------------------------------------------- #


def categorize_files_qt(data_dir: Path) -> dict[str, list[str]]:
    """Bin files in ``data_dir`` by category, including the new data types.

    Wraps :func:`omniviz.gui.panels.categorize_files` (which covers point
    clouds, boundary, vtk, patran and vector fields) and adds buckets for
    ``.vtu``, JOREK HDF5 (``.h5``), CARIDDI ``.dat`` mesh files and generic
    profile ``.dat`` files. Parsing/detection stays in io.py via the wrapped
    helper; here we only sort filenames into buckets.
    """
    from omniviz.gui.panels import categorize_files

    categories = categorize_files(data_dir)
    # Ensure every bucket the panels expect exists.
    for key in ("point_cloud", "boundary", "vtk", "patran", "vector_field"):
        categories.setdefault(key, [])
    categories.setdefault("hdf5", [])
    categories.setdefault("dat", [])

    if not data_dir.exists():
        return categories

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("."):
            continue
        lower = name.lower()
        if lower.endswith(".vtu"):
            # The VTK panel handles both .vtk and .vtu.
            if name not in categories["vtk"]:
                categories["vtk"].append(name)
        elif lower.endswith((".h5", ".hdf5")):
            categories["hdf5"].append(name)
        elif lower.endswith(".dat"):
            # CARIDDI mesh files and profiles are all .dat; surface them all in
            # one bucket so the CARIDDI/Profile panels can pick the right ones.
            if name not in categories["dat"]:
                categories["dat"].append(name)

    categories["vtk"].sort()
    return categories


# --------------------------------------------------------------------------- #
# Reusable widgets
# --------------------------------------------------------------------------- #


class FilePicker(QWidget):
    """A filterable, single-select list of file names."""

    def __init__(self, files: Sequence[str]) -> None:
        super().__init__()
        self._files = list(files)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter files…")
        self._search.textChanged.connect(self._refresh)
        layout.addWidget(self._search)

        self._list = QListWidget()
        layout.addWidget(self._list)
        self._refresh()

    @property
    def selected(self) -> str | None:
        item = self._list.currentItem()
        return item.text() if item is not None else None

    def set_files(self, files: Sequence[str]) -> None:
        self._files = list(files)
        self._refresh()

    def _refresh(self) -> None:
        query = self._search.text().lower().strip()
        files = [f for f in self._files if query in f.lower()] if query else self._files
        self._list.clear()
        self._list.addItems(files)


def _color_combo(default: str) -> QComboBox:
    combo = QComboBox()
    combo.addItems(list(COLOR_PALETTE))
    if default in COLOR_PALETTE:
        combo.setCurrentText(default)
    return combo


def _cmap_combo(default: str) -> QComboBox:
    combo = QComboBox()
    combo.addItems(list(COLORMAPS))
    if default in COLORMAPS:
        combo.setCurrentText(default)
    return combo


def _opacity_spin(default: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.05)
    spin.setValue(default)
    return spin


def _float_line_edit(default: str) -> QLineEdit:
    """A line edit that only accepts plain decimal numbers (no ``e`` exponent)."""
    le = QLineEdit(default)
    # Always use '.' as the decimal separator regardless of the system locale
    # (e.g. a German locale would otherwise treat '.' as a group separator and
    # turn "2.01" into "201").
    c_locale = QLocale.c()
    c_locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
    validator = QDoubleValidator(le)
    validator.setLocale(c_locale)
    validator.setNotation(QDoubleValidator.Notation.StandardNotation)  # forbid '1e5' input
    validator.setDecimals(15)  # keep full precision for physics values
    le.setLocale(c_locale)
    le.setValidator(validator)
    return le


# --------------------------------------------------------------------------- #
# Base panel
# --------------------------------------------------------------------------- #


class BasePanel(QWidget, Generic[T]):
    """Common scaffolding: optional file picker, an options form, add button."""

    title: str = ""
    button_label: str = "Add to scene"

    def __init__(self, on_add: OnAdd, files: Sequence[str] | None = None) -> None:
        super().__init__()
        self._on_add = on_add
        self._files = list(files or [])

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        if self.title:
            header = QLabel(self.title)
            header.setStyleSheet("font-weight: 700; font-size: 15px;")
            root.addWidget(header)

        self._picker: FilePicker | None = None
        if self._needs_file_picker():
            self._picker = FilePicker(self._files)
            root.addWidget(self._picker, stretch=1)

        options = QGroupBox("Options")
        self._form = QFormLayout(options)
        self._build_options(self._form)
        root.addWidget(options)

        add_btn = QPushButton(self.button_label)
        # Mark as the primary action so the QSS accent rule styles it.
        add_btn.setProperty("class", "primary")
        add_btn.clicked.connect(self._handle_add)
        root.addWidget(add_btn, alignment=Qt.AlignmentFlag.AlignRight)
        root.addStretch(0)

    # -- override hooks ------------------------------------------------------ #

    def _needs_file_picker(self) -> bool:
        return True

    def _build_options(self, form: QFormLayout) -> None:
        raise NotImplementedError

    def build_item(self) -> T | None:
        raise NotImplementedError

    # -- shared helpers ------------------------------------------------------ #

    def set_files(self, files: Sequence[str]) -> None:
        self._files = list(files)
        if self._picker is not None:
            self._picker.set_files(files)

    def _handle_add(self) -> None:
        item = self.build_item()
        if item is not None:
            self._on_add(item)


# --------------------------------------------------------------------------- #
# Concrete panels (ported from panels.py)
# --------------------------------------------------------------------------- #


class PointCloudPanel(BasePanel[PointCloudItem]):
    title = "Point Clouds"

    def _build_options(self, form: QFormLayout) -> None:
        self._color = _color_combo("red")
        form.addRow("Color", self._color)
        self._size = QSpinBox()
        self._size.setRange(1, 20)
        self._size.setValue(5)
        form.addRow("Point size", self._size)
        self._label = QLineEdit()
        self._label.setPlaceholderText("(auto)")
        form.addRow("Label", self._label)

    def build_item(self) -> PointCloudItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return PointCloudItem(
            file=f,
            label=self._label.text() or f,
            color=self._color.currentText(),
            point_size=self._size.value(),
        )


class BoundaryPanel(BasePanel[BoundaryItem]):
    title = "Boundary Surface"

    def _build_options(self, form: QFormLayout) -> None:
        self._color = _color_combo("cyan")
        form.addRow("Color", self._color)
        self._opacity = _opacity_spin(1.0)
        form.addRow("Opacity", self._opacity)
        self._nphi = QSpinBox()
        self._nphi.setRange(3, 720)
        self._nphi.setValue(81)
        form.addRow("n_phi", self._nphi)
        self._ns = QSpinBox()
        self._ns.setRange(2, 200)
        self._ns.setValue(10)
        form.addRow("n_s", self._ns)
        self._edges = QCheckBox("Show edges")
        form.addRow(self._edges)
        self._label = QLineEdit()
        self._label.setPlaceholderText("Boundary")
        form.addRow("Label", self._label)

    def build_item(self) -> BoundaryItem | None:
        if not self._picker or not self._picker.selected:
            return None
        return BoundaryItem(
            file=self._picker.selected,
            label=self._label.text() or "Boundary",
            color=self._color.currentText(),
            opacity=self._opacity.value(),
            n_phi=self._nphi.value(),
            n_s=self._ns.value(),
            show_edges=self._edges.isChecked(),
        )


class VtkMeshPanel(BasePanel[VtkMeshItem]):
    title = "VTK / VTU Mesh"

    def _build_options(self, form: QFormLayout) -> None:
        self._color = _color_combo("lightblue")
        form.addRow("Color", self._color)
        self._opacity = _opacity_spin(1.0)
        form.addRow("Opacity", self._opacity)
        self._edges = QCheckBox("Show edges")
        self._edges.setChecked(True)
        form.addRow(self._edges)
        self._label = QLineEdit()
        self._label.setPlaceholderText("VTK Mesh")
        form.addRow("Label", self._label)

    def build_item(self) -> VtkMeshItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return VtkMeshItem(
            file=f,
            label=self._label.text() or f,
            color=self._color.currentText(),
            opacity=self._opacity.value(),
            show_edges=self._edges.isChecked(),
        )


class PatranMeshPanel(BasePanel[PatranMeshItem]):
    title = "Patran Mesh (.msh)"

    def _build_options(self, form: QFormLayout) -> None:
        self._color = _color_combo("violet")
        form.addRow("Color", self._color)
        self._opacity = _opacity_spin(0.7)
        form.addRow("Opacity", self._opacity)
        self._edges = QCheckBox("Show edges")
        self._edges.setChecked(True)
        form.addRow(self._edges)
        self._label = QLineEdit()
        self._label.setPlaceholderText("Patran Mesh")
        form.addRow("Label", self._label)

    def build_item(self) -> PatranMeshItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return PatranMeshItem(
            file=f,
            label=self._label.text() or f,
            color=self._color.currentText(),
            opacity=self._opacity.value(),
            show_edges=self._edges.isChecked(),
        )


class VectorFieldPanel(BasePanel[VectorFieldItem]):
    title = "Vector Field  (x y z Bx By Bz)"

    def _build_options(self, form: QFormLayout) -> None:
        self._color = _color_combo("crimson")
        form.addRow("Arrow color", self._color)
        self._scale = QDoubleSpinBox()
        self._scale.setRange(0.0, 1e6)
        self._scale.setDecimals(4)
        self._scale.setValue(0.1)
        form.addRow("Scale", self._scale)
        self._sample = QDoubleSpinBox()
        self._sample.setRange(0.01, 1.0)
        self._sample.setSingleStep(0.05)
        self._sample.setValue(1.0)
        form.addRow("Subsample", self._sample)
        self._cmap = _cmap_combo("plasma")
        form.addRow("Colormap", self._cmap)
        self._color_by_mag = QCheckBox("Color by |B|")
        form.addRow(self._color_by_mag)
        self._label = QLineEdit()
        self._label.setPlaceholderText("B field")
        form.addRow("Label", self._label)

    def build_item(self) -> VectorFieldItem | None:
        if not self._picker or not self._picker.selected:
            return None
        return VectorFieldItem(
            file=self._picker.selected,
            label=self._label.text() or "B field",
            color=self._color.currentText(),
            scale=self._scale.value(),
            sample_frac=self._sample.value(),
            color_by_magnitude=self._color_by_mag.isChecked(),
            colormap=self._cmap.currentText(),
        )


class WirePanel(BasePanel[WireItem]):
    title = "Wire (current filament loop)"

    def _needs_file_picker(self) -> bool:
        return False

    def _build_options(self, form: QFormLayout) -> None:
        self._r0 = _float_line_edit("1.99141779000833")
        form.addRow("r0 (major radius) [m]", self._r0)
        self._z0 = _float_line_edit("0.0")
        form.addRow("z0 (axial pos) [m]", self._z0)
        self._alfa = _float_line_edit("3.0")
        form.addRow("alfa_wire [deg]", self._alfa)
        self._color = _color_combo("orange")
        form.addRow("Color", self._color)
        self._tube = _float_line_edit("0.0")
        form.addRow("Tube radius [m]", self._tube)
        self._label = QLineEdit()
        self._label.setPlaceholderText("Wire")
        form.addRow("Label", self._label)

    def build_item(self) -> WireItem | None:
        try:
            r0 = float(self._r0.text())
            z0 = float(self._z0.text())
            alfa = float(self._alfa.text())
            tube_in = float(self._tube.text())
        except ValueError:
            return None
        if r0 <= 0:
            return None
        return WireItem(
            label=self._label.text() or "Wire",
            r0=r0,
            z0=z0,
            alfa_wire_deg=alfa,
            color=self._color.currentText(),
            tube_radius=tube_in if tube_in > 0 else None,
        )


# --------------------------------------------------------------------------- #
# New panels: CARIDDI / JOREK / Profile
# --------------------------------------------------------------------------- #


def _dat_combo(files: Sequence[str], default: str, allow_blank: bool = False) -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    if allow_blank:
        combo.addItem("")
    combo.addItems(list(files))
    # Pre-select a sensible default if the file is present.
    idx = combo.findText(default)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    return combo


class CariddiMeshPanel(BasePanel[CariddiMeshItem]):
    """CARIDDI volume mesh from x.dat / ix.dat / ixtype.dat."""

    title = "CARIDDI Mesh"

    def _needs_file_picker(self) -> bool:
        # Three coordinated .dat pickers — no single-file list.
        return False

    def _build_options(self, form: QFormLayout) -> None:
        dat = self._files
        self._x = _dat_combo(dat, "x.dat")
        form.addRow("x.dat (nodes)", self._x)
        self._ix = _dat_combo(dat, "ix.dat")
        form.addRow("ix.dat (incidence)", self._ix)
        self._ixtype = _dat_combo(dat, "ixtype.dat")
        form.addRow("ixtype.dat (types)", self._ixtype)
        self._color = _color_combo("violet")
        form.addRow("Color", self._color)
        self._opacity = _opacity_spin(1.0)
        form.addRow("Opacity", self._opacity)
        self._edges = QCheckBox("Show edges")
        self._edges.setChecked(True)
        form.addRow(self._edges)
        self._label = QLineEdit()
        self._label.setPlaceholderText("CARIDDI mesh")
        form.addRow("Label", self._label)

    def set_files(self, files: Sequence[str]) -> None:
        super().set_files(files)
        for combo, default in (
            (self._x, "x.dat"),
            (self._ix, "ix.dat"),
            (self._ixtype, "ixtype.dat"),
        ):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(list(files))
            combo.setCurrentText(cur or default)

    def build_item(self) -> CariddiMeshItem | None:
        x, ix, ixtype = self._x.currentText(), self._ix.currentText(), self._ixtype.currentText()
        if not (x and ix and ixtype):
            return None
        return CariddiMeshItem(
            x_file=x,
            ix_file=ix,
            ixtype_file=ixtype,
            label=self._label.text() or "CARIDDI mesh",
            color=self._color.currentText(),
            opacity=self._opacity.value(),
            show_edges=self._edges.isChecked(),
        )


class CariddiCurrentDensityPanel(BasePanel[CariddiCurrentDensityItem]):
    """CARIDDI current-density glyphs: mesh files + optional J file."""

    title = "CARIDDI Current Density"

    def _needs_file_picker(self) -> bool:
        return False

    def _build_options(self, form: QFormLayout) -> None:
        dat = self._files
        self._x = _dat_combo(dat, "x.dat")
        form.addRow("x.dat (nodes)", self._x)
        self._ix = _dat_combo(dat, "ix.dat")
        form.addRow("ix.dat (incidence)", self._ix)
        self._ixtype = _dat_combo(dat, "ixtype.dat")
        form.addRow("ixtype.dat (types)", self._ixtype)
        self._j = _dat_combo(dat, "", allow_blank=True)
        form.addRow("J file (optional)", self._j)
        self._cmap = _cmap_combo("coolwarm")
        form.addRow("Colormap", self._cmap)
        self._scale = QDoubleSpinBox()
        self._scale.setRange(0.0, 1e9)
        self._scale.setDecimals(4)
        self._scale.setValue(1.0)
        form.addRow("Scale", self._scale)
        self._label = QLineEdit()
        self._label.setPlaceholderText("CARIDDI J")
        form.addRow("Label", self._label)

    def set_files(self, files: Sequence[str]) -> None:
        super().set_files(files)
        for combo, default in (
            (self._x, "x.dat"),
            (self._ix, "ix.dat"),
            (self._ixtype, "ixtype.dat"),
        ):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(list(files))
            combo.setCurrentText(cur or default)
        cur_j = self._j.currentText()
        self._j.clear()
        self._j.addItem("")
        self._j.addItems(list(files))
        self._j.setCurrentText(cur_j)

    def build_item(self) -> CariddiCurrentDensityItem | None:
        x, ix, ixtype = self._x.currentText(), self._ix.currentText(), self._ixtype.currentText()
        if not (x and ix and ixtype):
            return None
        return CariddiCurrentDensityItem(
            x_file=x,
            ix_file=ix,
            ixtype_file=ixtype,
            j_file=self._j.currentText(),
            label=self._label.text() or "CARIDDI J",
            colormap=self._cmap.currentText(),
            scale=self._scale.value(),
        )


class Hdf5RestartPanel(BasePanel[Hdf5RestartItem]):
    """JOREK HDF5 restart: .h5 picker + scalar name + n_sub / n_plane."""

    title = "JOREK Restart (.h5)"

    def _build_options(self, form: QFormLayout) -> None:
        self._scalar = QLineEdit("psi")
        form.addRow("Scalar", self._scalar)
        self._nsub = QSpinBox()
        self._nsub.setRange(2, 16)
        self._nsub.setValue(2)
        form.addRow("n_sub", self._nsub)
        self._nplane = QSpinBox()
        self._nplane.setRange(1, 64)
        self._nplane.setValue(4)
        form.addRow("n_plane", self._nplane)
        self._cmap = _cmap_combo("viridis")
        form.addRow("Colormap", self._cmap)
        self._label = QLineEdit()
        self._label.setPlaceholderText("JOREK restart")
        form.addRow("Label", self._label)

    def build_item(self) -> Hdf5RestartItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return Hdf5RestartItem(
            file=f,
            label=self._label.text() or f,
            scalar=self._scalar.text() or "psi",
            n_sub=self._nsub.value(),
            n_plane=self._nplane.value(),
            colormap=self._cmap.currentText(),
        )


class ProfilePanel(BasePanel[ProfileItem]):
    """A 2-column profile (.dat) rendered as a 2D matplotlib line."""

    title = "Profile (2D plot)"

    def _build_options(self, form: QFormLayout) -> None:
        self._label = QLineEdit()
        self._label.setPlaceholderText("Profile")
        form.addRow("Label", self._label)

    def build_item(self) -> ProfileItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return ProfileItem(file=f, label=self._label.text() or f)
