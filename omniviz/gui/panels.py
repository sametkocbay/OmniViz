"""Tab panels — one per visualization type — for the OmniViz GUI.

Each panel encapsulates a file picker (where relevant), the per-item options
form, and a factory method ``build_item`` that produces a typed
``ViewItem``.  Panels publish queue additions through a callback supplied by
the parent window.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

import customtkinter as ctk

from omniviz.gui.theme import COLOR_PALETTE, COLORMAPS, CORNER_RADIUS, PAD_X, PAD_Y
from omniviz.models import (
    BoundaryItem,
    PatranMeshItem,
    PointCloudItem,
    VectorFieldItem,
    ViewItem,
    VtkMeshItem,
    WireItem,
)

T = TypeVar("T", bound=ViewItem)

OnAdd = Callable[[ViewItem], None]


# --------------------------------------------------------------------------- #
# Reusable form widgets
# --------------------------------------------------------------------------- #


class FilePickerFrame(ctk.CTkFrame):
    """A searchable list of files with a single-select highlight."""

    def __init__(self, master, files: Sequence[str], height: int = 180) -> None:
        super().__init__(master, corner_radius=CORNER_RADIUS)
        self._files = list(files)
        self._buttons: list[ctk.CTkButton] = []
        self._selected: str | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._search = ctk.CTkEntry(self, placeholder_text="Filter files…")
        self._search.grid(row=0, column=0, sticky="ew", padx=PAD_X, pady=(PAD_Y, 4))
        self._search.bind("<KeyRelease>", lambda _e: self._refresh())

        self._scroll = ctk.CTkScrollableFrame(self, height=height, corner_radius=CORNER_RADIUS)
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=PAD_X, pady=(0, PAD_Y))
        self._scroll.grid_columnconfigure(0, weight=1)

        self._refresh()

    @property
    def selected(self) -> str | None:
        return self._selected

    def _refresh(self) -> None:
        for btn in self._buttons:
            btn.destroy()
        self._buttons.clear()

        query = self._search.get().lower().strip()
        files = [f for f in self._files if query in f.lower()] if query else self._files

        if not files:
            placeholder = ctk.CTkLabel(
                self._scroll, text="(no matching files)", text_color="gray60"
            )
            placeholder.grid(row=0, column=0, sticky="w", padx=PAD_X, pady=PAD_Y)
            self._buttons.append(placeholder)
            return

        for i, name in enumerate(files):
            btn = ctk.CTkButton(
                self._scroll,
                text=name,
                anchor="w",
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray85", "gray25"),
                command=lambda n=name: self._select(n),
            )
            btn.grid(row=i, column=0, sticky="ew", padx=4, pady=2)
            self._buttons.append(btn)
        self._apply_highlight()

    def _select(self, name: str) -> None:
        self._selected = name
        self._apply_highlight()

    def _apply_highlight(self) -> None:
        for btn in self._buttons:
            if not isinstance(btn, ctk.CTkButton):
                continue
            if btn.cget("text") == self._selected:
                btn.configure(fg_color=("gray80", "gray30"))
            else:
                btn.configure(fg_color="transparent")


def _row_label(master, row: int, col: int, text: str) -> None:
    ctk.CTkLabel(master, text=text, anchor="w").grid(
        row=row, column=col, padx=(PAD_X, 4), pady=4, sticky="w"
    )


# --------------------------------------------------------------------------- #
# Base panel
# --------------------------------------------------------------------------- #


class BasePanel(ctk.CTkFrame, Generic[T]):
    """Common scaffolding: header, options grid, file picker, add button."""

    title: str = ""
    button_label: str = "Add to queue"

    def __init__(self, master, on_add: OnAdd, files: Sequence[str] | None = None) -> None:
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self._on_add = on_add
        self._files = files or []

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=self.title, font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 4)
        )

        self._picker: FilePickerFrame | None = None
        next_row = 1
        if self._files is not None and self._needs_file_picker():
            self._picker = FilePickerFrame(self, self._files)
            self._picker.grid(row=next_row, column=0, sticky="nsew", padx=PAD_X, pady=4)
            self.grid_rowconfigure(next_row, weight=1)
            next_row += 1

        options = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        options.grid(row=next_row, column=0, sticky="ew", padx=PAD_X, pady=4)
        options.grid_columnconfigure(1, weight=1)
        options.grid_columnconfigure(3, weight=1)
        self._build_options(options)

        ctk.CTkButton(self, text=self.button_label, height=36, command=self._handle_add).grid(
            row=next_row + 1, column=0, sticky="e", padx=PAD_X, pady=PAD_Y
        )

    # -- override hooks ------------------------------------------------------ #

    def _needs_file_picker(self) -> bool:
        return True

    def _build_options(self, parent: ctk.CTkFrame) -> None:
        raise NotImplementedError

    def build_item(self) -> T | None:
        raise NotImplementedError

    # -- internal ------------------------------------------------------------ #

    def _handle_add(self) -> None:
        item = self.build_item()
        if item is not None:
            self._on_add(item)


# --------------------------------------------------------------------------- #
# Concrete panels
# --------------------------------------------------------------------------- #


class PointCloudPanel(BasePanel[PointCloudItem]):
    title = "Point Clouds"

    def _build_options(self, parent: ctk.CTkFrame) -> None:
        _row_label(parent, 0, 0, "Color")
        self._color = ctk.CTkOptionMenu(parent, values=list(COLOR_PALETTE))
        self._color.set("red")
        self._color.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 0, 2, "Point size")
        self._size = ctk.CTkSlider(parent, from_=1, to=20, number_of_steps=19)
        self._size.set(5)
        self._size.grid(row=0, column=3, sticky="ew", padx=4, pady=4)

        _row_label(parent, 1, 0, "Label")
        self._label = ctk.CTkEntry(parent, placeholder_text="(auto)")
        self._label.grid(row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

    def build_item(self) -> PointCloudItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return PointCloudItem(
            file=f,
            label=self._label.get() or f,
            color=self._color.get(),
            point_size=int(round(self._size.get())),
        )


class BoundaryPanel(BasePanel[BoundaryItem]):
    title = "Boundary Surface"

    def _build_options(self, parent: ctk.CTkFrame) -> None:
        _row_label(parent, 0, 0, "Color")
        self._color = ctk.CTkOptionMenu(parent, values=list(COLOR_PALETTE))
        self._color.set("cyan")
        self._color.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 0, 2, "Opacity")
        self._opacity = ctk.CTkSlider(parent, from_=0.1, to=1.0)
        self._opacity.set(1.0)
        self._opacity.grid(row=0, column=3, sticky="ew", padx=4, pady=4)

        _row_label(parent, 1, 0, "n_phi")
        self._nphi = ctk.CTkEntry(parent)
        self._nphi.insert(0, "81")
        self._nphi.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 1, 2, "n_s")
        self._ns = ctk.CTkEntry(parent)
        self._ns.insert(0, "10")
        self._ns.grid(row=1, column=3, sticky="ew", padx=4, pady=4)

        self._edges = ctk.CTkSwitch(parent, text="Show edges")
        self._edges.grid(row=2, column=0, columnspan=2, sticky="w", padx=PAD_X, pady=4)

        _row_label(parent, 2, 2, "Label")
        self._label = ctk.CTkEntry(parent, placeholder_text="Boundary")
        self._label.grid(row=2, column=3, sticky="ew", padx=4, pady=4)

    def build_item(self) -> BoundaryItem | None:
        if not self._picker or not self._picker.selected:
            return None
        try:
            n_phi = int(self._nphi.get())
            n_s = int(self._ns.get())
        except ValueError:
            return None
        return BoundaryItem(
            file=self._picker.selected,
            label=self._label.get() or "Boundary",
            color=self._color.get(),
            opacity=self._opacity.get(),
            n_phi=n_phi,
            n_s=n_s,
            show_edges=bool(self._edges.get()),
        )


class VtkMeshPanel(BasePanel[VtkMeshItem]):
    title = "VTK Mesh"

    def _build_options(self, parent: ctk.CTkFrame) -> None:
        _row_label(parent, 0, 0, "Color")
        self._color = ctk.CTkOptionMenu(parent, values=list(COLOR_PALETTE))
        self._color.set("lightblue")
        self._color.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 0, 2, "Opacity")
        self._opacity = ctk.CTkSlider(parent, from_=0.1, to=1.0)
        self._opacity.set(1.0)
        self._opacity.grid(row=0, column=3, sticky="ew", padx=4, pady=4)

        self._edges = ctk.CTkSwitch(parent, text="Show edges")
        self._edges.select()
        self._edges.grid(row=1, column=0, columnspan=2, sticky="w", padx=PAD_X, pady=4)

        _row_label(parent, 1, 2, "Label")
        self._label = ctk.CTkEntry(parent, placeholder_text="VTK Mesh")
        self._label.grid(row=1, column=3, sticky="ew", padx=4, pady=4)

    def build_item(self) -> VtkMeshItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return VtkMeshItem(
            file=f,
            label=self._label.get() or f,
            color=self._color.get(),
            opacity=self._opacity.get(),
            show_edges=bool(self._edges.get()),
        )


class PatranMeshPanel(BasePanel[PatranMeshItem]):
    title = "Patran Mesh (.msh)"

    def _build_options(self, parent: ctk.CTkFrame) -> None:
        _row_label(parent, 0, 0, "Color")
        self._color = ctk.CTkOptionMenu(parent, values=list(COLOR_PALETTE))
        self._color.set("violet")
        self._color.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 0, 2, "Opacity")
        self._opacity = ctk.CTkSlider(parent, from_=0.1, to=1.0)
        self._opacity.set(0.7)
        self._opacity.grid(row=0, column=3, sticky="ew", padx=4, pady=4)

        self._edges = ctk.CTkSwitch(parent, text="Show edges")
        self._edges.select()
        self._edges.grid(row=1, column=0, columnspan=2, sticky="w", padx=PAD_X, pady=4)

        _row_label(parent, 1, 2, "Label")
        self._label = ctk.CTkEntry(parent, placeholder_text="Patran Mesh")
        self._label.grid(row=1, column=3, sticky="ew", padx=4, pady=4)

    def build_item(self) -> PatranMeshItem | None:
        if not self._picker or not self._picker.selected:
            return None
        f = self._picker.selected
        return PatranMeshItem(
            file=f,
            label=self._label.get() or f,
            color=self._color.get(),
            opacity=self._opacity.get(),
            show_edges=bool(self._edges.get()),
        )


class VectorFieldPanel(BasePanel[VectorFieldItem]):
    title = "Vector Field  (x y z Bx By Bz)"

    def _build_options(self, parent: ctk.CTkFrame) -> None:
        _row_label(parent, 0, 0, "Arrow color")
        self._color = ctk.CTkOptionMenu(parent, values=list(COLOR_PALETTE))
        self._color.set("crimson")
        self._color.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 0, 2, "Scale")
        self._scale = ctk.CTkEntry(parent)
        self._scale.insert(0, "0.1")
        self._scale.grid(row=0, column=3, sticky="ew", padx=4, pady=4)

        _row_label(parent, 1, 0, "Subsample")
        self._sample = ctk.CTkSlider(parent, from_=0.01, to=1.0)
        self._sample.set(1.0)
        self._sample.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 1, 2, "Colormap")
        self._cmap = ctk.CTkOptionMenu(parent, values=list(COLORMAPS))
        self._cmap.set("plasma")
        self._cmap.grid(row=1, column=3, sticky="ew", padx=4, pady=4)

        self._color_by_mag = ctk.CTkSwitch(parent, text="Color by |B|")
        self._color_by_mag.grid(row=2, column=0, columnspan=2, sticky="w", padx=PAD_X, pady=4)

        _row_label(parent, 2, 2, "Label")
        self._label = ctk.CTkEntry(parent, placeholder_text="B field")
        self._label.grid(row=2, column=3, sticky="ew", padx=4, pady=4)

    def build_item(self) -> VectorFieldItem | None:
        if not self._picker or not self._picker.selected:
            return None
        try:
            scale = float(self._scale.get())
        except ValueError:
            return None
        return VectorFieldItem(
            file=self._picker.selected,
            label=self._label.get() or "B field",
            color=self._color.get(),
            scale=scale,
            sample_frac=float(self._sample.get()),
            color_by_magnitude=bool(self._color_by_mag.get()),
            colormap=self._cmap.get(),
        )


class WirePanel(BasePanel[WireItem]):
    title = "Wire (current filament loop)"

    def _needs_file_picker(self) -> bool:
        return False

    def _build_options(self, parent: ctk.CTkFrame) -> None:
        _row_label(parent, 0, 0, "r0 (major radius) [m]")
        self._r0 = ctk.CTkEntry(parent)
        self._r0.insert(0, "1.99141779000833")
        self._r0.grid(row=0, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        _row_label(parent, 1, 0, "z0 (axial pos) [m]")
        self._z0 = ctk.CTkEntry(parent)
        self._z0.insert(0, "0.0")
        self._z0.grid(row=1, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        _row_label(parent, 2, 0, "alfa_wire [deg]")
        self._alfa = ctk.CTkEntry(parent)
        self._alfa.insert(0, "3.0")
        self._alfa.grid(row=2, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

        _row_label(parent, 3, 0, "Color")
        self._color = ctk.CTkOptionMenu(parent, values=list(COLOR_PALETTE))
        self._color.set("orange")
        self._color.grid(row=3, column=1, sticky="ew", padx=4, pady=4)

        _row_label(parent, 3, 2, "Tube radius [m]")
        self._tube = ctk.CTkEntry(parent, placeholder_text="auto (0)")
        self._tube.insert(0, "0.0")
        self._tube.grid(row=3, column=3, sticky="ew", padx=4, pady=4)

        _row_label(parent, 4, 0, "Label")
        self._label = ctk.CTkEntry(parent, placeholder_text="Wire")
        self._label.grid(row=4, column=1, columnspan=3, sticky="ew", padx=4, pady=4)

    def build_item(self) -> WireItem | None:
        try:
            r0 = float(self._r0.get())
            z0 = float(self._z0.get())
            alfa = float(self._alfa.get())
            tube_in = float(self._tube.get())
        except ValueError:
            return None
        if r0 <= 0:
            return None
        return WireItem(
            label=self._label.get() or "Wire",
            r0=r0,
            z0=z0,
            alfa_wire_deg=alfa,
            color=self._color.get(),
            tube_radius=tube_in if tube_in > 0 else None,
        )


# --------------------------------------------------------------------------- #
# File categorization
# --------------------------------------------------------------------------- #


def categorize_files(data_dir: Path) -> dict[str, list[str]]:
    """Bin files in ``data_dir`` by their visualization category."""
    from omniviz.io import detect_file_columns  # local import keeps GUI cold-start fast

    categories: dict[str, list[str]] = {
        "point_cloud": [],
        "boundary": [],
        "vtk": [],
        "patran": [],
        "vector_field": [],
    }
    if not data_dir.exists():
        return categories

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("."):
            continue
        if name == "boundary.txt":
            categories["boundary"].append(name)
        elif name.endswith(".vtk"):
            categories["vtk"].append(name)
        elif name.endswith(".msh"):
            categories["patran"].append(name)
        elif name.endswith((".dat", ".txt")):
            if detect_file_columns(path) >= 6:
                categories["vector_field"].append(name)
            else:
                categories["point_cloud"].append(name)
    return categories
