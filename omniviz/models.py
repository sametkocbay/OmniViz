"""Typed visualization items used by the GUI render queue.

Each ``ViewItem`` subclass holds the parameters for a single renderable and
knows how to apply itself to a :class:`omniviz.plotter.UnifiedPlotter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from omniviz import io as oio
from omniviz.plotter import UnifiedPlotter


@dataclass
class ViewItem(ABC):
    """Base class for everything that can be queued for rendering."""

    label: str = ""

    #: short human-readable label for the type, e.g. ``"POINT CLOUD"``
    kind: str = ""

    @abstractmethod
    def summary(self) -> str:
        """Single-line description shown in the GUI queue."""

    @abstractmethod
    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        """Add this item to ``plotter``."""


@dataclass
class PointCloudItem(ViewItem):
    file: str = ""
    color: str = "red"
    point_size: int = 5
    kind: str = "POINT CLOUD"

    def summary(self) -> str:
        return f"[{self.kind}] {self.file} — {self.label or self.file}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        plotter.add_point_cloud(
            data_dir / self.file,
            color=self.color,
            point_size=self.point_size,
            label=self.label,
        )


@dataclass
class BoundaryItem(ViewItem):
    file: str = ""
    color: str = "cyan"
    opacity: float = 1.0
    n_phi: int = 81
    n_s: int = 10
    show_edges: bool = False
    kind: str = "BOUNDARY"

    def summary(self) -> str:
        return f"[{self.kind}] {self.file} — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        plotter.add_boundary(
            data_dir / self.file,
            color=self.color,
            opacity=self.opacity,
            n_phi=self.n_phi,
            n_s=self.n_s,
            show_edges=self.show_edges,
            label=self.label,
        )


@dataclass
class VtkMeshItem(ViewItem):
    file: str = ""
    color: str = "lightblue"
    opacity: float = 1.0
    show_edges: bool = True
    kind: str = "VTK"

    def summary(self) -> str:
        return f"[{self.kind}] {self.file} — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        plotter.add_vtk_mesh(
            data_dir / self.file,
            color=self.color,
            opacity=self.opacity,
            show_edges=self.show_edges,
            label=self.label,
        )


@dataclass
class PatranMeshItem(ViewItem):
    file: str = ""
    color: str = "violet"
    opacity: float = 0.7
    show_edges: bool = True
    kind: str = "PATRAN"

    def summary(self) -> str:
        return f"[{self.kind}] {self.file} — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        df_nodes, hexes = oio.read_patran_neutral(data_dir / self.file)
        plotter.add_hex_mesh(
            df_nodes,
            hexes,
            color=self.color,
            opacity=self.opacity,
            show_edges=self.show_edges,
            label=self.label,
        )


@dataclass
class VectorFieldItem(ViewItem):
    file: str = ""
    color: str = "crimson"
    scale: float = 0.1
    sample_frac: float = 1.0
    color_by_magnitude: bool = False
    colormap: str = "plasma"
    kind: str = "VECTOR FIELD"

    def summary(self) -> str:
        return f"[{self.kind}] {self.file}  scale={self.scale:g} — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        plotter.add_vector_field(
            data_dir / self.file,
            scale=self.scale,
            color=self.color,
            colormap=self.colormap,
            color_by_magnitude=self.color_by_magnitude,
            sample_frac=self.sample_frac,
            label=self.label,
        )


@dataclass
class WireItem(ViewItem):
    r0: float = 1.99141779
    z0: float = 0.0
    alfa_wire_deg: float = 3.0
    color: str = "orange"
    tube_radius: float | None = None
    kind: str = "WIRE"

    def summary(self) -> str:
        return (
            f"[{self.kind}] r0={self.r0:.4f}  z0={self.z0:.4f}  "
            f"α={self.alfa_wire_deg:.3f}° — {self.label}"
        )

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        plotter.add_wire(
            r0=self.r0,
            z0=self.z0,
            alfa_wire_deg=self.alfa_wire_deg,
            color=self.color,
            tube_radius=self.tube_radius,
            label=self.label,
        )
