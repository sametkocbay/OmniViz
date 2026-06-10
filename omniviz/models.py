"""Typed visualization items used by the GUI render queue.

Each ``ViewItem`` subclass holds the parameters for a single renderable and
knows how to apply itself to a :class:`omniviz.plotter.UnifiedPlotter`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pandas as pd

from omniviz import io as oio
from omniviz.plotter import UnifiedPlotter

log = logging.getLogger(__name__)


@dataclass
class ViewItem(ABC):
    """Base class for everything that can be queued for rendering."""

    label: str = ""

    #: short human-readable label for the type, e.g. ``"POINT CLOUD"``
    kind: str = ""

    #: stable identity, used to add/remove/update the live actor(s)
    id: str = field(default_factory=lambda: uuid4().hex)

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
            item_id=self.id,
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
            item_id=self.id,
        )


@dataclass
class VtkMeshItem(ViewItem):
    file: str = ""
    color: str = "lightblue"
    opacity: float = 1.0
    show_edges: bool = True
    kind: str = "VTK"

    #: file extensions this item can load (consumed by the GUI panel filter)
    extensions: tuple[str, ...] = (".vtk", ".vtu")

    def summary(self) -> str:
        return f"[{self.kind}] {self.file} (.vtk/.vtu) — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        plotter.add_vtk_mesh(
            data_dir / self.file,
            color=self.color,
            opacity=self.opacity,
            show_edges=self.show_edges,
            label=self.label,
            item_id=self.id,
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
            item_id=self.id,
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
            item_id=self.id,
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
            item_id=self.id,
        )


@dataclass
class CariddiMeshItem(ViewItem):
    """A CARIDDI volume mesh built from ``x``/``ix``/``ixtype`` files."""

    x_file: str = ""
    ix_file: str = ""
    ixtype_file: str = ""
    color: str = "violet"
    opacity: float = 1.0
    show_edges: bool = True
    kind: str = "CARIDDI MESH"

    def summary(self) -> str:
        return f"[{self.kind}] {self.x_file} — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        import pyvista as pv

        nodes, elements = oio.read_cariddi_mesh(
            data_dir / self.x_file,
            data_dir / self.ix_file,
            data_dir / self.ixtype_file,
        )
        cell_type_map = {
            "HEXAHEDRON": pv.CellType.HEXAHEDRON,
            "TETRA": pv.CellType.TETRA,
            "WEDGE": pv.CellType.WEDGE,
        }
        for name, cells in elements.items():
            if cells.size == 0:
                continue
            grid = pv.UnstructuredGrid({cell_type_map[name]: cells}, nodes)
            plotter.add_unstructured(
                grid,
                color=self.color,
                opacity=self.opacity,
                show_edges=self.show_edges,
                label=self.label,
                item_id=self.id,
            )


@dataclass
class CariddiCurrentDensityItem(ViewItem):
    """CARIDDI current-density glyphs.

    The current-density (``J``) computation is CARIDDI-run-specific and needs
    extra inputs (``gmat``/``I3D``); when those are absent we degrade
    gracefully: ``summary()`` says so and ``apply()`` no-ops with a warning.
    """

    x_file: str = ""
    ix_file: str = ""
    ixtype_file: str = ""
    j_file: str = ""
    color: str = "crimson"
    colormap: str = "coolwarm"
    scale: float = 1.0
    kind: str = "CARIDDI J"

    def summary(self) -> str:
        if not self.j_file:
            return f"[{self.kind}] (no J source — inactive) — {self.label}"
        return f"[{self.kind}] {self.j_file} — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        import numpy as np
        import pyvista as pv

        if not self.j_file:
            log.warning("CariddiCurrentDensityItem '%s' has no J source; skipping", self.label)
            return

        j_path = data_dir / self.j_file
        if not j_path.exists():
            log.warning("CARIDDI J source '%s' not found; skipping", j_path)
            return

        nodes, elements = oio.read_cariddi_mesh(
            data_dir / self.x_file,
            data_dir / self.ix_file,
            data_dir / self.ixtype_file,
        )
        hexes = elements.get("HEXAHEDRON")
        if hexes is None or hexes.size == 0:
            log.warning("CARIDDI J: no hexahedral elements; skipping")
            return

        centroids = oio.compute_element_centroids(nodes, hexes)
        j_vectors = np.loadtxt(j_path, dtype=np.float64)
        if j_vectors.ndim == 1:
            j_vectors = j_vectors.reshape(1, -1)
        j_vectors = j_vectors[:, :3] * self.scale
        n = min(len(centroids), len(j_vectors))

        cloud = pv.PolyData(centroids[:n])
        cloud["J"] = j_vectors[:n]
        cloud["magnitude"] = np.linalg.norm(j_vectors[:n], axis=1)
        glyphs = cloud.glyph(orient="J", scale="magnitude", geom=pv.Cone())
        plotter.add_unstructured(
            glyphs,
            scalars="magnitude",
            colormap=self.colormap,
            show_scalar_bar=True,
            show_edges=False,
            label=self.label,
            item_id=self.id,
        )


@dataclass
class ProfileItem(ViewItem):
    """A 2-column profile (rho/q/ffp/t) for a 2D matplotlib line plot.

    The 3D plotter cannot render this, so ``apply()`` is a no-op; the GUI's
    matplotlib panel consumes :meth:`load`.
    """

    file: str = ""
    kind: str = "PROFILE"

    def summary(self) -> str:
        return f"[{self.kind}] {self.file} — {self.label}"

    def load(self, data_dir: Path) -> pd.DataFrame:
        """Return the parsed ``x, value`` profile DataFrame."""
        return oio.read_profile(data_dir / self.file)

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        log.info("ProfileItem '%s' is a 2D profile; not rendered in the 3D view", self.label)


@dataclass
class Hdf5RestartItem(ViewItem):
    """A JOREK HDF5 restart rendered as a colored unstructured grid."""

    file: str = ""
    scalar: str = "psi"
    n_sub: int = 2
    n_plane: int = 4
    colormap: str = "viridis"
    kind: str = "JOREK HDF5"

    def summary(self) -> str:
        return f"[{self.kind}] {self.file}  scalar={self.scalar} — {self.label}"

    def apply(self, plotter: UnifiedPlotter, data_dir: Path) -> None:
        grid = oio.read_jorek_restart(
            data_dir / self.file,
            n_sub=self.n_sub,
            n_plane=self.n_plane,
            variables=[self.scalar],
        )
        plotter.add_unstructured(
            grid,
            scalars=self.scalar,
            colormap=self.colormap,
            show_scalar_bar=True,
            show_edges=False,
            label=self.label,
            item_id=self.id,
        )
