"""Unified PyVista plotter for OmniViz.

A single ``UnifiedPlotter`` accumulates renderables (point clouds, boundary
surfaces, hex meshes, VTK meshes, vector fields, wire loops) and presents
them in one window. Each ``add_*`` returns ``self`` for fluent chaining.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyvista as pv

if TYPE_CHECKING:
    from PIL import Image as PILImage

from omniviz import io as oio
from omniviz.io import BoundaryData

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Boundary reconstruction (Hermite poloidal × Fourier toroidal)
# --------------------------------------------------------------------------- #


def _hermite_basis(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s2 = s * s
    s3 = s2 * s
    return (
        1.0 - 3.0 * s2 + 2.0 * s3,
        s - 2.0 * s2 + s3,
        3.0 * s2 - 2.0 * s3,
        -s2 + s3,
    )


def _closest_points(query: np.ndarray, target: pv.DataSet) -> tuple[np.ndarray, np.ndarray]:
    """For each point in ``query`` return its distance to ``target`` and the
    closest point found on ``target``.

    Uses a cell locator (point-to-surface) when ``target`` has cells, otherwise
    falls back to a point-to-point locator. Both are bundled with VTK.
    """
    if target.n_cells > 0:
        _, closest = target.find_closest_cell(query, return_closest_point=True)
        closest = np.asarray(closest).reshape(-1, 3)
    else:
        tgt_pts = np.asarray(target.points)
        idx = [target.find_closest_point(p) for p in query]
        closest = tgt_pts[idx]
    dist = np.linalg.norm(query - closest, axis=1)
    return dist, closest


def min_distance_between(a: pv.DataSet, b: pv.DataSet) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimum distance between two geometries.

    Returns ``(distance, point_on_a, point_on_b)``. Computed symmetrically
    (A→B and B→A) so it is accurate for e.g. a sparsely-sampled wire against a
    boundary surface.
    """
    pts_a = np.asarray(a.points)
    pts_b = np.asarray(b.points)
    if pts_a.size == 0 or pts_b.size == 0:
        raise ValueError("both geometries must have points to measure distance")

    # A's points -> closest point on B
    d_ab, closest_on_b = _closest_points(pts_a, b)
    i = int(np.argmin(d_ab))
    best = (float(d_ab[i]), pts_a[i].copy(), closest_on_b[i].copy())

    # B's points -> closest point on A (catches the case where A is the coarse one)
    d_ba, closest_on_a = _closest_points(pts_b, a)
    j = int(np.argmin(d_ba))
    if d_ba[j] < best[0]:
        best = (float(d_ba[j]), closest_on_a[j].copy(), pts_b[j].copy())

    return best


def reconstruct_boundary_surface(
    data: BoundaryData,
    n_phi: int = 60,
    n_s: int = 10,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reconstruct (X, Y, Z) surface patches from a parsed boundary."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)

    fourier = np.zeros((data.n_harm, n_phi))
    fourier[0, :] = 1.0
    for k in range(1, (data.n_harm - 1) // 2 + 1):
        idx_cos = 2 * k - 1
        idx_sin = 2 * k
        if idx_sin < data.n_harm:
            arg = k * data.n_period * phi
            fourier[idx_cos, :] = np.cos(arg)
            fourier[idx_sin, :] = np.sin(arg)

    s_arr = np.linspace(0.0, 1.0, n_s)
    h1, h2, h3, h4 = _hermite_basis(s_arr)

    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for el in data.elements:
        sz1_R, sz1_Z = el.sizes[0]
        sz2_R, sz2_Z = el.sizes[1]

        R_s = (
            np.outer(el.vals_R[0], h1)
            + np.outer(el.deriv_R[0] * sz1_R, h2)
            + np.outer(el.vals_R[1], h3)
            + np.outer(el.deriv_R[1] * sz2_R, h4)
        )
        Z_s = (
            np.outer(el.vals_Z[0], h1)
            + np.outer(el.deriv_Z[0] * sz1_Z, h2)
            + np.outer(el.vals_Z[1], h3)
            + np.outer(el.deriv_Z[1] * sz2_Z, h4)
        )

        R_surf = R_s.T @ fourier
        Z_surf = Z_s.T @ fourier
        X_surf = R_surf * np.cos(phi)
        Y_surf = R_surf * np.sin(phi)
        patches.append((X_surf, Y_surf, Z_surf))

    return patches


# --------------------------------------------------------------------------- #
# UnifiedPlotter
# --------------------------------------------------------------------------- #

_AXIS_NORMALS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-x": (-1.0, 0.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "-z": (0.0, 0.0, -1.0),
}


class UnifiedPlotter:
    """Compose multiple data sources into one PyVista window."""

    def __init__(
        self,
        background: str = "white",
        title: str | None = None,
        plotter: Any | None = None,
    ) -> None:
        # ``plotter`` may be an externally supplied embedded plotter
        # (a ``pyvistaqt.QtInteractor`` or a ``pv.Plotter``). We duck-type it so
        # the core stays importable without Qt installed.
        self._plotter = plotter if plotter is not None else pv.Plotter()
        self._plotter.set_background(background)
        self._title = title
        self._has_content = False
        self._labels: list[str] = []
        self._clip: dict[str, Any] | None = None

        #: maps a stable item id -> list of actor handles it produced
        self._actors: dict[str, list] = {}

        # -- photo-mode / screenshot state
        self._axes_on = False
        self._bounds_on = False
        self._on_capture: Callable[[PILImage.Image], None] | None = None
        self._capture_scale = 3
        self._capture_busy = False
        self._camera_widget: Any = None
        self._camera_rep: Any = None
        self._camera_img: Any = None
        self._help_actor: Any = None
        self._orientation_widget: Any = None

    # -- public API ---------------------------------------------------------- #

    @property
    def plotter(self) -> pv.Plotter:
        """Underlying PyVista plotter for advanced customization."""
        return self._plotter

    def set_clip_plane(
        self,
        normal: str | Sequence[float] = "x",
        origin: Sequence[float] | None = None,
        invert: bool = False,
    ) -> UnifiedPlotter:
        """Activate a clip plane applied to subsequent meshes."""
        if isinstance(normal, str):
            normal_vec = _AXIS_NORMALS.get(normal.lower(), (1.0, 0.0, 0.0))
        else:
            normal_vec = tuple(float(c) for c in normal)
        self._clip = {"normal": normal_vec, "origin": origin, "invert": invert}
        return self

    def add_point_cloud(
        self,
        source: pd.DataFrame | np.ndarray | str | Path,
        *,
        color: str = "red",
        point_size: int = 5,
        opacity: float = 1.0,
        sample_frac: float = 1.0,
        label: str | None = None,
        render_as_spheres: bool = True,
        item_id: str | None = None,
    ) -> UnifiedPlotter:
        points = self._coerce_points(source, sample_frac)
        if points.size == 0:
            log.warning("Point cloud '%s' has no valid points", label or "<unnamed>")
            return self

        log.info(
            "Point cloud%s: %d points, bbox=%s..%s",
            f" ({label})" if label else "",
            len(points),
            np.round(points.min(axis=0), 3).tolist(),
            np.round(points.max(axis=0), 3).tolist(),
        )

        actor = self._plotter.add_points(
            pv.PolyData(points),
            color=color,
            point_size=point_size,
            opacity=opacity,
            render_points_as_spheres=render_as_spheres,
            label=label,
        )
        return self._mark(label, item_id, actor)

    def add_boundary(
        self,
        path: str | Path,
        *,
        color: str = "cyan",
        opacity: float = 0.5,
        show_edges: bool = True,
        edge_color: str = "black",
        line_width: float = 0.5,
        n_phi: int = 60,
        n_s: int = 10,
        label: str | None = None,
        item_id: str | None = None,
    ) -> UnifiedPlotter:
        data = oio.read_boundary(path)
        if data is None:
            log.error("Boundary file '%s' not found or invalid", path)
            return self

        patches = reconstruct_boundary_surface(data, n_phi=n_phi, n_s=n_s)
        multi = pv.MultiBlock([pv.StructuredGrid(x, y, z) for (x, y, z) in patches])
        mesh = self._maybe_clip(multi.combine() if self._clip else multi)

        actor = self._plotter.add_mesh(
            mesh,
            color=color,
            show_edges=show_edges,
            edge_color=edge_color,
            line_width=line_width,
            opacity=opacity,
            smooth_shading=True,
            specular=0.5,
            label=label,
        )

        if self._title is None:
            self._title = f"JOREK boundary  (harmonics {data.n_harm} · period {data.n_period})"
        return self._mark(label, item_id, actor)

    def add_hex_mesh(
        self,
        df_nodes: pd.DataFrame,
        elements_hex: Sequence[Sequence[int]],
        *,
        color: str = "violet",
        opacity: float = 1.0,
        show_edges: bool = True,
        label: str | None = None,
        item_id: str | None = None,
    ) -> UnifiedPlotter:
        df = df_nodes.sort_values("node_id")
        points = df[["x", "y", "z"]].to_numpy()
        id_to_idx = {nid: i for i, nid in enumerate(df["node_id"])}

        cells: list[list[int]] = []
        cell_types: list[int] = []
        for hex_nodes in elements_hex:
            try:
                idx = [id_to_idx[n] for n in hex_nodes]
            except KeyError:
                continue
            cells.append([8, *idx])
            cell_types.append(pv.CellType.HEXAHEDRON)

        if not cells:
            log.warning("No valid hex elements")
            return self

        grid = pv.UnstructuredGrid(np.hstack(cells), cell_types, points)
        log.info("Hex mesh: %d elements, %d nodes", len(elements_hex), len(df_nodes))

        actor = self._plotter.add_mesh(
            self._maybe_clip(grid),
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            label=label,
        )
        return self._mark(label, item_id, actor)

    def add_unstructured(
        self,
        grid: pv.UnstructuredGrid,
        *,
        color: str | None = None,
        opacity: float = 1.0,
        show_edges: bool = True,
        edge_color: str = "black",
        scalars: str | None = None,
        colormap: str | None = None,
        show_scalar_bar: bool = False,
        label: str | None = None,
        item_id: str | None = None,
    ) -> UnifiedPlotter:
        """Add a pre-built :class:`pyvista.UnstructuredGrid` to the scene.

        Generic entry point reused by the CARIDDI and JOREK HDF5 items.
        """
        kwargs: dict[str, Any] = {
            "opacity": opacity,
            "show_edges": show_edges,
            "edge_color": edge_color,
            "label": label,
        }
        if scalars is not None:
            kwargs["scalars"] = scalars
            kwargs["cmap"] = colormap or "viridis"
            kwargs["show_scalar_bar"] = show_scalar_bar
            if show_scalar_bar and label:
                # Title the bar by the item label so the GUI can show/hide it
                # later (PyVista keys scalar bars by title).
                kwargs["scalar_bar_args"] = {"title": label}
        else:
            kwargs["color"] = color or "lightgray"

        actor = self._plotter.add_mesh(self._maybe_clip(grid), **kwargs)
        log.info("Unstructured grid: %d points, %d cells", grid.n_points, grid.n_cells)
        return self._mark(label, item_id, actor)

    def add_vtk_mesh(
        self,
        path: str | Path,
        *,
        color: str = "lightblue",
        opacity: float = 1.0,
        show_edges: bool = True,
        label: str | None = None,
        item_id: str | None = None,
    ) -> UnifiedPlotter:
        mesh = pv.read(str(path))
        log.info("VTK mesh: %d points, %d cells", mesh.n_points, mesh.n_cells)
        actor = self._plotter.add_mesh(
            self._maybe_clip(mesh),
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            label=label,
        )
        return self._mark(label, item_id, actor)

    def add_wire(
        self,
        *,
        r0: float,
        z0: float,
        alfa_wire_deg: float,
        color: str = "orange",
        tube_radius: float | None = None,
        n_phi: int = 200,
        label: str | None = None,
        item_id: str | None = None,
    ) -> UnifiedPlotter:
        """Add a tilted circular current filament loop."""
        alfa = np.deg2rad(alfa_wire_deg)
        t = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        x_local = r0 * np.cos(t)
        y_local = r0 * np.sin(t)

        # Rotate the coil plane about the Y axis by alfa.
        x_rot = x_local * np.cos(alfa)
        y_rot = y_local
        z_rot = -x_local * np.sin(alfa) + z0

        # Close the loop for a clean spline tube.
        points = np.column_stack(
            [
                np.append(x_rot, x_rot[0]),
                np.append(y_rot, y_rot[0]),
                np.append(z_rot, z_rot[0]),
            ]
        )

        radius = tube_radius if (tube_radius and tube_radius > 0) else r0 * 0.02
        tube = pv.Spline(points, n_phi * 4).tube(radius=radius)
        log.info("Wire: r0=%.4f z0=%.4f alfa=%.3f° tube_r=%.4f", r0, z0, alfa_wire_deg, radius)

        actor = self._plotter.add_mesh(tube, color=color, label=label)
        return self._mark(label, item_id, actor)

    def add_vector_field(
        self,
        source: pd.DataFrame | str | Path,
        *,
        scale: float = 1.0,
        color: str = "crimson",
        colormap: str | None = None,
        color_by_magnitude: bool = False,
        sample_frac: float = 1.0,
        label: str | None = None,
        item_id: str | None = None,
    ) -> UnifiedPlotter:
        df = oio.read_vector_field(source) if isinstance(source, (str, Path)) else source.copy()

        required = {"x", "y", "z", "Bx", "By", "Bz"}
        if not required.issubset(df.columns):
            log.error("Vector field requires columns %s, got %s", required, list(df.columns))
            return self

        if 0.0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            log.warning("Vector field '%s' is empty after filtering", label or "<unnamed>")
            return self

        points = df[["x", "y", "z"]].to_numpy()
        vectors = df[["Bx", "By", "Bz"]].to_numpy() * scale
        magnitudes = np.linalg.norm(vectors, axis=1)
        log.info(
            "Vector field%s: %d arrows, |B| in [%.4g, %.4g]",
            f" ({label})" if label else "",
            len(points),
            magnitudes.min(),
            magnitudes.max(),
        )

        cloud = pv.PolyData(points)
        cloud["vectors"] = vectors
        cloud["magnitude"] = magnitudes
        glyphs = cloud.glyph(orient="vectors", scale="magnitude", factor=1.0)

        if color_by_magnitude:
            bar_args = {"title": label} if label else None
            actor = self._plotter.add_mesh(
                glyphs,
                scalars="magnitude",
                cmap=colormap or "plasma",
                label=label,
                scalar_bar_args=bar_args,
            )
        else:
            actor = self._plotter.add_mesh(glyphs, color=color, label=label)
        return self._mark(label, item_id, actor)

    def show(
        self,
        *,
        show_axes: bool = True,
        show_bounds: bool = True,
        show_legend: bool = True,
        photo_mode: bool = False,
        on_capture: Callable[[PILImage.Image], None] | None = None,
        capture_scale: int = 3,
    ) -> None:
        if not self._has_content:
            log.warning("Nothing to render")
            return

        # "Photo mode" starts with a clean canvas: no axes, grid, legend or text.
        if photo_mode:
            show_axes = show_bounds = show_legend = False

        self._axes_on = show_axes
        self._bounds_on = show_bounds

        if show_axes:
            self._plotter.add_axes()
        if show_bounds:
            self._plotter.show_bounds(grid="front", location="outer", all_edges=True)
        if self._title and not photo_mode:
            self._plotter.add_text(self._title, position="upper_left", font_size=10)
        if show_legend and self._labels:
            self._plotter.add_legend()
        if self._clip and not photo_mode:
            self._plotter.add_text(
                f"clip: normal={self._clip['normal']}",
                position="lower_left",
                font_size=8,
            )

        self._install_photo_tools(on_capture, capture_scale)
        self._install_view_tools()
        self._plotter.show()

    # -- photo mode / screenshots ------------------------------------------- #

    def capture_highres(self, scale: int | None = None) -> PILImage.Image:
        """Render the current view to a high-resolution PIL image.

        The on-screen camera button and key hints are hidden for the shot so
        they never end up in the exported figure.
        """
        from PIL import Image

        scale = scale or self._capture_scale

        hidden_widget = False
        if self._camera_widget is not None:
            try:
                self._camera_widget.Off()
                if self._camera_rep is not None:
                    self._camera_rep.VisibilityOff()
                hidden_widget = True
            except Exception:  # noqa: BLE001
                pass

        help_vis = None
        if self._help_actor is not None:
            try:
                help_vis = self._help_actor.GetVisibility()
                self._help_actor.SetVisibility(False)
            except Exception:  # noqa: BLE001
                help_vis = None

        orient_on = None
        if self._orientation_widget is not None:
            try:
                orient_on = bool(self._orientation_widget.GetEnabled())
                self._orientation_widget.EnabledOff()
            except Exception:  # noqa: BLE001
                orient_on = None

        self._plotter.render()
        try:
            arr = self._plotter.screenshot(return_img=True, scale=scale)
        finally:
            if hidden_widget:
                try:
                    if self._camera_rep is not None:
                        self._camera_rep.VisibilityOn()
                    self._camera_widget.On()
                except Exception:  # noqa: BLE001
                    pass
            if self._help_actor is not None and help_vis is not None:
                try:
                    self._help_actor.SetVisibility(help_vis)
                except Exception:  # noqa: BLE001
                    pass
            if self._orientation_widget is not None and orient_on:
                try:
                    self._orientation_widget.EnabledOn()
                except Exception:  # noqa: BLE001
                    pass
            self._plotter.render()

        return Image.fromarray(np.asarray(arr))

    def _do_capture(self, *_args: Any) -> None:
        """Camera-button / hot-key handler: grab a shot and hand it off."""
        if self._capture_busy:
            return
        self._capture_busy = True
        try:
            img = self.capture_highres()
        except Exception:  # noqa: BLE001
            log.exception("Screenshot capture failed")
            return
        finally:
            self._capture_busy = False

        if self._on_capture is not None:
            try:
                self._on_capture(img)
            except Exception:  # noqa: BLE001
                log.exception("Screenshot handler failed")
        else:
            from datetime import datetime

            name = f"omniviz_{datetime.now():%Y%m%d_%H%M%S}.png"
            img.save(name)
            log.info("Saved screenshot to %s", name)

    def _toggle_axes(self) -> None:
        self._axes_on = not self._axes_on
        try:
            self._plotter.show_axes() if self._axes_on else self._plotter.hide_axes()
            self._plotter.render()
        except Exception:  # noqa: BLE001
            log.debug("Axes toggle failed", exc_info=True)

    def _toggle_bounds(self) -> None:
        self._bounds_on = not self._bounds_on
        try:
            if self._bounds_on:
                self._plotter.show_bounds(grid="front", location="outer", all_edges=True)
            else:
                self._plotter.remove_bounds_axes()
            self._plotter.render()
        except Exception:  # noqa: BLE001
            log.debug("Bounds toggle failed", exc_info=True)

    def _set_view(self, name: str) -> None:
        """Snap the camera to a canonical orientation."""
        try:
            if name == "x":  # look down +X (shows the Y-Z plane)
                self._plotter.view_yz()
            elif name == "y":  # look down +Y (shows the X-Z plane)
                self._plotter.view_xz()
            elif name == "z":  # look down +Z (shows the X-Y plane)
                self._plotter.view_xy()
            elif name == "iso":
                self._plotter.view_isometric()
            elif name == "flip":  # spin 180° to view from the far side
                self._plotter.camera.Azimuth(180)
            # Re-assert mouse-rotation interaction: snapping to a view (or a
            # stray key/gizmo event) can leave the VTK interactor in a style
            # that no longer rotates. This is a no-op when already healthy.
            try:
                self._plotter.enable_trackball_style()
            except Exception:  # noqa: BLE001
                log.debug("enable_trackball_style failed", exc_info=True)
            self._plotter.reset_camera_clipping_range()
            self._plotter.render()
        except Exception:  # noqa: BLE001
            log.debug("Set view '%s' failed", name, exc_info=True)

    def _install_view_tools(self) -> None:
        """Bind view hot-keys and add the clickable axis orientation gizmo."""
        for key, view in (("x", "x"), ("y", "y"), ("z", "z"), ("i", "iso"), ("f", "flip")):
            try:
                self._plotter.add_key_event(key, lambda v=view: self._set_view(v))
            except Exception:  # noqa: BLE001
                log.debug("Could not bind view key '%s'", key, exc_info=True)

        # Interactive axis gizmo: click an arrow to snap to that axis / flip.
        try:
            self._orientation_widget = self._plotter.add_camera_orientation_widget()
        except Exception:  # noqa: BLE001
            self._orientation_widget = None
            log.debug("Camera orientation widget unavailable", exc_info=True)

    def _install_photo_tools(
        self,
        on_capture: Callable[[PILImage.Image], None] | None,
        capture_scale: int,
    ) -> None:
        self._on_capture = on_capture
        self._capture_scale = max(1, int(capture_scale))

        # Hot-keys: live cleanup + capture without leaving the 3-D view.
        try:
            self._plotter.add_key_event("c", self._do_capture)
            self._plotter.add_key_event("a", self._toggle_axes)
            self._plotter.add_key_event("g", self._toggle_bounds)
        except Exception:  # noqa: BLE001
            log.debug("Could not bind photo hot-keys", exc_info=True)

        # Faint key hint, hidden automatically while a shot is taken.
        try:
            self._help_actor = self._plotter.add_text(
                "[c] photo  [a] axes  [g] grid   view: [x][y][z] axis  [i] iso  [f] flip",
                position="lower_right",
                font_size=7,
                color="gray",
            )
        except Exception:  # noqa: BLE001
            self._help_actor = None

        self._make_camera_button()

    def _make_camera_button(self) -> None:
        """Add a clickable camera icon to the top-left of the render window."""
        try:
            import vtk

            iren = getattr(self._plotter.iren, "interactor", None)
            if iren is None:
                return

            self._camera_img = self._np_to_vtk_image(self._camera_icon(48))

            rep = vtk.vtkTexturedButtonRepresentation2D()
            rep.SetNumberOfStates(1)
            rep.SetButtonTexture(0, self._camera_img)
            rep.SetPlaceFactor(1)

            size, margin = 46, 14

            def _place() -> None:
                w, h = self._plotter.window_size
                rep.PlaceWidget([margin, margin + size, h - margin - size, h - margin, 0.0, 0.0])

            _place()

            widget = vtk.vtkButtonWidget()
            widget.SetInteractor(iren)
            widget.SetRepresentation(rep)
            widget.On()
            widget.AddObserver("StateChangedEvent", self._do_capture)

            # Keep the icon anchored to the top-left when the window resizes.
            iren.AddObserver("ConfigureEvent", lambda *_: _place())

            self._camera_widget = widget
            self._camera_rep = rep
        except Exception:  # noqa: BLE001
            log.debug("Camera button unavailable; use the [c] hot-key", exc_info=True)
            self._camera_widget = None

    @staticmethod
    def _camera_icon(size: int = 48) -> np.ndarray:
        """Draw a simple camera glyph as an RGBA array for the toolbar button."""
        from PIL import Image, ImageDraw

        s = size
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        accent = (31, 106, 165, 255)
        white = (255, 255, 255, 255)

        d.rounded_rectangle([1, 1, s - 2, s - 2], radius=s * 0.18, fill=accent)
        d.rounded_rectangle([s * 0.36, s * 0.24, s * 0.58, s * 0.36], radius=s * 0.03, fill=white)
        d.rounded_rectangle([s * 0.14, s * 0.34, s * 0.86, s * 0.76], radius=s * 0.08, fill=white)
        cx, cy, r = s * 0.5, s * 0.55, s * 0.135
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)
        d.ellipse([cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5], fill=white)
        return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def _np_to_vtk_image(arr: np.ndarray) -> Any:
        import vtk
        from vtkmodules.util.numpy_support import numpy_to_vtk

        h, w, c = arr.shape
        # VTK image origin is bottom-left, so flip vertically.
        flipped = np.ascontiguousarray(arr[::-1])
        image = vtk.vtkImageData()
        image.SetDimensions(w, h, 1)
        vtk_arr = numpy_to_vtk(flipped.reshape(-1, c), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_arr.SetNumberOfComponents(c)
        image.GetPointData().SetScalars(vtk_arr)
        return image

    # -- internal helpers ---------------------------------------------------- #

    def _mark(
        self,
        label: str | None,
        item_id: str | None = None,
        actor: Any | None = None,
    ) -> UnifiedPlotter:
        self._has_content = True
        if label:
            self._labels.append(label)
        if item_id is not None and actor is not None:
            self._actors.setdefault(item_id, []).append(actor)
        return self

    # -- live scene mutation ------------------------------------------------- #

    def remove_item(self, item_id: str) -> None:
        """Remove every actor associated with ``item_id`` and re-render."""
        handles = self._actors.pop(item_id, [])
        for handle in handles:
            try:
                self._plotter.remove_actor(handle)
            except Exception:  # noqa: BLE001
                log.debug("remove_actor failed for item '%s'", item_id, exc_info=True)
        self.render()

    def update_item(
        self,
        item_id: str,
        *,
        color: Any | None = None,
        opacity: float | None = None,
        visibility: bool | None = None,
    ) -> None:
        """Mutate stored actor properties for ``item_id`` in place."""
        handles = self._actors.get(item_id, [])
        for handle in handles:
            try:
                prop = getattr(handle, "prop", None) or handle.GetProperty()
                if color is not None:
                    prop.color = color
                if opacity is not None:
                    prop.opacity = opacity
                if visibility is not None:
                    handle.SetVisibility(bool(visibility))
            except Exception:  # noqa: BLE001
                log.debug("update_item failed for item '%s'", item_id, exc_info=True)
        self.render()

    def clear_items(self) -> None:
        """Remove every tracked actor from the scene."""
        for item_id in list(self._actors.keys()):
            self.remove_item(item_id)

    def render(self) -> None:
        """Re-render the scene if the underlying plotter supports it."""
        render = getattr(self._plotter, "render", None)
        if callable(render):
            try:
                render()
            except Exception:  # noqa: BLE001
                log.debug("render() failed", exc_info=True)

    def _maybe_clip(self, mesh: pv.DataSet) -> pv.DataSet:
        if not self._clip:
            return mesh
        origin = self._clip["origin"] or mesh.center
        try:
            return mesh.clip(
                normal=self._clip["normal"],
                origin=origin,
                invert=self._clip["invert"],
            )
        except Exception as exc:  # pragma: no cover
            log.warning("Clip failed (%s); rendering unclipped mesh", exc)
            return mesh

    @staticmethod
    def _coerce_points(
        source: pd.DataFrame | np.ndarray | str | Path,
        sample_frac: float,
    ) -> np.ndarray:
        if isinstance(source, (str, Path)):
            df = oio.read_xyz_data(source)
            points = df[["x", "y", "z"]].to_numpy()
        elif isinstance(source, pd.DataFrame):
            df = source.sample(frac=sample_frac, random_state=42) if sample_frac < 1.0 else source
            points = df[["x", "y", "z"]].to_numpy()
        else:
            points = np.asarray(source)
            if sample_frac < 1.0:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(points), int(len(points) * sample_frac), replace=False)
                points = points[idx]
        valid = np.all(np.isfinite(points), axis=1)
        return points[valid]
