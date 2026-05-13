"""
Unified PyVista Plotter Module

This module provides a flexible way to combine multiple data sources into a single plot.
Usage:
    plotter = UnifiedPlotter()
    plotter.add_point_cloud(df, color="red", label="Gauss Points")
    plotter.add_boundary("data/boundary.txt")
    plotter.add_hex_mesh(df_nodes, elements_hex)
    plotter.add_vtk_mesh("mesh.vtk")
    plotter.show()
"""

import pyvista as pv
import numpy as np
from typing import List, Union
import pandas as pd


class BoundaryReader:
    """Handles reading and reconstructing JOREK boundary files."""

    @staticmethod
    def basis_functions(s):
        """Cubic Hermite basis functions for s in [0, 1]."""
        s2 = s * s
        s3 = s2 * s
        h1 = 1.0 - 3.0 * s2 + 2.0 * s3
        h2 = s - 2.0 * s2 + s3
        h3 = 3.0 * s2 - 2.0 * s3
        h4 = -s2 + s3
        return h1, h2, h3, h4

    @staticmethod
    def read_boundary(filename: str):
        """Read boundary file and return elements, n_harm, n_period."""
        try:
            with open(filename, 'r') as f:
                content = f.read().split()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None

        iterator = iter(content)

        try:
            n_elem = int(next(iterator))
            n_nodes = int(next(iterator))
            n_harm = int(next(iterator))
            n_period = int(next(iterator))
            version = int(next(iterator))
        except StopIteration:
            print("Error: Empty file or invalid header.")
            return None

        print(f"Reading file: {n_elem} elements, {n_harm} harmonics, periodicity {n_period}")

        elements = []

        for _ in range(n_elem):
            try:
                idx = int(next(iterator))
                n1 = int(next(iterator))
                n2 = int(next(iterator))

                vals_n1_R = [float(next(iterator)) for _ in range(n_harm)]
                vals_n1_Z = [float(next(iterator)) for _ in range(n_harm)]
                deriv_n1_R = [float(next(iterator)) for _ in range(n_harm)]
                deriv_n1_Z = [float(next(iterator)) for _ in range(n_harm)]
                size_n1 = [float(next(iterator)), float(next(iterator))]

                vals_n2_R = [float(next(iterator)) for _ in range(n_harm)]
                vals_n2_Z = [float(next(iterator)) for _ in range(n_harm)]
                deriv_n2_R = [float(next(iterator)) for _ in range(n_harm)]
                deriv_n2_Z = [float(next(iterator)) for _ in range(n_harm)]
                size_n2 = [float(next(iterator)), float(next(iterator))]

                elements.append({
                    'vals_R': [np.array(vals_n1_R), np.array(vals_n2_R)],
                    'vals_Z': [np.array(vals_n1_Z), np.array(vals_n2_Z)],
                    'deriv_R': [np.array(deriv_n1_R), np.array(deriv_n2_R)],
                    'deriv_Z': [np.array(deriv_n1_Z), np.array(deriv_n2_Z)],
                    'sizes': [size_n1, size_n2]
                })

            except StopIteration:
                break

        return elements, n_harm, n_period

    @staticmethod
    def reconstruct_surface(elements, n_harm, n_period, n_phi=60, n_s=10):
        """Reconstruct the surface from boundary data."""
        all_X, all_Y, all_Z = [], [], []
        phi_arr = np.linspace(0, 2 * np.pi, n_phi)

        M = np.zeros((n_harm, n_phi))
        M[0, :] = 1.0

        for k in range(1, (n_harm - 1) // 2 + 1):
            idx_cos = 2 * k - 1
            idx_sin = 2 * k
            if idx_sin < n_harm:
                arg = k * n_period * phi_arr
                M[idx_cos, :] = np.cos(arg)
                M[idx_sin, :] = np.sin(arg)

        s_arr = np.linspace(0, 1, n_s)
        h1, h2, h3, h4 = BoundaryReader.basis_functions(s_arr)

        print("Reconstructing surface...")

        for el in elements:
            sz1_R, sz1_Z = el['sizes'][0]
            sz2_R, sz2_Z = el['sizes'][1]

            R_coeffs_s = (np.outer(el['vals_R'][0], h1) +
                          np.outer(el['deriv_R'][0] * sz1_R, h2) +
                          np.outer(el['vals_R'][1], h3) +
                          np.outer(el['deriv_R'][1] * sz2_R, h4))

            Z_coeffs_s = (np.outer(el['vals_Z'][0], h1) +
                          np.outer(el['deriv_Z'][0] * sz1_Z, h2) +
                          np.outer(el['vals_Z'][1], h3) +
                          np.outer(el['deriv_Z'][1] * sz2_Z, h4))

            R_surf = np.matmul(R_coeffs_s.T, M)
            Z_surf = np.matmul(Z_coeffs_s.T, M)

            X_surf = R_surf * np.cos(phi_arr)
            Y_surf = R_surf * np.sin(phi_arr)

            all_X.append(X_surf)
            all_Y.append(Y_surf)
            all_Z.append(Z_surf)

        return all_X, all_Y, all_Z


class UnifiedPlotter:
    """
    A unified plotter that combines multiple data sources into one PyVista plot.

    Example usage:
        plotter = UnifiedPlotter()
        plotter.add_point_cloud(df_points, color="red", point_size=5, label="Data 1")
        plotter.add_point_cloud(df_points2, color="blue", point_size=5, label="Data 2")
        plotter.add_boundary("data/boundary.txt", n_phi=60, n_s=10)
        plotter.add_hex_mesh(df_nodes, elements_hex)
        plotter.add_vtk_mesh("mesh.vtk")
        plotter.show()
    """

    def __init__(self, background: str = "white", title: str = None):
        """
        Initialize the unified plotter.

        Args:
            background: Background color for the plot
            title: Optional title for the plot
        """
        self._pv_plotter = pv.Plotter()
        self._pv_plotter.set_background(background)
        self._title = title
        self._has_content = False
        self._labels = []
        self._clip_plane = None  # Store clip plane settings
        self._meshes = []  # Store meshes for clipping

    def add_point_cloud(
        self,
        data: Union[pd.DataFrame, np.ndarray, str],
        color: str = "red",
        point_size: int = 5,
        opacity: float = 1.0,
        sample_frac: float = 1.0,
        label: str = None,
        render_as_spheres: bool = True
    ) -> 'UnifiedPlotter':
        """
        Add a point cloud to the plot.

        Args:
            data: DataFrame with x,y,z columns, numpy array (Nx3), or filename
            color: Point color
            point_size: Size of points
            opacity: Point opacity (0-1)
            sample_frac: Fraction of points to sample (0-1)
            label: Label for legend
            render_as_spheres: Whether to render points as spheres

        Returns:
            self for method chaining
        """
        # Handle different input types
        if isinstance(data, str):
            from . import parser
            df = parser.read_xyz_data(data)
            points = df[['x', 'y', 'z']].to_numpy()
        elif isinstance(data, pd.DataFrame):
            df = data
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
            points = df[['x', 'y', 'z']].to_numpy()
        else:
            points = np.asarray(data)
            if sample_frac < 1.0:
                n = int(len(points) * sample_frac)
                indices = np.random.RandomState(42).choice(len(points), n, replace=False)
                points = points[indices]

        # Filter out invalid points
        valid_mask = np.all(np.isfinite(points), axis=1)
        points = points[valid_mask]

        if len(points) == 0:
            print(f"Warning: No valid points in data{' (' + label + ')' if label else ''}")
            return self

        # Print diagnostics
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        print(f"Point cloud{' (' + label + ')' if label else ''}: {len(points)} points")
        print(f"  Bounding box: [{mins[0]:.3f}, {mins[1]:.3f}, {mins[2]:.3f}] to [{maxs[0]:.3f}, {maxs[1]:.3f}, {maxs[2]:.3f}]")

        cloud = pv.PolyData(points)
        self._pv_plotter.add_points(
            cloud,
            color=color,
            point_size=point_size,
            opacity=opacity,
            render_points_as_spheres=render_as_spheres,
            label=label
        )

        if label:
            self._labels.append(label)

        self._has_content = True
        return self

    def add_boundary(
        self,
        filename: str,
        color: str = "cyan",
        opacity: float = 0.5,
        show_edges: bool = True,
        edge_color: str = "black",
        line_width: float = 0.5,
        n_phi: int = 60,
        n_s: int = 10,
        label: str = None
    ) -> 'UnifiedPlotter':
        """
        Add a JOREK boundary surface to the plot.

        Args:
            filename: Path to boundary.txt file
            color: Surface color
            opacity: Surface opacity
            show_edges: Whether to show mesh edges
            edge_color: Color of mesh edges
            line_width: Width of edge lines
            n_phi: Resolution in toroidal direction
            n_s: Resolution in poloidal direction
            label: Label for legend

        Returns:
            self for method chaining
        """
        data = BoundaryReader.read_boundary(filename)
        if data is None:
            return self

        elements, n_harm, n_period = data
        X_list, Y_list, Z_list = BoundaryReader.reconstruct_surface(
            elements, n_harm, n_period, n_phi=n_phi, n_s=n_s
        )

        print("Creating PyVista meshes...")
        multi_block = pv.MultiBlock()

        for x, y, z in zip(X_list, Y_list, Z_list):
            grid = pv.StructuredGrid(x, y, z)
            multi_block.append(grid)

        # Apply clipping if set
        mesh_to_add = multi_block
        if self._clip_plane:
            try:
                # Merge multi_block for clipping
                merged = multi_block.combine()
                origin = self._clip_plane['origin']
                if origin is None:
                    origin = merged.center
                mesh_to_add = merged.clip(
                    normal=self._clip_plane['normal'],
                    origin=origin,
                    invert=self._clip_plane['invert']
                )
            except Exception as e:
                print(f"Warning: Could not clip boundary mesh: {e}")
                mesh_to_add = multi_block

        self._pv_plotter.add_mesh(
            mesh_to_add,
            color=color,
            show_edges=show_edges,
            edge_color=edge_color,
            line_width=line_width,
            opacity=opacity,
            smooth_shading=True,
            specular=0.5,
            label=label
        )

        if label:
            self._labels.append(label)

        # Add info text
        if self._title is None:
            self._title = f"JOREK Boundary (Harmonics: {n_harm}, Period: {n_period})"

        self._has_content = True
        return self

    def add_hex_mesh(
        self,
        df_nodes: pd.DataFrame,
        elements_hex: List[List[int]],
        color: str = "violet",
        opacity: float = 1.0,
        show_edges: bool = True,
        label: str = None
    ) -> 'UnifiedPlotter':
        """
        Add a hexahedral mesh to the plot.

        Args:
            df_nodes: DataFrame with columns [node_id, x, y, z]
            elements_hex: List of element connectivity (8 node IDs per element)
            color: Mesh color
            opacity: Mesh opacity
            show_edges: Whether to show mesh edges
            label: Label for legend

        Returns:
            self for method chaining
        """
        df_sorted = df_nodes.sort_values('node_id')
        points = df_sorted[['x', 'y', 'z']].values
        id_to_index = {node_id: i for i, node_id in enumerate(df_sorted['node_id'])}

        cells = []
        cell_types = []

        for hex_nodes in elements_hex:
            try:
                indices = [id_to_index[nid] for nid in hex_nodes]
                cells.append([8] + indices)
                cell_types.append(pv.CellType.HEXAHEDRON)
            except KeyError:
                continue

        if not cells:
            print("Warning: No valid hexahedral elements found")
            return self

        cells_flat = np.hstack(cells)
        grid = pv.UnstructuredGrid(cells_flat, cell_types, points)

        print(f"Hex mesh: {len(elements_hex)} elements, {len(df_nodes)} nodes")

        # Apply clipping if set
        mesh_to_add = grid
        if self._clip_plane:
            try:
                origin = self._clip_plane['origin']
                if origin is None:
                    origin = grid.center
                mesh_to_add = grid.clip(
                    normal=self._clip_plane['normal'],
                    origin=origin,
                    invert=self._clip_plane['invert']
                )
            except Exception as e:
                print(f"Warning: Could not clip hex mesh: {e}")
                mesh_to_add = grid

        self._pv_plotter.add_mesh(
            mesh_to_add,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            label=label
        )

        if label:
            self._labels.append(label)

        self._has_content = True
        return self

    def add_vtk_mesh(
        self,
        filename: str,
        color: str = "lightblue",
        opacity: float = 1.0,
        show_edges: bool = True,
        label: str = None
    ) -> 'UnifiedPlotter':
        """
        Add a VTK mesh file to the plot.

        Args:
            filename: Path to VTK file
            color: Mesh color
            opacity: Mesh opacity
            show_edges: Whether to show mesh edges
            label: Label for legend

        Returns:
            self for method chaining
        """
        try:
            mesh = pv.read(filename)
            print(f"VTK mesh: {mesh.n_points} points, {mesh.n_cells} cells")

            # Apply clipping if set
            mesh_to_add = mesh
            if self._clip_plane:
                try:
                    origin = self._clip_plane['origin']
                    if origin is None:
                        origin = mesh.center
                    mesh_to_add = mesh.clip(
                        normal=self._clip_plane['normal'],
                        origin=origin,
                        invert=self._clip_plane['invert']
                    )
                except Exception as e:
                    print(f"Warning: Could not clip VTK mesh: {e}")
                    mesh_to_add = mesh

            self._pv_plotter.add_mesh(
                mesh_to_add,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                label=label
            )

            if label:
                self._labels.append(label)

            self._has_content = True
        except Exception as e:
            print(f"Error loading VTK file '{filename}': {e}")

        return self

    def add_wire(
        self,
        r0: float,
        z0: float,
        alfa_wire_deg: float,
        color: str = "orange",
        tube_radius: float = None,
        n_phi: int = 200,
        label: str = None
    ) -> 'UnifiedPlotter':
        """
        Add a wire (current filament loop) to the plot.

        The wire is a circular current loop with major radius r0 centred on the
        Z-axis at height z0.  The coil plane is tilted by alfa_wire_deg degrees
        away from the horizontal (Z = const) plane, rotating about the Y-axis.

        Args:
            r0: Major radius of the wire loop [m]
            z0: Axial (vertical) position of the wire centre [m]
            alfa_wire_deg: Tilt of the coil plane from horizontal [deg].
                           0 = horizontal ring; positive tilts toward +X.
            color: Wire colour
            tube_radius: Radius of the tube used to represent the wire.
                         Defaults to 2 % of r0.
            n_phi: Number of sample points along the loop
            label: Label for legend

        Returns:
            self for method chaining
        """
        alfa_rad = np.deg2rad(alfa_wire_deg)

        # Parameterise the circular wire loop in its own (untilted) frame.
        t = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        x_local = r0 * np.cos(t)
        y_local = r0 * np.sin(t)
        z_local = np.zeros_like(t)

        # Tilt the coil plane by rotating about the Y-axis by alfa_rad.
        # The coil normal goes from (0,0,1) toward (sin(alfa), 0, cos(alfa)).
        x_rot = x_local * np.cos(alfa_rad) + z_local * np.sin(alfa_rad)
        y_rot = y_local
        z_rot = -x_local * np.sin(alfa_rad) + z_local * np.cos(alfa_rad) + z0

        # Close the loop for the spline
        x_rot = np.append(x_rot, x_rot[0])
        y_rot = np.append(y_rot, y_rot[0])
        z_rot = np.append(z_rot, z_rot[0])
        points = np.column_stack([x_rot, y_rot, z_rot])

        if tube_radius is None:
            tube_radius = r0 * 0.02

        spline = pv.Spline(points, n_phi * 4)
        tube = spline.tube(radius=tube_radius)

        print(f"Wire: r0={r0:.6f} m, z0={z0:.6f} m, alfa={alfa_wire_deg:.4f}°, tube_r={tube_radius:.4f} m")

        self._pv_plotter.add_mesh(tube, color=color, label=label)

        if label:
            self._labels.append(label)

        self._has_content = True
        return self

    def add_vector_field(
        self,
        data: Union[pd.DataFrame, str],
        scale: float = 1.0,
        color: str = "crimson",
        colormap: str = None,
        color_by_magnitude: bool = False,
        sample_frac: float = 1.0,
        label: str = None,
    ) -> 'UnifiedPlotter':
        """
        Add a vector field (arrows) to the plot.

        The input must contain columns x, y, z (positions) and Bx, By, Bz
        (vector components).  Each row becomes one arrow whose tail is placed
        at (x, y, z) and whose direction and length are given by (Bx, By, Bz).

        Args:
            data: DataFrame with columns [x, y, z, Bx, By, Bz], or a filename
                  to be parsed with parser.read_vector_field().
            scale: Uniform scale factor applied to every arrow length.
                   Use values < 1 to shorten arrows, > 1 to lengthen them.
            color: Arrow color (used when color_by_magnitude is False).
            colormap: Matplotlib colormap name used when color_by_magnitude is True.
            color_by_magnitude: If True, colour arrows by |B| using *colormap*.
            sample_frac: Fraction of rows to keep (0, 1].  Useful for dense data.
            label: Legend label.

        Returns:
            self for method chaining
        """
        if isinstance(data, str):
            from . import parser
            df = parser.read_vector_field(data)
        else:
            df = data.copy()

        required = {"x", "y", "z", "Bx", "By", "Bz"}
        if not required.issubset(df.columns):
            print(f"Warning: Vector-field data must contain columns {required}. Found: {list(df.columns)}")
            return self

        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        # Drop rows that contain non-finite values in any column
        df = df.replace([float('inf'), float('-inf')], float('nan')).dropna()

        if df.empty:
            print(f"Warning: No valid rows in vector field data{' (' + label + ')' if label else ''}")
            return self

        points = df[['x', 'y', 'z']].to_numpy()
        vectors = df[['Bx', 'By', 'Bz']].to_numpy() * scale

        magnitudes = np.linalg.norm(vectors, axis=1)
        print(
            f"Vector field{' (' + label + ')' if label else ''}: {len(points)} arrows, "
            f"|B| range [{magnitudes.min():.4g}, {magnitudes.max():.4g}]"
        )

        cloud = pv.PolyData(points)
        cloud['vectors'] = vectors
        cloud['magnitude'] = magnitudes

        glyphs = cloud.glyph(orient='vectors', scale='magnitude', factor=1.0)

        if color_by_magnitude:
            self._pv_plotter.add_mesh(
                glyphs,
                scalars='magnitude',
                cmap=colormap or 'plasma',
                label=label,
            )
        else:
            self._pv_plotter.add_mesh(
                glyphs,
                color=color,
                label=label,
            )

        if label:
            self._labels.append(label)

        self._has_content = True
        return self

    def add_custom_mesh(
        self,
        mesh: pv.DataSet,
        color: str = "white",
        opacity: float = 1.0,
        show_edges: bool = True,
        label: str = None,
        **kwargs
    ) -> 'UnifiedPlotter':
        """
        Add a custom PyVista mesh to the plot.

        Args:
            mesh: Any PyVista dataset
            color: Mesh color
            opacity: Mesh opacity
            show_edges: Whether to show mesh edges
            label: Label for legend
            **kwargs: Additional arguments passed to add_mesh

        Returns:
            self for method chaining
        """
        self._pv_plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            label=label,
            **kwargs
        )

        if label:
            self._labels.append(label)

        self._has_content = True
        return self

    def set_clip_plane(
        self,
        normal: str = 'x',
        origin: tuple = None,
        invert: bool = False
    ) -> 'UnifiedPlotter':
        """
        Set a clipping plane to create a cut view.

        Args:
            normal: Plane normal direction. Can be:
                - 'x', 'y', 'z', '-x', '-y', '-z' for axis-aligned planes
                - A tuple (nx, ny, nz) for custom normal
            origin: Origin point of the plane (x, y, z). If None, uses center of data.
            invert: If True, inverts which side is clipped

        Returns:
            self for method chaining

        Example:
            plotter.set_clip_plane('y')  # Cut along Y=0 plane
            plotter.set_clip_plane('x', origin=(1, 0, 0))  # Cut at X=1
            plotter.set_clip_plane((1, 1, 0))  # Diagonal cut
        """
        # Convert string normal to vector
        normal_map = {
            'x': (1, 0, 0),
            'y': (0, 1, 0),
            'z': (0, 0, 1),
            '-x': (-1, 0, 0),
            '-y': (0, -1, 0),
            '-z': (0, 0, -1),
        }

        if isinstance(normal, str):
            normal_vec = normal_map.get(normal.lower(), (1, 0, 0))
        else:
            normal_vec = tuple(normal)

        self._clip_plane = {
            'normal': normal_vec,
            'origin': origin,
            'invert': invert
        }

        print(f"Clip plane set: normal={normal_vec}, origin={origin}, invert={invert}")
        return self

    def show(
        self,
        show_axes: bool = True,
        show_bounds: bool = True,
        show_legend: bool = True,
        clip_widget: bool = False
    ):
        """
        Display the plot.

        Args:
            show_axes: Whether to show coordinate axes
            show_bounds: Whether to show bounding box with grid
            show_legend: Whether to show legend (if labels were provided)
            clip_widget: If True, shows an interactive clip plane widget
        """
        if not self._has_content:
            print("Warning: No content added to plotter")
            return

        if show_axes:
            self._pv_plotter.add_axes()

        if show_bounds:
            self._pv_plotter.show_bounds(
                grid='front',
                location='outer',
                all_edges=True
            )

        if self._title:
            self._pv_plotter.add_text(self._title, position='upper_left')

        if show_legend and self._labels:
            self._pv_plotter.add_legend()

        # Add clip plane info if set
        if self._clip_plane:
            clip_info = f"Clip: normal={self._clip_plane['normal']}"
            self._pv_plotter.add_text(clip_info, position='lower_left', font_size=8)

        print("Starting plotter...")
        self._pv_plotter.show()

    @property
    def plotter(self) -> pv.Plotter:
        """Access the underlying PyVista plotter for advanced customization."""
        return self._pv_plotter


# Convenience functions for standalone plots (backward compatibility)

def plot_xyz(df: pd.DataFrame, sample_frac: float = 1.0, color: str = "red", point_size: int = 5):
    """Quick plot of XYZ point cloud."""
    plotter = UnifiedPlotter()
    plotter.add_point_cloud(df, color=color, point_size=point_size, sample_frac=sample_frac)
    plotter.show()


def plot_boundary(filename: str = "data/boundary.txt", n_phi: int = 120, n_s: int = 10):
    """Quick plot of boundary file."""
    plotter = UnifiedPlotter()
    plotter.add_boundary(filename, n_phi=n_phi, n_s=n_s)
    plotter.show()


def plot_hex_mesh(df_nodes: pd.DataFrame, elements_hex: List[List[int]], color: str = "violet"):
    """Quick plot of hexahedral mesh."""
    plotter = UnifiedPlotter()
    plotter.add_hex_mesh(df_nodes, elements_hex, color=color)
    plotter.show()


def compare_point_clouds(df1: pd.DataFrame, df2: pd.DataFrame,
                         color1: str = "red", color2: str = "blue",
                         sample_frac: float = 0.5):
    """Quick comparison of two point clouds."""
    plotter = UnifiedPlotter()
    plotter.add_point_cloud(df1, color=color1, sample_frac=sample_frac, label="Dataset 1")
    plotter.add_point_cloud(df2, color=color2, sample_frac=sample_frac, label="Dataset 2")
    plotter.show()

