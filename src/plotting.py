import pyvista as pv
import numpy as np
from .visualizeBoundary import read_boundary, reconstruct_surface

def plot_xyz(df, p=0):
    """
    p = fraction of points to drop (0.0–1.0)
    """
    # Keep a random subset of points
    df_sampled = df.sample(frac=(1 - p), random_state=42)

    points = df_sampled[['x', 'y', 'z']].to_numpy()

    # For debugging: print bounding box and radius
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = np.linalg.norm(maxs - center)

    print("Bounding box min:", mins)
    print("Bounding box max:", maxs)
    print("Approx radius:", radius)

    # Point cloud
    cloud = pv.PolyData(points)

    plotter = pv.Plotter()
    plotter.add_points(
        cloud,
        render_points_as_spheres=True,
        point_size=5
    )

    # Make axes equal scale so radius is visually meaningful
    plotter.show_bounds(
        grid='front',
        location='outer',
        all_edges=True
    )
    plotter.add_axes()
    plotter.set_scale(1, 1, 1)  # ensure equal scale

    plotter.show()





def d_compare(df, df2, p=0.5):
    """
    p = fraction of points to drop (0.0–1.0)
    """

    # --- Sample points ---
    df_sampled  = df.sample(frac=(1 - p), random_state=42)
    df_sampled2 = df2.sample(frac=(1 - p), random_state=42)

    points  = df_sampled[['x', 'y', 'z']].to_numpy()
    points2 = df_sampled2[['x', 'y', 'z']].to_numpy()

    # --- Bounding boxes ---
    for pts, label in [(points, "Cloud 1"), (points2, "Cloud 2")]:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2
        radius = np.linalg.norm(maxs - center)

        print(f"\n{label} bounding box min:", mins)
        print(f"{label} bounding box max:", maxs)
        print(f"{label} approx radius:", radius)

    # --- Convert to PolyData ---
    cloud1 = pv.PolyData(points)
    cloud2 = pv.PolyData(points2)

    # --- Create single plotter ---
    plotter = pv.Plotter()

    plotter.add_points(
        cloud1,
        color="red",
        render_points_as_spheres=True,
        point_size=6
    )

    plotter.add_points(
        cloud2,
        color="blue",
        render_points_as_spheres=True,
        point_size=6
    )

    # Add axis & equal aspect ratio
    #plotter.add_axes()
    #plotter.show_bounds(all_edges=True)

    # Force equal aspect ratio
    plotter.camera.SetParallelProjection(True)
    plotter.set_scale(1, 1, 1)

    plotter.show()

def plot_xyz_and_boundary_txt(df, filename='boundary.txt'):
    data = read_boundary(filename)
    if data is None:
        return
    elements, n_harm, n_period = data

    # high resolution reconstruction
    X_list, Y_list, Z_list = reconstruct_surface(elements, n_harm, n_period,
                                                 n_phi=21, n_s=2)

    # PyVista Plotter initialisieren (ONE plotter only)
    plotter = pv.Plotter()
    plotter.set_background("white")

    print("Erstelle PyVista Meshes...")

    # MultiBlock for efficiency
    multi_block = pv.MultiBlock()

    for i, (x, y, z) in enumerate(zip(X_list, Y_list, Z_list)):
        grid = pv.StructuredGrid(x, y, z)
        multi_block.append(grid)

    # Add geometry mesh
    plotter.add_mesh(
        multi_block,
        color="cyan",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        opacity=0.5,
        smooth_shading=True,
        specular=0,
    )

    plotter.add_axes()
    plotter.add_text(
        f"JOREK Boundary (Harmonics: {n_harm}, Period: {n_period})",
        position='upper_left'
    )

    ###########################################
    # Sample df only once
    df_sampled = df.sample(frac=(1), random_state=42)

    points = df_sampled[['x', 'y', 'z']].to_numpy()

    # diagnostics
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = np.linalg.norm(maxs - center)

    print("Bounding box min:", mins)
    print("Bounding box max:", maxs)
    print("Approx radius:", radius)

    # Create point cloud
    cloud = pv.PolyData(points)

    # Add point cloud to SAME plotter
    plotter.add_points(
        cloud,
        render_points_as_spheres=True,
        point_size=5
    )

    # equal axes bounds
    plotter.show_bounds(
        grid='front',
        location='outer',
        all_edges=True
    )

    ###########################################
    print("Starte Plotter...")
    plotter.show()


def plot_patches(df_nodes, elements_hex):
    # 1. Prepare Points (Vertices)
    # We must ensure the order of points matches the index 0, 1, 2...
    # So we sort df_nodes by ID and extract just the XYZ columns
    df_sorted = df_nodes.sort_values('node_id')
    points = df_sorted[['x', 'y', 'z']].values

    # Map NodeID -> Array Index (0 to N)
    # This is needed because PyVista expects 0-based indices into the 'points' array
    id_to_index = {node_id: i for i, node_id in enumerate(df_sorted['node_id'])}

    # 2. Prepare Cells (Elements)
    # VTK requires a flat array: [n_nodes, node0, node1..., n_nodes, node0...]
    # For Hexahedrons, n_nodes is always 8.
    cells = []
    cell_types = []

    for hex_nodes in elements_hex:
        # Convert file NodeIDs to Array Indices
        try:
            indices = [id_to_index[nid] for nid in hex_nodes]
            # [Number of points, p1, p2, p3, p4, p5, p6, p7, p8]
            cells.append([8] + indices)
            cell_types.append(pv.CellType.HEXAHEDRON)
        except KeyError:
            continue

    # Flatten the list for VTK
    cells_flat = np.hstack(cells)

    # 3. Create Unstructured Grid
    grid = pv.UnstructuredGrid(cells_flat, cell_types, points)

    # 4. Plot
    # show_edges=True replicates the wireframe look of the MATLAB script
    grid.plot(show_edges=True, color='violet', notebook=False)





def plot_2_xyz_and_boundary_txt(df,df2, filename='boundary.txt'):
    data = read_boundary(filename)
    if data is None:
        return
    elements, n_harm, n_period = data

    # high resolution reconstruction
    X_list, Y_list, Z_list = reconstruct_surface(elements, n_harm, n_period,
                                                 n_phi=120, n_s=2)

    # PyVista Plotter initialisieren (ONE plotter only)
    plotter = pv.Plotter()
    plotter.set_background("white")

    print("Erstelle PyVista Meshes...")

    # MultiBlock for efficiency
    multi_block = pv.MultiBlock()

    for i, (x, y, z) in enumerate(zip(X_list, Y_list, Z_list)):
        grid = pv.StructuredGrid(x, y, z)
        multi_block.append(grid)

    # Add geometry mesh
    plotter.add_mesh(
        multi_block,
        color="cyan",
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        opacity=0.5,
        smooth_shading=True,
        specular=0,
    )

    plotter.add_axes()
    plotter.add_text(
        f"JOREK Boundary (Harmonics: {n_harm}, Period: {n_period})",
        position='upper_left'
    )

    ###########################################
    # Sample df only once
    df_sampled = df.sample(frac=(1), random_state=42)
    df_sampled2 = df2.sample(frac=(1), random_state=42)
    points = df_sampled[['x', 'y', 'z']].to_numpy()
    points2 = df_sampled2[['x', 'y', 'z']].to_numpy()
    # diagnostics
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = np.linalg.norm(maxs - center)

    print("Bounding box min:", mins)
    print("Bounding box max:", maxs)
    print("Approx radius:", radius)

    # Create point cloud
    cloud = pv.PolyData(points)
    cloud2 = pv.PolyData(points2)
    # Add point cloud to SAME plotter
    plotter.add_points(
        cloud,
        render_points_as_spheres=True,
        point_size=25,
        color="blue"
    )
    plotter.add_points(
        cloud2,
        render_points_as_spheres=True,
        point_size=25,
        color="red"
    )
    # equal axes bounds
    plotter.show_bounds(
        grid='front',
        location='outer',
        all_edges=True
    )

    ###########################################
    print("Starte Plotter...")
    plotter.show()