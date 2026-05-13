import numpy as np
import pyvista as pv


def basis_functions(s):
    """
    Kubische Hermite-Basisfunktionen für s im Intervall [0, 1].
    """
    s2 = s * s
    s3 = s2 * s

    h1 = 1.0 - 3.0 * s2 + 2.0 * s3
    h2 = s - 2.0 * s2 + s3
    h3 = 3.0 * s2 - 2.0 * s3
    h4 = -s2 + s3

    return h1, h2, h3, h4


def read_boundary(filename='boundary.txt'):
    try:
        with open(filename, 'r') as f:
            content = f.read().split()
    except FileNotFoundError:
        print(f"Fehler: Datei '{filename}' nicht gefunden.")
        return None

    iterator = iter(content)

    try:
        n_elem = int(next(iterator))
        n_nodes = int(next(iterator))
        n_harm = int(next(iterator))
        n_period = int(next(iterator))
        version = int(next(iterator))
    except StopIteration:
        print("Fehler: Leere Datei oder falscher Header.")
        return None

    print(f"Lese Datei: {n_elem} Elemente, {n_harm} Harmonische, Periodizität {n_period}")

    elements = []

    for _ in range(n_elem):
        try:
            idx = int(next(iterator))
            n1 = int(next(iterator))
            n2 = int(next(iterator))

            # Node 1 Daten
            vals_n1_R = [float(next(iterator)) for _ in range(n_harm)]
            vals_n1_Z = [float(next(iterator)) for _ in range(n_harm)]
            deriv_n1_R = [float(next(iterator)) for _ in range(n_harm)]
            deriv_n1_Z = [float(next(iterator)) for _ in range(n_harm)]
            size_n1 = [float(next(iterator)), float(next(iterator))]

            # Node 2 Daten
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


def reconstruct_surface(elements, n_harm, n_period, n_phi=60, n_s=10):
    """
    Rekonstruiert die Oberfläche.
    n_phi und n_s erhöht für glattere Darstellung in PyVista.
    """
    all_X = []
    all_Y = []
    all_Z = []

    phi_arr = np.linspace(0, 2 * np.pi, n_phi)

    # Matrix M für Fourier-Transformation vorbereiten
    M = np.zeros((n_harm, n_phi))
    M[0, :] = 1.0  # DC Komponente

    for k in range(1, (n_harm - 1) // 2 + 1):
        idx_cos = 2 * k - 1
        idx_sin = 2 * k
        if idx_sin < n_harm:
            # Beachte: n_period ist der Multiplikator für die Harmonischen
            arg = k * n_period * phi_arr
            M[idx_cos, :] = np.cos(arg)
            M[idx_sin, :] = np.sin(arg)

    s_arr = np.linspace(0, 1, n_s)
    h1, h2, h3, h4 = basis_functions(s_arr)

    print("Rekonstruiere Oberfläche...")

    for el in elements:
        sz1_R, sz1_Z = el['sizes'][0]
        sz2_R, sz2_Z = el['sizes'][1]

        # Poloidale Interpolation (Hermite)
        # Dimensionen: (n_harm, n_s)
        R_coeffs_s = (np.outer(el['vals_R'][0], h1) +
                      np.outer(el['deriv_R'][0] * sz1_R, h2) +
                      np.outer(el['vals_R'][1], h3) +
                      np.outer(el['deriv_R'][1] * sz2_R, h4))

        Z_coeffs_s = (np.outer(el['vals_Z'][0], h1) +
                      np.outer(el['deriv_Z'][0] * sz1_Z, h2) +
                      np.outer(el['vals_Z'][1], h3) +
                      np.outer(el['deriv_Z'][1] * sz2_Z, h4))

        # Toroidale Rekonstruktion (Fourier)
        # Matrix-Multiplikation: (n_s, n_harm) @ (n_harm, n_phi) -> (n_s, n_phi)
        R_surf = np.matmul(R_coeffs_s.T, M)
        Z_surf = np.matmul(Z_coeffs_s.T, M)

        # Konvertierung Zylinder -> Kartesisch
        X_surf = R_surf * np.cos(phi_arr)
        Y_surf = R_surf * np.sin(phi_arr)

        all_X.append(X_surf)
        all_Y.append(Y_surf)
        all_Z.append(Z_surf)

    return all_X, all_Y, all_Z


def plot_boundary_txt(filename='boundary.txt'):
    data = read_boundary(filename)
    if data is None: return
    elements, n_harm, n_period = data

    # Erhöhte Auflösung für PyVista (n_phi=120, n_s=10)
    X_list, Y_list, Z_list = reconstruct_surface(elements, n_harm, n_period, n_phi=120, n_s=10)

    # PyVista Plotter initialisieren
    plotter = pv.Plotter()
    plotter.set_background("white")

    print("Erstelle PyVista Meshes...")

    # Ein MultiBlock Datensatz ist effizienter für viele kleine Grids
    multi_block = pv.MultiBlock()

    for i, (x, y, z) in enumerate(zip(X_list, Y_list, Z_list)):
        # StructuredGrid erstellen für jedes Element
        # Dimensionen müssen (Nx, Ny, Nz) sein, hier (Ns, Nphi, 1)
        grid = pv.StructuredGrid(x, y, z)
        multi_block.append(grid)

    # Mesh zum Plotter hinzufügen
    # smooth_shading=True sorgt für eine schönere Oberfläche
    plotter.add_mesh(multi_block, color="cyan", show_edges=True, edge_color="black", line_width=0.5, opacity=1,
                     smooth_shading=True, specular =0.5)

    plotter.add_axes()
    plotter.add_text(f"JOREK Boundary (Harmonics: {n_harm}, Period: {n_period})", position='upper_left')

    print("Starte Plotter...")
    plotter.show()

