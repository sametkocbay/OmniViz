"""File parsers for OmniViz data sources.

Handles four formats:
    * XYZ ASCII point clouds (3 columns)
    * 6-column vector fields (x y z Bx By Bz)
    * JOREK boundary files (Hermite × Fourier representation)
    * Patran Neutral mesh files (.msh, .out)

Fortran-style malformed scientific notation (``1.234-309`` instead of
``1.234E-309``) is repaired transparently.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pyvista as pv

log = logging.getLogger(__name__)

_FORTRAN_FLOAT_RE = re.compile(r"^([+-]?\d+\.?\d*)([-+])(\d+)$")


def parse_fortran_float(token: str) -> float:
    """Parse a float that may use Fortran-style truncated exponents."""
    token = token.strip()
    if not token:
        return float("nan")

    match = _FORTRAN_FLOAT_RE.match(token)
    if match:
        mantissa, sign, exp = match.groups()
        token = f"{mantissa}E{sign}{exp}"

    try:
        return float(token)
    except ValueError:
        return float("nan")


def _iter_data_lines(path: Path | str) -> Iterable[list[str]]:
    """Yield whitespace-split tokens for each non-blank, non-comment line."""
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield stripped.split()


def detect_file_columns(path: Path | str) -> int:
    """Return the number of columns in the first data line, or 0 on failure."""
    try:
        for tokens in _iter_data_lines(path):
            return len(tokens)
    except OSError:
        pass
    return 0


def read_xyz_data(path: Path | str) -> pd.DataFrame:
    """Read a 3+ column XYZ ASCII file into a DataFrame with x/y/z columns."""
    rows: list[tuple[float, float, float]] = []
    for tokens in _iter_data_lines(path):
        if len(tokens) < 3:
            continue
        rows.append(
            (
                parse_fortran_float(tokens[0]),
                parse_fortran_float(tokens[1]),
                parse_fortran_float(tokens[2]),
            )
        )

    df = pd.DataFrame(rows, columns=["x", "y", "z"])
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def read_vector_field(path: Path | str) -> pd.DataFrame:
    """Read a 6-column vector field file: ``x y z Bx By Bz``."""
    rows: list[list[float]] = []
    for tokens in _iter_data_lines(path):
        if len(tokens) < 6:
            continue
        rows.append([parse_fortran_float(t) for t in tokens[:6]])

    df = pd.DataFrame(rows, columns=["x", "y", "z", "Bx", "By", "Bz"])
    return df.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass
class BoundaryElement:
    """One element of a JOREK boundary file."""

    vals_R: tuple[np.ndarray, np.ndarray]
    vals_Z: tuple[np.ndarray, np.ndarray]
    deriv_R: tuple[np.ndarray, np.ndarray]
    deriv_Z: tuple[np.ndarray, np.ndarray]
    sizes: tuple[tuple[float, float], tuple[float, float]]


@dataclass
class BoundaryData:
    """Parsed contents of a JOREK boundary file."""

    elements: list[BoundaryElement]
    n_harm: int
    n_period: int


def read_boundary(path: Path | str) -> BoundaryData | None:
    """Parse a JOREK boundary file. Returns ``None`` if the file is missing."""
    try:
        with open(path) as f:
            tokens = iter(f.read().split())
    except FileNotFoundError:
        return None

    try:
        n_elem = int(next(tokens))
        _ = int(next(tokens))  # n_nodes (unused, reserved for cross-checks)
        n_harm = int(next(tokens))
        n_period = int(next(tokens))
        _ = int(next(tokens))  # version
    except StopIteration:
        return None

    def take(n: int) -> list[float]:
        return [float(next(tokens)) for _ in range(n)]

    elements: list[BoundaryElement] = []
    for _ in range(n_elem):
        try:
            _ = int(next(tokens))  # element index
            _ = int(next(tokens))  # node 1
            _ = int(next(tokens))  # node 2

            vals_n1_R = np.array(take(n_harm))
            vals_n1_Z = np.array(take(n_harm))
            deriv_n1_R = np.array(take(n_harm))
            deriv_n1_Z = np.array(take(n_harm))
            size_n1 = (float(next(tokens)), float(next(tokens)))

            vals_n2_R = np.array(take(n_harm))
            vals_n2_Z = np.array(take(n_harm))
            deriv_n2_R = np.array(take(n_harm))
            deriv_n2_Z = np.array(take(n_harm))
            size_n2 = (float(next(tokens)), float(next(tokens)))

            elements.append(
                BoundaryElement(
                    vals_R=(vals_n1_R, vals_n2_R),
                    vals_Z=(vals_n1_Z, vals_n2_Z),
                    deriv_R=(deriv_n1_R, deriv_n2_R),
                    deriv_Z=(deriv_n1_Z, deriv_n2_Z),
                    sizes=(size_n1, size_n2),
                )
            )
        except StopIteration:
            break

    return BoundaryData(elements=elements, n_harm=n_harm, n_period=n_period)


def read_patran_neutral(path: Path | str) -> tuple[pd.DataFrame, list[list[int]]]:
    """Read a Patran Neutral file (``.msh``/``.out``).

    Returns a node DataFrame (``node_id, x, y, z``) and the list of
    hexahedral element connectivities (8 node IDs each).
    """
    nodes: list[list[float]] = []
    hexes: list[list[int]] = []

    def to_int(value: str) -> int:
        return int(float(value))

    with open(path) as f:
        it = iter(f)
        try:
            for line in it:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    label = to_int(parts[0])
                except ValueError:
                    continue

                if label == 25:  # title packet
                    next(it)
                elif label == 26:  # summary packet
                    next(it)
                elif label == 1:  # node
                    node_id = to_int(parts[1])
                    coords = list(map(float, next(it).split()))
                    next(it)  # constraints (ignored)
                    nodes.append([node_id, *coords])
                elif label == 2:  # element
                    shape_id = to_int(parts[2])
                    next(it)  # config (ignored)
                    node_ids = list(map(to_int, next(it).split()))
                    if shape_id == 8:  # hexahedron
                        hexes.append(node_ids[:8])
                elif label == 99:  # EOF
                    break
        except StopIteration:
            pass

    df = pd.DataFrame(nodes, columns=["node_id", "x", "y", "z"])
    df["node_id"] = df["node_id"].astype(int)
    df = df.sort_values("node_id").reset_index(drop=True)
    return df, hexes


# --------------------------------------------------------------------------- #
# CARIDDI mesh / current density
# --------------------------------------------------------------------------- #

#: CARIDDI ``ixtype`` value -> (node count, trailing-column count to strip).
#: ``ix`` rows carry connectivity followed by trailing material/flag columns;
#: the number of trailing columns differs by element type (see tools.md §5).
_CARIDDI_ELEMENT_RULES: dict[int, tuple[str, int, int]] = {
    1: ("HEXAHEDRON", 8, 1),  # hex: 8 nodes + 1 trailing material column
    2: ("TETRA", 4, 4),  # tetra: 4 nodes + 4 trailing columns
    3: ("WEDGE", 6, 3),  # penta/wedge: 6 nodes + 3 trailing columns
}


def read_cariddi_mesh(
    x_path: Path | str,
    ix_path: Path | str,
    ixtype_path: Path | str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a CARIDDI mesh.

    Returns the node coordinates ``(N, 3)`` and a mapping of element-type name
    (``"HEXAHEDRON"`` / ``"TETRA"`` / ``"WEDGE"``) to a 0-indexed connectivity
    array. 1-indexed Fortran node IDs are converted to 0-indexed and the
    trailing material/flag columns are stripped per element-type rules.
    """
    nodes = np.loadtxt(x_path, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    nodes = np.ascontiguousarray(nodes[:, :3], dtype=np.float64)

    # Rows in ``ix`` are ragged (the trailing-column count varies by element
    # type), so read line-by-line rather than via ``np.loadtxt``.
    incidence: list[list[int]] = []
    for tokens in _iter_data_lines(ix_path):
        incidence.append([int(float(t)) for t in tokens])

    element_types = np.atleast_1d(np.loadtxt(ixtype_path, dtype=np.int64))

    elements: dict[str, list[np.ndarray]] = {}
    for row, etype in zip(incidence, element_types, strict=False):
        rule = _CARIDDI_ELEMENT_RULES.get(int(etype))
        if rule is None:
            log.warning("Unknown CARIDDI element type %s; skipping", etype)
            continue
        name, n_nodes, _trailing = rule
        # Connectivity is the first ``n_nodes`` columns, converted to 0-indexed.
        conn = np.asarray(row[:n_nodes], dtype=np.int64) - 1
        elements.setdefault(name, []).append(conn)

    out: dict[str, np.ndarray] = {
        name: np.asarray(cells, dtype=np.int64) for name, cells in elements.items()
    }
    return nodes, out


def compute_element_centroids(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Return ``(M, 3)`` centroids: the mean coordinate of each element's nodes."""
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    if elements.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return nodes[elements].mean(axis=1)


def read_profile(path: Path | str) -> pd.DataFrame:
    """Read a 2-column ``.dat`` profile (rho/q/ffp/t) into ``x, value`` columns."""
    rows: list[tuple[float, float]] = []
    for tokens in _iter_data_lines(path):
        if len(tokens) < 2:
            continue
        rows.append((parse_fortran_float(tokens[0]), parse_fortran_float(tokens[1])))

    df = pd.DataFrame(rows, columns=["x", "value"])
    return df.replace([np.inf, -np.inf], np.nan).dropna()


# --------------------------------------------------------------------------- #
# JOREK HDF5 restart reader (optional, requires h5py)
# --------------------------------------------------------------------------- #

#: Default JOREK variable order (model-dependent; see tools.md §6c).
JOREK_VAR_NAMES = ["psi", "u", "j", "w", "rho", "T", "v_par", "T_e", "rho_n"]


def _jorek_basis_functions(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """2D cubic Bernstein/Hermite basis ``B[order, vertex]`` at ``(s, t)``.

    Ported verbatim from JOREK's ``jorek_read_h5.basis_functions`` (16 terms:
    4 orders x 4 vertices). ``s`` and ``t`` may be arrays for vectorised eval.
    """
    return np.asarray(
        [
            [
                (-1 + s) ** 2 * (1 + 2 * s) * (-1 + t) ** 2 * (1 + 2 * t),
                -(s**2 * (-3 + 2 * s) * (-1 + t) ** 2 * (1 + 2 * t)),
                s**2 * (-3 + 2 * s) * t**2 * (-3 + 2 * t),
                -((-1 + s) ** 2) * (1 + 2 * s) * t**2 * (-3 + 2 * t),
            ],
            [
                3 * (-1 + s) ** 2 * s * (-1 + t) ** 2 * (1 + 2 * t),
                -3 * (-1 + s) * s**2 * (-1 + t) ** 2 * (1 + 2 * t),
                3 * (-1 + s) * s**2 * t**2 * (-3 + 2 * t),
                -3 * (-1 + s) ** 2 * s * t**2 * (-3 + 2 * t),
            ],
            [
                3 * (-1 + s) ** 2 * (1 + 2 * s) * (-1 + t) ** 2 * t,
                -3 * s**2 * (-3 + 2 * s) * (-1 + t) ** 2 * t,
                3 * s**2 * (-3 + 2 * s) * (-1 + t) * t**2,
                -3 * (-1 + s) ** 2 * (1 + 2 * s) * (-1 + t) * t**2,
            ],
            [
                9 * (-1 + s) ** 2 * s * (-1 + t) ** 2 * t,
                -9 * (-1 + s) * s**2 * (-1 + t) ** 2 * t,
                9 * (-1 + s) * s**2 * (-1 + t) * t**2,
                -9 * (-1 + s) ** 2 * s * (-1 + t) * t**2,
            ],
        ]
    )


def _jorek_bf(n_sub: int) -> np.ndarray:
    """Evaluate the basis functions on a uniform ``n_sub x n_sub`` grid."""
    lin = np.linspace(0.0, 1.0, n_sub)
    s = np.tensordot(lin, [1] * n_sub, axes=0)
    t = s.transpose()
    return _jorek_basis_functions(s, t)


def _jorek_toroidal_basis(
    n_tor: int, n_period: int, phis: np.ndarray, without_n0_mode: bool = False
) -> np.ndarray:
    """Toroidal Fourier basis ``HZ[harmonic, plane]`` (tools.md §6c step 3)."""
    hz = np.zeros((n_tor, len(phis)))
    for i in range(n_tor):
        mode = np.floor((i + 1) / 2) * n_period
        if i == 0:
            if not without_n0_mode:
                hz[i, :] = 1.0
        elif i % 2 == 0:
            hz[i, :] = np.sin(mode * phis)
        else:
            hz[i, :] = np.cos(mode * phis)
    return hz


def read_jorek_restart(
    path: Path | str,
    *,
    n_sub: int = 2,
    n_plane: int = 4,
    phi: tuple[float, float] = (0.0, 360.0),
    variables: list[str] | None = None,
) -> pv.UnstructuredGrid:
    """Read a JOREK HDF5 restart into a :class:`pyvista.UnstructuredGrid`.

    This is the *minimal viable subset* described in tools.md §6c: linear
    poloidal subdivision, toroidal Fourier reconstruction, and a single VTK
    hexahedron (or quad) cell type. The full Bezier / quadratic higher-order
    path is intentionally **not** implemented — this just produces a reasonable
    colored grid. At least the ``psi`` scalar is attached.

    Requires the optional ``h5py`` dependency (``pip install omniviz[jorek]``).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise ImportError(
            "Reading JOREK HDF5 restarts requires h5py. Install it with: pip install omniviz[jorek]"
        ) from exc

    import pyvista as pv

    if variables is None:
        variables = ["psi"]

    with h5py.File(path, "r") as hf:
        n_period = int(hf["n_period"][0])
        n_tor = int(hf["n_tor"][0])
        n_elements = int(hf["n_elements"][0])
        vertex = np.array(hf["vertex"])  # (4, n_elements), 1-indexed node IDs
        x = np.array(hf["x"], dtype=np.float64)  # R/Z Bernstein coeffs
        size = np.array(hf["size"], dtype=np.float64)  # (order, vertex, element)
        values = np.array(hf["values"], dtype=np.float64)
        var_names = JOREK_VAR_NAMES

    # Resolve requested variable names to indices into the ``values`` array.
    var_idx = [var_names.index(v) for v in variables if v in var_names]
    if not var_idx:
        var_idx = [0]  # fall back to psi (index 0)

    # ------------------------------------------------------------------ #
    # Toroidal planes: build phi samples in radians.
    # ------------------------------------------------------------------ #
    periodic = abs((phi[0] - phi[1]) % 360.0) < 1e-9
    if n_plane <= 1:
        phis = np.asarray([phi[0]])
    else:
        phis = np.linspace(phi[0], phi[1], num=n_plane, endpoint=not periodic)
    phis = np.deg2rad(phis)
    n_plane = len(phis)

    bf = _jorek_bf(n_sub)  # (order, vertex, s, t)

    # ------------------------------------------------------------------ #
    # 2D (R, Z) reconstruction per element on the n_sub x n_sub grid.
    # Handle legacy (x.ndim == 3) vs post-2021 (x.ndim == 4) layouts.
    # ------------------------------------------------------------------ #
    # tmp[order, vertex, element, var(R/Z)]
    tmp = np.zeros((x.shape[0], x.shape[1], vertex.shape[0], vertex.shape[1]))
    if x.ndim == 4:  # post-Feb-2021: x[var, order, 0, node]
        for i in range(vertex.shape[0]):
            tmp[:, :, i, :] = x[:, :, 0, vertex[i, :] - 1]
    else:  # legacy: x[var, order, node]
        for i in range(vertex.shape[0]):
            tmp[:, :, i, :] = x[:, :, vertex[i, :] - 1]

    tmp[0, :, :, :] *= size
    tmp[1, :, :, :] *= size

    # RZ[element, s, t, var(R/Z)]
    rz = np.zeros((vertex.shape[1], n_sub, n_sub, 2))
    rz[:, :, :, 0] = np.tensordot(tmp[0, :, :, :], bf, axes=((0, 1), (0, 1)))
    rz[:, :, :, 1] = np.tensordot(tmp[1, :, :, :], bf, axes=((0, 1), (0, 1)))

    n_points = n_elements * n_sub * n_sub
    xyz = np.zeros((n_points * n_plane, 3))
    for i in range(n_plane):
        xyz[i * n_points : (i + 1) * n_points, 0] = np.ravel(rz[:, :, :, 0] * np.cos(phis[i]))
        xyz[i * n_points : (i + 1) * n_points, 1] = np.ravel(rz[:, :, :, 1])
        xyz[i * n_points : (i + 1) * n_points, 2] = np.ravel(rz[:, :, :, 0] * np.sin(phis[i]))

    # ------------------------------------------------------------------ #
    # Connectivity. Within each element build (n_sub-1)^2 quads, then either
    # keep them as quads (single plane) or sweep into hexahedra between planes.
    # ------------------------------------------------------------------ #
    block = np.zeros((n_sub - 1, n_sub - 1, 4), dtype=np.int64)
    for j in range(n_sub - 1):
        for k in range(n_sub - 1):
            block[j, k, :] = [
                n_sub * j + k,
                n_sub * (j + 1) + k,
                n_sub * (j + 1) + k + 1,
                n_sub * j + k + 1,
            ]
    i_start = np.arange(0, n_points, n_sub**2, dtype=np.int64)
    ien_2d = (i_start[:, None, None, None] + block[None, :, :, :]).reshape(-1, 4)

    n_cells_2d = ien_2d.shape[0]
    if n_plane > 1:
        n_cells_tor = n_plane if periodic else n_plane - 1
        cells = np.zeros((n_cells_2d * n_cells_tor, 9), dtype=np.int64)
        cells[:, 0] = 8
        for i in range(n_cells_tor):
            lo = i * n_points
            hi = ((i + 1) % n_plane) * n_points if periodic else (i + 1) * n_points
            offsets = np.concatenate(([lo] * 4, [hi] * 4))
            cells[i * n_cells_2d : (i + 1) * n_cells_2d, 1:9] = ien_2d + offsets
        cell_type = pv.CellType.HEXAHEDRON
    else:
        cells = np.hstack([np.full((n_cells_2d, 1), 4, dtype=np.int64), ien_2d])
        cell_type = pv.CellType.QUAD

    cell_types = np.full(cells.shape[0], cell_type, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells.ravel(), cell_types, xyz)

    # ------------------------------------------------------------------ #
    # Scalar reconstruction. values[var, order, harmonic, vertex, element].
    # Poloidal interp -> vals[var, harmonic, element, s, t], then apply HZ.
    # ------------------------------------------------------------------ #
    hz = _jorek_toroidal_basis(n_tor, n_period, phis)
    vals_pol = np.einsum("lihjk,ijk,ijmn->lhkmn", values[var_idx][:, :, :, vertex - 1], size, bf)
    vals_3d = np.einsum("lhkmn,hp->lpkmn", vals_pol, hz)  # [var, plane, elem, s, t]

    for vi, gi in enumerate(var_idx):
        # Flatten to match xyz point ordering (plane outermost).
        scalar = vals_3d[vi].reshape(-1)
        grid.point_data[var_names[gi]] = scalar

    # Ensure psi is present and active.
    if "psi" not in grid.point_data and var_names[var_idx[0]] != "psi":
        grid.point_data["psi"] = grid.point_data[var_names[var_idx[0]]]
    grid.set_active_scalars(var_names[var_idx[0]])
    return grid
