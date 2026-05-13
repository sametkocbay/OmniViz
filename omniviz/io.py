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

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
        rows.append((
            parse_fortran_float(tokens[0]),
            parse_fortran_float(tokens[1]),
            parse_fortran_float(tokens[2]),
        ))

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

            elements.append(BoundaryElement(
                vals_R=(vals_n1_R, vals_n2_R),
                vals_Z=(vals_n1_Z, vals_n2_Z),
                deriv_R=(deriv_n1_R, deriv_n2_R),
                deriv_Z=(deriv_n1_Z, deriv_n2_Z),
                sizes=(size_n1, size_n2),
            ))
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

                if label == 25:                       # title packet
                    next(it)
                elif label == 26:                     # summary packet
                    next(it)
                elif label == 1:                      # node
                    node_id = to_int(parts[1])
                    coords = list(map(float, next(it).split()))
                    next(it)                          # constraints (ignored)
                    nodes.append([node_id, *coords])
                elif label == 2:                      # element
                    shape_id = to_int(parts[2])
                    next(it)                          # config (ignored)
                    node_ids = list(map(to_int, next(it).split()))
                    if shape_id == 8:                 # hexahedron
                        hexes.append(node_ids[:8])
                elif label == 99:                     # EOF
                    break
        except StopIteration:
            pass

    df = pd.DataFrame(nodes, columns=["node_id", "x", "y", "z"])
    df["node_id"] = df["node_id"].astype(int)
    df = df.sort_values("node_id").reset_index(drop=True)
    return df, hexes
