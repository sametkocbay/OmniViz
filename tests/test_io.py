"""Tests for the OmniViz file parsers in :mod:`omniviz.io`."""

from __future__ import annotations

import math

import numpy as np

from omniviz import io as oio


def test_parse_fortran_float_normal() -> None:
    assert oio.parse_fortran_float("1.5") == 1.5
    assert oio.parse_fortran_float("-2.0E3") == -2000.0


def test_parse_fortran_float_truncated_exponent() -> None:
    # Fortran drops the 'E': "1.234-309" means 1.234E-309.
    assert oio.parse_fortran_float("1.234-309") == 1.234e-309
    assert oio.parse_fortran_float("5.0+2") == 500.0


def test_parse_fortran_float_invalid() -> None:
    assert math.isnan(oio.parse_fortran_float("not_a_number"))
    assert math.isnan(oio.parse_fortran_float(""))


def test_categorize_boundary_prefix(tmp_path) -> None:
    from omniviz.gui.panels import categorize_files

    (tmp_path / "boundary.txt").write_text("1 2 3\n")
    (tmp_path / "boundary_inner.txt").write_text("1 2 3\n")
    (tmp_path / "boundary_outer.txt").write_text("1 2 3\n")
    (tmp_path / "cloud.txt").write_text("1 2 3\n")

    cats = categorize_files(tmp_path)
    assert set(cats["boundary"]) == {
        "boundary.txt",
        "boundary_inner.txt",
        "boundary_outer.txt",
    }
    assert "cloud.txt" not in cats["boundary"]


def test_read_cariddi_mesh_indexing_and_type_split(tmp_path) -> None:
    # 8 nodes shared by a hex; reuse a subset for a tetra and a wedge.
    x = tmp_path / "x.dat"
    x.write_text("\n".join(f"{i}.0 {i}.0 {i}.0" for i in range(1, 9)) + "\n")

    # ix rows: connectivity (1-indexed) + trailing material/flag columns.
    #   hex   -> 8 nodes + 1 trailing
    #   tetra -> 4 nodes + 4 trailing
    #   wedge -> 6 nodes + 3 trailing
    ix = tmp_path / "ix.dat"
    ix.write_text(
        "1 2 3 4 5 6 7 8 99\n"  # hex (material 99)
        "1 2 3 4 11 12 13 14\n"  # tetra (4 trailing)
        "1 2 3 4 5 6 21 22 23\n"  # wedge (3 trailing)
    )

    ixtype = tmp_path / "ixtype.dat"
    ixtype.write_text("1\n2\n3\n")

    nodes, elements = oio.read_cariddi_mesh(x, ix, ixtype)

    assert nodes.shape == (8, 3)
    assert set(elements) == {"HEXAHEDRON", "TETRA", "WEDGE"}

    # 0-indexed conversion: first hex node was "1" -> index 0.
    np.testing.assert_array_equal(elements["HEXAHEDRON"], np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))
    np.testing.assert_array_equal(elements["TETRA"], np.array([[0, 1, 2, 3]]))
    np.testing.assert_array_equal(elements["WEDGE"], np.array([[0, 1, 2, 3, 4, 5]]))


def test_compute_element_centroids() -> None:
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    elements = np.array([[0, 1, 2, 3]])
    centroids = oio.compute_element_centroids(nodes, elements)
    np.testing.assert_allclose(centroids, [[1.0, 1.0, 0.0]])


def test_read_profile(tmp_path) -> None:
    path = tmp_path / "q.dat"
    path.write_text("# rho q\n0.0 1.0\n0.5 1.5\n1.0 3.0\n")
    df = oio.read_profile(path)
    assert list(df.columns) == ["x", "value"]
    assert len(df) == 3
    np.testing.assert_allclose(df["x"].to_numpy(), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(df["value"].to_numpy(), [1.0, 1.5, 3.0])
