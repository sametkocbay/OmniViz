"""Tests for actor tracking in :class:`omniviz.plotter.UnifiedPlotter`.

These run PyVista off-screen so they work without a display. If a usable GL
context is unavailable the plotter tests are skipped rather than failed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True

from omniviz.plotter import UnifiedPlotter, min_distance_between  # noqa: E402


@pytest.fixture
def plotter() -> UnifiedPlotter:
    try:
        return UnifiedPlotter()
    except Exception as exc:  # noqa: BLE001 - no GL context, etc.
        pytest.skip(f"PyVista plotter unavailable in this environment: {exc}")


def test_actor_tracking_add_and_remove(plotter: UnifiedPlotter) -> None:
    points = np.random.default_rng(0).random((20, 3))
    vectors = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "Bx": np.ones(20),
            "By": np.zeros(20),
            "Bz": np.zeros(20),
        }
    )

    plotter.add_point_cloud(points, item_id="a")
    plotter.add_vector_field(vectors, item_id="a")

    assert "a" in plotter._actors
    assert len(plotter._actors["a"]) == 2

    plotter.remove_item("a")
    assert "a" not in plotter._actors


def test_clear_items(plotter: UnifiedPlotter) -> None:
    points = np.random.default_rng(1).random((10, 3))
    plotter.add_point_cloud(points, item_id="x")
    plotter.add_point_cloud(points, item_id="y")
    assert set(plotter._actors) == {"x", "y"}

    plotter.clear_items()
    assert plotter._actors == {}


def test_min_distance_point_to_surface() -> None:
    # A plane at z=0 and a single point 2.0 above it: nearest distance is 2.0,
    # measured to the plane surface (not just its corner vertices).
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=4, j_size=4)
    point = pv.PolyData(np.array([[0.0, 0.0, 2.0]]))

    dist, pa, pb = min_distance_between(point, plane)
    assert dist == pytest.approx(2.0, abs=1e-6)
    assert pa[2] == pytest.approx(2.0)
    assert pb[2] == pytest.approx(0.0, abs=1e-6)


def test_min_distance_is_symmetric() -> None:
    a = pv.Sphere(radius=1.0, center=(0, 0, 0))
    b = pv.Sphere(radius=1.0, center=(5, 0, 0))
    d1, _, _ = min_distance_between(a, b)
    d2, _, _ = min_distance_between(b, a)
    # Surfaces are 1 unit from each centre; centres are 5 apart -> gap ~3.0.
    assert d1 == pytest.approx(d2, abs=1e-6)
    assert d1 == pytest.approx(3.0, abs=0.05)


def test_external_plotter_is_used() -> None:
    try:
        external = pv.Plotter()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PyVista plotter unavailable: {exc}")
    up = UnifiedPlotter(plotter=external)
    assert up.plotter is external
