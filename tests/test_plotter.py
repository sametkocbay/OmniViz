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

from omniviz.plotter import UnifiedPlotter  # noqa: E402


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


def test_external_plotter_is_used() -> None:
    try:
        external = pv.Plotter()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PyVista plotter unavailable: {exc}")
    up = UnifiedPlotter(plotter=external)
    assert up.plotter is external
