# OmniViz

> A modern Python GUI for visualizing stellarator and fusion-reactor
> geometries: JOREK boundary surfaces, Patran/VTK meshes, XYZ point clouds,
> 6-column vector fields, and current-filament wire loops — combined into a
> single interactive PyVista plot.

OmniViz wraps a small typed data-loading layer around
[PyVista](https://pyvista.org/) and exposes it through a
[CustomTkinter](https://customtkinter.tomschimansky.com/) GUI: pick what to
draw, queue it up, optionally enable a clip plane, and render.

---

## Features

- **Multi-source plotting.** Queue any combination of point clouds,
  JOREK boundary surfaces (Hermite × Fourier reconstruction), Patran
  neutral `.msh` hex meshes, generic `.vtk` meshes, vector fields, and
  tilted circular current loops.
- **Modern GUI.** CustomTkinter, dark/light/system theme, tabbed input
  panels with a live file filter, queue card view, and a one-click
  clip-plane toggle.
- **Robust parsers.** Handles Fortran-style truncated exponents
  (`1.234-309` → `1.234E-309`) found in many fusion codes.
- **Reproducible installs.** Either `uv` (`pyproject.toml` + `uv.lock`)
  or `conda` (`environment.yml`).
- **Scripting-friendly.** The `UnifiedPlotter` chain API can be used
  directly from notebooks/scripts without the GUI.

---

## Install

OmniViz requires **Python ≥ 3.10**.

### Option A — `uv` (recommended, fastest)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if
you do not have it, then:

```bash
git clone https://github.com/sametkocbay/OmniViz.git
cd OmniViz

# Create an isolated venv and install from the lock file
uv sync --frozen

# Run the GUI
uv run omniviz
```

`uv sync` reads `pyproject.toml` + `uv.lock` and produces a fully
reproducible `.venv/` in the project root.

### Option B — `conda` / `mamba`

```bash
conda env create -f environment.yml
conda activate omniviz
omniviz                 # or: python -m omniviz
```

This pulls `numpy`, `pandas`, `pyvista`, and `vtk` from `conda-forge`,
which is usually the smoothest path on Linux clusters that already use
conda.

### Option C — plain pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
omniviz
```

---

## Usage

### GUI

```bash
omniviz             # console script (after install)
python -m omniviz   # equivalent module form
./run.sh            # uses the local .venv created by `uv sync`
```

The window has:

1. **Left pane** — tabbed input forms (Point Cloud, Boundary, VTK,
   Patran, Vector Field, Wire). Each tab shows a filterable list of
   files from `data/`, plus the rendering options for that type.
2. **Right pane** — the *render queue* (cards showing each item you
   added), an optional **clip plane**, and the big **Render plot**
   button.
3. **Header** — appearance toggle (Dark / Light / System).

Items in the queue can be removed individually (`✕`) or cleared all at
once. Render hands the queue off to PyVista, which opens a separate
interactive 3D window.

### Programmatic use

```python
from omniviz.plotter import UnifiedPlotter

(UnifiedPlotter(background="white")
    .add_boundary("data/boundary.txt", n_phi=120, n_s=10, color="cyan")
    .add_point_cloud("data/xyz_gauss.dat", color="red", point_size=5)
    .add_vector_field("data/fields_xyz.dat", scale=0.1, color_by_magnitude=True)
    .add_wire(r0=1.99, z0=0.0, alfa_wire_deg=3.0)
    .set_clip_plane("y")
    .show())
```

---

## Data layout

The GUI auto-categorizes files dropped into `data/`:

| Pattern                          | Treated as     |
|----------------------------------|----------------|
| `boundary.txt`                   | JOREK boundary |
| `*.vtk`                          | VTK mesh       |
| `*.msh`                          | Patran neutral |
| `*.dat` / `*.txt` (≥ 6 columns)  | Vector field   |
| `*.dat` / `*.txt` (≤ 5 columns)  | XYZ point cloud |

You can point the GUI at a different directory via the Python API:

```python
from pathlib import Path
from omniviz.gui.app import run
run(data_dir=Path("/scratch/me/run42"))
```

---

## Project layout

```
OmniViz/
├── omniviz/                # the installable package
│   ├── __init__.py
│   ├── __main__.py         # `python -m omniviz`
│   ├── io.py               # parsers (XYZ, vector field, boundary, Patran)
│   ├── plotter.py          # UnifiedPlotter (PyVista wrapper)
│   ├── models.py           # ViewItem dataclasses (PointCloudItem, …)
│   └── gui/
│       ├── app.py          # main window
│       ├── panels.py       # one panel per data type
│       └── theme.py
├── data/                   # sample input files
├── pyproject.toml
├── uv.lock
├── environment.yml
├── run.sh
└── README.md
```

### Design rules followed

- **Typed dataclasses for items.** Each queueable thing is a
  `ViewItem` subclass with `summary()` + `apply()` — no untyped dicts.
- **Single source of truth for parsing.** `omniviz.io` is the only
  place that touches raw text files.
- **Boundary maths split from rendering.** `reconstruct_boundary_surface()`
  in `plotter.py` is pure NumPy and reusable outside the GUI.
- **GUI ↔ logic separation.** Panels only build items; rendering goes
  through the same `UnifiedPlotter` you would call from a script.
- **Non-blocking render.** Build-up of the plot runs off the Tk
  thread; the PyVista window is then shown from the main loop.
- **Lint config.** `ruff` configured in `pyproject.toml`
  (`E/F/W/I/UP/B/C4`) — run `uv run ruff check .`.

---

## Development

```bash
uv sync --extra dev          # install dev deps (ruff, pytest)
uv run ruff check .
uv run ruff format .
uv run pytest                # (tests live under tests/ — add as needed)
```

---

## License

MIT — see `LICENSE` if present, or the `license` field in
`pyproject.toml`.
