# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --frozen            # install from lock file into ./.venv
uv sync --extra dev         # add dev deps (ruff, pytest)
uv run omniviz              # launch the GUI (also: python -m omniviz, or ./run.sh)
uv run ruff check .         # lint
uv run ruff format .        # format
uv run pytest               # tests (none exist yet; tests/ is to be added)
```

`run.sh` falls back across `uv` → `./.venv/bin/omniviz` → `python -m omniviz`.

## Architecture

OmniViz is a GUI (CustomTkinter) over a script-usable core. Data flows in one direction:
**parse → typed item → queue → UnifiedPlotter → PyVista window.**

- `omniviz/io.py` — the **only** place that parses raw text files (XYZ point clouds,
  6-column vector fields, JOREK boundary, Patran `.msh`/`.out`). Add new file formats here.
  `parse_fortran_float` repairs Fortran-truncated exponents (`1.234-309` → `1.234E-309`),
  common in fusion codes. `detect_file_columns` drives auto-categorization (≥6 cols → vector
  field, ≤5 → point cloud).
- `omniviz/plotter.py` — `UnifiedPlotter`, a fluent PyVista wrapper where each `add_*`
  returns `self`. `reconstruct_boundary_surface()` is pure NumPy (Hermite poloidal × Fourier
  toroidal) and intentionally decoupled from rendering so it works outside the GUI. Also
  contains "photo mode" high-res capture used by the photo editor.
- `omniviz/models.py` — `ViewItem` ABC and its dataclass subclasses (PointCloudItem,
  BoundaryItem, VtkMeshItem, …). Each implements `summary()` (queue label) and
  `apply(plotter, data_dir)` (adds itself to a `UnifiedPlotter`).
- `omniviz/gui/` — `app.py` owns the window, the render queue, and a threaded render worker;
  `panels.py` has one panel per data type and only *builds* `ViewItem`s; `import_dialog.py`
  does chunked copy-with-progress into `data/`; `photo.py` is the screenshot caption/export
  editor; `file_dialog.py` and `theme.py` are GUI support.

Entry point is `omniviz.gui.app:run`; call `run(data_dir=Path(...))` to point at a different
data directory.

## Conventions / invariants

- Keep **all parsing** in `omniviz/io.py`; keep boundary math pure-NumPy in `plotter.py`.
- GUI panels build items only — never render directly. Rendering always goes through
  `UnifiedPlotter`, the same API used from scripts.
- Every renderable is a typed `ViewItem` subclass (`summary()` + `apply()`), never an
  untyped dict.
- Rendering is non-blocking: the plot is built off the Tk thread, then the PyVista window is
  shown from the main loop. Preserve this when touching the render path in `app.py`.
- Ruff is configured in `pyproject.toml` (rules E/F/W/I/UP/B/C4, line-length 100, E501 off).
