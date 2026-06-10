# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --frozen            # install from lock file into ./.venv
uv sync --extra dev         # add dev deps (ruff, pytest)
uv sync --extra jorek       # add h5py for the JOREK HDF5 restart reader
uv run omniviz              # launch the GUI (also: python -m omniviz, or ./run.sh)
uv run ruff check .         # lint
uv run ruff format .        # format
QT_QPA_PLATFORM=offscreen uv run pytest -q   # tests (headless; needs no display)
```

`run.sh` falls back across `uv` → `./.venv/bin/omniviz` → `python -m omniviz`.

## Architecture

OmniViz is a **single-window PySide6 GUI** with an **embedded** pyvistaqt 3D view, over a
script-usable core. The scene is persistent and live: adding/removing a queue item mutates the
one embedded view — there is no second window and no per-render rebuild. Data flows:
**parse → typed item (with id) → UnifiedPlotter.add_\*(item_id=…) → live actor in the embedded scene.**

- `omniviz/io.py` — the **only** place that parses raw files (XYZ point clouds, 6-col vector
  fields, JOREK boundary, Patran `.msh`/`.out`, CARIDDI mesh `x/ix/ixtype.dat`, 2-col profiles,
  generic VTK/VTU via `pv.read`, and JOREK HDF5 restarts via `read_jorek_restart`, which needs
  the optional `[jorek]` extra). `parse_fortran_float` repairs Fortran-truncated exponents
  (`1.234-309` → `1.234E-309`). `detect_file_columns` drives auto-categorization.
- `omniviz/plotter.py` — `UnifiedPlotter`. Constructed with `plotter=<QtInteractor>` to bind to
  the embedded view (defaults to a standalone `pv.Plotter()` for scripting). Each `add_*` takes
  `item_id=` and registers the returned actor in `self._actors[id]`; `remove_item`/`update_item`/
  `clear_items`/`render` drive live edits. `add_*` still return `self` (chaining preserved).
  `reconstruct_boundary_surface()` is pure NumPy and GUI-independent.
- `omniviz/models.py` — `ViewItem` ABC + dataclass subclasses, each with a stable `id`,
  `summary()`, and `apply(plotter, data_dir)` (which calls `add_*(item_id=self.id)`). Includes
  CARIDDI/JOREK/Profile items.
- `omniviz/gui/` — **Qt (current):** `main_window.py` (QMainWindow + central `QtInteractor`,
  scene-tree dock, toolbar, threaded loads), `qt_panels.py` (one panel per type → `ViewItem`),
  `qt_style.py` (dark/light QSS), `qt_icons.py`. **Legacy (unreferenced):** the CustomTkinter
  `app.py`, `panels.py` (still exports `categorize_files`, reused by `qt_panels`), `photo.py`,
  `file_dialog.py`, `import_dialog.py`, `theme.py`.

Entry point is `omniviz.gui.main_window:run`; call `run(data_dir=Path(...))` to point elsewhere.

## Conventions / invariants

- Keep **all parsing** in `omniviz/io.py`; keep boundary math pure-NumPy in `plotter.py`.
- GUI panels build items only — never render directly. Rendering goes through `UnifiedPlotter`.
- Every renderable is a typed `ViewItem` subclass with a stable `id`; queue ops map to
  `add_*(item_id=…)` / `remove_item(id)` / `update_item(id, …)` on the **shared** embedded plotter.
- VTK/Qt calls (`add_mesh`, `remove_actor`, `render`) run on the Qt main thread; heavy parsing/
  mesh-building runs in a worker, with the actor added on the main thread.
- Ruff is configured in `pyproject.toml` (rules E/F/W/I/UP/B/C4, line-length 100, E501 off).
