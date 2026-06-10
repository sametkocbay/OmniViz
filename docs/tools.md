# OmniViz — Tools & Implementation Research

Research backing the rewrite of OmniViz into a single-window PySide6 + pyvistaqt tool for
CARIDDI/JOREK visualization. This is the Phase 1 gate document; backend/frontend work follows
from here. Version pins below are *starting points* — let `uv` resolve a mutually compatible
set at install time and lock it; treat exact numbers as "verify on install", not gospel.

---

## 1. GUI stack: PySide6 + pyvistaqt (embedded single window)

**Decision:** central `pyvistaqt.QtInteractor` inside a `QMainWindow`, with dockable control
panels. This is the one persistent 3D scene; no per-render windows.

**Why PySide6:** LGPL (compatible with our MIT app); pyvistaqt talks to it through `qtpy`, so
the same code works on PyQt5/6/PySide2/6. Set `QT_API=pyside6` before Qt imports.

**Minimal embed pattern:**
```python
import os
os.environ["QT_API"] = "pyside6"
from PySide6.QtWidgets import QApplication, QMainWindow
from pyvistaqt import QtInteractor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interactor = QtInteractor(self)          # a QWidget
        self.setCentralWidget(self.interactor)
        self.plotter = self.interactor                # pyvistaqt exposes Plotter API directly
```
Notes:
- `QtInteractor` *is* a `QWidget` and also exposes the PyVista `Plotter` API. Depending on
  version you either call methods on the `QtInteractor` directly or via an inner `.plotter`;
  confirm against the installed version during Phase 2 and wrap behind our `UnifiedPlotter`.
- **Do not** also instantiate a standalone `pv.Plotter()` in the same process — known to
  conflict with the Qt interactor. All rendering goes through the one embedded plotter.
- Qt owns the event loop; no manual VTK render loop. Call `.render()` after mutations.

**Suggested deps (resolve/lock with uv):** `pyside6`, `pyvistaqt`, `qtpy`, with existing
`pyvista`, `numpy`, `pandas`. Put the GUI stack in the default deps and keep JOREK-HDF5 in an
optional extra (below).

---

## 2. Live scene mutation — the core requirement

Each "render" must mutate the persistent scene, not rebuild a window. The mechanism:

- `add_mesh(...)` / `add_points(...)` **return an actor handle**. Store it keyed by the
  `ViewItem` id.
- Remove: `plotter.remove_actor(handle)` then `plotter.render()`.
- Update in place (preferred over remove+re-add for the same geometry):
  - color/opacity/visibility via the actor's `prop`/properties,
  - scalars/colormap/clim via the actor's `mapper` (`mapper.scalar_range`, lookup-table cmap),
  - then `plotter.render()`.

**Implication for our code (Phase 2):**
- `UnifiedPlotter` is rebound to an *externally supplied* embedded plotter (constructor takes
  the `QtInteractor`), and keeps `dict[item_id -> actor handle(s)]`.
- Add `remove_item(id)`, `update_item(id, **props)`, `clear()` alongside the existing `add_*`.
- `omniviz/models.py` `ViewItem` gains a stable `id`; `apply()` returns the handle(s) so the
  GUI queue maps directly to live add/remove/update.

---

## 3. Colorbars, slicing, screenshots (in-window)

- **Scalar bar / magnitude coloring:** `add_mesh(..., scalars=..., cmap=..., clim=...,
  show_scalar_bar=True)`. Switch colormap/clim live via the actor mapper; toggle the bar via
  `add_scalar_bar` / `remove_scalar_bar`. Our `add_vector_field` already builds a `magnitude`
  scalar + `glyph(orient,scale)`; we just expose cmap/clim/scalar-bar in the panel.
- **Slicing/clipping:** `add_mesh_clip_plane`, `add_mesh_slice`, `add_mesh_slice_orthogonal`,
  `add_mesh_clip_box` give interactive widgets that coexist with persistent actors. Extends
  today's static `set_clip_plane`.
- **Views:** reuse existing `_set_view` logic (`view_yz/xz/xy/isometric`, azimuth flip) but
  bind to **Qt toolbar actions/QShortcut** instead of VTK key events.
- **High-res screenshots:** for publication output, prefer a **separate off-screen**
  `pv.Plotter(off_screen=True, window_size=...)` rebuilt from the same items, OR temporarily
  bump `plotter.window_size` then `screenshot()`. Keep the existing photo-mode caption/export
  (`omniviz/gui/photo.py`, `capture_highres`) re-homed onto the Qt window.

---

## 4. Threading rules

- VTK/Qt rendering (`add_mesh`, `remove_actor`, `render`) → **main thread only**.
- Heavy work (file parse, mesh build, HDF5 subdivision) → `QThread`/worker; emit the built
  PyVista object via a Qt signal; the main-thread slot adds the actor.
- This replaces today's `threading.Thread` + blocking `show()` model in `app.py`.

---

## 5. CARIDDI mesh + current density (high reuse)

Port from `../cariddi_j/python_postproc/` (`functions_for_plot.py`, `main_plot_mesh.py`,
`main_plot_current_density.py`).

**Files & formats:**
- `x.dat` → node coords `(N,3)` float (`np.loadtxt`).
- `ix.dat` → incidence matrix, 1-indexed Fortran node IDs + trailing material flag.
- `ixtype.dat` → element type per element: `1=hex, 2=tetra, 3=penta(wedge)`.
- (current density) `gmat.dat` (+ `ndofel.dat`, `iglobdof.dat`) build sparse maps from wall
  currents `I3D` → per-element `Jx/Jy/Jz`; this part is CARIDDI-run-specific.

**Mesh build (PyVista):** split by type, convert connectivity to 0-indexed, build
`UnstructuredGrid` with cell types `HEXAHEDRON(12)`, `TETRA(10)`, `WEDGE(13)`:
```python
grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: hex_cells}, nodes)  # etc.
```
**Current density glyphs:** centroids = mean of element nodes; `cell_data['J'] = stack(Jx,Jy,Jz)`;
`grid.glyph(orient='J', geom=pv.Cone(), factor=...)`; color by magnitude.

**Directly portable → `omniviz/io.py`:** `read_floating`/`read_integers` (numpy loaders),
`select_el` (type/material filtering), `compute_element_centroids`, the cell-type mapping.
**Not portable (run-specific):** `read_current`, `cariddi_to_realspace` — wrap behind a
"CARIDDI current density" item that needs the gmat/I3D inputs; degrade gracefully when absent.

New items: `CariddiMeshItem`, `CariddiCurrentDensityItem` (reuse `add_hex_mesh` / glyph path).

---

## 6. JOREK readers

### 6a. ASCII fields/profiles (quick wins)
- `fields_xyz.dat` (`x y z Bx By Bz`) already parses via existing `read_vector_field`.
- Profiles: 2-column `.dat` (rho/q/ffp/t). Add a small profile reader + a **2D line-plot**
  panel using matplotlib (already in the lockfile). New `ProfileItem` renders to a Qt-embedded
  matplotlib canvas (separate from the 3D view).

### 6b. Generic VTK/VTU (cheap interop)
- Lean on `pv.read()` for `.vtk`/`.vtu` from `jorek2vtk`/CARIDDI exports. Extend the VTK panel
  to `.vtu` + let the user pick the active scalar/vector array.

### 6c. HDF5 restarts (largest effort — optional extra, `h5py`)
Mirror `../jorek/util/paraview/` (`jorek_read_h5.py`) + `diagnostics/jorek2vtk.f90`.

**HDF5 layout (jorek*.h5):** scalars `n_var, n_period, n_tor, n_vertex_max, n_elements,
tstep, t_now`; arrays `vertex (n_vertex_max,n_elements)`, `x` (R/Z Bernstein coeffs), `size`
(basis scaling), `values (n_var, n_order, n_tor, n_vertex_max, n_elements)`. Variable order
typically `["psi","u","j","w","rho","T","v_par","T_e","rho_n"]` (model-dependent).

**Reconstruction algorithm (to port):**
1. 2D cubic Bernstein basis `B[order,vertex](s,t)` on a uniform `n_sub × n_sub` grid per
   element (the 16-term basis is given explicitly in `basis_functions()`).
2. `R(s,t)=Σ x[0,order,elem,vertex]·size·B`, same for `Z`.
3. Toroidal Fourier reconstruction: `n_mode=⌊(n+1)/2⌋·n_period`; `H[0]=1`, odd→`cos(n_mode·φ)`,
   even→`sin(n_mode·φ)`; `value(s,t,φ)=Σ_n values[var,order,n,vertex,elem]·H[n,φ]`.
4. Cartesian: `x=R·cos φ`, `z=R·sin φ`, `y=Z`.
5. Build `pv.UnstructuredGrid`; attach scalar(s) + assembled vector (e.g. B from BR/BZ/BP).

**Minimal viable subset (Phase 2 target):** linear subdivision (`n_sub=2`, `n_plane≈4`,
`phi=[0,360]`), one scalar (psi) + one vector field, n=0 (+ optionally a few harmonics).
Defer full VTK_BEZIER higher-order cells / rational weights / quadratic elements to a later
pass. Gate the whole HDF5 reader behind `pip install omniviz[jorek]` (`h5py`) so the base app
stays light.

---

## 7. Open risks / decisions for Phase 2
- Exact pyvistaqt↔PySide6↔pyvista version compatibility — resolve with `uv` and smoke-test the
  embed before building panels on top.
- `QtInteractor` direct-API vs inner-`.plotter` differs by version — isolate in `UnifiedPlotter`.
- JOREK HDF5 dataset layout has legacy vs post-2021 variants for `x`/`values` shapes — detect
  by rank and handle both, or target the format of the user's actual restart files first.
- High-res export path: off-screen rebuild vs window-size bump — pick after testing fidelity.
