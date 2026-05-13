"""Main window for the OmniViz GUI."""

from __future__ import annotations

import logging
import shutil
import threading
import tkinter as tk
import tkinter.messagebox as messagebox
from collections.abc import Iterable
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from omniviz import __version__
from omniviz.assets import LOGO_PATH
from omniviz.gui.file_dialog import ask_open_file
from omniviz.gui.import_dialog import CopyProgressDialog
from omniviz.gui.panels import (
    BoundaryPanel,
    PatranMeshPanel,
    PointCloudPanel,
    VectorFieldPanel,
    VtkMeshPanel,
    WirePanel,
    categorize_files,
)
from omniviz.gui.theme import CORNER_RADIUS, PAD_X, PAD_Y
from omniviz.models import ViewItem
from omniviz.plotter import UnifiedPlotter

log = logging.getLogger(__name__)


CLIP_DIRECTIONS = ("x", "y", "z", "-x", "-y", "-z")


def _project_root() -> Path:
    """Return the project root (parent of the ``omniviz`` package)."""
    return Path(__file__).resolve().parents[2]


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image as raw PNG bytes for ``tk.PhotoImage(data=…)``."""
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class App(ctk.CTk):
    """Main window."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__()
        self.data_dir = data_dir or _project_root() / "data"
        self.title(f"OmniViz {__version__}")
        self.geometry("1100x780")
        self.minsize(900, 640)

        self._items: list[ViewItem] = []
        self._categories = categorize_files(self.data_dir)
        self._last_import_dir: Path = self.data_dir if self.data_dir.exists() else Path.home()

        self._apply_window_icon()
        self._build_layout()

    # ---------------------------------------------------------------- branding

    def _apply_window_icon(self) -> None:
        """Set the Tk window/taskbar icon from the bundled logo, if available."""
        if not LOGO_PATH.is_file():
            log.debug("Logo not found at %s; skipping window icon", LOGO_PATH)
            return
        try:
            with Image.open(LOGO_PATH) as raw:
                # 64×64 is a sane window-icon size; Tk will rescale as the WM needs.
                icon = raw.convert("RGBA").resize((64, 64), Image.LANCZOS)
                self._icon_image = tk.PhotoImage(
                    master=self,
                    data=_pil_to_png_bytes(icon),
                )
            self.iconphoto(True, self._icon_image)
        except (OSError, tk.TclError) as exc:                  # noqa: BLE001
            log.debug("Could not set window icon: %s", exc)

    # ---------------------------------------------------------------- layout

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_pane()
        self._build_right_pane()
        self._build_status_bar()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=0, height=64,
                              fg_color=("gray92", "gray15"))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(2, weight=1)

        # Logo (falls back to a text-only title if the logo file is missing)
        self._header_logo: ctk.CTkImage | None = None
        if LOGO_PATH.is_file():
            try:
                with Image.open(LOGO_PATH) as raw:
                    pil_img = raw.convert("RGBA")
                self._header_logo = ctk.CTkImage(
                    light_image=pil_img, dark_image=pil_img, size=(48, 48),
                )
                ctk.CTkLabel(header, image=self._header_logo, text="").grid(
                    row=0, column=0, sticky="w", padx=(PAD_X * 2, 6),
                    pady=(PAD_Y // 2, PAD_Y // 2),
                )
            except OSError as exc:                             # noqa: BLE001
                log.debug("Could not load header logo: %s", exc)

        ctk.CTkLabel(
            header,
            text="OmniViz",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=1, sticky="w", padx=(4, PAD_X), pady=PAD_Y)

        ctk.CTkLabel(
            header,
            text="Stellarator & fusion-reactor geometry viewer",
            text_color=("gray35", "gray70"),
        ).grid(row=0, column=2, sticky="w", padx=4, pady=PAD_Y)

        ctk.CTkButton(
            header,
            text="Import file…",
            width=130,
            command=self._import_file,
        ).grid(row=0, column=3, sticky="e", padx=(PAD_X, 4), pady=PAD_Y)

        self._appearance = ctk.CTkSegmentedButton(
            header,
            values=["Dark", "Light", "System"],
            command=self._set_appearance,
        )
        self._appearance.set("Dark")
        self._appearance.grid(row=0, column=4, sticky="e", padx=PAD_X * 2, pady=PAD_Y)

    def _build_left_pane(self) -> None:
        self._left_pane = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self._left_pane.grid(row=1, column=0, sticky="nsew", padx=(PAD_X, 6), pady=PAD_Y)
        self._left_pane.grid_columnconfigure(0, weight=1)
        self._left_pane.grid_rowconfigure(0, weight=1)
        self._populate_tabs()

    def _populate_tabs(self) -> None:
        """(Re)build the tab view from ``self._categories``."""
        for child in self._left_pane.winfo_children():
            child.destroy()

        tabs = ctk.CTkTabview(self._left_pane, corner_radius=CORNER_RADIUS)
        tabs.grid(row=0, column=0, sticky="nsew")

        tab_specs = [
            ("Point Cloud",  PointCloudPanel,  "point_cloud"),
            ("Boundary",     BoundaryPanel,    "boundary"),
            ("VTK",          VtkMeshPanel,     "vtk"),
            ("Patran",       PatranMeshPanel,  "patran"),
            ("Vector Field", VectorFieldPanel, "vector_field"),
            ("Wire",         WirePanel,        None),
        ]
        for name, panel_cls, key in tab_specs:
            tab = tabs.add(name)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            files = self._categories.get(key, []) if key else []
            panel = panel_cls(tab, on_add=self._add_item, files=files)
            panel.grid(row=0, column=0, sticky="nsew")

    def _build_right_pane(self) -> None:
        pane = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        pane.grid(row=1, column=1, sticky="nsew", padx=(6, PAD_X), pady=PAD_Y)
        pane.grid_columnconfigure(0, weight=1)
        pane.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            pane,
            text="Render queue",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 4))

        self._queue_scroll = ctk.CTkScrollableFrame(pane, corner_radius=CORNER_RADIUS)
        self._queue_scroll.grid(row=1, column=0, sticky="nsew", padx=PAD_X, pady=4)
        self._queue_scroll.grid_columnconfigure(0, weight=1)

        self._empty_placeholder = ctk.CTkLabel(
            self._queue_scroll,
            text="No items queued.\nAdd visualizations from the tabs on the left.",
            text_color=("gray40", "gray60"),
            justify="center",
        )
        self._empty_placeholder.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y * 2)

        # --- queue actions
        actions = ctk.CTkFrame(pane, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=PAD_X, pady=4)
        actions.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(actions, text="Clear all",
                      command=self._clear_items).grid(row=0, column=1, sticky="e", padx=4)

        # --- clip plane
        clip = ctk.CTkFrame(pane, corner_radius=CORNER_RADIUS)
        clip.grid(row=3, column=0, sticky="ew", padx=PAD_X, pady=4)
        clip.grid_columnconfigure(1, weight=1)

        self._clip_enabled = ctk.CTkSwitch(clip, text="Enable clip plane")
        self._clip_enabled.grid(row=0, column=0, columnspan=2, sticky="w",
                                padx=PAD_X, pady=(PAD_Y, 4))

        ctk.CTkLabel(clip, text="Direction").grid(row=1, column=0, sticky="w",
                                                  padx=PAD_X, pady=(0, PAD_Y))
        self._clip_dir = ctk.CTkOptionMenu(clip, values=list(CLIP_DIRECTIONS))
        self._clip_dir.set("y")
        self._clip_dir.grid(row=1, column=1, sticky="ew", padx=PAD_X, pady=(0, PAD_Y))

        # --- big render button
        self._render_btn = ctk.CTkButton(
            pane,
            text="Render plot",
            height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._render,
        )
        self._render_btn.grid(row=4, column=0, sticky="ew", padx=PAD_X, pady=(PAD_Y, PAD_X))

    def _build_status_bar(self) -> None:
        self._status = ctk.CTkLabel(
            self, text=self._status_text(),
            anchor="w", text_color=("gray35", "gray70"),
        )
        self._status.grid(row=2, column=0, columnspan=2, sticky="ew",
                          padx=PAD_X * 2, pady=(0, 6))

    # ------------------------------------------------------------ queue ops

    def _add_item(self, item: ViewItem) -> None:
        self._items.append(item)
        self._refresh_queue()

    def _clear_items(self) -> None:
        self._items.clear()
        self._refresh_queue()

    def _remove_item(self, index: int) -> None:
        if 0 <= index < len(self._items):
            del self._items[index]
            self._refresh_queue()

    def _refresh_queue(self) -> None:
        for child in self._queue_scroll.winfo_children():
            child.destroy()

        if not self._items:
            self._empty_placeholder = ctk.CTkLabel(
                self._queue_scroll,
                text="No items queued.\nAdd visualizations from the tabs on the left.",
                text_color=("gray40", "gray60"),
                justify="center",
            )
            self._empty_placeholder.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y * 2)
        else:
            for i, item in enumerate(self._items):
                self._render_queue_card(i, item)

        self._status.configure(text=self._status_text())

    def _render_queue_card(self, index: int, item: ViewItem) -> None:
        card = ctk.CTkFrame(self._queue_scroll, corner_radius=CORNER_RADIUS)
        card.grid(row=index, column=0, sticky="ew", padx=4, pady=4)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card, text=item.kind,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray20", "gray80"),
        ).grid(row=0, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 0))

        ctk.CTkLabel(
            card, text=item.summary(),
            anchor="w", wraplength=320, justify="left",
        ).grid(row=1, column=0, sticky="w", padx=PAD_X, pady=(0, PAD_Y))

        ctk.CTkButton(
            card, text="✕", width=32,
            fg_color="transparent", hover_color=("gray80", "gray25"),
            command=lambda i=index: self._remove_item(i),
        ).grid(row=0, column=1, rowspan=2, padx=PAD_X, pady=PAD_Y)

    def _status_text(self) -> str:
        return f"{len(self._items)} item(s) queued · data dir: {self.data_dir}"

    # -------------------------------------------------------------- import

    _IMPORT_FILETYPES = (
        ("All supported", "*.dat *.txt *.vtk *.msh *.out"),
        ("XYZ / vector field data", "*.dat *.txt"),
        ("VTK mesh", "*.vtk"),
        ("Patran neutral mesh", "*.msh *.out"),
        ("All files", "*.*"),
    )

    def _import_file(self) -> None:
        """Pick a file from anywhere on disk and copy it into ``data/``."""
        src = ask_open_file(
            self,
            title="Choose a file to import into the data folder",
            initialdir=self._last_import_dir,
            filetypes=list(self._IMPORT_FILETYPES),
        )
        if src is None:
            return

        if not src.is_file():
            messagebox.showerror("Import failed", f"Not a regular file:\n{src}")
            return

        # Remember this directory for the next import.
        self._last_import_dir = src.parent

        self.data_dir.mkdir(parents=True, exist_ok=True)
        dst = self.data_dir / src.name

        # Same file? Nothing to do.
        try:
            if dst.exists() and dst.resolve() == src.resolve():
                messagebox.showinfo(
                    "Already in data folder",
                    f"{src.name} is already in {self.data_dir}.",
                )
                return
        except OSError:
            pass

        if dst.exists():
            overwrite = messagebox.askyesno(
                "Overwrite?",
                f"{dst.name} already exists in {self.data_dir}.\n\nReplace it?",
            )
            if not overwrite:
                return

        self._status.configure(text=f"Importing {src.name}…")
        # Tiny files: do a plain copy without the progress dialog.
        try:
            if src.stat().st_size < 64 * 1024:
                shutil.copy2(src, dst)
                self._on_import_done(dst)
                return
        except OSError as exc:
            messagebox.showerror("Import failed", str(exc))
            self._status.configure(text=self._status_text())
            return

        CopyProgressDialog(self, src=src, dst=dst, on_done=self._on_import_done)

    def _on_import_done(self, dst: Path | None) -> None:
        if dst is None:
            self._status.configure(text=self._status_text())
            return
        self._categories = categorize_files(self.data_dir)
        self._populate_tabs()
        self._status.configure(text=f"Imported {dst.name} · {self._status_text()}")

    # -------------------------------------------------------------- render

    def _render(self) -> None:
        if not self._items:
            messagebox.showwarning("Nothing queued", "Add at least one item first.")
            return

        items = list(self._items)
        clip_enabled = bool(self._clip_enabled.get())
        clip_dir = self._clip_dir.get()

        self._render_btn.configure(state="disabled", text="Rendering…")
        self._status.configure(text="Building plot…")

        thread = threading.Thread(
            target=self._render_worker,
            args=(items, clip_enabled, clip_dir),
            daemon=True,
        )
        thread.start()

    def _render_worker(
        self,
        items: Iterable[ViewItem],
        clip_enabled: bool,
        clip_dir: str,
    ) -> None:
        try:
            plotter = UnifiedPlotter(background="white", title=None)
            if clip_enabled:
                plotter.set_clip_plane(clip_dir)
            for item in items:
                item.apply(plotter, self.data_dir)
            self.after(0, self._finalize_render_ready, plotter)
        except Exception as exc:                      # noqa: BLE001
            log.exception("Rendering failed")
            self.after(0, self._finalize_render_error, exc)

    def _finalize_render_ready(self, plotter: UnifiedPlotter) -> None:
        self._render_btn.configure(state="normal", text="Render plot")
        self._status.configure(text="Opening PyVista window…")
        # PyVista's plotter.show() must run on the main thread
        # (the same thread that owns the Tk root), but it blocks until the
        # window is closed.  We schedule it as an idle callback so the GUI
        # can repaint first.
        self.after(50, lambda: self._show_plotter(plotter))

    def _finalize_render_error(self, exc: BaseException) -> None:
        self._render_btn.configure(state="normal", text="Render plot")
        self._status.configure(text=self._status_text())
        messagebox.showerror("Render failed", str(exc))

    def _show_plotter(self, plotter: UnifiedPlotter) -> None:
        try:
            plotter.show(show_axes=True, show_bounds=True, show_legend=True)
        except Exception as exc:                       # noqa: BLE001
            log.exception("PyVista show() failed")
            messagebox.showerror("Render failed", str(exc))
        finally:
            self._status.configure(text=self._status_text())

    # ------------------------------------------------------------ appearance

    @staticmethod
    def _set_appearance(mode: str) -> None:
        ctk.set_appearance_mode(mode.lower())


def run(data_dir: Path | str | None = None) -> None:
    """Configure CustomTkinter and run the OmniViz GUI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    try:
        # Prefer slightly larger fonts where the platform allows it.
        ctk.set_widget_scaling(1.0)
    except (tk.TclError, AttributeError):
        pass

    path = Path(data_dir) if data_dir else None
    App(data_dir=path).mainloop()


if __name__ == "__main__":
    run()
