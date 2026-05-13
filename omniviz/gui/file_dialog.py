"""Themed file-browser dialog for OmniViz.

CustomTkinter does not ship a file picker, and on Linux the stock
``tkinter.filedialog`` renders unreadably when paired with a dark theme —
folder labels are painted in the foreground colour on a same-colour
background and the only working click target is the tiny expand glyph.
This module provides a small, fully-themed replacement.
"""

from __future__ import annotations

import fnmatch
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path

import customtkinter as ctk

from omniviz.gui.theme import CORNER_RADIUS, PAD_X, PAD_Y

#: A file-type filter is a ``(label, "pattern1 pattern2 …")`` tuple, matching
#: the shape used by ``tkinter.filedialog``.
FileType = tuple[str, str]


def ask_open_file(
    master,
    *,
    title: str = "Open file",
    initialdir: Path | str | None = None,
    filetypes: Sequence[FileType] = (("All files", "*.*"),),
) -> Path | None:
    """Open a modal themed file picker and return the chosen path (or ``None``)."""
    dlg = FileBrowserDialog(master, title=title, initialdir=initialdir, filetypes=filetypes)
    return dlg.show()


class FileBrowserDialog(ctk.CTkToplevel):
    """Modal directory/file browser, fully themed under CustomTkinter."""

    def __init__(
        self,
        master,
        *,
        title: str,
        initialdir: Path | str | None,
        filetypes: Sequence[FileType],
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("820x560")
        self.minsize(560, 380)

        self._result: Path | None = None
        self._cwd: Path = self._resolve_initial(initialdir)
        self._filetypes: list[FileType] = list(filetypes) or [("All files", "*.*")]
        self._filter: FileType = self._filetypes[0]
        self._selected: Path | None = None
        self._entries: list[tuple[Path, ctk.CTkButton]] = []
        self._show_hidden = tk.BooleanVar(value=False)

        self.transient(master)
        self.after(50, self._safe_grab)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _e: self._on_cancel())

        self._build_widgets()
        self._populate()

    # ------------------------------------------------------------------ API

    def show(self) -> Path | None:
        self.wait_window()
        return self._result

    # ----------------------------------------------------------------- init

    @staticmethod
    def _resolve_initial(initialdir: Path | str | None) -> Path:
        if initialdir:
            try:
                p = Path(initialdir).expanduser().resolve()
                if p.is_dir():
                    return p
            except OSError:
                pass
        return Path.home()

    def _safe_grab(self) -> None:
        try:
            self.grab_set()
        except tk.TclError:                                    # window not viewable yet
            self.after(50, self._safe_grab)

    # ------------------------------------------------------------ build UI

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # -- top bar: Up + editable path + Go
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=PAD_X, pady=(PAD_X, 4))
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(top, text="↑ Up", width=70,
                      command=self._go_up).grid(row=0, column=0, padx=(0, PAD_X))

        self._path_var = tk.StringVar(value=str(self._cwd))
        self._path_entry = ctk.CTkEntry(top, textvariable=self._path_var)
        self._path_entry.grid(row=0, column=1, sticky="ew")
        self._path_entry.bind("<Return>", lambda _e: self._go_to_path())

        ctk.CTkButton(top, text="Go", width=60,
                      command=self._go_to_path).grid(row=0, column=2, padx=(PAD_X, 0))

        # -- shortcut bar
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", padx=PAD_X, pady=(0, 4))
        ctk.CTkButton(nav, text="Home", width=70,
                      command=lambda: self._navigate(Path.home())).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkButton(nav, text="Root  /", width=70,
                      command=lambda: self._navigate(Path("/"))).grid(row=0, column=1, padx=4)
        ctk.CTkCheckBox(
            nav, text="Show hidden", variable=self._show_hidden,
            command=self._populate,
        ).grid(row=0, column=2, padx=(PAD_X, 0))

        # -- file list
        self._list = ctk.CTkScrollableFrame(self, corner_radius=CORNER_RADIUS)
        self._list.grid(row=2, column=0, sticky="nsew", padx=PAD_X, pady=4)
        self._list.grid_columnconfigure(0, weight=1)

        # -- bottom bar: filter + selected name + Cancel/Open
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=PAD_X, pady=(4, PAD_X))
        bottom.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bottom, text="Filter").grid(row=0, column=0, padx=(0, 6), sticky="w")
        self._filter_menu = ctk.CTkOptionMenu(
            bottom,
            values=[label for label, _ in self._filetypes],
            command=self._on_filter_change,
        )
        self._filter_menu.set(self._filter[0])
        self._filter_menu.grid(row=0, column=1, sticky="ew")

        ctk.CTkButton(
            bottom, text="Cancel", width=100,
            fg_color="transparent", border_width=1,
            command=self._on_cancel,
        ).grid(row=0, column=2, padx=(PAD_X, 4))

        self._open_btn = ctk.CTkButton(
            bottom, text="Open", width=100, command=self._on_open, state="disabled",
        )
        self._open_btn.grid(row=0, column=3)

        # selected-file row
        self._selected_label = ctk.CTkLabel(
            bottom, text="No file selected.",
            text_color=("gray35", "gray70"), anchor="w",
        )
        self._selected_label.grid(row=1, column=0, columnspan=4,
                                  sticky="ew", padx=2, pady=(6, 0))

    # ------------------------------------------------------------ list view

    def _populate(self) -> None:
        for _, btn in self._entries:
            btn.destroy()
        self._entries.clear()
        self._selected = None
        self._open_btn.configure(state="disabled")
        self._selected_label.configure(text="No file selected.")
        self._path_var.set(str(self._cwd))

        try:
            children = list(self._cwd.iterdir())
        except (PermissionError, OSError) as exc:
            ctk.CTkLabel(
                self._list, text=f"(cannot list directory: {exc})",
                text_color=("gray35", "gray70"),
            ).grid(row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
            return

        children.sort(key=lambda p: (not self._is_dir(p), p.name.lower()))
        patterns = self._filter[1].split()
        show_hidden = bool(self._show_hidden.get())

        row = 0
        for p in children:
            if not show_hidden and p.name.startswith("."):
                continue
            is_dir = self._is_dir(p)
            if not is_dir and not self._matches(p.name, patterns):
                continue

            icon = "📁 " if is_dir else "📄 "
            btn = ctk.CTkButton(
                self._list, text=f"{icon} {p.name}", anchor="w",
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray85", "gray25"),
                command=lambda pp=p: self._on_click(pp),
            )
            btn.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
            btn.bind("<Double-Button-1>", lambda _e, pp=p: self._on_double_click(pp))
            self._entries.append((p, btn))
            row += 1

        if row == 0:
            ctk.CTkLabel(
                self._list, text="(empty)",
                text_color=("gray35", "gray70"),
            ).grid(row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")

    @staticmethod
    def _is_dir(p: Path) -> bool:
        try:
            return p.is_dir()
        except OSError:
            return False

    @staticmethod
    def _matches(name: str, patterns: list[str]) -> bool:
        if not patterns or "*.*" in patterns or "*" in patterns:
            return True
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)

    # ------------------------------------------------------------ actions

    def _navigate(self, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if resolved.is_dir():
            self._cwd = resolved
            self._populate()

    def _go_up(self) -> None:
        if self._cwd.parent != self._cwd:
            self._navigate(self._cwd.parent)

    def _go_to_path(self) -> None:
        target = Path(self._path_var.get()).expanduser()
        if target.is_dir():
            self._navigate(target)
        elif target.is_file():
            self._result = target.resolve()
            self.destroy()

    def _on_filter_change(self, label: str) -> None:
        for ft in self._filetypes:
            if ft[0] == label:
                self._filter = ft
                break
        self._populate()

    def _on_click(self, p: Path) -> None:
        if self._is_dir(p):
            self._navigate(p)
        else:
            self._select(p)

    def _on_double_click(self, p: Path) -> None:
        if self._is_dir(p):
            self._navigate(p)
        else:
            self._select(p)
            self._on_open()

    def _select(self, p: Path) -> None:
        self._selected = p
        self._open_btn.configure(state="normal")
        self._selected_label.configure(text=f"Selected: {p.name}")
        for path, btn in self._entries:
            btn.configure(
                fg_color=("gray80", "gray30") if path == p else "transparent",
            )

    def _on_open(self) -> None:
        if self._selected and self._selected.is_file():
            self._result = self._selected.resolve()
            self.destroy()

    def _on_cancel(self) -> None:
        self._result = None
        self.destroy()
