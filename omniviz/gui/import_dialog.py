"""Modal dialog that copies a file into ``data/`` with a live progress bar.

The copy runs on a worker thread so the GUI stays responsive; progress
updates are marshalled back to the Tk main loop with ``after()``.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from omniviz.gui.theme import CORNER_RADIUS, PAD_X, PAD_Y

_CHUNK_BYTES = 1 << 20  # 1 MiB


def _format_bytes(n: int) -> str:
    """Human-readable byte count (binary units)."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


class CopyProgressDialog(ctk.CTkToplevel):
    """Show a determinate progress bar while a file is copied.

    The dialog manages its own worker thread. The optional ``on_done``
    callback is invoked from the Tk main thread when the copy finishes
    (success or failure); it receives the destination ``Path`` on success
    or ``None`` if the copy was cancelled / errored.
    """

    def __init__(
        self,
        master,
        src: Path,
        dst: Path,
        on_done: Callable[[Path | None], None] | None = None,
    ) -> None:
        super().__init__(master)
        self._src = src
        self._dst = dst
        self._on_done = on_done
        self._total = max(src.stat().st_size, 1)
        self._cancel = threading.Event()
        self._error: BaseException | None = None
        self._start = time.monotonic()

        self.title("Importing file")
        self.geometry("440x180")
        self.resizable(False, False)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._request_cancel)
        self.after(50, lambda: self.grab_set())

        self._build_widgets()
        threading.Thread(target=self._copy, daemon=True).start()

    # ----------------------------------------------------------------- UI

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text=f"Copying  {self._src.name}",
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="ew", padx=PAD_X * 2, pady=(PAD_Y * 2, 4))

        self._sub = ctk.CTkLabel(
            self,
            text=f"0 / {_format_bytes(self._total)}",
            anchor="w",
            text_color=("gray35", "gray70"),
        )
        self._sub.grid(row=1, column=0, sticky="ew", padx=PAD_X * 2, pady=(0, PAD_Y))

        self._bar = ctk.CTkProgressBar(self, mode="determinate",
                                       corner_radius=CORNER_RADIUS)
        self._bar.set(0.0)
        self._bar.grid(row=2, column=0, sticky="ew", padx=PAD_X * 2, pady=4)

        self._cancel_btn = ctk.CTkButton(
            self, text="Cancel", width=100,
            fg_color="transparent", border_width=1,
            command=self._request_cancel,
        )
        self._cancel_btn.grid(row=3, column=0, sticky="e",
                              padx=PAD_X * 2, pady=PAD_Y * 2)

    # -------------------------------------------------------- worker side

    def _copy(self) -> None:
        copied = 0
        try:
            with open(self._src, "rb") as f_in, open(self._dst, "wb") as f_out:
                while True:
                    if self._cancel.is_set():
                        break
                    chunk = f_in.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    copied += len(chunk)
                    self.after(0, self._update, copied)
            if self._cancel.is_set():
                self._dst.unlink(missing_ok=True)
            else:
                # Preserve mtime/permissions, matching shutil.copy2 semantics.
                shutil.copystat(self._src, self._dst, follow_symlinks=False)
        except BaseException as exc:                          # noqa: BLE001
            self._error = exc
            self._dst.unlink(missing_ok=True)
        finally:
            self.after(0, self._finalize)

    # ---------------------------------------------------------- main side

    def _update(self, copied: int) -> None:
        frac = min(copied / self._total, 1.0)
        self._bar.set(frac)
        elapsed = max(time.monotonic() - self._start, 1e-3)
        speed = copied / elapsed
        self._sub.configure(
            text=(
                f"{_format_bytes(copied)} / {_format_bytes(self._total)}  "
                f"·  {_format_bytes(int(speed))}/s"
            )
        )

    def _request_cancel(self) -> None:
        self._cancel.set()
        self._cancel_btn.configure(text="Cancelling…", state="disabled")

    def _finalize(self) -> None:
        try:
            self.grab_release()
        except Exception:                                     # noqa: BLE001
            pass
        success = (not self._cancel.is_set()) and self._error is None
        result = self._dst if success else None
        callback = self._on_done
        error = self._error
        self.destroy()
        if error is not None:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Import failed", f"{self._src.name}: {error}")
        if callback:
            callback(result)
