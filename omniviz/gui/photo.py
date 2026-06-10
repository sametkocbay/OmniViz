"""Screenshot preview & label editor for OmniViz "photo mode".

After a high-resolution shot is captured from the 3-D window, :class:`PhotoEditor`
shows it in a preview, lets the user drop a caption in a corner (bottom-left by
default) with a live preview, and exports a publication-ready PNG/JPEG.
"""

from __future__ import annotations

import logging
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont, ImageTk

from omniviz.gui.theme import CORNER_RADIUS, PAD_X, PAD_Y

log = logging.getLogger(__name__)

# Largest edge of the on-screen preview (the export keeps full resolution).
_PREVIEW_MAX = 760

_CORNERS = {
    "Bottom-left": "bottom-left",
    "Bottom-right": "bottom-right",
    "Top-left": "top-left",
    "Top-right": "top-right",
}

_COLORS = {
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Yellow": (255, 214, 0),
    "Red": (220, 50, 47),
}

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A bold scalable font at ``px`` pixels, cached; falls back gracefully."""
    px = max(8, int(px))
    cached = _font_cache.get(px)
    if cached is not None:
        return cached

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(name, px)
            _font_cache[px] = font
            return font
        except OSError:
            continue
    try:
        import matplotlib.font_manager as fm

        font = ImageFont.truetype(fm.findfont("DejaVu Sans:bold"), px)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    _font_cache[px] = font
    return font


def composite_label(
    base: Image.Image,
    text: str,
    *,
    font_frac: float,
    color: tuple[int, int, int],
    corner: str = "bottom-left",
    box: bool = True,
    pad_frac: float = 0.035,
) -> Image.Image:
    """Return a copy of ``base`` with ``text`` drawn in the given corner.

    Sizes are expressed as fractions of the image height so the preview and the
    full-resolution export look identical regardless of scale.
    """
    if not text.strip():
        return base.convert("RGB")

    img = base.convert("RGBA")
    width, height = img.size
    font = _load_font(int(font_frac * height))
    pad = int(pad_frac * height)

    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = pad - bbox[0] if "left" in corner else width - tw - pad - bbox[0]
    y = height - th - pad - bbox[1] if "bottom" in corner else pad - bbox[1]

    if box:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        mx, my = pad * 0.45, pad * 0.3
        od.rounded_rectangle(
            [x + bbox[0] - mx, y + bbox[1] - my, x + bbox[0] + tw + mx, y + bbox[1] + th + my],
            radius=pad * 0.35,
            fill=(0, 0, 0, 150),
        )
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
    else:
        off = max(1, int(font_frac * height) // 18)
        draw.text((x + off, y + off), text, font=font, fill=(0, 0, 0, 170))

    draw.text((x, y), text, font=font, fill=(*color, 255))
    return img.convert("RGB")


class PhotoEditor(ctk.CTkToplevel):
    """Preview a captured screenshot and export it with an optional caption."""

    def __init__(
        self,
        master,
        image: Image.Image,
        *,
        initial_dir: Path | None = None,
    ) -> None:
        super().__init__(master)
        self._full = image.convert("RGB")
        self._initial_dir = initial_dir or Path.home()

        # Down-scaled working copy for a snappy live preview.
        self._preview_base = self._full.copy()
        self._preview_base.thumbnail((_PREVIEW_MAX, _PREVIEW_MAX), Image.LANCZOS)

        self._photo: ImageTk.PhotoImage | None = None

        self.title("Screenshot — add label & export")
        self.transient(master)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._build_widgets()
        self.after(50, self._grab)
        self._update_preview()

    # ----------------------------------------------------------------- setup

    def _grab(self) -> None:
        try:
            self.grab_set()
            self.focus_force()
        except Exception:  # noqa: BLE001
            pass

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # --- preview canvas
        preview_frame = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(PAD_X, 6), pady=PAD_X)
        self._preview_label = ctk.CTkLabel(preview_frame, text="")
        self._preview_label.pack(padx=PAD_X, pady=PAD_Y)

        # --- controls
        controls = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS, width=260)
        controls.grid(row=0, column=1, sticky="nsew", padx=(6, PAD_X), pady=PAD_X)
        controls.grid_columnconfigure(0, weight=1)

        w, h = self._full.size
        ctk.CTkLabel(
            controls,
            text=f"{w} × {h}px",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 2))

        ctk.CTkLabel(controls, text="Label").grid(
            row=1, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 0)
        )
        self._text_var = ctk.StringVar(value="")
        self._text_var.trace_add("write", lambda *_: self._update_preview())
        ctk.CTkEntry(
            controls,
            textvariable=self._text_var,
            placeholder_text="e.g. (a) W7-X boundary",
        ).grid(row=2, column=0, sticky="ew", padx=PAD_X, pady=(2, PAD_Y))

        ctk.CTkLabel(controls, text="Position").grid(
            row=3, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 0)
        )
        self._corner = ctk.CTkOptionMenu(
            controls, values=list(_CORNERS), command=lambda _: self._update_preview()
        )
        self._corner.set("Bottom-left")
        self._corner.grid(row=4, column=0, sticky="ew", padx=PAD_X, pady=(2, PAD_Y))

        ctk.CTkLabel(controls, text="Text color").grid(
            row=5, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 0)
        )
        self._color = ctk.CTkOptionMenu(
            controls, values=list(_COLORS), command=lambda _: self._update_preview()
        )
        self._color.set("White")
        self._color.grid(row=6, column=0, sticky="ew", padx=PAD_X, pady=(2, PAD_Y))

        self._font_var = ctk.DoubleVar(value=4.5)
        ctk.CTkLabel(controls, text="Font size").grid(
            row=7, column=0, sticky="w", padx=PAD_X, pady=(PAD_Y, 0)
        )
        ctk.CTkSlider(
            controls,
            from_=2.0,
            to=10.0,
            variable=self._font_var,
            command=lambda _: self._update_preview(),
        ).grid(row=8, column=0, sticky="ew", padx=PAD_X, pady=(2, PAD_Y))

        self._box_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            controls,
            text="Caption background",
            variable=self._box_var,
            command=self._update_preview,
        ).grid(row=9, column=0, sticky="w", padx=PAD_X, pady=PAD_Y)

        ctk.CTkButton(
            controls,
            text="Save image…",
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._save,
        ).grid(row=10, column=0, sticky="ew", padx=PAD_X, pady=(PAD_Y * 2, 4))

        ctk.CTkButton(
            controls,
            text="Close",
            fg_color="transparent",
            border_width=1,
            command=self._close,
        ).grid(row=11, column=0, sticky="ew", padx=PAD_X, pady=(4, PAD_Y))

    # --------------------------------------------------------------- preview

    def _current_kwargs(self) -> dict:
        return {
            "font_frac": float(self._font_var.get()) / 100.0,
            "color": _COLORS[self._color.get()],
            "corner": _CORNERS[self._corner.get()],
            "box": bool(self._box_var.get()),
        }

    def _update_preview(self) -> None:
        composed = composite_label(
            self._preview_base, self._text_var.get(), **self._current_kwargs()
        )
        self._photo = ImageTk.PhotoImage(composed)
        self._preview_label.configure(image=self._photo)

    # ----------------------------------------------------------------- save

    def _save(self) -> None:
        default = f"omniviz_{datetime.now():%Y%m%d_%H%M%S}.png"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save figure",
            initialdir=str(self._initial_dir),
            initialfile=default,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("TIFF image", "*.tif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        final = composite_label(self._full, self._text_var.get(), **self._current_kwargs())
        try:
            if path.lower().endswith((".jpg", ".jpeg")):
                final.save(path, quality=95)
            else:
                final.save(path)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return

        log.info("Saved figure to %s", path)
        self._close()

    def _close(self) -> None:
        try:
            self.grab_release()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()
