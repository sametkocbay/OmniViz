"""Theme tokens and palettes for the OmniViz GUI."""

from __future__ import annotations

# Color palettes for selectable plot tints (used in dropdowns)
COLOR_PALETTE: tuple[str, ...] = (
    "red", "orange", "yellow", "green", "cyan", "blue",
    "violet", "purple", "magenta", "white", "gray", "black",
    "lightblue", "crimson",
)

COLORMAPS: tuple[str, ...] = (
    "plasma", "viridis", "inferno", "magma", "cividis",
    "jet", "hot", "coolwarm", "rainbow", "turbo",
)

# Geometric constants
CORNER_RADIUS = 10
PAD_X = 12
PAD_Y = 8

# Accent color used for primary action button (matches CTk blue theme)
ACCENT = "#1f6aa5"
ACCENT_HOVER = "#144870"
SUCCESS = "#2fa572"
SUCCESS_HOVER = "#207a55"
