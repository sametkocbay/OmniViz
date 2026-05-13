"""Static assets bundled with OmniViz (logo, icons, …)."""

from __future__ import annotations

from pathlib import Path

#: Directory holding bundled assets — usable from both source checkouts
#: and installed wheels because the assets ride along with the package.
ASSETS_DIR: Path = Path(__file__).resolve().parent

LOGO_PATH: Path = ASSETS_DIR / "omniviz_logo.png"
