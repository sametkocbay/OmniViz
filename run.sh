#!/usr/bin/env bash
# Launch the OmniViz GUI.
#
# Order of preference:
#   1. `uv run omniviz`         — fastest, uses uv.lock
#   2. ./.venv/bin/omniviz      — local virtualenv created by `uv sync`/`pip install -e .`
#   3. python -m omniviz        — fall back to the active interpreter

set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    exec uv run omniviz "$@"
elif [[ -x ./.venv/bin/omniviz ]]; then
    exec ./.venv/bin/omniviz "$@"
elif [[ -x ./.venv/bin/python ]]; then
    exec ./.venv/bin/python -m omniviz "$@"
else
    exec python -m omniviz "$@"
fi
