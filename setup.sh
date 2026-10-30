#!/usr/bin/env bash
# Runs once on the machine (and in CI): creates .venv from uv.lock.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# uv downloads the pinned Python (.python-version) if the machine lacks it.
uv sync --frozen --no-dev
