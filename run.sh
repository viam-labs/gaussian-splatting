#!/usr/bin/env bash
# Module entrypoint; viam-server passes the socket path as $1.
# Only builds the venv when none exists, so a local dev venv (with dev deps) is left alone.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    ./setup.sh >&2
fi
exec .venv/bin/python src/main.py "$@"
