#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Launch aborted ==="
    echo "No virtualenv found at $VENV_DIR"
    echo "Please run setup first:  ./setup.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "=== Launching Diffucore UI ==="
exec python "$SCRIPT_DIR/app.py"
