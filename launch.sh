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

# Keep deps in sync with requirements.txt. Catches the common case of a user
# updating with `git pull` (instead of ./update.sh) and then hitting an import
# error for a newly added dependency. Hash-gated, so it's a no-op on every
# launch where requirements.txt hasn't changed.
REQ_FILE="$SCRIPT_DIR/requirements.txt"
STAMP="$VENV_DIR/.requirements.sha256"
REQ_HASH="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$REQ_HASH" ]; then
    echo "requirements.txt changed — syncing dependencies..."
    pip install -q -r "$REQ_FILE"
    # A new dep (e.g. spandrel) can pull torchvision from PyPI and clobber the
    # cu124 torch; repair from the cu124 index if CUDA broke.
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "Repairing CUDA torch..."
        pip uninstall -y -q torch torchvision
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
    fi
    echo "$REQ_HASH" > "$STAMP"
fi

echo "=== Launching Diffucore UI ==="
exec python "$SCRIPT_DIR/backend/app.py" --autolaunch "$@"
