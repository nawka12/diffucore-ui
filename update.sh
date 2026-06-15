#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== Diffucore UI Update ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "No virtualenv found at $VENV_DIR"
    echo "Please run setup first:  ./setup.sh"
    exit 1
fi

# --- pull latest UI code ---
echo "[1/4] Pulling latest changes..."
git -C "$SCRIPT_DIR" pull --ff-only

# --- sync submodule to the pinned revision ---
echo "[2/4] Updating submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

source "$VENV_DIR/bin/activate"

# --- refresh deps (requirements and the editable engine may have changed) ---
echo "[3/4] Updating Python dependencies..."
pip install --upgrade pip -q
pip install -q -r "$SCRIPT_DIR/requirements.txt"
pip install -q -e "$SCRIPT_DIR/diffucore"

# --- ensure CUDA torch is still present ---
# On failure, uninstall first: a bare `pip install torch` is a no-op when a
# mismatched wheel is already installed, so it could never repair a CPU build.
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && \
    echo "[4/4] CUDA torch OK." || \
    { echo "[4/4] Reinstalling CUDA torch..."; pip uninstall -y -q torch torchvision; pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124; }

echo ""
echo "=== Update complete ==="
echo "Relaunch the UI:  ./launch.sh"
