#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== Diffucore UI Setup ==="

# --- submodule ---
if [ ! -f "$SCRIPT_DIR/diffucore/src/diffucore/__init__.py" ]; then
    echo "[1/4] Initializing submodules..."
    git -C "$SCRIPT_DIR" submodule update --init --recursive
else
    echo "[1/4] Submodule already present."
fi

# --- venv ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "[2/4] Virtualenv already exists."
fi

source "$VENV_DIR/bin/activate"

# --- pip deps ---
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip -q
# Install the cu124 torch build first so requirements.txt / ultralytics don't
# pull the default PyPI wheel (built for a newer CUDA than many drivers run).
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q -r "$SCRIPT_DIR/requirements.txt"
pip install -q -e "$SCRIPT_DIR/diffucore"

# --- CUDA torch sanity check ---
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && \
    echo "[4/4] CUDA torch OK." || \
    echo "[4/4] WARNING: torch present but CUDA unavailable (check NVIDIA driver vs cu124)."

echo ""
echo "=== Setup complete ==="
echo "Activate the venv:  source $VENV_DIR/bin/activate"
echo "Run the UI:         python backend/app.py"
