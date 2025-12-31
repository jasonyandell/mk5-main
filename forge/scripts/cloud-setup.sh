#!/usr/bin/env bash
set -euo pipefail

# Crystal Forge Cloud Setup
# Usage: ./forge/scripts/cloud-setup.sh
#
# Run this ONCE when you first SSH into a Lambda Labs instance.
# Then run cloud-run.sh to start training.

REPO_URL="https://github.com/jasonyandell/mk5-main.git"
BRANCH="forge"
WORK_DIR="$HOME/crystal-forge"

echo "=== Crystal Forge Cloud Setup ==="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: System info
# ─────────────────────────────────────────────────────────────────────────────
echo "=== System Info ==="
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Clone repo
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Clone Repo ==="
if [ -d "$WORK_DIR" ]; then
    echo "Work dir exists, pulling latest..."
    cd "$WORK_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Cloning $REPO_URL (branch: $BRANCH)..."
    git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi
echo "Repo ready at: $WORK_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Install Dependencies ==="
pip install -q -r forge/requirements.txt
echo "Dependencies installed."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Verify GPU
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Verify GPU ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Wandb login
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Wandb Login ==="
if wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "Already logged in to wandb."
else
    echo "Please log in to wandb:"
    wandb login
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "  cd $WORK_DIR"
echo "  ./forge/scripts/cloud-run.sh"
echo ""
echo "=========================================="
