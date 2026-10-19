#!/bin/bash
# Train comparison models and report results
#
# 1. Train on 2025 only (20 epochs)
# 2. Train on 2020-2025 (20 epochs)
# Save to separate model files for comparison
#
# Usage: source .venv/bin/activate && bash scripts/train_comparison.sh

set -e

echo "=============================================="
echo "  Model Comparison Training"
echo "=============================================="
echo ""

DEVICE=${1:-cuda}

# --- Run 1: 2025 only ---
echo ">>> RUN 1: Training on 2025 only (device: $DEVICE)"
python3 -u scripts/train_model.py \
    --years 2025 \
    --epochs 20 \
    --batch-size 512 \
    --device "$DEVICE" \
    --patience 5

# Save the 2025 model
cp data/models/best_model.pt data/models/model_2025_only.pt
echo ">>> Saved: data/models/model_2025_only.pt"
echo ""

# --- Run 2: 2020-2025 ---
echo ">>> RUN 2: Training on 2020-2025 (device: $DEVICE)"
python3 -u scripts/train_model.py \
    --years 2020-2025 \
    --epochs 20 \
    --batch-size 512 \
    --device "$DEVICE" \
    --patience 5

# Save the all-years model
cp data/models/best_model.pt data/models/model_2020_2025.pt
echo ">>> Saved: data/models/model_2020_2025.pt"
echo ""

echo "=============================================="
echo "  Comparison Complete"
echo "=============================================="
echo "Files:"
echo "  data/models/model_2025_only.pt"
echo "  data/models/model_2020_2025.pt"
echo ""
echo "Review epoch logs above to compare:"
echo "  - Final val loss"
echo "  - R² values"
echo "  - σ_early vs σ_final"
echo "  - σ ratio (cloud shrinkage)"
