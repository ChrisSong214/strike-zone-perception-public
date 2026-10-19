#!/bin/bash
# =============================================================================
# Slurm script: Strike Zone Perception Full Pipeline (2020-2025)
#
# Steps:
#   1. Build swing decision features for all years 2020-2025 (40% decision point)
#   2. Train out-of-fold XGBoost baseline & compute Bayesian DVA per season
#   3. Export DVA leaderboards to CSV (data/leaderboards/ & root)
#   4. Generate top/bottom 10 leaderboard images for each season
# =============================================================================
#SBATCH --job-name=pipeline_all_years
#SBATCH --output=/mnt/scratch/user/chrsong/strike-zone-perception/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/strike-zone-perception/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --account=hpc_core
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="/mnt/scratch/user/chrsong/strike-zone-perception"
cd "${PROJECT_ROOT}"

ENV_PREFIX="/mnt/scratch/user/chrsong/envs/strike-perception"
export PATH="${ENV_PREFIX}/bin:${PATH}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p logs data/training data/leaderboards assets

echo "=========================================================="
echo "  Strike Zone Perception: Full Pipeline (2020-2025)"
echo "=========================================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-16}"
echo "Python:        $(which python3)"
echo "Start time:    $(date)"
echo "=========================================================="
echo ""

YEARS=(2020 2021 2022 2023 2024 2025)

# --- Step 1: Build Swing Features ---
echo "=========================================================="
echo ">>> STEP 1: Building swing features for 2020-2025..."
echo "=========================================================="
python3 -u scripts/build_swing_features.py --years 2020-2025 --device cuda --decision-frac 0.40

# --- Steps 2, 3, 4: Model Training, Leaderboard Export & Image Generation ---
for y in "${YEARS[@]}"; do
    echo ""
    echo "=========================================================="
    echo ">>> Processing Season ${y}"
    echo "=========================================================="
    
    echo ">>> [2/4] Training XGBoost baseline & computing Bayesian DVA ratings (${y})..."
    python3 -u scripts/train_swing_model.py --year "${y}" --kappa 30 --oof-splits 5
    
    echo ">>> [3/4] Exporting leaderboard to CSV (${y})..."
    python3 -u scripts/export_leaderboard.py --year "${y}" --output "data/leaderboards/${y}_dva_leaderboard.csv"
    cp "data/leaderboards/${y}_dva_leaderboard.csv" "${y}_dva_leaderboard.csv"
    
    echo ">>> [4/4] Generating leaderboard table images (${y})..."
    python3 -u analysis/generate_leaderboard_image.py --year "${y}"
done

echo ""
echo "=========================================================="
echo "All years (2020-2025) successfully completed at: $(date)"
echo "=========================================================="
