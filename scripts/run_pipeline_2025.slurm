#!/bin/bash
# =============================================================================
# Slurm script: 2025 Swing Decision Pipeline
#
# Steps:
#   1. Build swing decision features with LSTM perception at 40% flight
#   2. Train out-of-fold XGBoost model & compute Bayesian DVA ratings
#   3. Export 2025 DVA leaderboard to CSV
#   4. Generate top/bottom 10 leaderboard tables PNG
# =============================================================================
#SBATCH --job-name=pipeline_2025
#SBATCH --output=/mnt/scratch/user/chrsong/strike-zone-perception/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/strike-zone-perception/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --account=hpc_core
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="/mnt/scratch/user/chrsong/strike-zone-perception"
cd "${PROJECT_ROOT}"

ENV_PREFIX="/mnt/scratch/user/chrsong/envs/strike-perception"
export PATH="${ENV_PREFIX}/bin:${PATH}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=========================================================="
echo "  Strike Zone Perception: 2025 Pipeline"
echo "=========================================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-16}"
echo "Python:        $(which python3)"
echo "Start time:    $(date)"
echo "=========================================================="
echo ""

# 1. Build swing features
echo ">>> [1/4] Building swing features for 2025..."
python3 -u scripts/build_swing_features.py 2025 --device cuda --decision-frac 0.40

# 2. Train swing model and compute Bayesian DVA
echo ""
echo ">>> [2/4] Training XGBoost baseline & computing Bayesian DVA ratings..."
python3 -u scripts/train_swing_model.py --year 2025 --kappa 30 --oof-splits 5

# 3. Export leaderboard
echo ""
echo ">>> [3/4] Exporting leaderboard to CSV..."
python3 -u scripts/export_leaderboard.py --year 2025

# 4. Generate leaderboard images
echo ""
echo ">>> [4/4] Generating leaderboard table images..."
python3 -u analysis/generate_leaderboard_image.py --year 2025

echo ""
echo "=========================================================="
echo "All steps completed successfully at: $(date)"
echo "=========================================================="
