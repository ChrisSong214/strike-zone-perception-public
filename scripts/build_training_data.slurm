#!/bin/bash
# =============================================================================
# Slurm job script to generate training trajectory datasets (.pt)
#
# Partition: cpu (runs on gcpu2-* nodes)
# Years: 2020-2025 (processed across parallel workers)
#
# Submit:
#   sbatch scripts/build_training_data.slurm
#
# Or run with custom years:
#   sbatch --export=ALL,YEARS="2020-2025" scripts/build_training_data.slurm
# =============================================================================
#SBATCH --job-name=build_training_data
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

# Activate strike-perception conda environment
ENV_PREFIX="/mnt/scratch/user/chrsong/envs/strike-perception"
export PATH="${ENV_PREFIX}/bin:${PATH}"

YEARS="${YEARS:-2020-2025}"
WORKERS="${WORKERS:-6}"

echo "=========================================================="
echo "  Strike Zone Perception: Build Training Data"
echo "=========================================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-16}"
echo "Years:         ${YEARS}"
echo "Workers:       ${WORKERS}"
echo "Python:        $(which python3)"
echo "Start time:    $(date)"
echo "=========================================================="
echo ""

mkdir -p logs data/training

python3 -u scripts/build_training_data.py --years "${YEARS}" --workers "${WORKERS}"

echo ""
echo "=========================================================="
echo "Finished at: $(date)"
echo "=========================================================="
