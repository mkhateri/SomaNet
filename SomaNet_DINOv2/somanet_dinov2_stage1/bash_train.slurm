#!/bin/bash
#SBATCH --job-name=somanet_dino_train_s1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=train_s1_%j.out
#SBATCH --error=train_s1_%j.out

# ---------------------------------------------------------------------------
# Stage 1 — initial DINOv2 fine-tune training.
#   sbatch bash_train.slurm
# Reads train_sets/01..09 from the data root; hyperparameters live in
# run_trainer.py (identical to the original run). Uses 1 GPU (SW_NUM_GPUS).
# ---------------------------------------------------------------------------
set -o pipefail
STAGE1="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"   # .../SomaNet_DINOv2/somanet_dinov2_stage1
cd "$STAGE1"

# --- conda env (created from environment.yml); edit the name if yours differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate somanet

# --- DINOv2 backbone cache (see bash_infer.slurm) + vanilla attention ---
export TORCH_HOME="${TORCH_HOME:-$STAGE1/.torch_hub}"
export XFORMERS_DISABLED=1

# --- data root holding train_sets/ (defaults to <repo>/data) ---
export SOMANET_DATA="${SOMANET_DATA:-$(cd "$STAGE1/../.." && pwd)/data}"
# export SW_NUM_GPUS=1      # override GPU count if needed (must match --gres)

echo "[$(date)] host=$(hostname)  DINOv2 Stage-1 training  SOMANET_DATA=$SOMANET_DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python -u run_trainer.py
echo "[$(date)] TRAIN_S1_DONE rc=$?"
