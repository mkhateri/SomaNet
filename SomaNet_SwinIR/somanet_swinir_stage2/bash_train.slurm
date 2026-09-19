#!/bin/bash
#SBATCH --job-name=somanet_train_s2
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=train_s2_%j.out
#SBATCH --error=train_s2_%j.out

# ---------------------------------------------------------------------------
# Stage 2 — teacher-student SwinIR training (SomaNet).
#   sbatch bash_train.slurm
# run_trainer.py first auto-copies the Stage-1 checkpoint into
# pretrained_model/{teacher,student}/ (train Stage 1 first, or set SW_STAGE1_CKPT),
# then trains. Reads train_sets/01..09 from the data root. Uses 2 GPUs.
# ---------------------------------------------------------------------------
set -o pipefail
STAGE2="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"   # .../SomaNet_SwinIR/somanet_swinir_stage2
cd "$STAGE2"

# --- conda env (created from environment.yml); edit the name if yours differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate somanet

# --- data root holding train_sets/ (defaults to <repo>/data) ---
export SOMANET_DATA="${SOMANET_DATA:-$(cd "$STAGE2/../.." && pwd)/data}"
# export SW_STAGE1_CKPT=/path/to/stage1/checkpoint_best.pth   # override Stage-1 source
# export SW_NUM_GPUS=2      # override GPU count if needed (must match --gres)

echo "[$(date)] host=$(hostname)  Stage-2 training  SOMANET_DATA=$SOMANET_DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python -u run_trainer.py
echo "[$(date)] TRAIN_S2_DONE rc=$?"
