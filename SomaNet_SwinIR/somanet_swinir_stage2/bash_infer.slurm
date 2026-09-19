#!/bin/bash
#SBATCH --job-name=somanet_infer
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=inference_results/logs/infer_%A_%a.out
#SBATCH --error=inference_results/logs/infer_%A_%a.out

# ---------------------------------------------------------------------------
# SomaNet (SwinIR) inference, one volume per array task.
#   sbatch bash_infer.slurm                 -> the default --array (test 01, 02)
# To also run the training volumes, extend --array to match the table below
# (e.g. --array=0-10 for the 2 test + 9 train volumes) and resubmit.
#
# The (split, vol) table drives everything:
#   test  volumes -> output in inference_results/test_3d_stitched/<vol>/
#   train volumes -> SW_OUT_TAG=_train sends them to test_3d_stitched_train/<vol>/
#                    so they never overwrite the test-set predictions.
# ---------------------------------------------------------------------------
set -o pipefail
STAGE2="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"   # .../SomaNet_SwinIR/somanet_swinir_stage2
cd "$STAGE2"; mkdir -p inference_results/logs

# --- conda env (created from environment.yml); edit the name if yours differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate somanet

# --- data root holding test_sets/ and train_sets/ (defaults to <repo>/data) ---
export SOMANET_DATA="${SOMANET_DATA:-$(cd "$STAGE2/../.." && pwd)/data}"

# (split, vol) task table  -- index 0..1 = test, 2..10 = train
SPLITS=(test_sets test_sets train_sets train_sets train_sets train_sets train_sets train_sets train_sets train_sets train_sets)
VOLS=(  01        02        01         02         03         04         05         06         07         08         09        )
export SOMANET_SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
VOL=${VOLS[$SLURM_ARRAY_TASK_ID]}
# keep train predictions in a separate output dir
[ "$SOMANET_SPLIT" = "train_sets" ] && export SW_OUT_TAG=_train || unset SW_OUT_TAG

echo "[$(date)] host=$(hostname)  inference  split=$SOMANET_SPLIT  vol=$VOL  (task $SLURM_ARRAY_TASK_ID)"
echo "  SOMANET_DATA=$SOMANET_DATA   out tag='${SW_OUT_TAG:-}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python -u ../run_infer.py "$VOL"
echo "[$(date)] INFER_DONE split=$SOMANET_SPLIT vol=$VOL rc=$?"
