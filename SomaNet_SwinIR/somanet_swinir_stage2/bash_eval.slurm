#!/bin/bash
#SBATCH --job-name=somanet_eval
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=inference_results/logs/eval_%A_%a.out
#SBATCH --error=inference_results/logs/eval_%A_%a.out

# ---------------------------------------------------------------------------
# Score SomaNet predictions against the GT mask (CPU only). Run AFTER inference.
#   sbatch bash_eval.slurm                  -> the default --array (test 01, 02)
# Uses the SAME (split, vol) table as bash_infer.slurm, so extend --array the
# same way to also score the training volumes. run_eval.py writes the 7 metrics
# to pred_FINAL_evaluation.txt next to each prediction.
# ---------------------------------------------------------------------------
set -o pipefail
STAGE2="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"   # .../SomaNet_SwinIR/somanet_swinir_stage2
cd "$STAGE2"; mkdir -p inference_results/logs

# --- conda env (created from environment.yml); edit the name if yours differs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate somanet

# --- data root holding test_sets/ and train_sets/ (defaults to <repo>/data) ---
export SOMANET_DATA="${SOMANET_DATA:-$(cd "$STAGE2/../.." && pwd)/data}"

# (split, vol) task table  -- must match bash_infer.slurm
SPLITS=(test_sets test_sets train_sets train_sets train_sets train_sets train_sets train_sets train_sets train_sets train_sets)
VOLS=(  01        02        01         02         03         04         05         06         07         08         09        )
export SOMANET_SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
VOL=${VOLS[$SLURM_ARRAY_TASK_ID]}
[ "$SOMANET_SPLIT" = "train_sets" ] && export SW_OUT_TAG=_train || unset SW_OUT_TAG

echo "[$(date)] host=$(hostname)  eval  split=$SOMANET_SPLIT  vol=$VOL  (task $SLURM_ARRAY_TASK_ID)"
python -u ../run_eval.py "$VOL"
echo "[$(date)] EVAL_DONE split=$SOMANET_SPLIT vol=$VOL rc=$?"
