"""Evaluate SomaNet (DINOv2) predictions against the GT mask and save the metrics.

Usage:  python -u run_eval.py 01        (default volume id, for example 01)

Scores somanet_dinov2_stage2/inference_results/test_3d_stitched/<vol>/pred_FINAL.nii.gz
against the volume's GT mask using evaluation.py and writes the
7 metrics (Dice, IoU, PQ, ARAND, VOItotal/split/merge) to pred_FINAL_evaluation.txt
next to the prediction. No GPU needed; run run_infer.py first to produce pred_FINAL.
"""
import os, sys, glob, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.join(HERE, 'somanet_dinov2_stage2')
os.chdir(STAGE2)                               # relative paths resolve against stage2

# ---- data location (must match run_infer.py) -------------------------------
VOL = sys.argv[1].strip() if len(sys.argv) > 1 else '01'
DATA_ROOT = os.environ.get('SOMANET_DATA', os.path.join(os.path.dirname(HERE), 'data'))
SPLIT = os.environ.get('SOMANET_SPLIT', 'test_sets')   # 'test_sets' or 'train_sets'
DATA = os.path.join(DATA_ROOT, SPLIT)         # GT masks live in <split>/<VOL>
# ---------------------------------------------------------------------------

gts = sorted(glob.glob(os.path.join(DATA, VOL, '*_mask.nii.gz')))
pred = os.path.join('inference_results',
                    'test_3d_stitched' + os.environ.get('DINO_OUT_TAG', ''),
                    VOL, 'pred_FINAL.nii.gz')
if not gts:
    sys.exit(f"[eval] no GT mask in {os.path.join(DATA, VOL)}")
if not os.path.isfile(pred):
    sys.exit(f"[eval] no prediction at {pred} -- run 'python run_infer.py {VOL}' first")

metrics = subprocess.run([sys.executable, 'evaluation.py', gts[0], pred],
                         capture_output=True, text=True).stdout
out_txt = os.path.join(os.path.dirname(pred), 'pred_FINAL_evaluation.txt')
with open(out_txt, 'w') as f:
    f.write(metrics)
print(metrics + f"metrics saved -> {out_txt}", flush=True)
