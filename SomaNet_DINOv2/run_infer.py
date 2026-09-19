"""Run SomaNet (DINOv2) inference on a test or (partially annotated) training volume.

Usage:  python -u run_infer.py 01        (default volume id, for example 01)
"""

import os, sys
# xFormers here is not built with CUDA; disable it so the DINOv2 backbone falls
# back to vanilla (numerically-equivalent) attention. Must be set before import.
os.environ.setdefault('XFORMERS_DISABLED', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.join(HERE, 'somanet_dinov2_stage2')
sys.path.insert(0, STAGE2)                     
os.chdir(STAGE2)                               
from scripts.inference_2dstack_to_3d import Inference3DStitcher
from scripts.infer_config import build_config

# ---- data location -------------------------------------------------------
# Data root holds  train_sets/  and  test_sets/  (see preprocessing/README.md).
# Default: the shared  data/  folder at the repo root. Put your data there, or
# point SOMANET_DATA at it:   export SOMANET_DATA=/path/to/data_root
DATA_ROOT = os.environ.get('SOMANET_DATA', os.path.join(os.path.dirname(HERE), 'data'))

# which split to run: 'test_sets' (default) or 'train_sets'  -> reads <root>/<split>/<VOL>
SPLIT = os.environ.get('SOMANET_SPLIT', 'test_sets')
DATA = os.path.join(DATA_ROOT, SPLIT)

# which volume (sub-folder) to run:
VOL = sys.argv[1].strip() if len(sys.argv) > 1 else '01'

#=========================================================================
if __name__ == '__main__':
    print(f"2D DINOv2 inference + 2D->3D stitch (vol {VOL}).", flush=True)
    Inference3DStitcher(build_config(VOL, DATA)).infer_3d()
    # To score against GT (test sets): python run_eval.py <vol>
