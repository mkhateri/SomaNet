"""Initial SwinIR training

Run from this dir:  python -u run_trainer.py

Env overrides (all optional; defaults reproduce the original run):
  SOMANET_DATA   data root holding train_sets/01..09
                 (default: the shared  data/  folder at the repo root)
  SW_NUM_GPUS    number of GPUs           (default: 2)
  SW_EPOCHS      number of epochs         (default: 250)
  SW_RESUME      '1' to resume from checkpoint_dir (default: '0' -> fresh)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # make `scripts`, `model`, `utils` importable
from scripts.trainer import Trainer

# Data root (SOMANET_DATA). Expected layout — the loader reads the img/ + mask/
# PNG slices; the .nii.gz volumes are the sources they were generated from:
#
#   <SOMANET_DATA>/
#   ├── train_sets/                 # N training volumes -> 'train_dir'
#   │   ├── 01/
#   │   │   ├── <stem>_raw.nii.gz    # source raw volume
#   │   │   ├── <stem>_mask.nii.gz   # source instance-label volume
#   │   │   ├── img/  <stem>_<z>.png # per-slice raw  (normalized 0-255)
#   │   │   └── mask/ <stem>_<z>.png # per-slice mask (labels 1..N), same names as img/
#   │   ├── 02/ ... └── 09/
#   └── (each set dir has the two .nii.gz + img/ + mask/)
#
# PNGs are produced by preprocessing/nii_to_png.py. Default: the shared  data/
# folder at the repo root; point SOMANET_DATA at a local copy to train elsewhere.
DATA = os.environ.get('SOMANET_DATA', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data'))

config_dict = {
    # MODE
    'mode': 'train',

    # DATA
    'train_dir': [f'{DATA}/train_sets/{i:02d}' for i in range(1, 10)],   # 01..09
    'data_mode': {'img': True, 'mask': True, 'prompt': False},
    'img_format': ['png', 'jpg', 'jpeg'],
    'task_mode': 'SEG',
    'val_split': 0.3,
    'batch_size': 4,
    'num_workers': 2,
    'random_seed': 120,
    'pin_memory': True,
    'shuffle': True,
    'drop_last': True,
    'mask_nonzero_ratio_threshold': 0.10,   # the least non-zero ratio in cropped masks

    # MODEL (Option A: predefined SwinIR, built-in decoder)
    'model': 'SwinIR',
    'in_channels': 1,       # 1 for grayscale, 3 for RGB
    'num_class': 2,         # number of segmentation classes
    'emb_dim': 32,          # embedding dim for affinity learning

    # TRAIN
    'resume': os.environ.get('SW_RESUME', '0') == '1',
    'checkpoint_type': 'latest',
    'checkpoint_dir': './output_training/checkpoints',
    'pretrained_model_dir': './output_training/checkpoints',
    'epochs': int(os.environ.get('SW_EPOCHS', '250')),
    'num_gpus': int(os.environ.get('SW_NUM_GPUS', '2')),
    'single_precision': False,
    'optimizer_name': 'adam',
    'lr_base': 5e-4, 'lr_mode': 'steplr', 'step_size': 10000, 'gamma': 0.9,
    'data_precision': 'bfloat16',

    'losses': {'CrossEntropyLoss'},
    'loss_weights': {'CrossEntropyLoss': 1.0},
    'log_dir': './output_training/logs',

    # Total Loss: embedding_loss + mask_loss
    'embedding_weight': 0.1, 'mask_weight': 1.0,

    'save_image_frequency': 1, 'save_checkpoint_frequency': 250,

    # AFFINITY PARAMETERS
    'separate_weight_affinity': True,
    'shifts_affinity': [1, 3, 5, 9, 11, 19, 27, 35], 'neighbor_affinity': 8,

    # AUGMENTATION
    'crop_size': 400, 'padding_size': 0, 'cuting_corner_size': 7,

    # LOGGING
    'log_gpu_utilization': True, 'print_config': True,
}

if __name__ == '__main__':
    print(f"Initial SwinIR training (data root: {DATA}).", flush=True)
    Trainer(config_dict).train()
