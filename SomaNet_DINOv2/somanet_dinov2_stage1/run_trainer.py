"""Initial DINOv2 (finetune)

Run from this dir:  python -u run_trainer.py

Env overrides (all optional; defaults reproduce the original run):
  SOMANET_DATA   data root holding train_sets/01..09
                 (default: the shared  data/  folder at the repo root)
  SW_NUM_GPUS    number of GPUs           (default: 1)
  SW_EPOCHS      number of epochs         (default: 250)
  SW_RESUME      '1' to resume from checkpoint_dir (default: '0' -> fresh)
"""

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # make `scripts`, `model`, `utils` importable
from scripts.trainer import Trainer

# data root holding train_sets/01..09 (see preprocessing/README.md).
# Default: the shared  data/  folder at the repo root; point SOMANET_DATA elsewhere to override.
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
    'mask_nonzero_ratio_threshold': 0.10,

    # MODEL: DINOv2Seg (self-contained, auto-selects ViT by VRAM)
    'model': 'DINOv2Seg',
    'dinov2_variant': 'auto',
    'embed_dim': 64,
    'decoder_mid_dim': 128,
    'adapter_depth': 4,
    'use_checkpoint': False,
    'pretrained': True,
    'n_unfrozen_blocks': 4,
    'in_channels': 1,
    'num_class': 2,
    'emb_dim': 32,

    # TRAIN
    'resume': os.environ.get('SW_RESUME', '0') == '1',
    'checkpoint_type': 'latest',
    'checkpoint_dir': './output_training/checkpoints',
    'pretrained_model_dir': './output_training/checkpoints',
    'epochs': int(os.environ.get('SW_EPOCHS', '250')),
    'num_gpus': int(os.environ.get('SW_NUM_GPUS', '1')),
    'single_precision': False,
    'optimizer_name': 'adamw',
    'lr_base': 5e-4,
    'lr_backbone': 1e-5,
    'lr_mode': 'steplr', 'step_size': 10000, 'gamma': 0.9,
    'data_precision': 'bfloat16',

    'losses': {'CrossEntropyLoss'},
    'loss_weights': {'CrossEntropyLoss': 1.0},
    'log_dir': './output_training/logs',

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
    print(f"Initial DINOv2 training (data root: {DATA}).", flush=True)
    Trainer(config_dict).train()
