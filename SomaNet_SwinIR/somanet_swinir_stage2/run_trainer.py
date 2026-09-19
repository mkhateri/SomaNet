"""Teacher-student SwinIR training entry point.

Mirrors run_infer.py: sets up imports, builds the training config, and launches
the Trainer. Hyperparameters are identical to scripts/trainer.py __main__ (kept
verbatim for reproducibility); only the data root and a few operational knobs are
env-overridable, with defaults that reproduce the original run exactly.

Run from this dir:  python -u run_trainer.py

Env overrides (all optional; defaults reproduce the original run):
  SOMANET_DATA   data root holding train_sets/01..09
                 (default: the shared  data/  folder at the repo root)
  SW_NUM_GPUS    number of GPUs           (default: 2)
  SW_EPOCHS      number of epochs         (default: 100)
  SW_RESUME      '1' to resume from checkpoint_dir (default: '0' -> fresh)
"""
import os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # make `scripts`, `model`, `utils` importable
from scripts.trainer import Trainer


def _init_from_stage1():
    """Initialize teacher & student from the Stage-1 pretraining checkpoint.

    Stage 2 starts both the teacher and the student from the Stage-1 output. Instead
    of copying that checkpoint by hand, we copy it here (once) into the paths the
    config expects. Point SW_STAGE1_CKPT at the Stage-1 checkpoint; by default it is
    the sibling Stage-1 folder's best checkpoint. Existing files are left untouched.
    """
    src = os.environ.get(
        'SW_STAGE1_CKPT',
        os.path.join(HERE, '..', 'somanet_swinir_stage1',
                     'output_training', 'checkpoints', 'checkpoint_best.pth'))
    for role in ('teacher', 'student'):
        dst = os.path.join(HERE, 'pretrained_model', f'{role}_pretrained_model',
                           'checkpoints', 'checkpoint_best.pth')
        if os.path.exists(dst):
            continue
        if not os.path.isfile(src):
            print(f"[pretrained] WARNING: Stage-1 checkpoint not found at {src}.\n"
                  f"             Train Stage 1 first, or set SW_STAGE1_CKPT.", flush=True)
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[pretrained] initialized {role} from Stage-1 -> {dst}", flush=True)

# data root holding train_sets/01..09 (see preprocessing/README.md).
# Default: the shared  data/  folder at the repo root. Put your data there, or
# point SOMANET_DATA at it:   export SOMANET_DATA=/path/to/data_root
DATA = os.environ.get('SOMANET_DATA', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data'))

config_dict = {
    # MODE
    'mode': 'train',

    # DATA
    'train_dir': [f'{DATA}/train_sets/{i:02d}' for i in range(1, 10)],   # 01..09
    'data_mode': {'img': True, 'mask': True, 'prompt': False},
    'img_format': ['png', 'jpg', 'jpeg'],
    'task_mode': 'SEG',
    'val_split': 0.2,
    'batch_size': 2,
    'num_workers': 1,
    'random_seed': 120,
    'pin_memory': True,
    'shuffle': True,
    'drop_last': True,
    'mask_nonzero_ratio_threshold': 0.15,   # the least non-zero ratio in cropped masks

    # MODEL
    'model': 'SwinIR', 'student_model': 'SwinIR', 'teacher_model': 'SwinIR',
    'in_channels': 1, 'num_class': 2,

    # TRAIN
    'resume': os.environ.get('SW_RESUME', '0') == '1',
    'fine_tune': False,
    'checkpoint_type': 'latest',
    'checkpoint_dir': './output_training/checkpoints',
    'pretrained_model_dir': './output_training/checkpoints',
    'epochs': int(os.environ.get('SW_EPOCHS', '100')),
    'num_gpus': int(os.environ.get('SW_NUM_GPUS', '2')),
    'data_precision': 'bfloat16',
    'optimizer_name': 'adam',
    'lr_base': 1e-4, 'lr_mode': 'steplr', 'step_size': 2500, 'gamma': 0.9,

    'losses': {'CrossEntropyLoss'},
    'loss_weights': {'CrossEntropyLoss': 1.0},
    'log_dir': './output_training/logs',

    'ema_decay': 0.999,
    'pretrained_teacher_model_dir': './pretrained_model/teacher_pretrained_model/checkpoints/checkpoint_best.pth',
    'pretrained_student_model_dir': './pretrained_model/student_pretrained_model/checkpoints/checkpoint_best.pth',

    # LOSS: embedding_weight*embedding_loss + mask_weight*mask_loss
    'embedding_weight': 0.005, 'mask_weight': 1.0,
    # TOTAL: Supervised_LOSS + weakly_supervised_weight*weakly_supervised_LOSS
    'weakly_supervised_weight': 0.5,
    # Threshold of knowledge transfer teacher->student (0.5 == argmax)
    'teacher_threshold_WGT': 0.5,

    # GRADIENT CHECKPOINTING (memory optimization; ~40-50% less GPU mem, ~30% more compute)
    'use_checkpoint': True,

    # GRADIENT CLIPPING AND STABILITY
    'grad_clip_enabled': True, 'grad_clip_max_norm': 1.0,
    'max_nan_skips_per_epoch': 10, 'loss_scale_factor': 1.0,

    # FEATURE DISTILLATION (teacher-student embedding consistency)
    'feature_distillation_enabled': True, 'feature_distillation_weight': 0.1,

    'save_image_frequency': 1, 'save_checkpoint_frequency': 2500,

    # AFFINITY PARAMETERS
    'separate_weight_affinity': True,
    'shifts_affinity': [1, 3, 5, 9, 11, 19, 27, 35], 'neighbor_affinity': 8,

    # AUGMENTATION
    'crop_size': 400, 'padding_size': 0, 'cuting_corner_size': 7,

    # LOGGING
    'log_gpu_utilization': True, 'print_config': True,
}

if __name__ == '__main__':
    print(f"Teacher-student SwinIR training (data root: {DATA}).", flush=True)
    _init_from_stage1()          # auto-copy Stage-1 -> pretrained_model/{teacher,student}
    Trainer(config_dict).train()
