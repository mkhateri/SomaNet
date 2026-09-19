# SomaNet (DINOv2)

3D soma instance segmentation. SomaNet is trained as a **two-step pipeline**: a
DINOv2 backbone is finetuned to **initialize** the model, then a teacher–student
stage refines it. The reported model is the **final teacher** from Stage 2.

## Layout
```
SomaNet_DINOv2/
├── run_infer.py             # inference entry point (runs the Stage-2 model)
├── run_eval.py              # score pred_FINAL vs GT, save metrics
├── somanet_dinov2_stage1/   # Stage 1 — DINOv2 finetune pretraining (initialization)
└── somanet_dinov2_stage2/   # Stage 2 — teacher–student (SomaNet) + inference / evaluation
```

## Environment
Uses the shared `somanet` conda env (torch 2.3.x). **No extra Python packages** — the
code sets `XFORMERS_DISABLED=1`, so xformers is not required.

The DINOv2 backbone is fetched once via `torch.hub.load('facebookresearch/dinov2', ...)`.
Compute nodes usually lack internet, so pre-cache the hub repo into a shared dir:
```bash
export TORCH_HOME=/path/to/shared/.torch_hub
export XFORMERS_DISABLED=1
python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=False)"
```

## Run

**Inference, then evaluation** (run from the repo root):
```bash
export TORCH_HOME=/path/to/shared/.torch_hub XFORMERS_DISABLED=1
python -u run_infer.py 01                # -> somanet_dinov2_stage2/inference_results/test_3d_stitched/01/pred_FINAL.nii.gz
python -u run_eval.py  01                # score pred_FINAL vs GT -> pred_FINAL_evaluation.txt (7 metrics)
```

**Training** (from scratch — stage 1, then stage 2):
```bash
cd somanet_dinov2_stage1 && python -u run_trainer.py   # stage 1 (initialization)
cd ../somanet_dinov2_stage2 && python -u run_trainer.py   # stage 2 (teacher–student)
```

`run_infer.py` only runs inference; `run_eval.py` is a
separate, CPU-only step that scores `pred_FINAL` against the volume's GT mask with
`evaluation.py` and writes the 7 metrics (Dice, IoU, PQ, ARAND,
VOItotal/split/merge) to `pred_FINAL_evaluation.txt`.

The data root is set by `SOMANET_DATA` (default `../data`) and the split by
`SOMANET_SPLIT` (`test_sets` or `train_sets`). Only the data root and volume id are
configurable in `run_infer.py`; all model / post-processing hyperparameters are fixed
in `somanet_dinov2_stage2/scripts/infer_config.py` for reproducibility.

## Acknowledgment

The DINOv2 backbone and its pretrained weights are from Meta AI
([facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), Apache‑2.0),
loaded at runtime via `torch.hub`.
