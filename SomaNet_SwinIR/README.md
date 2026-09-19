# SomaNet (SwinIR)

3D soma instance segmentation. SomaNet is trained as a **two-step pipeline**: a
supervised SwinIR is pretrained to **initialize** the model, then a teacher–student
stage refines it. The reported model is the **final teacher** from Stage 2.


## Layout
```
SomaNet_SwinIR/
├── run_infer.py             # inference entry point (runs the Stage-2 model)
├── run_eval.py              # score pred_FINAL vs GT, save metrics
├── somanet_swinir_stage1/   # Stage 1 — supervised SwinIR pretraining (initialization)
└── somanet_swinir_stage2/   # Stage 2 — teacher–student (SomaNet) + inference / evaluation
```

## Environment
```bash
conda activate <env>        # torch 2.3.x + numpy<2, scipy, scikit-image, nibabel,
                            # opencv, timm, monai, pyyaml, pillow, einops, tensorboard, yacs
```

## Run

**Inference, then evaluation** (run from the repo root):
```bash
python -u run_infer.py 01                # -> somanet_swinir_stage2/inference_results/test_3d_stitched/01/pred_FINAL.nii.gz
python -u run_eval.py  01                # score pred_FINAL vs GT -> pred_FINAL_evaluation.txt (7 metrics)
```

**Training** (from scratch — stage 1, then stage 2):
```bash
cd somanet_swinir_stage1 && python -u run_trainer.py   # stage 1 (initialization)
cd ../somanet_swinir_stage2 && python -u run_trainer.py   # stage 2 (teacher–student)
```

`run_infer.py` only runs inference; `run_eval.py` is a
separate, CPU-only step that scores `pred_FINAL` against the volume's GT mask with
`evaluation.py` and writes the 7 metrics (Dice, IoU, PQ, ARAND,
VOItotal/split/merge) to `pred_FINAL_evaluation.txt`.

The data root is set by `SOMANET_DATA` (default `../data`) and the split by
`SOMANET_SPLIT` (`test_sets` or `train_sets`). Only the data root and volume id are
configurable in `run_infer.py`; all model / post-processing hyperparameters are fixed
in `somanet_swinir_stage2/scripts/infer_config.py` for reproducibility.

## Acknowledgment

The SwinIR backbone architecture is from Liang et al.
([JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR), Apache‑2.0),
adapted here for segmentation.
