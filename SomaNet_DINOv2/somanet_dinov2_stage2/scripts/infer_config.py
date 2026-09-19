"""Fixed inference config for SomaNet (DINOv2).

All model / affinity / post-processing hyperparameters live here, kept verbatim
for reproducibility. The top-level run_infer.py only chooses the data directory
and the volume; it calls build_config(vol, data) to get this dict unchanged.
A handful of values remain env-overridable (DINO_* sweeps); defaults reproduce
the reported run exactly.
"""
import os


def build_config(vol: str, data: str) -> dict:
    """Return the inference config for one test volume (`vol`) under `data`."""
    return {
        'mode': 'inference',
        'test_dir': [f'{data}/{vol}'],
        'data_mode': {'img': True, 'mask': True, 'prompt': False},
        'img_format': ['png', 'jpg', 'jpeg'], 'task_mode': 'SEG',
        # REPRODUCIBILITY: this seed is passed to the wrapped 2D Inference, whose set_seed()
        # (utils/utils.py) fixes python/numpy/torch/cuda RNGs AND sets
        # torch.backends.cudnn.deterministic=True, cudnn.benchmark=False. Inference is a
        # forward pass only (fixed teacher weights, no dropout/augmentation) and all
        # post-processing is pure numpy/scipy/skimage, so on the same GPU (a100) + module
        # (pytorch/2.3, XFORMERS_DISABLED=1 for DINOv2 vanilla attention) pred_FINAL.nii.gz is
        # deterministic. With the g5 z-stitch (relink_max_gap=5, utils/postprocess.py) it
        # reproduces the official table: DINOv2 PQ 0.706 (Block1) / 0.774 (Block2).
        # Once _pass1_cache.npz exists, infer_3d() skips the GPU and re-stitch is bit-identical.
        'batch_size': 1, 'num_workers': 1, 'random_seed': 120,
        'pin_memory': True, 'shuffle': False, 'drop_last': False,

        # --- model: DINOv2Seg (drop-in replacement for SwinIR, returns (emb, seg)) ---
        'model': 'DINOv2Seg', 'in_channels': 1, 'num_class': 2,
        # architecture params must match the trained checkpoint:
        'dinov2_variant': 'dinov2_vits14', 'embed_dim': 64, 'emb_dim': 32,
        'decoder_mid_dim': 128, 'adapter_depth': 4,
        'use_checkpoint': False,
        'pretrained': False,          # backbone weights come from our checkpoint, not torch-hub

        # DINOv2 tiled inference (positional embedding is locked to the training crop).
        # Overlap is tunable via env DINO_TILE_OVERLAP to suppress seam over-segmentation.
        'infer_tile_size': 400,
        'infer_tile_overlap': int(os.environ.get('DINO_TILE_OVERLAP', '250')),  # 250 = tuned best
        # Gaussian tile-blend width (fraction of tile); smaller = more peaked at centre,
        # stronger seam suppression. Tunable via env DINO_TILE_SIGMA_FRAC.
        'infer_tile_sigma_frac': float(os.environ.get('DINO_TILE_SIGMA_FRAC', '0.25')),

        'separate_weight_affinity': True,
        'shifts_affinity': [1, 3, 5, 9, 11, 19, 27, 35], 'neighbor_affinity': 8,
        'crop_size': 0,
        'checkpoint_path': 'latest',      # use the last checkpoint (checkpoint_latest.pth)
        # TEACHER weights from our training run; checkpoints store teacher_state_dict
        # + student_state_dict -> select the teacher explicitly.
        'checkpoint_dir': './output_training/checkpoints',
        # teacher_state_dict (default) or student_state_dict, selectable via env.
        'checkpoint_state_key': os.environ.get('DINO_CKPT_KEY', 'teacher_state_dict'),
        'test_results_dir': './inference_results/test_2d',
        'log_dir': './inference_results/logs',
        'num_gpus': 1, 'single_precision': False,
        'threshold_probability_map': float(os.environ.get('DINO_SEMTHR', '0.3')),  # sweepable

        # semantic mask post-processing
        'apply_morphology': True, 'morph_kernel_size': (5, 5), 'min_object_size': 250,
        'apply_hole_filling': False, 'apply_watershed': True,

        # affinity-based 2D instance segmentation (seeded watershed)
        # Merge-oriented knobs are env-tunable to sweep DINOv2's 2D over-segmentation:
        #   DINO_AGGLOM   -> agglomeration_boundary_thresh (higher = merge more fragments)
        #   DINO_MININST  -> min_instance_size (higher = drop small 2D detections)
        #   DINO_SEEDMERGE-> seed_merge_distance (higher = merge nearby seeds)
        'affinity_seed_threshold': 0.92, 'affinity_seed_erode_radius': 6,
        'affinity_peak_min_distance': 10, 'affinity_medium_range_weight': 0.3,
        'use_semantic_for_seeds': True, 'semantic_seed_gate_thresh': 0.2,
        'seed_boundary_overlap_thresh': 0.5,
        'seed_merge_distance': int(os.environ.get('DINO_SEEDMERGE', '5')),
        'seed_merge_boundary_thresh': 0.3,
        'apply_agglomeration': True,
        'agglomeration_boundary_thresh': float(os.environ.get('DINO_AGGLOM', '0.35')),
        'agglomeration_min_boundary_length': 5,
        'min_instance_size': int(os.environ.get('DINO_MININST', '1000')),  # FINAL: 1000
        'instance_fill_holes': True, 'instance_smooth_radius': 2,

        # 2D->3D z-stitch: the gap that builds pred_FINAL is relink_max_gap=5 (g5),
        # set in utils/postprocess.py (ZPP). No stitch knob is configured here.
        # DINO_OUT_TAG lets parallel overlap sweeps write to distinct dirs (avoids
        # clobbering + Pass-1 cache reuse across configs).
        'output_3d_dir': './inference_results/test_3d_stitched' + os.environ.get('DINO_OUT_TAG', ''),
        'save_nifti': True,

        # geometric post-processing chain -> pred_FINAL.nii.gz
        'apply_postprocess': True,
        'postprocess_obj_gate': 400, 'postprocess_min_span': 3,
        # (pred_FINAL comes from the geometric chain above; the embedding-cosine
        #  Pass-2 stitch is not used, so its knobs are left at their off defaults.)
    }
