import os


def build_config(vol: str, data: str) -> dict:
    """Return the inference config for one test volume (`vol`) under `data`."""
    return {
        'mode': 'inference',
        'test_dir': [f'{data}/{vol}'],
        'data_mode': {'img': True, 'mask': True, 'prompt': False},
        'img_format': ['png', 'jpg', 'jpeg'], 'task_mode': 'SEG',

        'batch_size': 1, 'num_workers': 1, 'random_seed': 120,
        'pin_memory': True, 'shuffle': False, 'drop_last': False,
        'model': 'SwinIR', 'in_channels': 1, 'num_class': 2,
        'separate_weight_affinity': True,
        'shifts_affinity': [1, 3, 5, 9, 11, 19, 27, 35], 'neighbor_affinity': 8,
        'crop_size': 0,
        'checkpoint_path': os.environ.get('SW_CKPT_PATH', 'latest'),

        # TS teacher-student from output_training 
        'checkpoint_dir': os.environ.get('SW_CKPT_DIR', './output_training/checkpoints'),
        'checkpoint_state_key': os.environ.get('SW_CKPT_KEY', 'teacher_state_dict'),
        'test_results_dir': './inference_results/test_2d',
        'log_dir': './inference_results/logs',
        'num_gpus': 1, 'single_precision': False,
        'threshold_probability_map': float(os.environ.get('SW_SEMTHR', '0.3')),  # FINAL: 0.3

        # semantic mask post-processing
        'apply_morphology': True, 'morph_kernel_size': (5, 5), 'min_object_size': 250,
        'apply_hole_filling': False, 'apply_watershed': True,

        # affinity-based 2D instance segmentation (seeded watershed)
        'affinity_seed_threshold': 0.92, 'affinity_seed_erode_radius': 6,
        'affinity_peak_min_distance': 10, 'affinity_medium_range_weight': 0.3,
        'use_semantic_for_seeds': True, 'semantic_seed_gate_thresh': 0.2,
        'seed_boundary_overlap_thresh': 0.5, 'seed_merge_distance': 5,
        'seed_merge_boundary_thresh': 0.3,
        'apply_agglomeration': True, 'agglomeration_boundary_thresh': 0.35,
        'agglomeration_min_boundary_length': 5,
        'min_instance_size': int(os.environ.get('SW_MININST', '1000')), 'instance_fill_holes': True, 'instance_smooth_radius': 2,  

        # 2D->3D z-stitch: the gap that builds pred_FINAL is relink_max_gap=5 (g5), set in utils/postprocess.py 
        'output_3d_dir': './inference_results/test_3d_stitched' + os.environ.get('SW_OUT_TAG', ''),
        'save_nifti': True,

        # geometric post-processing chain: auto-run after the stitch -> pred_FINAL.nii.gz
        'apply_postprocess': True,
        'postprocess_obj_gate': 400, 'postprocess_min_span': 3,
    }
