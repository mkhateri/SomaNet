"""
3D instance segmentation by stitching 2D per-slice predictions.

Composition wrapper around the ``Inference`` class in ``scripts/inference.py``.

Pipeline
--------
Pass 1  Per-slice 2D inference (model forward -> affinity seeded watershed ->
         agglomeration -> semantic/morphological post-processing).  Caches the
         per-slice instance labels and per-instance mean L2-normalised
         embeddings, so re-stitching does not re-run the GPU.
Stitch  Geometric post-processing chain from the Pass-1 cache
         (utils/postprocess.py): z-overlap re-link (relink_max_gap = 5, "g5"),
         bad-slice bridging, hole filling, transient / z-span
         filtering.  Produces the final 3D (H, W, Z) int32 label volume.

Outputs (per test volume, under output_3d_dir/<vol>/)
-----------------------------------------------------
* pred_FINAL.nii.gz         final 3D instance segmentation (NIfTI)
* <stem>_pred.nii.gz        same volume under the input's coordinate name
* _pass1_cache.npz          cached Pass-1 per-slice labels (gap-independent)
* _pass1_embs.npy           cached per-instance embeddings
"""

import sys

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from skimage.morphology import remove_small_objects

from scripts.inference import Inference
from model.losses.losses import embedding_loss
from model.handlers.data_handler import DataHandler
from utils.affinity import multi_offset, gen_affs_ours, weight_binary_ratio

# Optional: nibabel for NIfTI I/O
try:
    import nibabel as nib
    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False
    print("[WARN] nibabel not installed – NIfTI save/load disabled.", flush=True)

# ======================================================================
# Main 3D stitcher
# ======================================================================
class Inference3DStitcher:
    """Wraps the existing 2D ``Inference`` class and stitches slices into
    a coherent 3D instance segmentation volume."""

    def __init__(self, config: Dict[str, Any]):
        # ---- 3D-specific defaults ----
        self.output_3d_dir = config.get(
            'output_3d_dir', './inference_results/test_3d_stitching',
        )
        self.save_nifti = config.get('save_nifti', True)

        # ---- Instantiate the existing 2D inference engine ----
        self.infer2d = Inference(config)
        self.config = self.infer2d.config  # resolved config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def infer_3d(self):
        """Main entry: loop over test dirs, run Pass-1 + geometric post-processing, save pred_FINAL."""
        print("\n" + "=" * 70, flush=True)
        print("3D STITCHING PIPELINE", flush=True)
        print("=" * 70, flush=True)

        all_test_dirs = list(self.config['test_dir'])

        for test_dir in all_test_dirs:
            print(f"\n>>> Processing test directory: {test_dir}", flush=True)

            # Point the 2D engine at a single test directory
            self.config['test_dir'] = [test_dir]
            self.infer2d.test_loader = DataHandler(
                self.config,
            )._get_inference_dataloaders()

            test_dir_name = os.path.basename(test_dir)
            save_dir = os.path.join(self.output_3d_dir, test_dir_name)
            os.makedirs(save_dir, exist_ok=True)

            # Pass 1 — per-slice 2D inference (with caching)
            cache_path = os.path.join(save_dir, '_pass1_cache.npz')
            emb_cache_path = os.path.join(save_dir, '_pass1_embs.npy')

            if os.path.isfile(cache_path) and os.path.isfile(emb_cache_path):
                print("  Loading cached Pass 1 results...", flush=True)
                z_to_labels, z_to_inst_embs, sorted_z = \
                    self._load_pass1_cache(cache_path, emb_cache_path)
            else:
                z_to_labels, z_to_inst_embs, sorted_z = \
                    self._run_pass1_per_slice()
                # Save cache for future re-runs
                self._save_pass1_cache(
                    z_to_labels, z_to_inst_embs, sorted_z,
                    cache_path, emb_cache_path,
                )

            if len(sorted_z) == 0:
                print("[WARN] No slices processed – skipping.", flush=True)
                continue

            print(
                f"Pass 1 complete: {len(sorted_z)} slices, "
                f"z range [{sorted_z[0]}..{sorted_z[-1]}]",
                flush=True,
            )

            # Geometric post-processing chain (re-stitch from cache) = the result
            if self.config.get('apply_postprocess', True):
                self._run_postprocess_chain(cache_path, test_dir, save_dir)

        print("\n3D stitching pipeline completed.", flush=True)

    def _run_postprocess_chain(self, cache_path, test_dir, save_dir):
        """Geometric post-processing chain from the Pass-1 cache -> pred_FINAL.nii.gz.
        Auto-run after stitching; re-tunable standalone via run_postprocess.py (same cache)."""
        from utils import postprocess as pp
        img_dir = os.path.join(test_dir, 'img')
        img_files = pp.sort_by_z([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                  if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        out = pp.run_chain(cache_path, img_files,
                           obj_gate=self.config.get('postprocess_obj_gate', 400),
                           min_span=self.config.get('postprocess_min_span', 3))
        final_path = os.path.join(save_dir, 'pred_FINAL.nii.gz')
        pp.save_nii(out, final_path)
        # also save under the input's long coordinate name (parallel to *_raw/_mask.nii.gz)
        stem = os.path.basename(img_files[0]).rsplit('_', 1)[0] if img_files else ''
        if stem:
            named_path = os.path.join(save_dir, f'{stem}_pred.nii.gz')
            pp.save_nii(out, named_path)
            print(f"  also saved {named_path}", flush=True)
        n = len(set(int(x) for x in np.unique(out)) - {0})
        print(f"  post-processing chain -> {n} instances, saved {final_path}", flush=True)

    # ------------------------------------------------------------------
    # Pass 1: per-slice 2D inference
    # ------------------------------------------------------------------
    def _run_pass1_per_slice(self):
        """Run model forward + 2D segmentation for every slice.

        Returns
        -------
        z_to_labels : dict[int, np.ndarray]
            z-index → (H, W) int32 instance labels
        z_to_inst_embs : dict[int, dict[int, np.ndarray]]
            z-index → {inst_id: mean_embedding (16,)}
        sorted_z : list[int]
            Numerically sorted z-indices
        """
        z_to_labels: Dict[int, np.ndarray] = {}
        z_to_inst_embs: Dict[int, Dict[int, np.ndarray]] = {}

        self.infer2d.model.eval()
        n_batches = len(self.infer2d.test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.infer2d.test_loader):
                # --- extract z-index from filename ---
                basename = batch['img_base_name'][0]  # e.g. '…_01_100.png'
                z_idx = int(
                    os.path.splitext(basename)[0].split('_')[-1]
                )

                # --- forward pass (identical to inference_modified.py) ---
                processed = self.infer2d.batch_processing(batch)
                inp_img = processed['img']
                gt_labels_affs = processed['labels_affs']
                labels_affs_mask = processed['affs_mask']
                weight_map = processed['weight_map']

                with torch.amp.autocast(
                    device_type=self.infer2d.device.type,
                    enabled=not self.config.get('single_precision', True),
                ):
                    embedding, pred_mask_probability = self.infer2d.model(
                        inp_img,
                    )

                # Compute affinities via embedding_loss (also L2-normalises)
                _, pred_affinity, _ = embedding_loss(
                    embedding, gt_labels_affs, weight_map, labels_affs_mask,
                    offsets=self.infer2d.offsets, criterion='WeightedMSE',
                    affs0_weight=1, mode='AAAI',
                )

                pred_aff_np = pred_affinity[0].cpu().numpy()  # (C, H, W)

                # Semantic foreground
                _, binary_thresh = self.infer2d.generate_binary_masks(
                    pred_mask_probability,
                )
                semantic_fg = binary_thresh[0].cpu().numpy().astype(np.uint8)
                semantic_prob = torch.softmax(
                    pred_mask_probability, dim=1,
                )[0, 1].cpu().numpy().astype(np.float64)

                # Clean semantic fg
                import cv2
                if self.config.get('apply_morphology', True):
                    ks = self.config.get('morph_kernel_size', (5, 5))
                    kernel = np.ones(ks, np.uint8)
                    semantic_fg = cv2.morphologyEx(
                        semantic_fg, cv2.MORPH_OPEN, kernel,
                    )
                fg_min_size = self.config.get('min_object_size', 250)
                if fg_min_size > 0:
                    semantic_fg = remove_small_objects(
                        semantic_fg > 0, min_size=fg_min_size,
                    ).astype(np.uint8)

                # Instance segmentation
                labels_2d = self.infer2d.affinity_to_instances(
                    pred_aff_np, semantic_fg, semantic_prob,
                )

                # Agglomeration
                if self.config.get('apply_agglomeration', True):
                    boundary_map = self.infer2d._compute_boundary_map(
                        pred_aff_np,
                    )
                    labels_2d = self.infer2d._agglomerate_instances(
                        labels_2d, boundary_map,
                    )

                labels_2d = self.infer2d.post_process_instances(labels_2d)

                # --- per-instance mean embedding ---
                # L2-normalise embedding (same as embedding_loss does)
                emb_normed = F.normalize(embedding, p=2, dim=1)
                emb_np = emb_normed[0].cpu().numpy()  # (16, H, W)

                inst_embs = self._compute_instance_mean_embeddings(
                    emb_np, labels_2d,
                )

                # Cache
                z_to_labels[z_idx] = labels_2d.astype(np.int32)
                z_to_inst_embs[z_idx] = inst_embs

                if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                    n_inst = len(set(np.unique(labels_2d)) - {0})
                    print(
                        f"  Pass 1: {batch_idx + 1}/{n_batches} | "
                        f"z={z_idx} | instances={n_inst}",
                        flush=True,
                    )

        sorted_z = sorted(z_to_labels.keys())
        return z_to_labels, z_to_inst_embs, sorted_z

    # ------------------------------------------------------------------
    # Pass 1 cache save / load
    # ------------------------------------------------------------------
    @staticmethod
    def _save_pass1_cache(z_to_labels, z_to_inst_embs, sorted_z,
                          cache_path, emb_cache_path):
        """Persist Pass 1 results so re-runs skip the GPU pass."""
        # Labels: save as compressed npz with z-indices as keys
        label_arrays = {str(z): z_to_labels[z] for z in sorted_z}
        np.savez_compressed(cache_path, sorted_z=np.array(sorted_z),
                            **label_arrays)

        # Embeddings: serialize as a list of (z, inst_id, emb_vector)
        rows = []
        for z in sorted_z:
            for inst_id, emb_vec in z_to_inst_embs[z].items():
                rows.append(np.concatenate([[z, inst_id], emb_vec]))
        if rows:
            np.save(emb_cache_path, np.array(rows))
        else:
            np.save(emb_cache_path, np.zeros((0, 18)))
        print(f"  Pass 1 cache saved.", flush=True)

    @staticmethod
    def _load_pass1_cache(cache_path, emb_cache_path):
        """Load cached Pass 1 results."""
        data = np.load(cache_path)
        sorted_z = list(data['sorted_z'].astype(int))

        z_to_labels = {}
        for z in sorted_z:
            z_to_labels[z] = data[str(z)].astype(np.int32)

        z_to_inst_embs: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        emb_rows = np.load(emb_cache_path)
        for row in emb_rows:
            z = int(row[0])
            inst_id = int(row[1])
            emb_vec = row[2:]
            z_to_inst_embs[z][inst_id] = emb_vec

        print(
            f"  Loaded cache: {len(sorted_z)} slices, "
            f"{len(emb_rows)} instance embeddings.",
            flush=True,
        )
        return z_to_labels, dict(z_to_inst_embs), sorted_z

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_instance_mean_embeddings(
        emb: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Compute per-instance mean L2-normalised embedding.

        Parameters
        ----------
        emb : (C, H, W) L2-normalised embedding
        labels : (H, W) int32 instance labels (0 = background)

        Returns
        -------
        dict mapping inst_id → (C,) mean embedding (re-normalised)
        """
        inst_embs: Dict[int, np.ndarray] = {}
        for inst_id in np.unique(labels):
            if inst_id == 0:
                continue
            mask = labels == inst_id  # (H, W)
            # emb[:, mask] → (C, N)
            mean_vec = emb[:, mask].mean(axis=1)  # (C,)
            norm = np.linalg.norm(mean_vec) + 1e-8
            inst_embs[int(inst_id)] = mean_vec / norm
        return inst_embs

# ======================================================================
# CLI entry point
# ======================================================================

