import os
import torch
import numpy as np
import importlib
import cv2
from torch import Tensor

from typing import Any, Dict, Union, Tuple, Optional
from utils.utils import set_seed
from model.handlers.config_handler import ConfigHandler
from model.handlers.parallel_handler import ParallelHandler
from utils.affinity import multi_offset, gen_affs_ours, weight_binary_ratio

from skimage.morphology import binary_erosion, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label as sk_label, regionprops
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy import ndimage as ndi

class Inference:
    """Inference class with affinity-based instance segmentation."""

    # Number of affinity channels per shift (neighbor=8 -> 4 directions)
    _DIRS_PER_SHIFT = 4  # [-s,0], [0,-s], [-s,-s], [-s,s]

    def __init__(self, config: Union[str, Dict[str, Any]]):
        print("Initializing Inference...", flush=True)
        self.config = ConfigHandler(config).config
        set_seed(self.config.get('random_seed', 321))
        print("Seed set.", flush=True)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = min(torch.cuda.device_count(), self.config.get('num_gpus', 1))
        print(f"Using device: {self.device} with {self.num_gpus} GPU(s)", flush=True)

        # Model setup
        print("Setting up model...", flush=True)
        self.model = self._get_model(
            model_module='model.architectures',
            model_name=self.config['model'],
            in_channels=self.config['in_channels'],
            num_classes=self.config['num_class']
        )
        print(f"Model {self.config['model']} created and setup complete.", flush=True)

        if self.num_gpus > 1:
            self.model = ParallelHandler.apply_parallel(self.model, self.num_gpus)
            print("Parallelization applied.", flush=True)

        self.model.to(self.device)
        print("Model moved to device.", flush=True)

        checkpoint_path = self.get_checkpoint_path(self.config['checkpoint_path'])
        print("checkpoint_path:", checkpoint_path)
        self.load_checkpoint(checkpoint_path)
        print("Checkpoint loaded.", flush=True)

        self.offsets = multi_offset(
            shifts=self.config['shifts_affinity'],
            neighbor=self.config['neighbor_affinity']
        )
        self.shifts = self.config['shifts_affinity']

        # Pre-compute offset channel ranges per shift for convenience
        d = self._DIRS_PER_SHIFT
        self._shift_ranges = {}
        for idx, s in enumerate(self.shifts):
            self._shift_ranges[s] = (idx * d, (idx + 1) * d)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def _get_model(self, model_module, model_name, in_channels, num_classes):
        module = importlib.import_module(model_module)
        Model = getattr(module, model_name)
        kwargs = dict(in_channels=in_channels, num_classes=num_classes)
        if model_name == 'DINOv2Seg':
            # DINOv2Seg needs its backbone variant + head dims to match the
            # trained checkpoint. pretrained=False: the backbone weights are
            # overwritten by our checkpoint, so no torch-hub weight download.
            kwargs.update(
                model_name=self.config.get('dinov2_variant', 'dinov2_vits14'),
                emb_dim=self.config.get('emb_dim', 32),
                embed_dim=self.config.get('embed_dim', 64),
                decoder_mid_dim=self.config.get('decoder_mid_dim', 128),
                adapter_depth=self.config.get('adapter_depth', 4),
                use_checkpoint=self.config.get('use_checkpoint', False),
                pretrained=self.config.get('pretrained', False),
            )
        return Model(**kwargs).to(self.device)

    def get_checkpoint_path(self, checkpoint_path):
        if checkpoint_path in ['latest', 'best']:
            return os.path.join(self.config['checkpoint_dir'], f'checkpoint_{checkpoint_path}.pth')
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Which weights to load. Training checkpoints hold both 'teacher_state_dict'
        # and 'student_state_dict'; the pretrained-init checkpoints hold 'state_dict'.
        # 'checkpoint_state_key' (config) selects explicitly; otherwise auto-detect.
        key = self.config.get('checkpoint_state_key')
        if key is None:
            for cand in ('state_dict', 'teacher_state_dict', 'student_state_dict'):
                if cand in checkpoint:
                    key = cand
                    break
        if key not in checkpoint:
            raise KeyError(f"'{key}' not in checkpoint; available keys: {list(checkpoint.keys())}")
        print(f"Loading weights from checkpoint key '{key}'"
              f" (epoch={checkpoint.get('epoch', '?')})")
        state_dict = checkpoint[key]
        if list(state_dict.keys())[0].startswith('module.'):
            print("Checkpoint trained on multiple GPUs. Adjusting state_dict for single GPU...")
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # DINOv2: the spatial positional embedding is created lazily (registered
        # as None at build), so it is absent from the model state and would be
        # dropped as an "unexpected key" -- the trained weights lost. Pre-register
        # it with the checkpoint's shape so it is loaded, and pin its grid.
        if ('sp_pos_embed' in state_dict
                and getattr(self.model, 'sp_pos_embed', None) is None
                and hasattr(self.model, '_sp_pos_shape')):
            import torch.nn as _nn, math as _math
            _sp = state_dict['sp_pos_embed']
            self.model.register_parameter('sp_pos_embed', _nn.Parameter(torch.zeros_like(_sp)))
            _n = int(round(_math.sqrt(int(_sp.shape[1]))))
            self.model._sp_pos_shape = (_n, _n)
            print(f"Pre-registered DINOv2 sp_pos_embed {tuple(_sp.shape)}, grid ({_n},{_n}).")

        model_state = self.model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape != model_state[k].shape:
                    skipped_keys.append(k)
                    continue
            filtered_state_dict[k] = v

        if skipped_keys:
            print(f"Skipped keys due to size mismatch: {skipped_keys}")

        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        # DINOv2: the learned spatial positional embedding (sp_pos_embed) is
        # size-locked to the training crop. It starts as None, so its grid shape
        # (_sp_pos_shape) is not restored by load_state_dict and the model would
        # lazily RE-INITIALISE it on the first forward, discarding the trained
        # weights. Pin the grid here so the loaded embedding is used. Inputs must
        # match the training tile size (see tiled_forward).
        sp = getattr(self.model, 'sp_pos_embed', None)
        if sp is not None and hasattr(self.model, '_sp_pos_shape'):
            import math as _math
            n = int(round(_math.sqrt(int(sp.shape[1]))))
            self.model._sp_pos_shape = (n, n)
            print(f"Pinned DINOv2 sp_pos_embed grid to ({n},{n}) from checkpoint.")

    # ------------------------------------------------------------------
    # Tiled forward (DINOv2 fixed-size positional embedding)
    # ------------------------------------------------------------------
    def tiled_forward(self, inp_img, tile=None, overlap=None):
        """Run the model on fixed-size overlapping tiles and blend the outputs.

        DINOv2's positional embedding is locked to the training crop, so the full
        slice is processed as `tile`x`tile` windows (matching training) and the
        per-tile (embedding, seg) outputs are blended with a smooth Hann window.
        Returns (embedding_full, seg_full) at the input resolution — identical
        interface to calling ``self.model(inp_img)`` directly.
        """
        import torch.nn.functional as F
        tile = tile or int(self.config.get('infer_tile_size', 400))
        overlap = int(self.config.get('infer_tile_overlap', 150) if overlap is None else overlap)
        B, C, H, W = inp_img.shape
        assert B == 1, "tiled_forward expects batch size 1"
        stride = max(1, tile - overlap)

        # tile start positions; the last tile is snapped to the far edge so every
        # window is exactly `tile` wide (no partial tiles when H,W >= tile).
        def _starts(L):
            if L <= tile:
                return [0]
            s = list(range(0, L - tile + 1, stride))
            if s[-1] != L - tile:
                s.append(L - tile)
            return s
        ys, xs = _starts(H), _starts(W)

        # Gaussian tile weighting: peaks at the tile centre and tapers smoothly to
        # the borders, so each seam pixel is dominated by the tile where it sits
        # most central -> smoother blends, fewer boundary fragments than a Hann
        # window. sigma = infer_tile_sigma_frac * tile.
        _coords = torch.arange(tile, device=inp_img.device, dtype=torch.float32) - (tile - 1) / 2.0
        _sigma = tile * float(self.config.get('infer_tile_sigma_frac', 0.25))
        _g1d = torch.exp(-(_coords ** 2) / (2.0 * _sigma ** 2))
        win2d = (_g1d[:, None] * _g1d[None, :]).clamp_min(1e-6)     # (tile, tile)

        emb_acc = seg_acc = wsum = None
        for y in ys:
            for x in xs:
                patch = inp_img[:, :, y:y + tile, x:x + tile]
                ph, pw = patch.shape[-2:]
                if (ph, pw) != (tile, tile):                       # only if H/W < tile
                    patch = F.pad(patch, (0, tile - pw, 0, tile - ph))
                emb_t, seg_t = self.model(patch)
                emb_t = emb_t.float()[:, :, :ph, :pw]
                seg_t = seg_t.float()[:, :, :ph, :pw]
                w = win2d[:ph, :pw]
                if emb_acc is None:
                    emb_acc = inp_img.new_zeros((B, emb_t.shape[1], H, W), dtype=torch.float32)
                    seg_acc = inp_img.new_zeros((B, seg_t.shape[1], H, W), dtype=torch.float32)
                    wsum = inp_img.new_zeros((B, 1, H, W), dtype=torch.float32)
                emb_acc[:, :, y:y + ph, x:x + pw] += emb_t * w
                seg_acc[:, :, y:y + ph, x:x + pw] += seg_t * w
                wsum[:, :, y:y + ph, x:x + pw] += w
        wsum = wsum.clamp_min(1e-6)
        return emb_acc / wsum, seg_acc / wsum

    def batch_processing(self, batch):
        precision = torch.float32 if self.config.get('single_precision', True) else torch.float16

        labels_affs_list, affs_mask_list, weight_map_list = [], [], []
        for i in range(batch['mask'].shape[0]):
            labels_affs, affs_mask = gen_affs_ours(
                batch['mask'][i], offsets=self.offsets, ignore=False, padding=True
            )
            labels_affs_list.append(labels_affs)
            affs_mask_list.append(affs_mask)
            weight_map = weight_binary_ratio(labels_affs, mask=affs_mask, alpha=1.0)
            weight_map_list.append(weight_map)

        batch['labels_affs'] = torch.from_numpy(np.stack(labels_affs_list, axis=0)).to(self.device, dtype=precision)
        batch['affs_mask'] = torch.from_numpy(np.stack(affs_mask_list, axis=0)).to(self.device, dtype=precision)
        batch['weight_map'] = torch.from_numpy(np.stack(weight_map_list, axis=0)).to(self.device, dtype=precision)
        batch['mask_semantic'] = (batch['mask'] > 0).to(torch.long).to(self.device, dtype=precision)
        batch['img'] = batch['img'].to(self.device, dtype=precision)
        batch['mask'] = batch['mask'].to(self.device, dtype=precision)
        return batch

    def generate_binary_masks(self, pred_mask_probability: torch.Tensor):
        probability_map = torch.softmax(pred_mask_probability, dim=1)
        binary_mask_argmax = torch.argmax(probability_map, dim=1).squeeze(1)
        threshold = self.config.get('threshold_probability_map', 0.5)
        binary_mask_thresholded = (probability_map[:, 1, :, :] > threshold).int()
        return binary_mask_argmax, binary_mask_thresholded

    # ==================================================================
    # Affinity -> boundary / interior maps (helpers for the watershed)
    # ==================================================================
    def _get_affinity_channels(self, pred_affinity_np: np.ndarray,
                               shift_indices: list) -> np.ndarray:
        """Select affinity channels belonging to the given shift indices.

        Args:
            pred_affinity_np: (C, H, W) predicted affinities
            shift_indices: list of indices into self.shifts (e.g. [0] for shift=1)

        Returns:
            (N, H, W) selected channels
        """
        d = self._DIRS_PER_SHIFT
        channels = []
        for si in shift_indices:
            channels.append(pred_affinity_np[si * d:(si + 1) * d])
        return np.concatenate(channels, axis=0)

    def _compute_boundary_map(self, pred_affinity_np: np.ndarray) -> np.ndarray:
        """Compute a boundary probability map from predicted affinities.

        Uses short-range affinities (shift=1) as primary boundary signal,
        and optionally blends in medium-range (shift=3,5) for robustness.

        Returns:
            boundary: (H, W) float in [0,1], high = boundary
        """
        # Short-range: shift=1 (channels 0:4)
        short = self._get_affinity_channels(pred_affinity_np, [0])  # (4,H,W)
        boundary_short = 1.0 - np.mean(short, axis=0)  # high at boundaries

        # Medium-range blend
        medium_weight = self.config.get('affinity_medium_range_weight', 0.3)
        if medium_weight > 0 and len(self.shifts) >= 3:
            # shifts[1]=3, shifts[2]=5
            med_indices = [i for i in [1, 2] if i < len(self.shifts)]
            if med_indices:
                med = self._get_affinity_channels(pred_affinity_np, med_indices)
                boundary_med = 1.0 - np.mean(med, axis=0)
                boundary = (1.0 - medium_weight) * boundary_short + medium_weight * boundary_med
            else:
                boundary = boundary_short
        else:
            boundary = boundary_short

        return np.clip(boundary, 0, 1).astype(np.float64)

    def _compute_interior_map(self, pred_affinity_np: np.ndarray,
                              semantic_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute interior probability from short-range affinities.

        When use_semantic_for_seeds is True, the semantic probability is used
        as a **mask** (zero out background) rather than a multiplier, so that
        the affinity values keep their original range for thresholding.

        Args:
            pred_affinity_np: (C, H, W) predicted affinities
            semantic_prob: (H, W) optional semantic foreground probability.

        Returns:
            interior: (H, W) float in [0,1], high = inside an instance
        """
        short = self._get_affinity_channels(pred_affinity_np, [0])
        interior = np.clip(np.mean(short, axis=0), 0, 1).astype(np.float64)

        if semantic_prob is not None and self.config.get('use_semantic_for_seeds', True):
            # Use semantic prob as a gate: zero out low-confidence background
            # but preserve affinity magnitude in foreground regions
            sem_thresh = self.config.get('semantic_seed_gate_thresh', 0.2)
            semantic_gate = (semantic_prob > sem_thresh).astype(np.float64)
            interior = interior * semantic_gate

        return interior

    def _filter_seeds_by_boundary(self, markers: np.ndarray,
                                   boundary: np.ndarray) -> np.ndarray:
        """Remove seed regions that overlap heavily with high-boundary areas.

        A seed sitting on a boundary is likely a false seed created by noise.
        If the mean boundary value within a seed region exceeds the threshold,
        that seed is removed.

        Args:
            markers: (H, W) labeled seed map
            boundary: (H, W) boundary probability map

        Returns:
            markers with bad seeds zeroed out
        """
        thresh = self.config.get('seed_boundary_overlap_thresh', 0.5)
        if thresh <= 0:
            return markers

        for region in regionprops(markers):
            seed_mask = markers == region.label
            mean_boundary = boundary[seed_mask].mean()
            if mean_boundary > thresh:
                markers[seed_mask] = 0
        return markers

    def _merge_close_seeds(self, markers: np.ndarray,
                           boundary: np.ndarray) -> np.ndarray:
        """Merge seed regions that are very close AND have low boundary between them.

        Two nearby seeds that are NOT separated by a real boundary are likely
        fragments of the same object and should be merged.

        Args:
            markers: (H, W) labeled seed map
            boundary: (H, W) boundary probability map

        Returns:
            markers with merged seeds
        """
        merge_dist = self.config.get('seed_merge_distance', 5)
        merge_boundary_thresh = self.config.get('seed_merge_boundary_thresh', 0.3)

        if merge_dist <= 0:
            return markers

        props = regionprops(markers)
        if len(props) < 2:
            return markers

        # Build centroid array for fast distance computation
        centroids = np.array([r.centroid for r in props])
        label_ids = np.array([r.label for r in props])

        # Union-Find for merging
        parent = {lid: lid for lid in label_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Check all pairs within merge_dist
        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                dist = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                if dist > merge_dist * 3:
                    # Quick skip for clearly distant seeds
                    continue

                # Compute min distance between the two seed regions
                mask_i = markers == label_ids[i]
                mask_j = markers == label_ids[j]
                dt_i = distance_transform_edt(~mask_i)
                min_gap = dt_i[mask_j].min() if mask_j.any() else float('inf')

                if min_gap > merge_dist:
                    continue

                # Check boundary strength between them: sample boundary along
                # the line between centroids
                ci, cj = centroids[i], centroids[j]
                n_samples = max(int(dist), 3)
                rows = np.linspace(ci[0], cj[0], n_samples).astype(int)
                cols = np.linspace(ci[1], cj[1], n_samples).astype(int)
                rows = np.clip(rows, 0, boundary.shape[0] - 1)
                cols = np.clip(cols, 0, boundary.shape[1] - 1)
                mean_boundary_between = boundary[rows, cols].mean()

                if mean_boundary_between < merge_boundary_thresh:
                    union(label_ids[i], label_ids[j])

        # Apply merges: remap all labels to their union-find root
        merged = markers.copy()
        for lid in label_ids:
            root = find(lid)
            if root != lid:
                merged[markers == lid] = root

        # Relabel sequentially WITHOUT re-running connected components
        # (sk_label would undo the merge for non-adjacent regions)
        unique_ids = sorted(set(int(x) for x in np.unique(merged) if x != 0))
        relabeled = np.zeros_like(merged)
        for new_id, old_id in enumerate(unique_ids, start=1):
            relabeled[merged == old_id] = new_id
        return relabeled.astype(np.int32)

    def affinity_watershed(self, pred_affinity_np: np.ndarray,
                           semantic_fg: np.ndarray,
                           semantic_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """Affinity-based seeded watershed with seed merging and filtering.

        Pipeline:
        1. Compute boundary map from short+medium range affinities
        2. Compute interior map (optionally weighted by semantic probability)
        3. Threshold interior to find seed regions, erode to separate
        4. Filter seeds: remove those overlapping high-boundary areas
        5. Merge seeds: combine nearby seeds with low boundary between them
        6. Watershed on boundary map, masked by semantic foreground

        Args:
            pred_affinity_np: (C, H, W) predicted affinities, values in [0,1]
            semantic_fg: (H, W) binary foreground mask
            semantic_prob: (H, W) optional semantic foreground probability [0,1]

        Returns:
            labels: (H, W) int instance label map (0 = background)
        """
        boundary = self._compute_boundary_map(pred_affinity_np)
        interior = self._compute_interior_map(pred_affinity_np, semantic_prob)

        # Threshold interior to get seed regions
        seed_thresh = self.config.get('affinity_seed_threshold', 0.92)
        seed_mask = (interior > seed_thresh) & (semantic_fg > 0)

        # Erode seeds to separate touching instances
        seed_erode_radius = self.config.get('affinity_seed_erode_radius', 6)
        if seed_erode_radius > 0:
            seed_mask = binary_erosion(seed_mask, disk(seed_erode_radius))

        # Label seed connected components
        markers = sk_label(seed_mask).astype(np.int32)

        # Filter: remove seeds sitting on boundaries
        markers = self._filter_seeds_by_boundary(markers, boundary)

        # Merge: combine close seeds with weak boundary between them
        markers = self._merge_close_seeds(markers, boundary)

        # If no seeds found, fall back to distance-transform seeds
        if markers.max() == 0:
            distance = distance_transform_edt(semantic_fg)
            min_dist = self.config.get('affinity_peak_min_distance', 10)
            coords = peak_local_max(
                distance, min_distance=min_dist,
                labels=semantic_fg, footprint=np.ones((3, 3))
            )
            local_maxi = np.zeros(distance.shape, dtype=bool)
            if coords.size > 0:
                local_maxi[tuple(coords.T)] = True
            markers = ndi.label(local_maxi)[0]

        # Watershed on boundary map with affinity-derived seeds
        labels = watershed(boundary, markers, mask=semantic_fg)
        return labels.astype(np.int32)

    def affinity_to_instances(self, pred_affinity_np: np.ndarray,
                              semantic_fg: np.ndarray,
                              semantic_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """Instance segmentation from predicted affinities: a seeded watershed on
        the boundary map (seeds from the interior/affinity map, grown under the
        semantic foreground). This is the only method used in the paper."""
        return self.affinity_watershed(pred_affinity_np, semantic_fg, semantic_prob)

    # ==================================================================
    # Agglomerative merging of over-segmented instances
    # ==================================================================
    def _agglomerate_instances(self, labels: np.ndarray,
                                boundary: np.ndarray) -> np.ndarray:
        """Hierarchical agglomerative merging of over-segmented instances.

        After watershed, adjacent instances whose shared boundary has weak
        affinity signal (model does not predict a real boundary between them)
        are iteratively merged.  Each iteration:
          1. Find all adjacent instance pairs (4-connected) and compute the
             mean boundary strength along their shared edge.
          2. Merge the pair with the weakest boundary, if below threshold.
          3. Repeat until no more merges qualify.

        Args:
            labels:   (H, W) int instance label map (0 = background)
            boundary: (H, W) float boundary probability map [0,1]

        Returns:
            labels: (H, W) int instance label map after merging
        """
        agglom_thresh = self.config.get('agglomeration_boundary_thresh', 0.35)
        min_edge_len = self.config.get('agglomeration_min_boundary_length', 5)

        labels = labels.copy().astype(np.int32)
        n_merges = 0

        while True:
            # --- Build adjacency graph (vectorised) ---
            edges = {}  # (lo_label, hi_label) -> (sum_boundary, count)
            max_id = int(labels.max()) + 1

            for dy, dx in [(0, 1), (1, 0)]:
                if dy == 1:
                    a = labels[:-1, :].ravel()
                    b = labels[1:, :].ravel()
                    bnd = (0.5 * (boundary[:-1, :] + boundary[1:, :])).ravel()
                else:
                    a = labels[:, :-1].ravel()
                    b = labels[:, 1:].ravel()
                    bnd = (0.5 * (boundary[:, :-1] + boundary[:, 1:])).ravel()

                diff_mask = (a != b) & (a > 0) & (b > 0)
                if not np.any(diff_mask):
                    continue

                lo = np.minimum(a[diff_mask], b[diff_mask]).astype(np.int64)
                hi = np.maximum(a[diff_mask], b[diff_mask]).astype(np.int64)
                bv = bnd[diff_mask]

                pair_key = lo * max_id + hi
                unique_keys = np.unique(pair_key)

                for key in unique_keys:
                    la_val = int(key // max_id)
                    lb_val = int(key % max_id)
                    mask = pair_key == key
                    s = float(bv[mask].sum())
                    c = int(mask.sum())
                    p = (la_val, lb_val)
                    if p in edges:
                        edges[p] = (edges[p][0] + s, edges[p][1] + c)
                    else:
                        edges[p] = (s, c)

            # --- Find the weakest-boundary adjacent pair ---
            best_pair = None
            best_mean = float('inf')

            for pair, (s, c) in edges.items():
                if c < min_edge_len:
                    continue
                mean_bnd = s / c
                if mean_bnd < agglom_thresh and mean_bnd < best_mean:
                    best_mean = mean_bnd
                    best_pair = pair

            if best_pair is None:
                break  # no more merges qualify

            # Merge: relabel the higher-ID instance to the lower-ID
            labels[labels == best_pair[1]] = best_pair[0]
            n_merges += 1

        # Relabel sequentially
        if n_merges > 0:
            unique = sorted(set(int(x) for x in np.unique(labels) if x != 0))
            relabeled = np.zeros_like(labels)
            for new_id, old_id in enumerate(unique, start=1):
                relabeled[labels == old_id] = new_id
            labels = relabeled

        return labels

    # ==================================================================
    # Instance label post-processing
    # ==================================================================
    def post_process_instances(self, labels: np.ndarray) -> np.ndarray:
        """Clean up instance labels.

        1. Remove small fragments (min_instance_size)
        2. Fill holes within each instance
        3. Smooth instance boundaries with morphological closing
        4. Relabel sequentially
        """
        min_size = self.config.get('min_instance_size',
                                   self.config.get('min_object_size', 250))

        # Remove small instances
        if min_size > 0:
            for region in regionprops(labels):
                if region.area < min_size:
                    labels[labels == region.label] = 0

        # Fill holes within each instance
        if self.config.get('instance_fill_holes', True):
            out = np.zeros_like(labels)
            for lbl in np.unique(labels):
                if lbl == 0:
                    continue
                mask = labels == lbl
                mask = binary_fill_holes(mask)
                out[mask] = lbl
            labels = out

        # Smooth instance boundaries
        smooth_radius = self.config.get('instance_smooth_radius', 2)
        if smooth_radius > 0:
            se = disk(smooth_radius)
            out = np.zeros_like(labels)
            for lbl in np.unique(labels):
                if lbl == 0:
                    continue
                mask = (labels == lbl).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se.astype(np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se.astype(np.uint8))
                out[mask > 0] = lbl
            labels = out

        # Relabel sequentially (without re-running CC which would merge touching)
        unique = np.unique(labels)
        relabeled = np.zeros_like(labels)
        new_id = 0
        for old_id in unique:
            if old_id == 0:
                continue
            new_id += 1
            relabeled[labels == old_id] = new_id
        return relabeled

    # ==================================================================
    # Instance segmentation metrics
    # ==================================================================
