"""Trimmed linking_core: ONLY the geometric z_postprocess used by the
pure-2D post-processing chain (extracted verbatim from the full module)."""
import numpy as np


def _slice_fg_iou(m1, m2):
    """Foreground IoU between two boolean slice masks."""
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return inter / union if union > 0 else 0.0


def z_postprocess_relabel(volume_3d,
                          min_overlap=30,
                          enforce_one_to_one=True,
                          fragment_max_span=2,
                          bad_min_fg_frac=0.001,
                          bad_max_fg_frac=0.9,
                          bad_collapsed_max_count=5,
                          bad_count_outlier_ratio=3.0,
                          bad_neighbor_overlap_thresh=0.3,
                          relink_max_gap=2,
                          relink_min_overlap=None,
                          col_solidity_min_present=0.6,
                          col_solidity_max_gap=2,
                          verbose=True):
    """Geometric z-direction post-processing on a per-slice-labelled volume.

    Pure geometry — no model, no z-affinity. Repairs horizontal banding (a
    cell's column shattered into many short-lived IDs) by propagating one
    consistent ID down each cell's column via maximal slice-to-slice overlap,
    bridging "bad" slices, re-linking columns broken by short gaps, and filling
    brief interior holes so columns become solid.

    Input: ``volume_3d`` (N, H, W) where each slice is independently labelled
    (e.g. the per-slice 2D instance volume ``instance_labels_2d.npy``). Each
    distinct non-zero label within a slice is treated as one instance. Returns
    ``(out_volume (N, H, W) int32, info)``.

    Stage 1 - bad-slice detection + bridging.
        Slice z is flagged "bad" if ANY of:
          * foreground fraction < ``bad_min_fg_frac`` (empty / near-empty), or
          * foreground > ``bad_max_fg_frac`` AND instance count <=
            ``bad_collapsed_max_count`` (collapsed into a near-full blob — dense
            *valid* tissue also fills the frame, so the low-count condition is
            required to avoid flagging it), or
          * its instance count is a strong outlier vs the median of its nearest
            good neighbours (ratio > ``bad_count_outlier_ratio`` either way), or
          * it overlaps poorly with BOTH neighbours (fg-IoU <
            ``bad_neighbor_overlap_thresh``) while the two neighbours overlap
            each other well.
        Bad slices are excluded from identity propagation (previous good slice
        links straight to next good slice) and filled from the nearest good
        slice so the volume stays complete.

    Stage 2 - overlap re-stitch over good slices. Each instance on the next good
        slice inherits the ID of the previous good slice's instance it overlaps
        most (>= ``min_overlap``). ``enforce_one_to_one`` (default) lets each
        previous ID be inherited by only its best successor — a geometry-only
        stand-in for the "don't merge across a strong 2D boundary" guard
        (boundary maps are not persisted). Unmatched instances start a new ID.

    Stage 3 - re-link across short gaps. A column whose ID flickers (the cell
        vanishes for 1..``relink_max_gap`` slices then returns) is reconnected:
        a column ending at z is linked to one starting at z+1..z+1+gap when
        their footprints overlap >= ``relink_min_overlap`` (defaults to
        ``min_overlap``). Endpoint+overlap matching, so two continuous adjacent
        cells (no end/start gap, no spatial overlap) are NOT merged.

    Stage 4 - fragment reabsorb + column-solidity fill. Tiny z-fragments (span
        <= ``fragment_max_span``) are reabsorbed into the larger column they
        most overlap; then for each column present on >=
        ``col_solidity_min_present`` of its z-range, interior holes up to
        ``col_solidity_max_gap`` long are filled by copying the nearest present
        slice's footprint (background pixels only).

    info = {'bad_slices', 'bad_reason', 'n_before', 'n_after_stage2',
            'n_after_relink', 'n_after', 'n_relink_merges', 'n_holes_filled',
            'fg_frac', 'counts'}.
    """
    vol = volume_3d.astype(np.int32)
    N, H, W = vol.shape
    relink_ov = relink_min_overlap if relink_min_overlap is not None else min_overlap

    if verbose:
        print("  z_postprocess params: "
              f"min_overlap={min_overlap} one_to_one={enforce_one_to_one} "
              f"fragment_max_span={fragment_max_span}", flush=True)
        print(f"    bad-slice: min_fg={bad_min_fg_frac} "
              f"max_fg={bad_max_fg_frac}&count<={bad_collapsed_max_count} "
              f"count_ratio={bad_count_outlier_ratio} "
              f"neigh_overlap={bad_neighbor_overlap_thresh}", flush=True)
        print(f"    relink: max_gap={relink_max_gap} min_overlap={relink_ov} | "
              f"solidity: min_present={col_solidity_min_present} "
              f"max_gap={col_solidity_max_gap}", flush=True)

    def n_inst(a):
        u = np.unique(a)
        return int(len(u) - (1 if 0 in u else 0))

    fg_masks = [vol[z] > 0 for z in range(N)]
    fg_frac = np.array([float(m.mean()) for m in fg_masks])
    counts = np.array([n_inst(vol[z]) for z in range(N)])
    n_before = int(counts.sum())

    # ---- Stage 1: detect bad slices ----
    bad_reason = {}
    near_empty = fg_frac < bad_min_fg_frac
    # Collapsed = near-full frame AND abnormally few instances. Dense valid
    # tissue also fills the frame but has many instances, so it is NOT flagged.
    collapsed = (fg_frac > bad_max_fg_frac) & (counts <= bad_collapsed_max_count)
    good0 = [z for z in range(N) if not (near_empty[z] or collapsed[z])]
    good0_set = set(good0)
    bad = set()
    for z in np.where(near_empty)[0].tolist():
        bad.add(z); bad_reason[z] = 'empty'
    for z in np.where(collapsed)[0].tolist():
        bad.add(z); bad_reason[z] = 'collapsed'

    def nearest_in(side_iter):
        for zz in side_iter:
            if zz in good0_set:
                return zz
        return None

    for z in good0:
        prev_g = nearest_in(range(z - 1, -1, -1))
        next_g = nearest_in(range(z + 1, N))
        reason = None

        neigh = [g for g in (prev_g, next_g) if g is not None]
        if neigh and counts[z] > 0:
            med = float(np.median([counts[g] for g in neigh]))
            if med > 0:
                ratio = max(counts[z] / med, med / counts[z])
                if ratio > bad_count_outlier_ratio:
                    reason = 'count_outlier'

        if reason is None and prev_g is not None and next_g is not None:
            iou_prev = _slice_fg_iou(fg_masks[z], fg_masks[prev_g])
            iou_next = _slice_fg_iou(fg_masks[z], fg_masks[next_g])
            iou_sides = _slice_fg_iou(fg_masks[prev_g], fg_masks[next_g])
            if (iou_prev < bad_neighbor_overlap_thresh and
                    iou_next < bad_neighbor_overlap_thresh and
                    iou_sides > bad_neighbor_overlap_thresh):
                reason = 'neighbor_break'

        if reason is not None:
            bad.add(z); bad_reason[z] = reason

    good = [z for z in range(N) if z not in bad]
    bad_slices = sorted(bad)

    if not good:
        if verbose:
            print("  z_postprocess: every slice flagged bad — returning input.",
                  flush=True)
        return vol.copy(), {'bad_slices': bad_slices, 'bad_reason': bad_reason,
                            'n_before': n_before, 'n_after_stage2': n_before,
                            'n_after_relink': n_before, 'n_after': n_before,
                            'n_relink_merges': 0, 'n_holes_filled': 0,
                            'fg_frac': fg_frac, 'counts': counts}

    # ---- Stage 2: overlap re-stitch over good slices ----
    out = np.zeros((N, H, W), dtype=np.int32)
    global_next = 0

    # Seed the first good slice.
    g0 = good[0]
    remap0 = {}
    for lid in np.unique(vol[g0]):
        if lid == 0:
            continue
        global_next += 1
        remap0[int(lid)] = global_next
    max_l0 = int(vol[g0].max())
    lut0 = np.zeros(max_l0 + 1, dtype=np.int32)
    for lid, gid in remap0.items():
        lut0[lid] = gid
    out[g0] = lut0[vol[g0]]
    prev_g = g0

    for g in good[1:]:
        prev_lab = out[prev_g]          # already global IDs
        cur_lab = vol[g]                # per-slice local IDs
        both = (prev_lab > 0) & (cur_lab > 0)

        assign = {}  # cur_local -> prev_global
        if both.any():
            a = prev_lab[both].astype(np.int64)   # prev global
            b = cur_lab[both].astype(np.int64)    # cur local
            max_b = int(cur_lab.max()) + 1
            keys = a * max_b + b
            uk, cnts = np.unique(keys, return_counts=True)
            cands = []  # (count, prev_global, cur_local)
            for k, c in zip(uk, cnts):
                if int(c) < min_overlap:
                    continue
                cands.append((int(c), int(k // max_b), int(k % max_b)))
            cands.sort(reverse=True)
            used_prev = set()
            used_cur = set()
            for c, pg, cl in cands:
                if cl in used_cur:
                    continue
                if enforce_one_to_one and pg in used_prev:
                    continue
                assign[cl] = pg
                used_cur.add(cl)
                used_prev.add(pg)

        max_lc = int(cur_lab.max())
        lut = np.zeros(max_lc + 1, dtype=np.int32)
        for lid in np.unique(cur_lab):
            if lid == 0:
                continue
            lid = int(lid)
            if lid in assign:
                lut[lid] = assign[lid]
            else:
                global_next += 1
                lut[lid] = global_next
        out[g] = lut[cur_lab]
        prev_g = g

    # Bridge/fill bad slices by copying the nearest good slice's final labels.
    good_arr = np.array(good)
    for z in bad_slices:
        nearest = int(good_arr[np.argmin(np.abs(good_arr - z))])
        out[z] = out[nearest]

    n_after_stage2 = n_inst(out)

    # ---- Stage 3: re-link columns broken by short (<= relink_max_gap) gaps ----
    n_relink_merges = 0
    if relink_max_gap >= 1 and int(out.max()) > 0:
        slices_of = {}
        for z in range(N):
            for lid in np.unique(out[z]):
                if lid == 0:
                    continue
                slices_of.setdefault(int(lid), []).append(z)
        zmax = {g: s[-1] for g, s in slices_of.items()}
        zmin = {g: s[0] for g, s in slices_of.items()}
        starts_by_z = {}
        for g, mn in zmin.items():
            starts_by_z.setdefault(mn, []).append(g)

        max_gid = int(out.max())
        parent3 = list(range(max_gid + 1))

        def find3(x):
            while parent3[x] != x:
                parent3[x] = parent3[parent3[x]]
                x = parent3[x]
            return x

        def union3(a, b):
            ra, rb = find3(a), find3(b)
            if ra != rb:
                parent3[max(ra, rb)] = min(ra, rb)

        used_target = set()
        # Link each column's bottom end to a column that starts within the gap
        # window and overlaps its footprint (endpoint + overlap only — two
        # continuous adjacent cells have no end/start gap and won't merge).
        for g in sorted(slices_of, key=lambda x: zmax[x]):
            za = zmax[g]
            a_mask = out[za] == g
            best, best_ov = None, 0
            for zb in range(za + 1, za + 2 + relink_max_gap):
                if zb >= N:
                    break
                for h in starts_by_z.get(zb, []):
                    if h in used_target or find3(h) == find3(g):
                        continue
                    ov = int((a_mask & (out[zb] == h)).sum())
                    if ov >= relink_ov and ov > best_ov:
                        best_ov, best = ov, h
            if best is not None:
                union3(g, best)
                used_target.add(best)
                n_relink_merges += 1
        if n_relink_merges:
            lut3 = np.array([find3(i) for i in range(max_gid + 1)],
                            dtype=np.int32)
            out = lut3[out]

    n_after_relink = n_inst(out)

    # ---- Stage 4a: reabsorb tiny z-fragments into larger overlapping cols ----
    slices_of = {}
    for z in range(N):
        for lid in np.unique(out[z]):
            if lid == 0:
                continue
            slices_of.setdefault(int(lid), set()).add(z)

    frags = [lid for lid, zs in slices_of.items()
             if len(zs) <= fragment_max_span]
    merge_map = {}

    def resolve(x):
        while x in merge_map:
            x = merge_map[x]
        return x

    for f in sorted(frags, key=lambda x: len(slices_of[x])):
        best_target, best_ov = None, 0
        for z in sorted(slices_of[f]):
            fmask = out[z] == f
            for zz in (z - 1, z + 1):
                if zz < 0 or zz >= N:
                    continue
                neigh_ids = out[zz][fmask]
                neigh_ids = neigh_ids[neigh_ids > 0]
                neigh_ids = neigh_ids[neigh_ids != f]
                if neigh_ids.size == 0:
                    continue
                vals, cs = np.unique(neigh_ids, return_counts=True)
                for v, c in zip(vals, cs):
                    v = int(resolve(int(v)))
                    if v == f:
                        continue
                    if len(slices_of.get(v, ())) <= len(slices_of[f]):
                        continue
                    if int(c) > best_ov:
                        best_ov = int(c)
                        best_target = v
        if best_target is not None and best_ov > 0:
            merge_map[f] = best_target

    if merge_map:
        max_id = max(slices_of.keys())
        lut = np.arange(max_id + 1, dtype=np.int32)
        for f in merge_map:
            lut[f] = resolve(f)
        out = lut[out]

    # ---- Stage 4b: column-solidity fill of brief interior holes ----
    n_holes_filled = 0
    slices_of = {}
    for z in range(N):
        for lid in np.unique(out[z]):
            if lid == 0:
                continue
            slices_of.setdefault(int(lid), []).append(z)
    for g, present in slices_of.items():
        zmn, zmx = present[0], present[-1]
        span = zmx - zmn + 1
        if span <= 1 or (len(present) / span) < col_solidity_min_present:
            continue
        present_set = set(present)
        holes = [z for z in range(zmn, zmx + 1) if z not in present_set]
        if not holes:
            continue
        runs, cur = [], [holes[0]]
        for z in holes[1:]:
            if z == cur[-1] + 1:
                cur.append(z)
            else:
                runs.append(cur)
                cur = [z]
        runs.append(cur)
        for run in runs:
            if len(run) > col_solidity_max_gap:
                continue
            for z in run:
                npz = min(present, key=lambda p: abs(p - z))
                fill = (out[npz] == g) & (out[z] == 0)
                if fill.any():
                    out[z][fill] = g
                    n_holes_filled += 1

    # Relabel consecutively.
    uniq = sorted(set(int(x) for x in np.unique(out) if x != 0))
    relabel = np.zeros((max(uniq) + 1) if uniq else 1, dtype=np.int32)
    for new_id, old in enumerate(uniq, start=1):
        relabel[old] = new_id
    out = relabel[out]

    n_after = n_inst(out)

    if verbose:
        show = bad_slices if len(bad_slices) <= 30 else bad_slices[:30] + ['...']
        print(f"  z_postprocess: {len(bad_slices)} bad slices flagged {show}",
              flush=True)
        print(f"  z_postprocess: instances {n_before} -> {n_after_stage2} "
              f"(re-stitch) -> {n_after_relink} (gap re-link, "
              f"{n_relink_merges} merges) -> {n_after} (reabsorb+solidity, "
              f"{n_holes_filled} holes filled)", flush=True)

    return out, {'bad_slices': bad_slices, 'bad_reason': bad_reason,
                 'n_before': n_before, 'n_after_stage2': n_after_stage2,
                 'n_after_relink': n_after_relink, 'n_after': n_after,
                 'n_relink_merges': n_relink_merges,
                 'n_holes_filled': n_holes_filled,
                 'fg_frac': fg_frac, 'counts': counts}
