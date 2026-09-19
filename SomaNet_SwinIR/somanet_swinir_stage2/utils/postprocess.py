""" post-processing chain.
Order: z_postprocess (geometric stitch) -> v4_jump_replace -> z_consistency ->
object_transient_remove -> zspan_filter.
"""
import os
import numpy as np
import nibabel as nib
from PIL import Image
from scipy import ndimage as ndi

from utils.linking_core import z_postprocess_relabel

# z-stitch (2D->3D) parameters. relink_max_gap is the REAL 3D stitch gap that
# produces pred_FINAL; it is fixed to 5 (the "g5" setting reported in the paper).
ZPP = dict(min_overlap=int(os.environ.get('SW_ZPP_MINOV', '20')),
           fragment_max_span=int(os.environ.get('SW_ZPP_SPAN', '10')),
           relink_max_gap=5)   # paper: g5


# ------ Helpers
def sort_by_z(files):
    def kz(p):
        b = os.path.splitext(os.path.basename(p))[0]
        for part in reversed(b.split('_')):
            try: return int(part)
            except ValueError: pass
        return 0
    return sorted(files, key=kz)


def relabel(v):
    uq = sorted(set(int(x) for x in np.unique(v) if x != 0))
    lut = np.zeros((max(uq) + 1) if uq else 1, dtype=np.int32)
    for n, o in enumerate(uq, 1):
        lut[o] = n
    return lut[v]


def save_nii(vol, path):
    """Amira-ready: uint16 + explicit cal window (avoids the black-in-Amira bug)."""
    arr = np.transpose(vol, (1, 2, 0)); vmax = int(arr.max())
    arr = arr.astype(np.uint16) if vmax <= 65535 else arr.astype(np.int32)
    img = nib.Nifti1Image(arr, np.eye(4))
    img.header['cal_min'] = 0; img.header['cal_max'] = vmax
    nib.save(img, path)


def _local_median(x, half=2):
    n = len(x); m = np.zeros(n)
    for z in range(n):
        nb = [x[j] for j in range(max(0, z - half), min(n, z + half + 1)) if j != z]
        m[z] = np.median(nb) if nb else x[z]
    return m


def _fg_iou(a, b):
    A = a > 0; B = b > 0; u = int((A | B).sum())
    return int((A & B).sum()) / u if u else 1.0


# -------  pipeline
def stitch(cache_npz):
    """Geometric z_postprocess on the cached per-slice 2D instances -> (Z,H,W)."""
    z = np.load(cache_npz, allow_pickle=True)
    n = len(z['sorted_z'])
    stack = np.stack([z[str(k)] for k in range(1, n + 1)], axis=0).astype(np.int32)
    out, _ = z_postprocess_relabel(stack, **ZPP)
    return out


def v4_jump_replace(out, img_files, half=2, mean_jump=25., black_abs=20.,
                    blur_frac=0.4, big_ratio=1.8, iou_bad=0.3, iou_ok=0.3):
    """Replace corrupt/balloon slices (black/white/blurry/big-jump/discontinuity)
    with the before/after consensus of the nearest good slices."""
    N = out.shape[0]
    em_mean = np.zeros(N); em_focus = np.zeros(N)
    for i, f in enumerate(img_files[:N]):
        e = np.array(Image.open(f).convert('L')).astype(np.float32)
        em_mean[i] = e.mean(); em_focus[i] = ndi.laplace(e).var()
    lm, lf = _local_median(em_mean, half), _local_median(em_focus, half)
    black = em_mean < black_abs
    white = (em_mean - lm) > mean_jump
    blurry = (lf > 0) & (em_focus < blur_frac * lf)
    fg = np.array([float((out[i] > 0).mean()) for i in range(N)])
    ma = np.array([max([c for l, c in zip(*np.unique(out[i], return_counts=True)) if l] or [0]) for i in range(N)])
    lfg, lma = _local_median(fg, half), _local_median(ma, half)
    big = ((lfg > 0) & (fg > big_ratio * lfg)) | ((lma > 0) & (ma > big_ratio * lma))
    ip = np.array([_fg_iou(out[i], out[i - 1]) if i else 1 for i in range(N)])
    nx = np.array([_fg_iou(out[i], out[i + 1]) if i < N - 1 else 1 for i in range(N)])
    sd = np.array([_fg_iou(out[i - 1], out[i + 1]) if 0 < i < N - 1 else 1 for i in range(N)])
    jump = black | white | blurry | big | ((np.minimum(ip, nx) < iou_bad) & (sd > iou_ok))
    good = np.array([i for i in range(N) if not jump[i]])
    for i in np.where(jump)[0]:
        b, a = good[good < i], good[good > i]
        if b.size and a.size:
            bb, aa = out[int(b[-1])], out[int(a[0])]
            out[i] = np.where((bb > 0) & (aa > 0), bb, 0)
    return out


def z_consistency(out, passes=2):
    """Voxel-level: where both z-neighbours agree and the centre differs, snap it."""
    N = out.shape[0]
    for _ in range(passes):
        o = out.copy(); ch = 0
        for i in range(1, N - 1):
            ag = (out[i - 1] == out[i + 1]) & (out[i - 1] != out[i])
            if ag.any():
                o[i][ag] = out[i - 1][ag]; ch += 1
        out = o
        if ch == 0:
            break
    return out


def object_transient_remove(out, gate=400):
    """Remove single-slice transient parts (flanges/specks) smaller than `gate`,
    snapping them to the both-neighbour consensus."""
    N = out.shape[0]
    for i in range(1, N - 1):
        cur, pv, nx = out[i], out[i - 1], out[i + 1]
        trans = (cur > 0) & (cur != pv) & (cur != nx)
        if not trans.any():
            continue
        lab, nl = ndi.label(trans)
        if nl == 0:
            continue
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, nl + 1))
        cons = np.where(pv == nx, pv, 0)
        small = np.where(sizes < gate)[0] + 1
        if small.size:
            m = np.isin(lab, small); out[i][m] = cons[m]
    return out


def zspan_filter(out, min_span=3):
    """Remove instances present in fewer than `min_span` slices (FP specks)."""
    N = out.shape[0]; span = {}
    for i in range(N):
        for l in np.unique(out[i]):
            if l:
                span[int(l)] = span.get(int(l), 0) + 1
    rm = {l for l, s in span.items() if s < min_span}
    if rm:
        lut = np.arange(int(out.max()) + 1, dtype=np.int32)
        for r in rm:
            lut[r] = 0
        out = lut[out]
    return out


def run_chain(cache_npz, img_files, obj_gate=400, min_span=3):
    """Full chain -> final (Z,H,W) int32 label volume (consecutive labels)."""
    out = stitch(cache_npz)
    out = v4_jump_replace(out, img_files)
    out = z_consistency(out)
    out = object_transient_remove(out, gate=obj_gate)
    out = zspan_filter(out, min_span=min_span)
    return relabel(out)
