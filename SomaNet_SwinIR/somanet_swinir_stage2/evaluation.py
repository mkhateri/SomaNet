"""evaluation.py - 3D instance-segmentation metrics.

Reports 7 metrics, in this order:
  Dice (up)   - monai.metrics.DiceMetric   (semantic foreground, 3D)
  IoU  (up)   - monai.metrics.MeanIoU      (semantic foreground, 3D)
  PQ   (up)   - custom 3D panoptic quality (PQ = SQ x RQ, matched at instance IoU >= 0.5)
  ARAND (down)     - skimage.metrics.adapted_rand_error        (3D)
  VOItotal (down)  - variation of information, split + merge    (skimage, 3D)
  VOIsplit (down)  - over-segmentation part of VOI
  VOImerge (down)  - under-segmentation part of VOI

Metrics are computed only where GT > 0 (the annotated region).

Usage:  python -u evaluation.py <gt_mask.nii.gz> <pred.nii.gz>
"""
import sys
import numpy as np
import nibabel as nib
import torch
from scipy.optimize import linear_sum_assignment
from monai.metrics import DiceMetric, MeanIoU
from skimage.metrics import variation_of_information, adapted_rand_error


def _load(path):
    return np.asarray(nib.load(path).dataobj).astype(np.int32)


def _dice_iou(gt, pred):
    """MONAI Dice + IoU on the binary foreground (B=1, C=1)."""
    yp = torch.from_numpy((pred > 0).astype(np.float32))[None, None]
    yt = torch.from_numpy((gt > 0).astype(np.float32))[None, None]
    dm = DiceMetric(include_background=True, reduction="mean"); dm(yp, yt)
    im = MeanIoU(include_background=True, reduction="mean"); im(yp, yt)
    return float(dm.aggregate().item()), float(im.aggregate().item())


def _pq(gt, pred):
    """3D panoptic quality (PQ = SQ x RQ) at instance IoU >= 0.5, Hungarian one-to-one."""
    min_size = 100
    eps = 1e-9           # float drift guard: 0.5 >= 0.5 - eps stays True
    pid = np.unique(pred[pred > 0])
    gid = np.unique(gt[gt > 0])
    parea = np.bincount(pred.ravel())[pid] if pid.size else np.zeros(0, np.int64)
    garea = np.bincount(gt.ravel())[gid] if gid.size else np.zeros(0, np.int64)
    if min_size > 1 and pid.size:                       # drop edge-clips / debris
        keep = parea >= min_size
        pid, parea = pid[keep], parea[keep]
    if pid.size == 0 or gid.size == 0:
        return 0.0

    # dense (n_pred, n_gt) IoU matrix in a single pass over the overlapping voxels
    prow = {int(p): i for i, p in enumerate(pid)}
    gcol = {int(g): j for j, g in enumerate(gid)}
    both = (pred > 0) & (gt > 0)
    K = int(gid.max()) + 1
    keys, counts = np.unique(pred[both].astype(np.int64) * K + gt[both].astype(np.int64),
                             return_counts=True)
    inter = np.zeros((pid.size, gid.size), np.float64)
    for k, c in zip(keys, counts):
        p, g = int(k // K), int(k % K)
        if p in prow:
            inter[prow[p], gcol[g]] = c
    union = parea[:, None] + garea[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    # optimal one-to-one match at IoU >= 0.5
    ok = iou >= 0.5 - eps
    r, c = linear_sum_assignment(-(iou * ok))
    sel = ok[r, c]
    tp = int(sel.sum())
    fp, fn = pid.size - tp, gid.size - tp
    if tp == 0:
        return 0.0
    sq = float(iou[r, c][sel].sum()) / tp                # avg IoU of matched pairs
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return sq * rq


def evaluate(gt, pred, enable=True):
    """Return the 7 reported metrics as a dict (scored where GT > 0)."""
    pred = pred.copy()
    if enable:
        pred[gt == 0] = 0                               # score only the annotated region
        m = gt > 0
        voi = variation_of_information(gt[m], pred[m])
        arand = adapted_rand_error(gt[m], pred[m])[0]
    else:
        voi = variation_of_information(gt, pred)
        arand = adapted_rand_error(gt, pred)[0]

    dice, iou = _dice_iou(gt, pred)
    return dict(
        Dice=dice, IoU=iou, PQ=_pq(gt, pred), aRand=float(arand),
        VI_split=float(voi[0]), VI_merge=float(voi[1]), VI_total=float(voi[0] + voi[1]),
    )


def _fmt(r):
    return (f"    Dice     = {r['Dice']:.3f}   (up)\n"
            f"    IoU      = {r['IoU']:.3f}   (up)\n"
            f"    PQ       = {r['PQ']:.3f}   (up)\n"
            f"    ARAND    = {r['aRand']:.3f}   (down)\n"
            f"    VOItotal = {r['VI_total']:.3f}   (down)\n"
            f"    VOIsplit = {r['VI_split']:.3f}   (down)\n"
            f"    VOImerge = {r['VI_merge']:.3f}   (down)")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluation.py <gt_mask.nii.gz> <pred.nii.gz>")
        sys.exit(1)
    gt, pred = _load(sys.argv[1]), _load(sys.argv[2])
    assert gt.shape == pred.shape, f"shape mismatch: gt {gt.shape} vs pred {pred.shape}"
    print(f"gt   = {sys.argv[1]}\npred = {sys.argv[2]}\nshape={gt.shape}\n")
    print(_fmt(evaluate(gt, pred)))
