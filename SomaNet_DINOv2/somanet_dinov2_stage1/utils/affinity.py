import numpy as np
from scipy.ndimage import shift
import torch

# Copied from https://github.com/weih527/Pixel-Embedded-Affinity, and adapted to our codebase
# Learning to Model Pixel-Embedded Affinity for Homogeneous Instance Segmentation (DOI: https://doi.org/10.1609/aaai.v36i1.19984)

def gen_offsets(shift, neighbor=4):
    assert neighbor == 4 or neighbor == 8, 'neigbor must be 4 or 8!'
    if neighbor == 4:
        return [[-shift, 0], [0, -shift]]
    else:
        return [[-shift, 0], [0, -shift], [-shift, -shift], [-shift, shift]]

def multi_offset(shifts, neighbor=4):
    out = []
    for shift in shifts:
        out += gen_offsets(shift, neighbor=neighbor)
    return out



def gen_affs_ours(labels, offsets=[[-1, 0], [0, -1]], ignore=False, padding=False):
    # If the input has shape [1, w, h], squeeze it to [w, h]
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels.squeeze(0)

    n_channels = len(offsets)
    
    # Assuming labels are PyTorch tensors, we create empty tensors for affinities and masks
    affinities = torch.zeros((n_channels,) + labels.shape, dtype=torch.float32)
    masks = torch.zeros((n_channels,) + labels.shape, dtype=torch.uint8)
    
    # Convert labels to NumPy array for processing with scipy.ndimage.shift
    labels_np = labels.numpy()
    
    for cid, off in enumerate(offsets):
        shift_off = [-x for x in off]
        
        # Shift the labels using scipy's shift function
        shifted_np = shift(labels_np, shift_off, order=0, prefilter=False)
        shifted = torch.from_numpy(shifted_np)  # Convert back to tensor
        
        mask = torch.ones_like(labels)
        mask_np = shift(mask.numpy(), shift_off, order=0, prefilter=False)
        mask = torch.from_numpy(mask_np)
        
        dif = labels - shifted
        out = torch.zeros_like(dif)
        out[dif == 0] = 1
        out[dif != 0] = 0
        
        if ignore:
            out[labels == 0] = 0
            out[shifted == 0] = 0
        
        if padding:
            out[mask == 0] = 1
        else:
            out[mask == 0] = 0
        
        affinities[cid] = out
        masks[cid] = mask
    
    return affinities, masks

def weight_binary_ratio(label, mask=None, alpha=1.0):
    """Binary-class rebalancing."""
    if label.max() == label.min():  # Uniform weights for single-label volume
        weight_factor = 1.0
        weight = torch.ones_like(label, dtype=torch.float32)
    else:
        label = (label != 0).int()
        if mask is None:
            weight_factor = float(label.sum()) / label.numel()
        else:
            weight_factor = float((label * mask).sum()) / mask.sum()
        weight_factor = np.clip(weight_factor, a_min=5e-2, a_max=0.99)

        if weight_factor > 0.5:
            weight = label.float() + alpha * weight_factor / (1 - weight_factor) * (1 - label.float())
        else:
            weight = alpha * (1 - weight_factor) / weight_factor * label.float() + (1 - label.float())

        if mask is not None:
            weight = weight * mask.float()

    return weight.float()

