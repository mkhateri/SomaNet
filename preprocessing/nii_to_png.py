"""nii_to_png.py -- convert 3D .nii.gz volumes to per-slice PNGs for TRAINING.

The per-slice PNGs produced here are used for TRAINING ONLY (the training
dataloader reads them from the img/ and mask/ subfolders). Inference runs
directly on the *_raw.nii.gz volumes and does not need these PNGs.

For each set folder it reads *_raw.nii.gz and *_mask.nii.gz and writes:
  * raw  -> img/   : normalized to [0,255] over the whole volume, uint8
  * mask -> mask/  : unique nonzero labels remapped to sequential 1..N, uint8
  * one PNG per z-slice (axis 2), named <prefix>_<i+1>.png
  * prefix = the file name with the trailing "_raw"/"_mask" + ".nii.gz" dropped

Usage:
    python nii_to_png.py <path>              # converts every set folder found under <path>
    # examples:
    #   python nii_to_png.py data            # both train_sets/ and test_sets/, all volumes
    #   python nii_to_png.py data/train_sets # just the training volumes
    #   python nii_to_png.py data/test_sets/01   # a single set folder

Then set that data root and train (SwinIR and DINOv2 use the same variable):

    <data_root>/               # <-- put this address in SW_TRAIN_DATA
    └── train_sets/
        ├── 01/  <stem>_raw.nii.gz  <stem>_mask.nii.gz  img/  mask/
        └── ...

    export SW_TRAIN_DATA=/path/to/data_root
"""

import os, sys, argparse
import numpy as np
import nibabel as nib
import imageio
from tqdm import tqdm


def save_as_png(nifti_image, save_path, prefix, flag):
    """Save each z-slice as a PNG. flag in {'raw','masks'}."""
    image_data = nifti_image.get_fdata()

    if flag == 'raw':
        image_data = 255 * (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        image_data = image_data.astype(np.uint8)

    if flag == 'masks':
        unique_values = np.unique(image_data)
        unique_values = unique_values[unique_values != 0]           # exclude background 0
        value_mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}
        vectorized_map = np.vectorize(lambda v: value_mapping.get(v, 0))
        image_data = vectorized_map(image_data).astype(np.uint8)

    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(image_data.shape[2]), desc=f"{flag}:{prefix}", leave=False):
        imageio.imwrite(os.path.join(save_path, f"{prefix}_{i + 1}.png"), image_data[:, :, i])


def convert_set(set_dir, out_dir=None):
    """Convert one set folder: *_raw.nii.gz -> <out>/img/, *_mask.nii.gz -> <out>/mask/."""
    out_dir = out_dir or set_dir
    for file_name in sorted(os.listdir(set_dir)):
        if not file_name.endswith('.nii.gz'):
            continue
        prefix = '_'.join(file_name.split('_')[:-1])
        nifti_image = nib.load(os.path.join(set_dir, file_name))
        if '_mask' in file_name:
            print(f"  mask -> {os.path.join(out_dir, 'mask')}  ({file_name})", flush=True)
            save_as_png(nifti_image, os.path.join(out_dir, 'mask'), prefix, 'masks')
        elif '_raw' in file_name:
            print(f"  raw  -> {os.path.join(out_dir, 'img')}   ({file_name})", flush=True)
            save_as_png(nifti_image, os.path.join(out_dir, 'img'), prefix, 'raw')


def find_set_dirs(root):
    """Every folder at or under `root` that directly holds a *_raw.nii.gz volume."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.endswith('_raw.nii.gz') for f in filenames):
            yield dirpath


def main():
    ap = argparse.ArgumentParser(
        description="Convert *_raw.nii.gz / *_mask.nii.gz volumes to per-slice PNGs (img/ + mask/).")
    ap.add_argument('path', help='a set folder, or any parent of them '
                                 '(e.g. data, data/train_sets); every set under it is converted')
    ap.add_argument('--all', action='store_true', help=argparse.SUPPRESS)  # kept for backward compat (no-op)
    args = ap.parse_args()

    set_dirs = sorted(find_set_dirs(args.path))
    if not set_dirs:
        sys.exit(f"no *_raw.nii.gz found under {args.path}")
    for set_dir in set_dirs:
        print(f"[set] {set_dir}", flush=True)
        convert_set(set_dir)


if __name__ == '__main__':
    main()
