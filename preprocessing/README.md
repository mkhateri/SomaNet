# Preprocessing — prepare the data

SomaNet reads 3D EM volumes stored as `.nii.gz` and converts them to per‑slice
PNGs. This is a two‑step process:

1. **Place your volumes** in the directory tree shown below.
2. **Run `nii_to_png.py`** to generate the `img/` and `mask/` PNGs the
   dataloader reads.

## Step 1 — arrange the volumes

Download the dataset (see the link at the bottom) and place each volume in a
numbered set folder, exactly like this — one raw volume and its instance mask
per folder (`test_sets/` = evaluation volumes, `train_sets/` = training volumes):

```
##############################################################################
# <data_root>/
# ├── train_sets/                              └── test_sets/
# │   ├── 01/                                      ├── 01/
# │   │   ├── <stem>_raw.nii.gz                    │   ├── <stem>_raw.nii.gz
# │   │   └── <stem>_mask.nii.gz                   │   └── <stem>_mask.nii.gz
# │   ├── 02/                                      └── 02/
# │   │   ├── <stem>_raw.nii.gz                        ├── <stem>_raw.nii.gz
# │   │   └── <stem>_mask.nii.gz                       └── <stem>_mask.nii.gz
# │   └── ...
# │       ├── <stem>_raw.nii.gz
# │       └── <stem>_mask.nii.gz
##############################################################################
```

- `<stem>` is any name; the two files in a set folder must share it and differ
  only by the `_raw` / `_mask` suffix (e.g. `z-4100_y-10015_x-32853_01_raw.nii.gz`
  and `..._01_mask.nii.gz`).

## Step 2 — convert to PNGs

```bash
# convert every set folder found under the path (recursively):
python nii_to_png.py <data_root>            # both train_sets/ and test_sets/, all volumes

# …or narrow it down:
python nii_to_png.py <data_root>/train_sets     # only the training volumes
python nii_to_png.py <data_root>/test_sets/01   # a single set folder
```

This fills in an `img/` and a `mask/` subfolder inside every set folder:

```
##############################################################################
# <data_root>/train_sets/01/            <data_root>/test_sets/01/
# ├── <stem>_raw.nii.gz                 ├── <stem>_raw.nii.gz
# ├── <stem>_mask.nii.gz                ├── <stem>_mask.nii.gz
# ├── img/         (generated)          ├── img/         (generated)
# │   ├── <stem>_1.png                  │   ├── <stem>_1.png
# │   ├── <stem>_2.png                  │   ├── <stem>_2.png
# │   └── ...                           │   └── ...
# └── mask/        (generated)          └── mask/        (generated)
#     ├── <stem>_1.png                      ├── <stem>_1.png
#     ├── <stem>_2.png                      ├── <stem>_2.png
#     └── ...                               └── ...
##############################################################################
```

**Conversion details** (`nii_to_png.py`):
- **raw** → normalised to `[0,255]` over the whole volume, saved `uint8` in `img/`.
- **mask** → unique non‑zero labels remapped to sequential `1..N` (`uint8`) in `mask/`.
- slices are taken along axis 2 (z); filename `<stem>_<i+1>.png`, where `<stem>`
  is the file name with the trailing `_raw`/`_mask` + `.nii.gz` dropped.

Inference and training read this `<data_root>` from `SOMANET_DATA` (default: the
shared `data/` folder at the repo root), with the split chosen by `SOMANET_SPLIT`
(`test_sets` or `train_sets`).

---

## Dataset

<https://drive.google.com/drive/folders/1WLVaU3sGd8RQfwsBIBomZyNwl4m2D8pc>
(`Test dataset/` and `Training dataset/` sub‑folders, one folder per volume —
each contains the `*_raw.nii.gz` and `*_mask.nii.gz` to place under the
matching `test_sets/NN/` or `train_sets/NN/` above.)

*Optional:* `download_dataset.sh` automates the download + layout + conversion
for a few demo volumes (needs `pip install gdown` and the Drive folder to be
public). Edit the `SETS` map inside it to choose which volumes to fetch.
