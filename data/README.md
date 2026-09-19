# data/ — the SomaNet data root (`SOMANET_DATA`)

This is the default data root both models read (inference and training). The
folders are committed **empty** (real volumes are too large for git); put your
data into the matching folders and everything runs with no configuration.

To use a different location instead, point the env var at it:
`export SOMANET_DATA=/path/to/data_root`.

## Layout

```
data/
├── train_sets/          # training volumes  -> run_trainer.py (train_sets/01..09)
│   ├── 01/  ...  09/
└── test_sets/           # evaluation volumes -> run_infer.py / run_eval.py
    ├── 01/
    └── 02/
```

Each set folder holds one volume: `<stem>_raw.nii.gz` + `<stem>_mask.nii.gz`,
plus the `img/` and `mask/` PNG subfolders produced by
`preprocessing/nii_to_png.py` (PNGs are for **training**; inference reads the
`*_raw.nii.gz` directly). See `preprocessing/README.md` for the two steps.

```
data/train_sets/01/
├── <stem>_raw.nii.gz
├── <stem>_mask.nii.gz
├── img/    <stem>_1.png … <stem>_N.png
└── mask/   <stem>_1.png … <stem>_N.png
```

## Which volume goes in which folder

The dataset (see `preprocessing/README.md` for the download link) maps to the
set folders as follows:

| folder            | volume (`<stem>`)            |
|-------------------|------------------------------|
| `train_sets/01`   | z-1000_y-15328_x-42084_1     |
| `train_sets/02`   | z-1500_y-15935_x-27828_1     |
| `train_sets/03`   | z-1800_y-21032_x-47685_1     |
| `train_sets/04`   | z-2000_y-22763_x-49197_1     |
| `train_sets/05`   | z-2500_y-14825_x-47290_1     |
| `train_sets/06`   | z-3500_y-21034_x-24866_1     |
| `train_sets/07`   | z-4800_y-12155_x-22527_1     |
| `train_sets/08`   | z-5500_y-15095_x-29783_1     |
| `train_sets/09`   | z-4000_y-10548_x-38450_1     |
| `test_sets/01`    | z-4100_y-10015_x-32853_01    |
| `test_sets/02`    | z-4100_y-10015_x-32853_11    |

The `.gitkeep` files only keep the empty folders in git; you can leave or
delete them once the real volumes are in place.
