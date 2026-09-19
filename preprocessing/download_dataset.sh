#!/bin/bash
# Download selected SomaNet dataset volumes from Google Drive and lay them out as
#   <OUT>/test_sets/NN/  and  <OUT>/train_sets/NN/
# each containing <stem>_raw.nii.gz + <stem>_mask.nii.gz, then generate img/ + mask/
# PNGs via nii_to_png.py (same format the dataloader reads).
#
# Dataset Drive folder [1]:
#   https://drive.google.com/drive/folders/1WLVaU3sGd8RQfwsBIBomZyNwl4m2D8pc
#     Test dataset/     z-4100..._01, z-4100..._11
#     Training dataset/ z-1000..._1, z-1500..._1, ...
#
# Only a FEW sets are listed below (extend the SETS map as needed).
# Requires gdown:  pip install gdown

set -e
OUT="${1:-./data_demo}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"

# target_relpath : gdrive_folder_id
declare -A SETS=(
  ["test_sets/01"]="15IRAUWuQ_ll-XkYf8-yvD5_zZZyNb7_4"    # z-4100_y-10015_x-32853_01
  ["test_sets/02"]="13PYGWEfANOUVdDNJV-OKL89N4KJ1Ua0n"    # z-4100_y-10015_x-32853_11
  ["train_sets/01"]="1TsYjbwKFtEoUYMWeOTuC-iaIvMjKsM99"   # z-1000_y-15328_x-42084_1
  ["train_sets/02"]="1YLyX7wcJlyfYlAvG6zZvzQfCybtF7z_o"   # z-1500_y-15935_x-27828_1
)

command -v gdown >/dev/null 2>&1 || { echo "gdown not found. pip install gdown" >&2; exit 1; }

for rel in "${!SETS[@]}"; do
  id="${SETS[$rel]}"
  dst="$OUT/$rel"; mkdir -p "$dst"
  echo "=== $rel  <-  gdrive:$id ==="
  tmp="$(mktemp -d)"
  gdown --folder "https://drive.google.com/drive/folders/$id" -O "$tmp" >/dev/null 2>&1 || \
    gdown --folder "https://drive.google.com/drive/folders/$id" -O "$tmp"
  # copy ONLY the top-level whole-volume nii.gz (ignore any *_crop/ per-instance folders)
  find "$tmp" -maxdepth 2 -name '*_raw.nii.gz'  ! -path '*crop*' -exec cp -f {} "$dst/" \;
  find "$tmp" -maxdepth 2 -name '*_mask.nii.gz' ! -path '*crop*' -exec cp -f {} "$dst/" \;
  rm -rf "$tmp"
  echo "  nii.gz: $(ls "$dst"/*.nii.gz 2>/dev/null | wc -l)  -> generating img/ + mask/ PNGs"
  "$PY" "$HERE/nii_to_png.py" "$dst"
done
echo "DONE -> $OUT"

#[1] https://github.com/liuxy1103/EMADS