#!/bin/bash
# Download the SomaNet (SwinIR) inference checkpoint from Google Drive and place it
# where run_infer.py expects it: ./output_training/checkpoints/checkpoint_latest.pth
#
# The weights live in this shared Google Drive folder (a single .pth inside it):
#   https://drive.google.com/drive/folders/1I_9Oql6Cqg6SZPQIR2OlaBVVFg9qYre2   (SomaNet_models/swinir)
# The Drive file may be named anything (e.g. checkpoint.pth); it is saved locally as
# checkpoint_latest.pth, the name run_infer.py loads.
#
# Requires gdown:  pip install gdown
set -e
FOLDER_URL="https://drive.google.com/drive/folders/1I_9Oql6Cqg6SZPQIR2OlaBVVFg9qYre2"   # SomaNet_models/swinir
LOCAL_NAME="checkpoint_latest.pth"       # what run_infer.py expects locally

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$HERE/output_training/checkpoints"
mkdir -p "$DEST_DIR"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install it with:  pip install gdown" >&2
  exit 1
fi

echo "Downloading weights from Google Drive into $DEST_DIR ..."
TMP="$(mktemp -d)"
gdown --folder "$FOLDER_URL" -O "$TMP"

# Grab the single .pth checkpoint from the folder, whatever it is named
SRC="$(find "$TMP" -name '*.pth' | head -1)"
if [ -z "$SRC" ]; then
  echo "ERROR: no .pth checkpoint found in the Drive folder. Uploaded yet?" >&2
  echo "Files fetched:"; find "$TMP" -type f
  exit 1
fi
mv -f "$SRC" "$DEST_DIR/$LOCAL_NAME"
rm -rf "$TMP"
echo "Done -> $DEST_DIR/$LOCAL_NAME"
echo "Inference will now find the checkpoint automatically (run_infer.py)."
