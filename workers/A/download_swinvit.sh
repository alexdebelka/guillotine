#!/usr/bin/env bash
# Download MONAI SwinUNETR SSL pretrained weights (model_swinvit.pt, ~411 MB).
# Source: MONAI research-contributions release.
set -euo pipefail

DEST="${1:-/shared-docker/work/weights/model_swinvit.pt}"
mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
  echo "weights already at $DEST ($(du -h "$DEST" | cut -f1))"
  exit 0
fi

URL="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
echo "downloading model_swinvit.pt -> $DEST"
curl -L --fail --progress-bar -o "$DEST" "$URL"
echo "done: $(du -h "$DEST" | cut -f1)"
