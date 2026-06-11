#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/home/zsyy/下载/deeparuco-main}"
ENV_NAME="${2:-deeparuco39}"

if [[ ! -d "$REPO" ]]; then
  echo "DeepArUco++ repo not found: $REPO" >&2
  exit 1
fi

conda create -y -n "$ENV_NAME" python=3.9
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -r "$REPO/requirements.txt"

echo "Environment ready: $ENV_NAME"
echo "Run:"
echo "  conda run -n $ENV_NAME python /home/zsyy/桌面/JuMarker/detect_deeparuco_cam.py --repo '$REPO' --camera 0"
