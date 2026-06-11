#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="$project_dir/build/utils/jumarker_test"

if [[ ! -x "$binary" ]]; then
    cat >&2 <<EOF
jumarker_test not found: $binary

Build JuMarker on Ubuntu first:
  cd "$project_dir"
  mkdir -p build
  cd build
  cmake .. -DOPENCV_PATH=/path/to/opencv/installation
  make -j"\$(nproc)"

If OpenCV is installed from apt, this is often enough:
  cmake ..
  make -j"\$(nproc)"
EOF
    exit 1
fi

exec python3 "$project_dir/detect_rubik_8.py" --project "$project_dir" "$@"
