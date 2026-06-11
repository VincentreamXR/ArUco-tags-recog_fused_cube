#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

template="marker_designs/seaberyMarker.svg"
output_dir="output_seabery_8"
count=8

if [[ ! -f "$template" ]]; then
    echo "Template not found: $template" >&2
    exit 1
fi

if [[ ! -x "build/utils/create_marker" ]]; then
    echo "Generator not found: build/utils/create_marker" >&2
    echo "Build JuMarker first:" >&2
    echo "  mkdir -p build" >&2
    echo "  cd build" >&2
    echo "  cmake .. -DOPENCV_PATH=/usr" >&2
    echo "  make -j\$(nproc)" >&2
    exit 1
fi

mkdir -p "$output_dir"
./build/utils/create_marker "$template" "$output_dir/" "$count"

echo "Generated $count Seabery-based markers in: $project_dir/$output_dir"
echo "Use id-bits=3 when detecting these 8 generated IDs."
