#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2


def load_deeparuco_aruco(repo_dir):
    impl_dir = Path(repo_dir).expanduser().resolve() / "impl"
    if not impl_dir.exists():
        raise FileNotFoundError(f"DeepArUco impl directory not found: {impl_dir}")
    sys.path.insert(0, str(impl_dir))
    import aruco  # noqa: PLC0415

    return aruco


def main():
    parser = argparse.ArgumentParser(
        description="Generate DeepArUco MIP36h12 tag PNG images."
    )
    parser.add_argument(
        "--repo-dir",
        default="/home/zsyy/下载/deeparuco-main",
        help="Path to the DeepArUco repository.",
    )
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=31)
    parser.add_argument("--size-px", type=int, default=1000)
    parser.add_argument(
        "--border-width",
        type=float,
        default=1.0,
        help="DeepArUco get_marker border_width value, normally 1.0 for printing.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/zsyy/deeparuco_mip36h12_tags_0_31",
    )
    args = parser.parse_args()

    if args.start_id < 0 or args.end_id < args.start_id or args.end_id > 249:
        raise ValueError("ID range must satisfy 0 <= start-id <= end-id <= 249")

    aruco = load_deeparuco_aruco(args.repo_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for marker_id in range(args.start_id, args.end_id + 1):
        marker, _ = aruco.get_marker(
            marker_id,
            size=args.size_px,
            border_width=args.border_width,
        )
        path = out_dir / f"deeparuco_mip36h12_id{marker_id:04d}.png"
        if not cv2.imwrite(str(path), marker):
            raise RuntimeError(f"Failed to write {path}")
        print(path)


if __name__ == "__main__":
    main()
