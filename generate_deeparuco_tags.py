#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def main():
    parser = argparse.ArgumentParser(description="Generate printable ArUco tags for DeepArUco++ experiments.")
    parser.add_argument("--dictionary", default="DICT_6X6_250", choices=sorted(DICTIONARIES))
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--tag-size-px", type=int, default=800)
    parser.add_argument("--border-px", type=int, default=120)
    parser.add_argument("--out-dir", default="/home/zsyy/deeparuco_tags")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dictionary = get_aruco_dictionary(args.dictionary)

    for marker_id in range(args.start_id, args.start_id + args.count):
        marker = generate_marker(dictionary, marker_id, args.tag_size_px)
        canvas_size = args.tag_size_px + 2 * args.border_px
        canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
        canvas[
            args.border_px : args.border_px + args.tag_size_px,
            args.border_px : args.border_px + args.tag_size_px,
        ] = marker
        path = out_dir / f"{args.dictionary}_id{marker_id:04d}.png"
        cv2.imwrite(str(path), canvas)
        print(path)


def get_aruco_dictionary(name):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])
    return cv2.aruco.Dictionary_get(DICTIONARIES[name])


def generate_marker(dictionary, marker_id, size):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    return cv2.aruco.drawMarker(dictionary, marker_id, size)


if __name__ == "__main__":
    main()
