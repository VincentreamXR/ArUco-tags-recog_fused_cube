#!/usr/bin/env python3
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import detect_aruco3 as aruco_base


def main():
    args = aruco_base.parse_args()
    args.dictionary = "DICT_6X6_250"

    dictionary = aruco_base.get_aruco_dictionary(args.dictionary)
    params = aruco_base.create_detector_parameters(args)
    detector = aruco_base.create_detector(dictionary, params)

    camera_matrix = None
    dist_coeffs = None
    if args.calibration:
        camera_matrix, dist_coeffs = aruco_base.read_calibration(Path(args.calibration).expanduser())

    if args.image:
        aruco_base.run_image(args, dictionary, params, detector, camera_matrix, dist_coeffs)
    else:
        aruco_base.run_camera(args, dictionary, params, detector, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Exception:", exc, file=sys.stderr)
        sys.exit(1)
