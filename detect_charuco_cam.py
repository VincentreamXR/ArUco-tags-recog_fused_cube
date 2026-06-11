#!/usr/bin/python3
import argparse
import sys

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


def draw_text_panel(image, lines):
    if not lines:
        return
    overlay = image.copy()
    height = 24 + 28 * len(lines)
    cv2.rectangle(overlay, (12, 12), (520, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, image, 0.58, 0, image)
    y = 40
    for line in lines:
        cv2.putText(image, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def read_calibration(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError("Cannot open calibration file: {}".format(path))

    camera_matrix = None
    dist_coeffs = None
    for key in ("camera_matrix", "Camera_Matrix", "cameraMatrix", "CameraMatrix"):
        node = fs.getNode(key)
        if not node.empty():
            camera_matrix = node.mat()
            break
    for key in ("distortion_coefficients", "Distortion_Coefficients", "dist_coeffs", "distCoeffs", "distortion"):
        node = fs.getNode(key)
        if not node.empty():
            dist_coeffs = node.mat()
            break
    fs.release()

    if camera_matrix is None or dist_coeffs is None:
        raise RuntimeError("Calibration file must contain camera matrix and distortion coefficients")
    return camera_matrix, dist_coeffs


def create_detector_parameters(args):
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = args.adaptive_min
    params.adaptiveThreshWinSizeMax = args.adaptive_max
    params.adaptiveThreshWinSizeStep = args.adaptive_step
    params.minMarkerPerimeterRate = args.min_marker_perimeter_rate
    params.maxMarkerPerimeterRate = args.max_marker_perimeter_rate
    return params


def create_aruco_detector(dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None


def create_charuco_detector(board, params):
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        detector.setDetectorParameters(params)
        return detector
    return None


def detect_markers(gray, dictionary, params, detector):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def detect_charuco(gray, board, marker_corners, marker_ids, camera_matrix, dist_coeffs, detector):
    if detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board, camera_matrix, dist_coeffs
    )
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def estimate_charuco_pose(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs):
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        return None

    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
        )
        if ok:
            return rvec, tvec
        return None

    chessboard_corners = board.getChessboardCorners()
    ids = charuco_ids.reshape(-1).astype(int)
    object_points = chessboard_corners[ids].astype(np.float32)
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not ok:
        return None
    return rvec, tvec


def draw_pose_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length):
    if hasattr(cv2.aruco, "drawAxis"):
        cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
    else:
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)


def main():
    parser = argparse.ArgumentParser(description="Detect ChArUco boards from a camera using OpenCV aruco.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width. Default: 1280.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height. Default: 720.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--dictionary", default="DICT_5X5_100", choices=sorted(DICTIONARIES.keys()))
    parser.add_argument("--squares-x", type=int, default=5, help="Number of chessboard squares in X/columns.")
    parser.add_argument("--squares-y", type=int, default=7, help="Number of chessboard squares in Y/rows.")
    parser.add_argument("--square-length", type=float, default=0.04, help="Square side length in meters.")
    parser.add_argument("--marker-length", type=float, default=0.02, help="Aruco marker side length in meters.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file.")
    parser.add_argument("--adaptive-min", type=int, default=3)
    parser.add_argument("--adaptive-max", type=int, default=53)
    parser.add_argument("--adaptive-step", type=int, default=10)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.015)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    parser.add_argument("--probe-only", action="store_true", help="Initialize detectors and exit without opening the camera.")
    args = parser.parse_args()

    if args.marker_length >= args.square_length:
        raise ValueError("--marker-length must be smaller than --square-length")

    if hasattr(cv2.aruco, "Dictionary_get"):
        dictionary = cv2.aruco.Dictionary_get(DICTIONARIES[args.dictionary])
    else:
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[args.dictionary])
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            args.squares_x,
            args.squares_y,
            args.square_length,
            args.marker_length,
            dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            args.square_length,
            args.marker_length,
            dictionary,
        )
    params = create_detector_parameters(args)
    aruco_detector = create_aruco_detector(dictionary, params)
    charuco_detector = create_charuco_detector(board, params)

    if args.probe_only:
        print("OpenCV:", cv2.__version__, flush=True)
        print("Aruco detector:", type(aruco_detector).__name__ if aruco_detector is not None else "legacy", flush=True)
        print("ChArUco detector:", type(charuco_detector).__name__ if charuco_detector is not None else "legacy", flush=True)
        return

    camera_matrix = None
    dist_coeffs = None
    if args.calibration:
        camera_matrix, dist_coeffs = read_calibration(args.calibration)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, args.autofocus)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera index {}".format(args.camera))

    print("OpenCV:", cv2.__version__, flush=True)
    print("Camera: live:{} requested={}x{} autofocus={}".format(args.camera, args.width, args.height, args.autofocus), flush=True)
    print(
        "ChArUco: {} squares={}x{} square_length={} marker_length={}".format(
            args.dictionary, args.squares_x, args.squares_y, args.square_length, args.marker_length
        ),
        flush=True,
    )
    print("Press ESC or q to quit.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected = detect_markers(gray, dictionary, params, aruco_detector)

        charuco_corners = None
        charuco_ids = None
        if marker_ids is not None and len(marker_ids) > 0:
            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
                gray, board, marker_corners, marker_ids, camera_matrix, dist_coeffs, charuco_detector
            )
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            if charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 255))

                if camera_matrix is not None and dist_coeffs is not None and len(charuco_ids) >= 4:
                    pose = estimate_charuco_pose(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)
                    if pose is not None:
                        rvec, tvec = pose
                        draw_pose_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, args.square_length * 2.0)

        marker_count = 0 if marker_ids is None else len(marker_ids)
        charuco_count = 0 if charuco_ids is None else len(charuco_ids)
        lines = [
            "markers: {}  charuco corners: {}".format(marker_count, charuco_count),
            "dictionary: {}  board: {}x{}".format(args.dictionary, args.squares_x, args.squares_y),
            "resolution: {}x{}".format(frame.shape[1], frame.shape[0]),
        ]
        draw_text_panel(frame, lines)
        cv2.imshow("ChArUco Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Exception:", exc, file=sys.stderr)
        sys.exit(1)
