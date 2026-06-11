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
    cv2.rectangle(overlay, (12, 12), (700, height), (0, 0, 0), -1)
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


def get_dictionary(name):
    if hasattr(cv2.aruco, "Dictionary_get"):
        return cv2.aruco.Dictionary_get(DICTIONARIES[name])
    return cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])


def detect_markers(gray, dictionary, params):
    if hasattr(cv2.aruco, "detectMarkers"):
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector.detectMarkers(gray)


def estimate_pose(marker_corners, marker_length, camera_matrix, dist_coeffs):
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners, marker_length, camera_matrix, dist_coeffs
        )
        return rvecs, tvecs

    half = marker_length / 2.0
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )
    rvecs = []
    tvecs = []
    for corners in marker_corners:
        ok, rvec, tvec = cv2.solvePnP(object_points, corners.reshape(-1, 2), camera_matrix, dist_coeffs)
        if ok:
            rvecs.append(rvec.reshape(1, 3))
            tvecs.append(tvec.reshape(1, 3))
    return np.array(rvecs), np.array(tvecs)


def draw_axis(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length):
    if hasattr(cv2.aruco, "drawAxis"):
        cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
        return
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)


def main():
    parser = argparse.ArgumentParser(description="Detect ArUco/DeepArUco++ printable tags from a camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width. Default: 1280.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height. Default: 720.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--dictionary", default="DICT_6X6_250", choices=sorted(DICTIONARIES.keys()))
    parser.add_argument("--marker-length", type=float, default=0.04, help="Marker side length in meters for pose.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file.")
    parser.add_argument("--adaptive-min", type=int, default=3)
    parser.add_argument("--adaptive-max", type=int, default=53)
    parser.add_argument("--adaptive-step", type=int, default=10)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.015)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    args = parser.parse_args()

    dictionary = get_dictionary(args.dictionary)
    params = create_detector_parameters(args)

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
    print("Tag dictionary: {} marker_length={}".format(args.dictionary, args.marker_length), flush=True)
    print("Press ESC or q to quit.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected = detect_markers(gray, dictionary, params)

        marker_count = 0 if marker_ids is None else len(marker_ids)
        ids_text = []
        if marker_count > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            ids_text = marker_ids.flatten().tolist()

            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs = estimate_pose(marker_corners, args.marker_length, camera_matrix, dist_coeffs)
                for rvec, tvec in zip(rvecs, tvecs):
                    draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, args.marker_length * 0.75)

        lines = [
            "tags: {} ids: {}".format(marker_count, ids_text[:8]),
            "dictionary: {} rejected: {}".format(args.dictionary, len(rejected)),
            "resolution: {}x{}".format(frame.shape[1], frame.shape[0]),
        ]
        draw_text_panel(frame, lines)
        cv2.imshow("DeepArUco++ Tags Detection", frame)

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
