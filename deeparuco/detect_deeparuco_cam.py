#!/usr/bin/env python3
import argparse
import sys
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


def draw_text_panel(image, lines):
    if not lines:
        return
    overlay = image.copy()
    width = 760
    height = 24 + 28 * len(lines)
    cv2.rectangle(overlay, (12, 12), (width, height), (0, 0, 0), -1)
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


def norm(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-9)


def ensure_repo(repo):
    if not repo.exists():
        raise RuntimeError(
            "DeepArUco++ repo not found: {}\n"
            "Clone/install it first, for example:\n"
            "  git clone https://github.com/AVAuco/deeparuco.git {}\n"
            "Then run this script with a Python environment containing TensorFlow, ultralytics, torch and OpenCV.".format(
                repo, repo
            )
        )
    for rel in ("impl/aruco.py", "impl/heatmaps.py", "impl/losses.py", "impl/utils.py"):
        if not (repo / rel).exists():
            raise RuntimeError("DeepArUco++ repo is missing expected file: {}".format(repo / rel))


def load_deeparuco(repo, detector_name, regressor_name):
    ensure_repo(repo)
    sys.path.insert(0, str(repo))

    try:
        import tensorflow as tf
        from impl.aruco import find_id
        from impl.heatmaps import pos_from_heatmap
        from impl.losses import weighted_loss
        from impl.utils import marker_from_corners, ordered_corners
        from tensorflow.keras.models import load_model
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Cannot import DeepArUco++ dependencies. Use the DeepArUco++ Python 3.9 environment from its requirements.txt.\n"
            "Import error: {}".format(exc)
        )

    model_dir = repo / "models"
    detector_path = model_dir / f"{detector_name}.pt"
    regressor_path = model_dir / f"{regressor_name}.h5"
    decoder_path = model_dir / "dec_new.h5"
    for path in (detector_path, regressor_path, decoder_path):
        if not path.exists():
            raise RuntimeError("DeepArUco++ model file not found: {}".format(path))

    detector = YOLO(str(detector_path))
    regressor = load_model(str(regressor_path), custom_objects={"weighted_loss": weighted_loss})
    decoder = load_model(str(decoder_path))

    @tf.function(reduce_retracing=True)
    def refine_corners(crops):
        return regressor(crops)

    @tf.function(reduce_retracing=True)
    def decode_markers(markers):
        return decoder(markers)

    return {
        "find_id": find_id,
        "marker_from_corners": marker_from_corners,
        "ordered_corners": ordered_corners,
        "pos_from_heatmap": pos_from_heatmap,
        "detector": detector,
        "refine_corners": refine_corners,
        "decode_markers": decode_markers,
    }


def hmap_to_corners(predictions, pos_from_heatmap):
    area = 75
    kp_params = cv2.SimpleBlobDetector_Params()
    kp_params.filterByArea = True
    kp_params.minArea = area * 0.8
    kp_params.maxArea = area * 1.2
    kp_detector = cv2.SimpleBlobDetector_create(kp_params)
    return [[(x, y) for x, y in zip(*pos_from_heatmap(pred, kp_detector))] for pred in predictions]


def detect_deeparuco(frame, bundle, regressor_name, threshold, detector_conf, detector_iou):
    detector = bundle["detector"]
    detections = detector(frame, verbose=False, iou=detector_iou, conf=detector_conf)[0].cpu().boxes
    if not len(detections):
        return []

    xyxy = []
    for det in detections:
        x1, y1, x2, y2 = [int(val) for val in det.xyxy.cpu().numpy()[0]]
        pad_x = int(0.2 * (x2 - x1) + 0.5)
        pad_y = int(0.2 * (y2 - y1) + 0.5)
        xyxy.append(
            [
                max(x1 - pad_x, 0),
                max(y1 - pad_y, 0),
                min(x2 + pad_x, frame.shape[1] - 1),
                min(y2 + pad_y, frame.shape[0] - 1),
            ]
        )

    crops_ori = [cv2.resize(frame[y1:y2, x1:x2], (64, 64)) for x1, y1, x2, y2 in xyxy]
    if regressor_name != "reg_baseline":
        crops = [norm(crop) for crop in crops_ori]
    else:
        crops = crops_ori.copy()

    corner_predictions = bundle["refine_corners"](np.array(crops)).numpy()
    if regressor_name.split("_")[1] == "hmap":
        corners = hmap_to_corners(corner_predictions, bundle["pos_from_heatmap"])
        keep = [len(cs) == 4 for cs in corners]
        reorg = [(det, crop, cs) for det, crop, cs, k in zip(xyxy, crops_ori, corners, keep) if k]
        if not reorg:
            return []
        xyxy, crops_ori, corners = zip(*reorg)
    else:
        corners = [[(pred[i], pred[i + 1]) for i in range(0, 8, 2)] for pred in corner_predictions]

    corners = [
        bundle["ordered_corners"]([c[0] for c in cs], [c[1] for c in cs])
        for cs in corners
    ]

    markers = []
    for crop, cs in zip(crops_ori, corners):
        marker = bundle["marker_from_corners"](crop, cs, 32)
        markers.append(norm(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))
    decoder_out = np.round(bundle["decode_markers"](np.array(markers)).numpy())
    ids, dists = zip(*[bundle["find_id"](out) for out in decoder_out])

    results = []
    for det, cs, marker_id, dist in zip(xyxy, corners, ids, dists):
        x1, y1, x2, y2 = det
        width = x2 - x1
        height = y2 - y1
        packed = [(cs[i], cs[i + 1]) for i in range(0, 8, 2)]
        image_corners = np.asarray(
            [[x1 + x * width, y1 + y * height] for x, y in packed],
            dtype=np.float32,
        )
        results.append(
            {
                "id": int(marker_id),
                "dist": float(dist),
                "accepted": float(dist) < threshold,
                "bbox": det,
                "corners": image_corners,
            }
        )
    return results


def estimate_pose(corners, marker_length, camera_matrix, dist_coeffs):
    half = marker_length / 2.0
    object_points = np.asarray(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float32,
    )
    ok, rvec, tvec = cv2.solvePnP(object_points, corners.astype(np.float32), camera_matrix, dist_coeffs)
    if not ok:
        return None
    return rvec, tvec


def draw_marker(frame, result, camera_matrix, dist_coeffs, marker_length):
    color = (0, 255, 0) if result["accepted"] else (0, 0, 255)
    corners = result["corners"].astype(int)
    for i in range(4):
        cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, 2, cv2.LINE_AA)
    center = np.mean(corners, axis=0).astype(int)
    cv2.putText(frame, str(result["id"]), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    if camera_matrix is None or dist_coeffs is None or not result["accepted"]:
        return None
    pose = estimate_pose(result["corners"], marker_length, camera_matrix, dist_coeffs)
    if pose is None:
        return None
    rvec, tvec = pose
    axis = np.asarray(
        [[0, 0, 0], [marker_length, 0, 0], [0, marker_length, 0], [0, 0, -marker_length]],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)
    o, x, y, z = projected
    cv2.line(frame, tuple(o), tuple(x), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(frame, tuple(o), tuple(y), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, tuple(o), tuple(z), (0, 0, 255), 2, cv2.LINE_AA)
    return rvec, tvec


def create_opencv_aruco(args):
    dictionary = get_aruco_dictionary(args.dictionary)
    params = create_detector_parameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = args.adaptive_min
    params.adaptiveThreshWinSizeMax = args.adaptive_max
    params.adaptiveThreshWinSizeStep = args.adaptive_step
    params.minMarkerPerimeterRate = args.min_marker_perimeter_rate
    params.maxMarkerPerimeterRate = args.max_marker_perimeter_rate
    return dictionary, params


def detect_opencv_aruco(gray, dictionary, params):
    if hasattr(cv2.aruco, "detectMarkers"):
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector.detectMarkers(gray)


def get_aruco_dictionary(name):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])
    return cv2.aruco.Dictionary_get(DICTIONARIES[name])


def create_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def main():
    parser = argparse.ArgumentParser(description="Detect DeepArUco++ markers from a camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width. Default: 1280.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height. Default: 720.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--repo", default="/home/zsyy/下载/deeparuco-main", help="DeepArUco++ repository root. Default: /home/zsyy/下载/deeparuco-main.")
    parser.add_argument("--detector", default="det_luma_bc_s", help="YOLO detector model name in repo/models. Default: det_luma_bc_s.")
    parser.add_argument("--regressor", default="reg_hmap_8", help="Corner refinement model name in repo/models. Default: reg_hmap_8.")
    parser.add_argument("--threshold", type=float, default=9.0, help="Reject decoded markers with distance >= threshold. Default: 9.")
    parser.add_argument("--detector-conf", type=float, default=0.03, help="YOLO confidence threshold. Default: 0.03.")
    parser.add_argument("--detector-iou", type=float, default=0.5, help="YOLO IoU threshold. Default: 0.5.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file.")
    parser.add_argument("--marker-length", type=float, default=0.04, help="Marker side length in meters for pose. Default: 0.04.")
    parser.add_argument("--fallback-opencv", action="store_true", help="Also run OpenCV ArUco detection as a fallback/side-by-side baseline.")
    parser.add_argument("--dictionary", default="DICT_6X6_250", choices=sorted(DICTIONARIES.keys()))
    parser.add_argument("--adaptive-min", type=int, default=3)
    parser.add_argument("--adaptive-max", type=int, default=53)
    parser.add_argument("--adaptive-step", type=int, default=10)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.015)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    bundle = load_deeparuco(repo, args.detector, args.regressor)

    camera_matrix = None
    dist_coeffs = None
    if args.calibration:
        camera_matrix, dist_coeffs = read_calibration(args.calibration)

    opencv_dictionary = None
    opencv_params = None
    if args.fallback_opencv:
        opencv_dictionary, opencv_params = create_opencv_aruco(args)

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
    print("DeepArUco++ repo:", repo, flush=True)
    print("Models: detector={} regressor={} decoder=dec_new".format(args.detector, args.regressor), flush=True)
    print("Press ESC or q to quit.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        results = detect_deeparuco(
            frame,
            bundle,
            args.regressor,
            args.threshold,
            args.detector_conf,
            args.detector_iou,
        )

        accepted = 0
        pose_count = 0
        ids = []
        for result in results:
            pose = draw_marker(frame, result, camera_matrix, dist_coeffs, args.marker_length)
            if result["accepted"]:
                accepted += 1
                ids.append(result["id"])
            if pose is not None:
                pose_count += 1

        opencv_count = 0
        if args.fallback_opencv:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = detect_opencv_aruco(gray, opencv_dictionary, opencv_params)
            if marker_ids is not None and len(marker_ids) > 0:
                opencv_count = len(marker_ids)
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        lines = [
            "DeepArUco++ detections: {} accepted: {} ids: {}".format(len(results), accepted, ids[:8]),
            "threshold: {}  pose: {}".format(args.threshold, pose_count if camera_matrix is not None else "no calibration"),
            "OpenCV ArUco baseline: {}".format(opencv_count) if args.fallback_opencv else "OpenCV ArUco baseline: off",
        ]
        draw_text_panel(frame, lines)
        if accepted or results:
            print(
                "detections={} accepted={} ids={} dists={}".format(
                    len(results),
                    accepted,
                    ids,
                    ["{:.1f}".format(r["dist"]) for r in results],
                ),
                flush=True,
            )

        cv2.imshow("DeepArUco++ Detection", frame)
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
