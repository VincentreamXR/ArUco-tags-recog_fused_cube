#!/usr/bin/python3
import argparse
import sys
import time

import cv2
import numpy as np

from calib_targets import (
    PuzzleBoardParams,
    PuzzleBoardSearchMode,
    PuzzleBoardSpec,
    detect_puzzleboard_best,
)


PUZZLEPOLE_SPECS = {
    12: (73, 14),
    18: (7, 20),
    24: (242, 26),
    30: (176, 32),
    36: (325, 38),
    42: (410, 44),
    48: (115, 50),
}


def draw_text_panel(image, lines):
    if not lines:
        return
    overlay = image.copy()
    width = 780
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


def make_params(period, cols, square_size_mm, px_per_square):
    origin_row, rows = PUZZLEPOLE_SPECS[period]
    board = PuzzleBoardSpec(
        rows=rows,
        cols=cols,
        cell_size=square_size_mm,
        origin_row=origin_row,
        origin_col=0,
    )
    params = PuzzleBoardParams.sweep_for_board(board)
    for cfg in params:
        cfg.px_per_square = px_per_square
        cfg.decode.search_mode = PuzzleBoardSearchMode.fixed_board()
    return params


def build_configurations(periods, cols, square_size_mm, px_per_square):
    configs = []
    for period in periods:
        configs.append(
            {
                "period": period,
                "origin_row": PUZZLEPOLE_SPECS[period][0],
                "rows": PUZZLEPOLE_SPECS[period][1],
                "params": make_params(period, cols, square_size_mm, px_per_square),
            }
        )
    return configs


def detect_best(gray, configs):
    best = None
    for cfg in configs:
        try:
            result = detect_puzzleboard_best(gray, cfg["params"])
        except Exception:
            continue
        corner_count = len(result.corners)
        if corner_count == 0:
            continue
        score = (
            corner_count,
            result.decode.edges_matched,
            result.decode.mean_confidence,
            -result.decode.bit_error_rate,
        )
        if best is None or score > best["score"]:
            best = {"cfg": cfg, "result": result, "score": score}
    return best


def draw_detection(frame, detection, show_ids):
    result = detection["result"]
    for corner in result.corners:
        x, y = corner.position
        center = (int(round(x)), int(round(y)))
        cv2.circle(frame, center, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 8, (0, 80, 255), 1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(frame, str(corner.id), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)


def local_object_points(result, origin_row, origin_col, square_size_mm):
    object_points = []
    image_points = []
    for corner in result.corners:
        x_img, y_img = corner.position
        x_obj = (corner.grid.i - origin_col) * square_size_mm
        y_obj = (corner.grid.j - origin_row) * square_size_mm
        object_points.append([x_obj, y_obj, 0.0])
        image_points.append([x_img, y_img])
    return np.asarray(object_points, dtype=np.float32), np.asarray(image_points, dtype=np.float32)


def draw_pose(frame, result, cfg, camera_matrix, dist_coeffs, square_size_mm):
    if camera_matrix is None or dist_coeffs is None or len(result.corners) < 4:
        return None
    object_points, image_points = local_object_points(result, cfg["origin_row"], 0, square_size_mm)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=4.0,
        confidence=0.99,
    )
    if not ok:
        return None
    axis_len = square_size_mm * 2.0
    axis = np.asarray([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]], dtype=np.float32)
    projected, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)
    o, x, y, z = projected
    cv2.line(frame, tuple(o), tuple(x), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(frame, tuple(o), tuple(y), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, tuple(o), tuple(z), (0, 0, 255), 2, cv2.LINE_AA)
    inlier_count = 0 if inliers is None else len(inliers)
    return rvec, tvec, inlier_count


def parse_periods(value):
    if value.lower() == "auto":
        return sorted(PUZZLEPOLE_SPECS)
    period = int(value)
    if period not in PUZZLEPOLE_SPECS:
        raise argparse.ArgumentTypeError("period must be one of {} or auto".format(sorted(PUZZLEPOLE_SPECS)))
    return [period]


def main():
    parser = argparse.ArgumentParser(description="Detect A4-printed PuzzlePole/PuzzleBoard patterns from a camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width. Default: 1280.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height. Default: 720.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--period", type=parse_periods, default=parse_periods("auto"), help="PuzzlePole period: 12,18,24,30,36,42,48 or auto. Default: auto.")
    parser.add_argument("--cols", type=int, default=7, help="Printed board columns/puzzle pieces. Default: 7.")
    parser.add_argument("--square-size-mm", type=float, default=21.0, help="Physical square size on printed A4 paper, in mm. Used for pose only. Default: 21.0.")
    parser.add_argument("--px-per-square", type=float, default=60.0, help="Detector scale hint. Default: 60.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file for pose.")
    parser.add_argument("--show-ids", action="store_true", help="Draw corner IDs.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Seconds between terminal status prints. Default: 1.0.")
    args = parser.parse_args()

    configs = build_configurations(args.period, args.cols, args.square_size_mm, args.px_per_square)

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
    print("PuzzlePoles: periods={} cols={} square_size_mm={}".format(args.period, args.cols, args.square_size_mm), flush=True)
    print("Press ESC or q to quit.", flush=True)

    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detect_best(gray, configs)

        lines = [
            "resolution: {}x{}  period: {}".format(frame.shape[1], frame.shape[0], "none"),
            "corners: 0  edges: 0/0  ber: -  conf: -",
            "pose: no calibration" if camera_matrix is None else "pose: no detection",
        ]

        if detection is not None:
            cfg = detection["cfg"]
            result = detection["result"]
            draw_detection(frame, detection, args.show_ids)
            pose = draw_pose(frame, result, cfg, camera_matrix, dist_coeffs, args.square_size_mm)

            pose_text = "pose: no calibration"
            if camera_matrix is not None:
                pose_text = "pose: failed"
                if pose is not None:
                    _, tvec, inliers = pose
                    tx, ty, tz = [float(v) for v in tvec.reshape(-1)]
                    pose_text = "pose: t=({:.1f},{:.1f},{:.1f})mm inliers={}".format(tx, ty, tz, inliers)

            lines = [
                "resolution: {}x{}  period: {}".format(frame.shape[1], frame.shape[0], cfg["period"]),
                "corners: {}  edges: {}/{}  ber: {:.3f}  conf: {:.3f}".format(
                    len(result.corners),
                    result.decode.edges_matched,
                    result.decode.edges_observed,
                    result.decode.bit_error_rate,
                    result.decode.mean_confidence,
                ),
                "origin: row={} col={}  {}".format(result.decode.master_origin_row, result.decode.master_origin_col, pose_text),
            ]

            now = time.time()
            if args.print_every > 0 and now - last_print >= args.print_every:
                print(
                    "period={} corners={} edges={}/{} ber={:.3f} conf={:.3f} origin=({}, {})".format(
                        cfg["period"],
                        len(result.corners),
                        result.decode.edges_matched,
                        result.decode.edges_observed,
                        result.decode.bit_error_rate,
                        result.decode.mean_confidence,
                        result.decode.master_origin_row,
                        result.decode.master_origin_col,
                    ),
                    flush=True,
                )
                last_print = now

        draw_text_panel(frame, lines)
        cv2.imshow("PuzzlePole Detection", frame)

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
