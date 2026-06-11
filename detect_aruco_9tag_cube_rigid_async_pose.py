#!/usr/bin/env python3
import argparse
import itertools
import os
import sys
import threading
import time
from pathlib import Path


DEEPPARUCO_PYTHON = "/home/zsyy/anaconda3/envs/deeparuco39/bin/python"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "utils" / "camera_calibration.yml"
UPPER_CUBE_COLOR = (0, 255, 255)
LOWER_CUBE_COLOR = (255, 0, 0)


def ensure_opencv_python():
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        env_python = Path(DEEPPARUCO_PYTHON)
        if exc.name == "cv2" and env_python.exists() and Path(sys.executable) != env_python:
            os.execv(str(env_python), [str(env_python), __file__, *sys.argv[1:]])
        raise
    return cv2, np


cv2, np = ensure_opencv_python()


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


FACE_SPECS = {
    # normal, marker horizontal axis, marker vertical-down axis when viewed from outside the cube
    "front": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "right": (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "back": (np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "left": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "top": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
}
SIDE_FACE_NAMES = {"front", "right", "back", "left"}


class LatestFrameSlot:
    def __init__(self):
        self._condition = threading.Condition()
        self._item = None
        self._sequence = 0
        self._closed = False

    def put(self, item):
        with self._condition:
            self._sequence += 1
            self._item = item
            self._condition.notify_all()

    def get_latest(self, last_sequence=0, timeout=0.05):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._closed and self._sequence == last_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._closed:
                return None, self._sequence
            return self._item, self._sequence

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class RateCounter:
    def __init__(self, smoothing=0.9):
        self.smoothing = float(smoothing)
        self.last_time = None
        self.rate = 0.0

    def tick(self):
        now = time.monotonic()
        if self.last_time is not None:
            dt = max(now - self.last_time, 1e-6)
            instant = 1.0 / dt
            if self.rate <= 0.0:
                self.rate = instant
            else:
                self.rate = self.smoothing * self.rate + (1.0 - self.smoothing) * instant
        self.last_time = now
        return self.rate


class PoseEmaFilter:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.rvec = None
        self.tvec = None

    def update(self, rvec, tvec):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if self.rvec is None:
            self.rvec = rvec.copy()
            self.tvec = tvec.copy()
            return self.rvec.copy(), self.tvec.copy()

        if float(np.dot(self.rvec.reshape(3), rvec.reshape(3))) < 0.0:
            rvec = -rvec
        self.rvec = (1.0 - self.alpha) * self.rvec + self.alpha * rvec
        self.tvec = (1.0 - self.alpha) * self.tvec + self.alpha * tvec
        return self.rvec.copy(), self.tvec.copy()


class PoseTracker:
    def __init__(self, args):
        self.pose_filter = PoseEmaFilter(args.ema_alpha) if args.pose_filter == "ema" else None
        self.max_hold_time = float(args.hold_last_seconds)
        self.max_reprojection_error = float(args.max_stable_reprojection_error)
        self.max_rotation_jump_deg = float(args.max_rotation_jump_deg)
        self.max_translation_jump = float(args.max_translation_jump)
        self.last_raw_rvec = None
        self.last_raw_tvec = None
        self.last_output_rvec = None
        self.last_output_tvec = None
        self.last_update_time = None
        self.last_rolls = {}

    def get_initial_guess(self):
        if self.last_raw_rvec is None or self.last_raw_tvec is None:
            return None
        return self.last_raw_rvec.copy(), self.last_raw_tvec.copy()

    def update(self, rvec, tvec, mean_error, dynamic_rolls):
        if mean_error > self.max_reprojection_error:
            return self.get_held_pose(), "held_bad_reproj"
        if self.last_raw_rvec is not None and self.last_raw_tvec is not None:
            if self.is_pose_jump(rvec, tvec):
                return self.get_held_pose(), "held_pose_jump"
        self.last_raw_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
        self.last_raw_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
        if self.pose_filter is not None:
            out_rvec, out_tvec = self.pose_filter.update(self.last_raw_rvec, self.last_raw_tvec)
        else:
            out_rvec, out_tvec = self.last_raw_rvec.copy(), self.last_raw_tvec.copy()
        self.last_output_rvec = out_rvec.copy()
        self.last_output_tvec = out_tvec.copy()
        self.last_update_time = time.monotonic()
        self.last_rolls = dict(dynamic_rolls)
        return (self.last_output_rvec.copy(), self.last_output_tvec.copy()), "measured"

    def is_pose_jump(self, rvec, tvec):
        current_rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        previous_rotation, _ = cv2.Rodrigues(self.last_raw_rvec)
        relative_rotation = current_rotation @ previous_rotation.T
        trace = float(np.trace(relative_rotation))
        angle = np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        translation_delta = float(np.linalg.norm(np.asarray(tvec, dtype=np.float64).reshape(3, 1) - self.last_raw_tvec))
        return angle > self.max_rotation_jump_deg or translation_delta > self.max_translation_jump

    def get_held_pose(self):
        if self.last_output_rvec is None or self.last_output_tvec is None or self.last_update_time is None:
            return None
        if time.monotonic() - self.last_update_time > self.max_hold_time:
            return None
        return self.last_output_rvec.copy(), self.last_output_tvec.copy()


def get_dictionary(name):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])
    return cv2.aruco.Dictionary_get(DICTIONARIES[name])


def create_detector_parameters(args):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = args.adaptive_min
    params.adaptiveThreshWinSizeMax = args.adaptive_max
    params.adaptiveThreshWinSizeStep = args.adaptive_step
    params.minMarkerPerimeterRate = args.min_marker_perimeter_rate
    params.maxMarkerPerimeterRate = args.max_marker_perimeter_rate
    if hasattr(params, "useAruco3Detection"):
        params.useAruco3Detection = args.aruco3
    return params


def create_detector(dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None


def detect_markers(gray, dictionary, params, detector):
    if detector is not None:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def read_calibration(path):
    if str(path) == "/path/to/camera.yml":
        raise RuntimeError(
            "/path/to/camera.yml is only an example. Use the real file: {}".format(DEFAULT_CALIBRATION)
        )
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
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
    return camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64)


def approximate_calibration(width, height):
    focal = float(max(width, height))
    camera_matrix = np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def parse_face_order(text):
    faces = [part.strip().lower() for part in text.split(",") if part.strip()]
    if len(faces) != 4 or any(face not in SIDE_FACE_NAMES for face in faces):
        raise ValueError("--face-order must contain four faces from: front,right,back,left")
    if len(set(faces)) != 4:
        raise ValueError("--face-order must not repeat faces")
    return faces


def parse_id_list(text):
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return set()
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not ids:
        raise ValueError("ID list must not be empty")
    return set(ids)


def parse_corner_rolls(text):
    rolls = {}
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return rolls
    for part in text.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError("--corner-rolls entries must look like id:roll, for example 0:1,3:-1")
        marker_id_text, roll_text = part.split(":", 1)
        rolls[int(marker_id_text.strip())] = int(roll_text.strip()) % 4
    return rolls


def parse_id_face_map(text):
    mapping = {}
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return mapping
    for part in text.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError("--id-face-map entries must look like id:face, for example 0:front,3:left")
        marker_id_text, face = part.split(":", 1)
        face = face.strip().lower()
        if face not in FACE_SPECS:
            raise ValueError("--id-face-map face must be one of: front,right,back,left,top")
        mapping[int(marker_id_text.strip())] = face
    return mapping


def roll_corners(corners, roll):
    roll = int(roll) % 4
    if roll == 0:
        return corners
    return np.roll(corners, -roll, axis=0).copy()


def marker_object_corners(face, center_z, cube_size, marker_length):
    normal, u_axis, v_axis = FACE_SPECS[face]
    face_offset = cube_size * 0.5
    center = normal * face_offset + np.array([0.0, 0.0, center_z])
    half = marker_length * 0.5
    return np.array(
        [
            center - half * u_axis - half * v_axis,
            center + half * u_axis - half * v_axis,
            center + half * u_axis + half * v_axis,
            center - half * u_axis + half * v_axis,
        ],
        dtype=np.float32,
    )


def rotate_points_z(points, angle_deg):
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return np.asarray(points, dtype=np.float32) @ rotation.T


def rotated_marker_object_corners(face, center_z, cube_size, marker_length, angle_deg):
    return rotate_points_z(
        marker_object_corners(face, center_z, cube_size, marker_length),
        angle_deg,
    )


def build_marker_layout(args):
    face_order = parse_face_order(args.face_order)
    args.lower_ids = parse_id_list(args.lower_ids)
    args.corner_rolls = parse_corner_rolls(args.corner_rolls)
    args.auto_roll_ids = parse_id_list(args.auto_roll_ids)
    args.id_face_map = parse_id_face_map(args.id_face_map)
    upper_z = (args.cube_size + args.vertical_gap) * 0.5
    lower_z = -(args.cube_size + args.vertical_gap) * 0.5
    layout = {}
    for index, face in enumerate(face_order):
        marker_id = index
        marker_face = args.id_face_map.get(marker_id, face)
        layout[marker_id] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                upper_z,
                args.cube_size,
                args.marker_length,
                args.upper_rotation_deg,
            ),
            args.corner_rolls.get(marker_id, 0),
        )
        lower_id = index + 4
        if lower_id in args.lower_ids:
            marker_face = args.id_face_map.get(lower_id, face)
            layout[lower_id] = roll_corners(
                rotated_marker_object_corners(
                    marker_face,
                    lower_z,
                    args.cube_size,
                    args.marker_length,
                    args.lower_rotation_deg,
                ),
                args.corner_rolls.get(lower_id, 0),
            )
    if args.top_id >= 0:
        marker_face = args.id_face_map.get(args.top_id, "top")
        layout[args.top_id] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                upper_z,
                args.cube_size,
                args.marker_length,
                args.upper_rotation_deg,
            ),
            args.corner_rolls.get(args.top_id, 0),
        )
    return layout


def make_candidate_layout(layout, rolls):
    if not rolls:
        return layout
    candidate = dict(layout)
    for marker_id, roll in rolls.items():
        if marker_id in candidate:
            candidate[marker_id] = roll_corners(layout[marker_id], roll)
    return candidate


def collect_correspondences(marker_corners, marker_ids, layout):
    if marker_ids is None:
        return None, None, []

    object_points = []
    image_points = []
    used_ids = []
    for corners, marker_id_array in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(marker_id_array)
        if marker_id not in layout:
            continue
        object_points.append(layout[marker_id])
        image_points.append(corners.reshape(4, 2).astype(np.float32))
        used_ids.append(marker_id)

    if not object_points:
        return None, None, []
    return np.vstack(object_points).astype(np.float32), np.vstack(image_points).astype(np.float32), used_ids


def estimate_best_fused_pose(marker_corners, marker_ids, layout, camera_matrix, dist_coeffs, args, initial_guess=None):
    auto_ids = []
    if marker_ids is not None:
        detected_ids = [int(marker_id) for marker_id in marker_ids.reshape(-1)]
        auto_ids = [marker_id for marker_id in detected_ids if marker_id in args.auto_roll_ids and marker_id in layout]
    auto_ids = sorted(set(auto_ids))

    roll_sets = [{}]
    if auto_ids:
        roll_sets = []
        for values in itertools.product(range(4), repeat=len(auto_ids)):
            roll_sets.append(dict(zip(auto_ids, values)))
            if len(roll_sets) >= args.auto_roll_max_candidates:
                break

    best = None
    for dynamic_rolls in roll_sets:
        candidate_layout = make_candidate_layout(layout, dynamic_rolls)
        object_points, image_points, used_ids = collect_correspondences(marker_corners, marker_ids, candidate_layout)
        pose = estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, args, initial_guess)
        if pose is None:
            continue
        _, _, _, mean_error, _ = pose
        if best is None or mean_error < best[3][3]:
            best = (object_points, image_points, used_ids, pose, dynamic_rolls)

    if best is None:
        object_points, image_points, used_ids = collect_correspondences(marker_corners, marker_ids, layout)
        return object_points, image_points, used_ids, None, {}
    return best


def estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, args, initial_guess=None):
    if object_points is None or len(object_points) < 4 or len(set(used_ids)) < args.min_tags_for_pose:
        return None

    if args.ransac and len(object_points) >= 8:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            reprojectionError=args.reprojection_error,
            iterationsCount=args.ransac_iterations,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        inlier_count = 0 if inliers is None else int(len(inliers))
    else:
        if initial_guess is not None:
            guess_rvec, guess_tvec = initial_guess
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                guess_rvec.copy(),
                guess_tvec.copy(),
                True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            return None
        inlier_count = len(object_points)

    if hasattr(cv2, "solvePnPRefineLM") and len(object_points) >= 8:
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
    return rvec, tvec, inlier_count, float(np.mean(errors)), float(np.max(errors))


def cube_vertices(center_z, cube_size, rotation_deg):
    half = cube_size * 0.5
    vertices = np.array(
        [
            [-half, -half, center_z - half],
            [half, -half, center_z - half],
            [half, half, center_z - half],
            [-half, half, center_z - half],
            [-half, -half, center_z + half],
            [half, -half, center_z + half],
            [half, half, center_z + half],
            [-half, half, center_z + half],
        ],
        dtype=np.float32,
    )
    return rotate_points_z(vertices, rotation_deg)


def draw_projected_polyline(frame, points, color, closed=True):
    pts = np.round(points).astype(int).reshape(-1, 2)
    for idx in range(len(pts) - 1):
        pt0 = point_tuple(pts[idx])
        pt1 = point_tuple(pts[idx + 1])
        if pt0 is not None and pt1 is not None:
            cv2.line(frame, pt0, pt1, color, 2, cv2.LINE_AA)
    if closed and len(pts) > 2:
        pt0 = point_tuple(pts[-1])
        pt1 = point_tuple(pts[0])
        if pt0 is not None and pt1 is not None:
            cv2.line(frame, pt0, pt1, color, 2, cv2.LINE_AA)


def draw_rigid_tag_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs):
    for marker_id, object_corners in sorted(layout.items()):
        projected, _ = cv2.projectPoints(object_corners, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        draw_projected_polyline(frame, projected, (0, 220, 0), closed=True)
        center = point_tuple(np.mean(projected, axis=0))
        if center is not None:
            cv2.putText(
                frame,
                str(marker_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


def point_tuple(point, max_abs=1000000):
    point = np.round(point).reshape(-1)
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        return None
    if abs(point[0]) > max_abs or abs(point[1]) > max_abs:
        return None
    return int(point[0]), int(point[1])


def draw_fused_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs, args):
    upper_z = (args.cube_size + args.vertical_gap) * 0.5
    lower_z = -(args.cube_size + args.vertical_gap) * 0.5
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

    for vertices, color in (
        (cube_vertices(upper_z, args.cube_size, args.upper_rotation_deg), UPPER_CUBE_COLOR),
        (cube_vertices(lower_z, args.cube_size, args.lower_rotation_deg), LOWER_CUBE_COLOR),
    ):
        projected, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        for i0, i1 in edges:
            pt0 = point_tuple(projected[i0])
            pt1 = point_tuple(projected[i1])
            if pt0 is None or pt1 is None:
                continue
            cv2.line(
                frame,
                pt0,
                pt1,
                color,
                args.cube_line_thickness,
                cv2.LINE_AA,
            )

    if args.draw_model_tags:
        draw_rigid_tag_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs)

    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, args.axis_length)


def rotation_to_euler_xyz(rvec):
    rotation, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        y = np.arctan2(-rotation[2, 0], sy)
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        y = np.arctan2(-rotation[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def pose_matrix_object_to_camera(rvec, tvec):
    rotation, _ = cv2.Rodrigues(rvec)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = tvec.reshape(3)
    return matrix


def format_matrix(matrix):
    rows = []
    for row in matrix:
        rows.append("[{}]".format(", ".join("{:.6f}".format(float(value)) for value in row)))
    return "[{}]".format(", ".join(rows))


def draw_text_panel(image, lines):
    overlay = image.copy()
    width = min(image.shape[1] - 24, 980)
    height = 24 + 28 * len(lines)
    cv2.rectangle(overlay, (12, 12), (12 + width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (24, 40 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_frame(
    frame,
    dictionary,
    params,
    detector,
    layout,
    camera_matrix,
    dist_coeffs,
    args,
    pose_tracker=None,
    capture_fps=0.0,
    process_fps=0.0,
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, rejected = detect_markers(gray, dictionary, params, detector)

    marker_count = 0 if marker_ids is None else len(marker_ids)
    ids = [] if marker_ids is None else marker_ids.flatten().astype(int).tolist()
    if marker_count > 0:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    object_points, image_points, used_ids, pose, dynamic_rolls = estimate_best_fused_pose(
        marker_corners,
        marker_ids,
        layout,
        camera_matrix,
        dist_coeffs,
        args,
        pose_tracker.get_initial_guess() if pose_tracker is not None else None,
    )
    draw_layout = make_candidate_layout(layout, dynamic_rolls)

    pose_text = "pose: need at least {} configured tags".format(args.min_tags_for_pose)
    pose_status = "none"
    if pose is not None:
        rvec, tvec, inliers, mean_error, max_error = pose
        if pose_tracker is not None:
            tracked_pose, pose_status = pose_tracker.update(rvec, tvec, mean_error, dynamic_rolls)
            if tracked_pose is None:
                pose_text = "pose: rejected {} err={:.2f}px".format(pose_status, mean_error)
                tracked_rvec = None
                tracked_tvec = None
            else:
                tracked_rvec, tracked_tvec = tracked_pose
        else:
            tracked_rvec, tracked_tvec = rvec, tvec
            pose_status = "measured"

        if tracked_rvec is None or tracked_tvec is None:
            lines = [
                "ArUco rigid object fused pose  dictionary: {}  capture={:.1f}fps process={:.1f}fps".format(
                    args.dictionary, capture_fps, process_fps
                ),
                "detected ids: {}  used ids: {}  rejected: {}".format(ids[:12], used_ids, len(rejected)),
                pose_text,
                "auto corner rolls: {}".format(dynamic_rolls if dynamic_rolls else "off/unchanged"),
                "yellow=upper cube  blue=lower cube  cube_size={}m marker_length={}m top_id={}".format(
                    args.cube_size, args.marker_length, args.top_id
                ),
                "green=projected rigid tag model  one object axis only",
            ]
            draw_text_panel(frame, lines)
            return pose, ids, used_ids

        rvec, tvec = tracked_rvec, tracked_tvec
        draw_fused_model(frame, draw_layout, rvec, tvec, camera_matrix, dist_coeffs, args)
        tx, ty, tz = tvec.reshape(-1)
        roll, pitch, yaw = rotation_to_euler_xyz(rvec)
        object_to_camera = pose_matrix_object_to_camera(rvec, tvec)
        pose_text = (
            "rigid object pose t=({:.3f},{:.3f},{:.3f})m rpy=({:.1f},{:.1f},{:.1f})deg inliers={} err={:.2f}px {}"
            .format(tx, ty, tz, roll, pitch, yaw, inliers, mean_error, pose_status)
        )
        if args.print_pose:
            print(
                "object_pose ids={} t=({:.6f},{:.6f},{:.6f}) rvec=({:.6f},{:.6f},{:.6f}) "
                "rpy=({:.3f},{:.3f},{:.3f}) rolls={} object_to_camera={} mean_err={:.3f} max_err={:.3f}".format(
                    used_ids,
                    tx,
                    ty,
                    tz,
                    *rvec.reshape(-1),
                    roll,
                    pitch,
                    yaw,
                    dynamic_rolls if dynamic_rolls else {},
                    format_matrix(object_to_camera),
                    mean_error,
                    max_error,
                ),
                flush=True,
            )

    lines = [
        "ArUco rigid object fused pose  dictionary: {}  capture={:.1f}fps process={:.1f}fps".format(
            args.dictionary, capture_fps, process_fps
        ),
        "detected ids: {}  used ids: {}  rejected: {}".format(ids[:12], used_ids, len(rejected)),
        pose_text,
        "auto corner rolls: {}".format(dynamic_rolls if dynamic_rolls else "off/unchanged"),
        "yellow=upper cube  blue=lower cube  cube_size={}m marker_length={}m top_id={}".format(
            args.cube_size, args.marker_length, args.top_id
        ),
        "green=projected rigid tag model  one object axis only",
    ]
    draw_text_panel(frame, lines)
    return pose, ids, used_ids


def run_image(args, dictionary, params, detector, layout, camera_matrix, dist_coeffs):
    frame = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Cannot read image: {}".format(args.image))
    if camera_matrix is None:
        camera_matrix, dist_coeffs = approximate_calibration(frame.shape[1], frame.shape[0])

    pose, ids, used_ids = process_frame(frame, dictionary, params, detector, layout, camera_matrix, dist_coeffs, args)
    print("image: {}".format(args.image), flush=True)
    print("detected ids: {} used ids: {} pose: {}".format(ids, used_ids, "ok" if pose is not None else "failed"), flush=True)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print("output: {}".format(output_path), flush=True)
    if args.show:
        cv2.imshow("deeparuco detect", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args, dictionary, params, detector, layout, camera_matrix, dist_coeffs):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, args.autofocus)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera index {}".format(args.camera))

    print("OpenCV: {}".format(cv2.__version__), flush=True)
    print(
        "Camera: live:{} requested={}x{} autofocus={} async_capture=on".format(
            args.camera, args.width, args.height, args.autofocus
        ),
        flush=True,
    )
    print(
        "IDs: upper side 0-3, lower side {}, upper top {}; face order: {}; upper rotation={}deg".format(
            sorted(args.lower_ids), args.top_id, args.face_order, args.upper_rotation_deg
        ),
        flush=True,
    )
    if camera_matrix is None:
        print("Calibration: approximate from frame size. Use --calibration for metric pose.", flush=True)
    print("Press ESC or q to quit.", flush=True)

    stop_event = threading.Event()
    raw_frames = LatestFrameSlot()
    annotated_frames = LatestFrameSlot()
    capture_rate = RateCounter()
    process_rate = RateCounter()
    pose_tracker = PoseTracker(args)
    window_name = "ArUco Cube Rigid Async Fused Pose"

    def capture_loop():
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.002)
                    continue
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                fps = capture_rate.tick()
                raw_frames.put((frame, fps))
        finally:
            raw_frames.close()

    def processing_loop():
        last_sequence = 0
        try:
            while not stop_event.is_set():
                item, sequence = raw_frames.get_latest(last_sequence, timeout=0.05)
                if item is None:
                    continue
                if sequence == last_sequence:
                    continue
                last_sequence = sequence
                frame, current_capture_fps = item
                frame = frame.copy()
                if camera_matrix is None:
                    current_camera_matrix, current_dist_coeffs = approximate_calibration(frame.shape[1], frame.shape[0])
                else:
                    current_camera_matrix, current_dist_coeffs = camera_matrix, dist_coeffs

                process_fps = process_rate.tick()
                process_frame(
                    frame,
                    dictionary,
                    params,
                    detector,
                    layout,
                    current_camera_matrix,
                    current_dist_coeffs,
                    args,
                    pose_tracker=pose_tracker,
                    capture_fps=current_capture_fps,
                    process_fps=process_fps,
                )
                annotated_frames.put(frame)
        finally:
            annotated_frames.close()

    capture_thread = threading.Thread(target=capture_loop, name="camera-capture", daemon=True)
    process_thread = threading.Thread(target=processing_loop, name="aruco-pose-processing", daemon=True)
    capture_thread.start()
    process_thread.start()

    last_annotated_sequence = 0
    try:
        while not stop_event.is_set():
            frame, sequence = annotated_frames.get_latest(last_annotated_sequence, timeout=0.01)
            if frame is not None and sequence != last_annotated_sequence:
                last_annotated_sequence = sequence
                cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                stop_event.set()
                break
    finally:
        stop_event.set()
        raw_frames.close()
        annotated_frames.close()
        capture_thread.join(timeout=1.0)
        process_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect nine rigid ArUco tags on upper/lower cube faces and estimate one fused model pose asynchronously."
    )
    parser.add_argument("--dictionary", default="DICT_6X6_250", choices=sorted(DICTIONARIES))
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width. Default: 1280.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height. Default: 720.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, open camera.")
    parser.add_argument("--output", default=None, help="Optional annotated image output path for --image.")
    parser.add_argument("--show", action="store_true", help="Show the annotated image when using --image.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file.")
    parser.add_argument("--cube-size", type=float, default=0.04, help="Single cube side length in meters. Default: 0.04.")
    parser.add_argument("--vertical-gap", type=float, default=0.0, help="Gap between lower and upper cube in meters.")
    parser.add_argument("--marker-length", type=float, default=0.04, help="ArUco marker side length in meters. Default: 0.04.")
    parser.add_argument(
        "--face-order",
        default="front,right,back,left",
        help="Face order for upper ids 0-3 and lower ids 4-7 if present. Default: front,right,back,left.",
    )
    parser.add_argument("--lower-ids", default="4,5,6,7", help="Lower cube ArUco IDs actually present. Default: 4,5,6,7.")
    parser.add_argument("--top-id", type=int, default=8, help="Upper cube top-face ArUco ID. Use -1 to disable. Default: 8.")
    parser.add_argument("--upper-rotation-deg", type=float, default=45.0, help="Upper cube rotation around model Z axis. Default: 45.")
    parser.add_argument("--lower-rotation-deg", type=float, default=0.0, help="Lower cube rotation around model Z axis. Default: 0.")
    parser.add_argument(
        "--corner-rolls",
        default="",
        help="Static per-ID 3D corner order roll, e.g. 0:1,3:-1. Use this when a printed tag is rotated on its face.",
    )
    parser.add_argument(
        "--auto-roll-ids",
        default="none",
        help="IDs whose corner order is searched each frame. Disabled by default to avoid pose jumps. Example: 8 or 0,3,8.",
    )
    parser.add_argument(
        "--auto-roll-max-candidates",
        type=int,
        default=64,
        help="Maximum dynamic corner-roll candidates tested per frame.",
    )
    parser.add_argument(
        "--id-face-map",
        default="",
        help="Override ID to cube face assignment, e.g. 0:left,3:front,8:top. Faces: front,right,back,left,top.",
    )
    parser.add_argument("--axis-length", type=float, default=0.04, help="Drawn model axis length in meters.")
    parser.add_argument("--cube-line-thickness", type=int, default=3, help="Projected cube wireframe line thickness.")
    parser.add_argument("--draw-model-tags", action="store_true", default=True, help="Project all configured rigid tag outlines.")
    parser.add_argument("--no-draw-model-tags", dest="draw_model_tags", action="store_false", help="Hide projected rigid tag outlines.")
    parser.add_argument("--aruco3", action="store_true", help="Enable OpenCV ArUco3 detector path when supported.")
    parser.add_argument("--adaptive-min", type=int, default=3)
    parser.add_argument("--adaptive-max", type=int, default=53)
    parser.add_argument("--adaptive-step", type=int, default=10)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.015)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    parser.add_argument("--ransac", action="store_true", default=False, help="Use solvePnPRansac when enough points exist.")
    parser.add_argument("--no-ransac", dest="ransac", action="store_false", help="Use plain solvePnP.")
    parser.add_argument("--ransac-iterations", type=int, default=100)
    parser.add_argument("--reprojection-error", type=float, default=5.0)
    parser.add_argument("--min-tags-for-pose", type=int, default=2, help="Minimum configured tags needed for fused object pose.")
    parser.add_argument("--pose-filter", choices=("none", "ema"), default="ema", help="Smooth the final fused rvec/tvec. Default: ema.")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="EMA pose smoothing alpha. Higher follows motion faster.")
    parser.add_argument("--hold-last-seconds", type=float, default=0.25, help="Hold last stable pose briefly when a frame is rejected.")
    parser.add_argument(
        "--max-stable-reprojection-error",
        type=float,
        default=8.0,
        help="Reject fused poses above this mean reprojection error in pixels.",
    )
    parser.add_argument(
        "--max-rotation-jump-deg",
        type=float,
        default=70.0,
        help="Reject one-frame rotation jumps larger than this angle.",
    )
    parser.add_argument(
        "--max-translation-jump",
        type=float,
        default=0.15,
        help="Reject one-frame translation jumps larger than this many meters.",
    )
    parser.add_argument("--print-pose", action="store_true", help="Print pose every frame when available.")
    return parser.parse_args()


def main():
    args = parse_args()
    dictionary = get_dictionary(args.dictionary)
    params = create_detector_parameters(args)
    detector = create_detector(dictionary, params)
    layout = build_marker_layout(args)

    camera_matrix = None
    dist_coeffs = None
    calibration_path = None
    if args.calibration:
        calibration_path = Path(args.calibration).expanduser()
    elif DEFAULT_CALIBRATION.exists():
        calibration_path = DEFAULT_CALIBRATION
    if calibration_path is not None:
        camera_matrix, dist_coeffs = read_calibration(calibration_path)

    if args.image:
        run_image(args, dictionary, params, detector, layout, camera_matrix, dist_coeffs)
    else:
        run_camera(args, dictionary, params, detector, layout, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Exception:", exc, file=sys.stderr)
        sys.exit(1)
