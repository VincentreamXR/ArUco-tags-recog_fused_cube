#!/usr/bin/env python3
import argparse
import itertools
import os
import sys
import threading
import time
import traceback
from pathlib import Path


DEEPPARUCO_PYTHON = "/home/zsyy/anaconda3/envs/deeparuco39/bin/python"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "utils" / "camera_calibration.yml"
PRISM_COLOR = (0, 220, 255)
UPPER_CUBE_COLOR = (0, 255, 255)
LOWER_CUBE_COLOR = (255, 0, 0)
TAG_FACE_COLORS = [
    (60, 180, 75),
    (230, 25, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (188, 246, 12),
]
DETECTED_CORNER_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 255, 0),
]


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
    # Object frame origin is centered between the upper and lower cube centers:
    # +X -> right face, +Y -> front face, +Z -> top face.
    # Each face stores: outward normal, marker horizontal axis, marker vertical axis.
    "front": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "right": (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "back": (np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "left": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "top": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
}

ADJACENT_TAG_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (8, 0), (8, 1), (8, 2), (8, 3),
]


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


class LowPassFilter:
    def __init__(self):
        self.value = None

    def reset(self):
        self.value = None

    def filter(self, value, alpha):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self.value is None:
            self.value = value.copy()
        else:
            self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value.copy()


class OneEuroVectorFilter:
    def __init__(self, min_cutoff=1.2, beta=0.04, derivate_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivate_cutoff = float(derivate_cutoff)
        self.last_time = None
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    @staticmethod
    def smoothing_factor(dt, cutoff):
        r = 2.0 * np.pi * float(cutoff) * float(dt)
        return r / (r + 1.0)

    def reset(self):
        self.last_time = None
        self.x_filter.reset()
        self.dx_filter.reset()

    def filter(self, value, timestamp=None, cutoff_scale=1.0):
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.last_time is None:
            self.last_time = now
            self.x_filter.reset()
            self.dx_filter.reset()
            return self.x_filter.filter(value, 1.0)

        dt = max(now - self.last_time, 1e-6)
        self.last_time = now
        previous = self.x_filter.value if self.x_filter.value is not None else value
        derivative = (value - previous) / dt
        dx_alpha = self.smoothing_factor(dt, self.derivate_cutoff)
        dx_hat = self.dx_filter.filter(derivative, dx_alpha)
        speed = float(np.max(np.abs(dx_hat)))
        cutoff = max(1e-4, (self.min_cutoff * float(cutoff_scale)) + self.beta * speed)
        alpha = self.smoothing_factor(dt, cutoff)
        return self.x_filter.filter(value, alpha)


class PoseTracker:
    def __init__(self, args):
        if args.pose_filter == "ema":
            self.pose_filter = PoseEmaFilter(args.ema_alpha)
        elif args.pose_filter == "one_euro":
            self.pose_filter = (
                OneEuroVectorFilter(args.one_euro_min_cutoff, args.one_euro_beta, args.one_euro_derivate_cutoff),
                OneEuroVectorFilter(args.one_euro_min_cutoff, args.one_euro_beta, args.one_euro_derivate_cutoff),
            )
        else:
            self.pose_filter = None
        self.max_hold_time = float(args.hold_last_seconds)
        self.max_reprojection_error = float(args.max_stable_reprojection_error)
        self.min_follow = float(args.min_pose_follow)
        self.low_confidence_extra_smoothing = float(args.low_confidence_extra_smoothing)
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

    def compute_confidence(self, num_tags, mean_error):
        tag_score = np.clip((float(num_tags) - 1.0) / 3.0, 0.0, 1.0)
        error_score = np.clip(1.0 - float(mean_error) / max(self.max_reprojection_error, 1e-6), 0.0, 1.0)
        return float(0.6 * tag_score + 0.4 * error_score)

    def filter_pose(self, rvec, tvec, confidence):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if self.last_raw_rvec is not None and float(np.dot(self.last_raw_rvec.reshape(3), rvec.reshape(3))) < 0.0:
            rvec = -rvec
        if self.pose_filter is None:
            return rvec.copy(), tvec.copy()
        if isinstance(self.pose_filter, PoseEmaFilter):
            return self.pose_filter.update(rvec, tvec)
        r_filter, t_filter = self.pose_filter
        cutoff_scale = 1.0 + (1.0 - confidence) * self.low_confidence_extra_smoothing
        filtered_rvec = r_filter.filter(rvec.reshape(-1), cutoff_scale=cutoff_scale).reshape(3, 1)
        filtered_tvec = t_filter.filter(tvec.reshape(-1), cutoff_scale=cutoff_scale).reshape(3, 1)
        return filtered_rvec, filtered_tvec

    def update(self, rvec, tvec, mean_error, dynamic_rolls, num_tags):
        if mean_error > self.max_reprojection_error:
            return self.get_held_pose(), "held_bad_reproj"
        self.last_raw_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
        self.last_raw_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
        confidence = self.compute_confidence(num_tags, mean_error)
        out_rvec, out_tvec = self.filter_pose(self.last_raw_rvec, self.last_raw_tvec, confidence)
        if self.last_output_rvec is not None and self.last_output_tvec is not None:
            follow = self.min_follow + (1.0 - self.min_follow) * confidence
            out_rvec = (1.0 - follow) * self.last_output_rvec + follow * out_rvec
            out_tvec = (1.0 - follow) * self.last_output_tvec + follow * out_tvec
        self.last_output_rvec = out_rvec.copy()
        self.last_output_tvec = out_tvec.copy()
        self.last_update_time = time.monotonic()
        self.last_rolls = dict(dynamic_rolls)
        return (self.last_output_rvec.copy(), self.last_output_tvec.copy()), "measured"

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


def probe_camera_index(index, width, height, autofocus):
    cap = cv2.VideoCapture(int(index))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    if not cap.isOpened():
        cap.release()
        return False
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None)


def discover_camera_indices(max_index, width, height, autofocus):
    available = []
    for index in range(int(max_index)):
        if probe_camera_index(index, width, height, autofocus):
            available.append(index)
    return available


def parse_face_order(text):
    faces = [part.strip().lower() for part in text.split(",") if part.strip()]
    if len(faces) != 4 or any(face not in FACE_SPECS for face in faces):
        raise ValueError("--face-order must contain four faces from: front,right,back,left")
    if len(set(faces)) != 4:
        raise ValueError("--face-order must not repeat faces")
    return faces


def parse_id_list(text):
    if isinstance(text, (set, list, tuple)):
        return {int(marker_id) for marker_id in text}
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return set()
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not ids:
        raise ValueError("ID list must not be empty")
    return set(ids)


def parse_corner_rolls(text):
    rolls = {}
    if isinstance(text, dict):
        return {int(marker_id): int(roll) % 4 for marker_id, roll in text.items()}
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
    if isinstance(text, dict):
        return {int(marker_id): str(face).strip().lower() for marker_id, face in text.items()}
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


def parse_id_rotation_map(text):
    mapping = {}
    if isinstance(text, dict):
        return {int(marker_id): float(angle) for marker_id, angle in text.items()}
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return mapping
    for part in text.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError("--id-rotation-map entries must look like id:deg, for example 0:45,8:0")
        marker_id_text, angle_text = part.split(":", 1)
        mapping[int(marker_id_text.strip())] = float(angle_text.strip())
    return mapping


def roll_corners(corners, roll):
    roll = int(roll) % 4
    if roll == 0:
        return corners
    return np.roll(corners, -roll, axis=0).copy()


def face_center(face, side_center_z, prism_width, prism_height, prism_depth):
    if face == "front":
        return np.array([0.0, prism_depth * 0.5, side_center_z], dtype=np.float32)
    if face == "right":
        return np.array([prism_width * 0.5, 0.0, side_center_z], dtype=np.float32)
    if face == "back":
        return np.array([0.0, -prism_depth * 0.5, side_center_z], dtype=np.float32)
    if face == "left":
        return np.array([-prism_width * 0.5, 0.0, side_center_z], dtype=np.float32)
    if face == "top":
        return np.array([0.0, 0.0, prism_height * 0.5], dtype=np.float32)
    raise ValueError("Unsupported face: {}".format(face))


def get_cube_size(args):
    cube_size = getattr(args, "cube_size", None)
    if cube_size is None:
        cube_size = getattr(args, "prism_width", 0.04)
    return float(cube_size)


def get_vertical_gap(args):
    vertical_gap = getattr(args, "vertical_gap", None)
    if vertical_gap is None:
        vertical_gap = getattr(args, "tag_row_gap", 0.0)
    return float(vertical_gap)


def get_cube_center_z(args):
    cube_size = get_cube_size(args)
    vertical_gap = get_vertical_gap(args)
    center_offset = 0.5 * (cube_size + vertical_gap)
    upper_z = center_offset
    lower_z = -center_offset
    return upper_z, lower_z


def rotate_points_z(points, angle_deg):
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return np.asarray(points, dtype=np.float32) @ rotation.T


def marker_object_corners(face, center, marker_length, inplane_rotation_deg=0.0):
    _, base_u_axis, base_v_axis = FACE_SPECS[face]
    angle = np.deg2rad(inplane_rotation_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    u_axis = cos_a * base_u_axis + sin_a * base_v_axis
    v_axis = -sin_a * base_u_axis + cos_a * base_v_axis
    half = marker_length * 0.5
    center = np.asarray(center, dtype=np.float32).reshape(3)
    return np.array(
        [
            center - half * u_axis - half * v_axis,
            center + half * u_axis - half * v_axis,
            center + half * u_axis + half * v_axis,
            center - half * u_axis + half * v_axis,
        ],
        dtype=np.float32,
    )


def marker_object_corners_on_cube(face, center_z, cube_size, marker_length, inplane_rotation_deg=0.0):
    normal, _, _ = FACE_SPECS[face]
    center = normal * (cube_size * 0.5) + np.array([0.0, 0.0, center_z], dtype=np.float32)
    return marker_object_corners(face, center, marker_length, inplane_rotation_deg)


def rotated_marker_object_corners(face, center_z, cube_size, marker_length, cube_rotation_deg, inplane_rotation_deg=0.0):
    corners = marker_object_corners_on_cube(face, center_z, cube_size, marker_length, inplane_rotation_deg)
    return rotate_points_z(corners, cube_rotation_deg)


def build_marker_layout(args):
    face_order = parse_face_order(args.face_order)
    args.lower_ids = parse_id_list(args.lower_ids)
    args.corner_rolls = parse_corner_rolls(args.corner_rolls)
    args.auto_roll_ids = parse_id_list(args.auto_roll_ids)
    args.id_face_map = parse_id_face_map(args.id_face_map)
    args.id_rotation_map = parse_id_rotation_map(args.id_rotation_map)
    cube_size = get_cube_size(args)
    upper_z, lower_z = get_cube_center_z(args)
    layout = {}

    for index, face in enumerate(face_order):
        marker_face = args.id_face_map.get(index, face)
        layout[index] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                upper_z,
                cube_size,
                args.marker_length,
                args.upper_rotation_deg,
                args.id_rotation_map.get(index, 0.0),
            ),
            args.corner_rolls.get(index, 0),
        )

        lower_id = index + 4
        if lower_id in args.lower_ids:
            marker_face = args.id_face_map.get(lower_id, face)
            layout[lower_id] = roll_corners(
                rotated_marker_object_corners(
                    marker_face,
                    lower_z,
                    cube_size,
                    args.marker_length,
                    args.lower_rotation_deg,
                    args.id_rotation_map.get(lower_id, 0.0),
                ),
                args.corner_rolls.get(lower_id, 0),
            )

    if args.top_id >= 0:
        marker_face = args.id_face_map.get(args.top_id, "top")
        layout[args.top_id] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                upper_z,
                cube_size,
                args.marker_length,
                args.upper_rotation_deg,
                args.id_rotation_map.get(args.top_id, 0.0),
            ),
            args.corner_rolls.get(args.top_id, 0),
        )
    return layout


def visible_adjacent_tag_pairs(visible_ids):
    visible = {int(marker_id) for marker_id in visible_ids}
    return [(a, b) for a, b in ADJACENT_TAG_PAIRS if a in visible and b in visible]


def score_pose_candidate_for_selection(mean_error, used_ids, previous_pose=None, rvec=None, tvec=None):
    tag_bonus = 0.35 * max(0, len(set(used_ids)) - 1)
    score = float(mean_error) - tag_bonus
    if previous_pose is not None and tvec is not None:
        _, previous_tvec = previous_pose
        translation_delta = float(np.linalg.norm(np.asarray(tvec, dtype=np.float64).reshape(3, 1) - previous_tvec))
        score += min(translation_delta * 3.0, 3.0)
    return score


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


def collect_correspondences_for_ids(marker_corners, marker_ids, layout, selected_ids):
    if marker_ids is None:
        return None, None, []

    detected = {}
    for corners, marker_id_array in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(marker_id_array)
        if marker_id in layout and marker_id not in detected:
            detected[marker_id] = corners.reshape(4, 2).astype(np.float32)

    object_points = []
    image_points = []
    used_ids = []
    for marker_id in selected_ids:
        marker_id = int(marker_id)
        if marker_id not in layout or marker_id not in detected:
            continue
        object_points.append(layout[marker_id])
        image_points.append(detected[marker_id])
        used_ids.append(marker_id)

    if not object_points:
        return None, None, []
    return np.vstack(object_points).astype(np.float32), np.vstack(image_points).astype(np.float32), used_ids


def compute_tag_reprojection_errors(object_points, image_points, used_ids, rvec, tvec, camera_matrix, dist_coeffs):
    if object_points is None or image_points is None or not used_ids:
        return {}
    if not hasattr(cv2, "projectPoints"):
        return {}
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    point_errors = np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
    tag_errors = {}
    offset = 0
    for marker_id in used_ids:
        marker_point_errors = point_errors[offset:offset + 4]
        offset += 4
        if len(marker_point_errors) == 0:
            continue
        tag_errors[int(marker_id)] = (
            float(np.mean(marker_point_errors)),
            float(np.max(marker_point_errors)),
        )
    return tag_errors


def select_inlier_tag_ids(tag_errors, used_ids, args):
    ordered_ids = []
    for marker_id in used_ids:
        marker_id = int(marker_id)
        if marker_id not in ordered_ids:
            ordered_ids.append(marker_id)

    min_tags = max(1, int(getattr(args, "min_tags_for_pose", 1)))
    if (
        not tag_errors
        or any(marker_id not in tag_errors for marker_id in ordered_ids)
        or not getattr(args, "reject_outlier_tags", True)
        or len(ordered_ids) <= min_tags
    ):
        return ordered_ids

    mean_errors = np.array([tag_errors[marker_id][0] for marker_id in ordered_ids], dtype=np.float64)
    median_error = float(np.median(mean_errors))
    ratio = max(float(getattr(args, "tag_outlier_ratio", 2.8)), 1.0)
    min_threshold = max(float(getattr(args, "min_tag_mean_reprojection_error", 4.0)), 0.0)
    max_threshold = float(getattr(args, "max_tag_mean_reprojection_error", 12.0))
    threshold = max(min_threshold, median_error * ratio)
    if max_threshold > 0.0:
        threshold = min(threshold, max_threshold)

    inlier_ids = [marker_id for marker_id in ordered_ids if tag_errors[marker_id][0] <= threshold]
    if len(inlier_ids) < min_tags:
        return ordered_ids
    return inlier_ids


def refine_pose_by_tag_outliers(
    marker_corners,
    marker_ids,
    layout,
    object_points,
    image_points,
    used_ids,
    pose,
    camera_matrix,
    dist_coeffs,
    args,
):
    if pose is None or not getattr(args, "reject_outlier_tags", True) or len(set(used_ids)) <= 1:
        return object_points, image_points, used_ids, pose

    rvec, tvec, _, _, _ = pose
    tag_errors = compute_tag_reprojection_errors(
        object_points,
        image_points,
        used_ids,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    inlier_ids = select_inlier_tag_ids(tag_errors, used_ids, args)
    if tuple(inlier_ids) == tuple(used_ids):
        return object_points, image_points, used_ids, pose

    refined_object_points, refined_image_points, refined_used_ids = collect_correspondences_for_ids(
        marker_corners,
        marker_ids,
        layout,
        inlier_ids,
    )
    refined_pose = estimate_fused_pose(
        refined_object_points,
        refined_image_points,
        refined_used_ids,
        camera_matrix,
        dist_coeffs,
        args,
        initial_guess=(rvec, tvec),
    )
    if refined_pose is None:
        return object_points, image_points, used_ids, pose
    return refined_object_points, refined_image_points, refined_used_ids, refined_pose


def rotation_distance_degrees(rvec_a, rvec_b):
    rotation_a, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    rotation_b, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    delta = rotation_a @ rotation_b.T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def score_pose_candidate(pose, used_ids, dynamic_rolls, previous_rolls, top_pose, args, previous_pose=None):
    rvec, _, _, mean_error, _ = pose
    switch_count = sum(1 for marker_id, roll in dynamic_rolls.items() if previous_rolls.get(marker_id, 0) != roll)
    score = score_pose_candidate_for_selection(
        mean_error,
        used_ids,
        previous_pose=previous_pose,
        rvec=pose[0],
        tvec=pose[1],
    )
    score += args.roll_switch_penalty * float(switch_count)
    if args.lock_top_pose_to_top_tag and top_pose is not None:
        score += args.top_disambiguation_weight * rotation_distance_degrees(rvec, top_pose[0])
    return float(score)


def estimate_best_fused_pose(marker_corners, marker_ids, layout, camera_matrix, dist_coeffs, args, initial_guess, previous_rolls):
    auto_ids = []
    if marker_ids is not None:
        detected_ids = [int(marker_id) for marker_id in marker_ids.reshape(-1)]
        auto_ids = [marker_id for marker_id in detected_ids if marker_id in args.auto_roll_ids and marker_id in layout]
    auto_ids = sorted(set(auto_ids))
    top_image_corners = None
    top_pose_cache = {}
    if args.lock_top_pose_to_top_tag and args.top_id >= 0:
        top_image_corners = extract_marker_image_corners(marker_corners, marker_ids, args.top_id)

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
        top_pose = None
        if top_image_corners is not None and args.top_id in candidate_layout:
            top_roll = int(dynamic_rolls.get(args.top_id, 0))
            if top_roll not in top_pose_cache:
                top_pose_cache[top_roll] = estimate_pose_from_points(
                    candidate_layout[args.top_id].astype(np.float32),
                    top_image_corners,
                    camera_matrix,
                    dist_coeffs,
                )
            top_pose = top_pose_cache[top_roll]
        solve_guess = initial_guess
        if solve_guess is None and top_pose is not None:
            solve_guess = top_pose

        candidate_id_sets = []
        if used_ids:
            candidate_id_sets.append(tuple(used_ids))
            candidate_id_sets.extend(visible_adjacent_tag_pairs(used_ids))

        seen_id_sets = set()
        for candidate_ids in candidate_id_sets:
            candidate_ids = tuple(int(marker_id) for marker_id in candidate_ids)
            if candidate_ids in seen_id_sets:
                continue
            seen_id_sets.add(candidate_ids)
            if len(candidate_ids) == len(used_ids) and tuple(used_ids) == candidate_ids:
                candidate_object_points, candidate_image_points, candidate_used_ids = object_points, image_points, used_ids
            else:
                candidate_object_points, candidate_image_points, candidate_used_ids = collect_correspondences_for_ids(
                    marker_corners,
                    marker_ids,
                    candidate_layout,
                    candidate_ids,
                )

            pose = estimate_fused_pose(
                candidate_object_points,
                candidate_image_points,
                candidate_used_ids,
                camera_matrix,
                dist_coeffs,
                args,
                initial_guess=solve_guess,
            )
            if pose is None:
                continue
            if top_pose is not None and initial_guess is not None:
                top_mismatch_deg = rotation_distance_degrees(pose[0], top_pose[0])
                if top_mismatch_deg >= args.top_disambiguation_retry_deg:
                    top_seed_pose = estimate_fused_pose(
                        candidate_object_points,
                        candidate_image_points,
                        candidate_used_ids,
                        camera_matrix,
                        dist_coeffs,
                        args,
                        initial_guess=top_pose,
                    )
                    if top_seed_pose is not None:
                        current_score = score_pose_candidate(
                            pose,
                            candidate_used_ids,
                            dynamic_rolls,
                            previous_rolls,
                            top_pose,
                            args,
                            previous_pose=initial_guess,
                        )
                        top_seed_score = score_pose_candidate(
                            top_seed_pose,
                            candidate_used_ids,
                            dynamic_rolls,
                            previous_rolls,
                            top_pose,
                            args,
                            previous_pose=initial_guess,
                        )
                        if top_seed_score < current_score:
                            pose = top_seed_pose
            candidate_object_points, candidate_image_points, candidate_used_ids, pose = refine_pose_by_tag_outliers(
                marker_corners,
                marker_ids,
                candidate_layout,
                candidate_object_points,
                candidate_image_points,
                candidate_used_ids,
                pose,
                camera_matrix,
                dist_coeffs,
                args,
            )
            score_top_pose = top_pose if args.top_id in candidate_used_ids else None
            score = score_pose_candidate(
                pose,
                candidate_used_ids,
                dynamic_rolls,
                previous_rolls,
                score_top_pose,
                args,
                previous_pose=initial_guess,
            )
            if best is None or score < best[5]:
                best = (candidate_object_points, candidate_image_points, candidate_used_ids, pose, dynamic_rolls, score)

    if best is None:
        object_points, image_points, used_ids = collect_correspondences(marker_corners, marker_ids, layout)
        return object_points, image_points, used_ids, None, {}
    return best[:5]


def estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, args, initial_guess=None):
    if object_points is None or len(object_points) < 4 or len(set(used_ids)) < args.min_tags_for_pose:
        return None

    if initial_guess is not None:
        guess_rvec, guess_tvec = initial_guess
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec=np.asarray(guess_rvec, dtype=np.float64).reshape(3, 1).copy(),
            tvec=np.asarray(guess_tvec, dtype=np.float64).reshape(3, 1).copy(),
            useExtrinsicGuess=True,
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


def estimate_pose_from_points(object_points, image_points, camera_matrix, dist_coeffs):
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec


def extract_marker_image_corners(marker_corners, marker_ids, target_id):
    if marker_ids is None:
        return None
    for corners, marker_id_array in zip(marker_corners, marker_ids.reshape(-1)):
        if int(marker_id_array) == int(target_id):
            return corners.reshape(4, 2).astype(np.float32)
    return None


def estimate_single_marker_pose(marker_id, marker_corners, marker_ids, layout, camera_matrix, dist_coeffs):
    if marker_id not in layout:
        return None
    image_corners = extract_marker_image_corners(marker_corners, marker_ids, marker_id)
    if image_corners is None:
        return None
    return estimate_pose_from_points(layout[marker_id].astype(np.float32), image_corners, camera_matrix, dist_coeffs)


def rotation_matrix_aligning_vectors(source, target):
    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    source /= max(np.linalg.norm(source), 1e-9)
    target /= max(np.linalg.norm(target), 1e-9)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-9:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - source * np.dot(axis, source)
        axis /= max(np.linalg.norm(axis), 1e-9)
        rvec = axis.reshape(3, 1) * np.pi
        rotation, _ = cv2.Rodrigues(rvec)
        return rotation

    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    rvec = axis.reshape(3, 1) * angle
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation


def prism_vertices(width, depth, height):
    hx = width * 0.5
    hy = depth * 0.5
    hz = height * 0.5
    return np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )


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


def draw_translucent_polygon(frame, points, color, alpha):
    pts = np.round(points).astype(np.int32).reshape(-1, 2)
    if len(pts) < 3 or not np.all(np.isfinite(pts)):
        return
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, pts, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0, frame)


def draw_rigid_tag_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs):
    rotation, _ = cv2.Rodrigues(rvec)
    entries = []
    for marker_id, object_corners in sorted(layout.items()):
        camera_points = object_corners.astype(np.float64) @ rotation.T + tvec.reshape(1, 3)
        projected, _ = cv2.projectPoints(object_corners, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        depth = float(np.mean(camera_points[:, 2]))
        entries.append((depth, marker_id, projected))

    for _, marker_id, projected in sorted(entries, key=lambda item: item[0], reverse=True):
        color = TAG_FACE_COLORS[marker_id % len(TAG_FACE_COLORS)]
        draw_translucent_polygon(frame, projected, color, alpha=0.28)
        draw_projected_polyline(frame, projected, (0, 220, 0), closed=True)
        center = point_tuple(np.mean(projected, axis=0))
        if center is not None:
            cv2.putText(
                frame,
                str(marker_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
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


def draw_detected_corners(frame, marker_corners, marker_ids, args):
    if marker_ids is None:
        return
    for corners, marker_id_value in zip(marker_corners, marker_ids.reshape(-1)):
        corners = corners.reshape(4, 2)
        marker_id = int(marker_id_value)
        for corner_index, point in enumerate(corners):
            center = point_tuple(point)
            if center is None:
                continue
            cv2.circle(
                frame,
                center,
                args.corner_radius,
                DETECTED_CORNER_COLORS[corner_index % len(DETECTED_CORNER_COLORS)],
                -1,
                cv2.LINE_AA,
            )
            if args.draw_corner_index:
                cv2.putText(
                    frame,
                    "{}:c{}".format(marker_id, corner_index),
                    (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )


def draw_fused_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs, args):
    if args.draw_prism_wireframe:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        cube_size = get_cube_size(args)
        upper_z, lower_z = get_cube_center_z(args)
        for vertices, color in (
            (cube_vertices(upper_z, cube_size, args.upper_rotation_deg), UPPER_CUBE_COLOR),
            (cube_vertices(lower_z, cube_size, args.lower_rotation_deg), LOWER_CUBE_COLOR),
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
        if args.draw_detected_corners:
            draw_detected_corners(frame, marker_corners, marker_ids, args)

    object_points, image_points, used_ids, pose, dynamic_rolls = estimate_best_fused_pose(
        marker_corners,
        marker_ids,
        layout,
        camera_matrix,
        dist_coeffs,
        args,
        initial_guess=None if pose_tracker is None else pose_tracker.get_initial_guess(),
        previous_rolls={} if pose_tracker is None else pose_tracker.last_rolls,
    )
    draw_layout = make_candidate_layout(layout, dynamic_rolls)

    pose_text = "pose: need at least {} configured tags".format(args.min_tags_for_pose)
    pose_state_text = "lost"
    if pose is not None:
        rvec, tvec, inliers, mean_error, max_error = pose
        stable_pose = (rvec, tvec)
        if pose_tracker is not None:
            tracked_pose, pose_state_text = pose_tracker.update(
                rvec,
                tvec,
                mean_error,
                dynamic_rolls,
                num_tags=len(set(used_ids)),
            )
            if tracked_pose is not None:
                stable_pose = tracked_pose
            else:
                stable_pose = pose_tracker.get_held_pose()
                if stable_pose is None:
                    stable_pose = (rvec, tvec)
                    pose_state_text = "{}_raw_drawn".format(pose_state_text)
        if stable_pose is not None:
            rvec, tvec = stable_pose
            active_rolls = dynamic_rolls if pose_tracker is None or pose_state_text == "measured" else pose_tracker.last_rolls
            draw_layout = make_candidate_layout(layout, active_rolls)
            draw_fused_model(frame, draw_layout, rvec, tvec, camera_matrix, dist_coeffs, args)
            tx, ty, tz = tvec.reshape(-1)
            roll, pitch, yaw = rotation_to_euler_xyz(rvec)
            object_to_camera = pose_matrix_object_to_camera(rvec, tvec)
            pose_text = (
                "rigid object pose t=({:.3f},{:.3f},{:.3f})m rpy=({:.1f},{:.1f},{:.1f})deg pts={} err={:.2f}px state={}"
                .format(tx, ty, tz, roll, pitch, yaw, inliers, mean_error, pose_state_text)
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
    if pose is None and pose_tracker is not None:
        held_pose = pose_tracker.get_held_pose()
        if held_pose is not None:
            rvec, tvec = held_pose
            draw_fused_model(frame, make_candidate_layout(layout, pose_tracker.last_rolls), rvec, tvec, camera_matrix, dist_coeffs, args)
            tx, ty, tz = tvec.reshape(-1)
            roll, pitch, yaw = rotation_to_euler_xyz(rvec)
            pose_text = (
                "rigid object pose t=({:.3f},{:.3f},{:.3f})m rpy=({:.1f},{:.1f},{:.1f})deg state=held_no_detection"
                .format(tx, ty, tz, roll, pitch, yaw)
            )
            pose_state_text = "held_no_detection"

    rolls_text = dynamic_rolls if dynamic_rolls else (pose_tracker.last_rolls if pose_tracker is not None and pose_tracker.last_rolls else "off/unchanged")
    cube_size = get_cube_size(args)
    vertical_gap = get_vertical_gap(args)
    lines = [
        "ArUco rigid object fused pose  dictionary: {}  capture={:.1f}fps process={:.1f}fps".format(
            args.dictionary, capture_fps, process_fps
        ),
        "detected ids: {}  used ids: {}  rejected: {}".format(ids[:12], used_ids, len(rejected)),
        pose_text,
        "auto corner rolls: {}".format(rolls_text),
        "cube_size={:.4f}m  vertical_gap={:.4f}m  marker_length={:.4f}m  hold={:.2f}s".format(
            cube_size, vertical_gap, args.marker_length, args.hold_last_seconds
        ),
        "detected corners show id:c0-c3  adjacent faces add fused pose candidates",
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
    camera_text = str(args.camera).strip().lower()
    scanned = []
    if camera_text in ("auto", "any"):
        scanned = discover_camera_indices(args.camera_scan_max, args.width, args.height, args.autofocus)
        if not scanned:
            raise RuntimeError(
                "No working camera found in 0..{}; check /dev/video* or pass --camera N".format(
                    args.camera_scan_max - 1
                )
            )
        camera_index = scanned[0]
    else:
        camera_index = int(args.camera)
        if not probe_camera_index(camera_index, args.width, args.height, args.autofocus):
            scanned = discover_camera_indices(args.camera_scan_max, args.width, args.height, args.autofocus)
            if camera_index not in scanned and scanned:
                camera_index = scanned[0]
            elif not scanned:
                raise RuntimeError(
                    "Cannot open camera index {}. No working cameras found in 0..{}.".format(
                        args.camera, args.camera_scan_max - 1
                    )
                )

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, args.autofocus)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera index {}. Available candidates: {}".format(
                camera_index, scanned if scanned else "none"
            )
        )

    print("OpenCV: {}".format(cv2.__version__), flush=True)
    print(
        "Camera: live:{} requested={}x{} autofocus={} async_capture=on".format(
            camera_index, args.width, args.height, args.autofocus
        ),
        flush=True,
    )
    if scanned:
        print("Camera scan candidates: {}".format(scanned), flush=True)
    print(
        "Model: 9-tag cube stack {:.3f}x{:.3f}x{:.3f}m, side ids [0-7], top id {}, iterative PnP".format(
            args.prism_width, args.prism_height, args.prism_depth, args.top_id
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
                try:
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
                except Exception:
                    traceback.print_exc()
                    draw_text_panel(
                        frame,
                        [
                            "processing exception",
                            "see terminal traceback",
                            "pose pipeline paused for this frame",
                        ],
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
        description="Detect a rigid 9-tag ArUco cube stack and estimate one fused model pose asynchronously."
    )
    parser.add_argument("--dictionary", default="DICT_6X6_250", choices=sorted(DICTIONARIES))
    parser.add_argument("--camera", default="0", help="Camera index or 'auto'. Default: 0.")
    parser.add_argument("--camera-scan-max", type=int, default=10, help="Max camera index to probe when scanning. Default: 10.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width. Default: 1280.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height. Default: 720.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, open camera.")
    parser.add_argument("--output", default=None, help="Optional annotated image output path for --image.")
    parser.add_argument("--show", action="store_true", help="Show the annotated image when using --image.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file.")
    parser.add_argument("--prism-width", type=float, default=0.04, help="Legacy rigid object width in meters. Default: 0.04.")
    parser.add_argument("--prism-height", type=float, default=0.08, help="Legacy rigid object height in meters. Default: 0.08.")
    parser.add_argument("--prism-depth", type=float, default=0.04, help="Legacy rigid object depth in meters. Default: 0.04.")
    parser.add_argument("--tag-row-gap", type=float, default=0.0, help="Vertical gap between upper and lower tag rows in meters.")
    parser.add_argument("--cube-size", type=float, default=0.04, help="Single cube edge length in meters. Default: 0.04.")
    parser.add_argument("--vertical-gap", type=float, default=0.0, help="Gap between lower and upper cube in meters.")
    parser.add_argument("--marker-length", type=float, default=0.032, help="ArUco marker side length in meters. Default: 0.032.")
    parser.add_argument(
        "--face-order",
        default="front,right,back,left",
        help="Legacy option kept for compatibility. Explicit per-ID layout is used by default.",
    )
    parser.add_argument("--lower-ids", default="4,5,6,7", help="Lower cube ArUco IDs actually present. Default: 4,5,6,7.")
    parser.add_argument("--upper-rotation-deg", type=float, default=45.0, help="Upper cube rotation around object Z axis. Default: 45.")
    parser.add_argument("--lower-rotation-deg", type=float, default=0.0, help="Lower cube rotation around object Z axis. Default: 0.")
    parser.add_argument("--top-id", type=int, default=8, help="Top face ArUco ID. Set negative to disable top face tag. Default: 8.")
    parser.add_argument("--top-rotation-deg", type=float, default=0.0, help="Legacy option kept for compatibility. Prefer --id-rotation-map for per-ID control.")
    parser.add_argument(
        "--corner-rolls",
        default="",
        help="Static per-ID 3D corner order roll, e.g. 0:1,3:-1. Use this when a printed tag is rotated on its face.",
    )
    parser.add_argument(
        "--auto-roll-ids",
        default="none",
        help="IDs whose corner order is searched each frame. Default: none; use c0-c3 labels and --corner-rolls for fixed corrections.",
    )
    parser.add_argument(
        "--auto-roll-max-candidates",
        type=int,
        default=64,
        help="Maximum dynamic corner-roll candidates tested per frame. Default: 64 covers IDs 0, 3 and 8 together.",
    )
    parser.add_argument(
        "--id-face-map",
        default="0:right,1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:top",
        help="Override ID to prism face assignment. Faces: front,right,back,left,top.",
    )
    parser.add_argument(
        "--id-rotation-map",
        default="",
        help="Override per-ID in-plane rotation in degrees, e.g. 0:0,1:0,2:0,3:0,8:0.",
    )
    parser.add_argument("--axis-length", type=float, default=0.04, help="Drawn model axis length in meters.")
    parser.add_argument("--cube-line-thickness", type=int, default=3, help="Projected cube wireframe line thickness.")
    parser.add_argument("--draw-prism-wireframe", action="store_true", default=True, help="Draw the upper/lower cube wireframes. Default: on.")
    parser.add_argument("--no-draw-prism-wireframe", dest="draw_prism_wireframe", action="store_false", help="Hide cube wireframes.")
    parser.add_argument("--draw-model-tags", action="store_true", default=True, help="Project all configured rigid tag outlines. Default: on.")
    parser.add_argument("--no-draw-model-tags", dest="draw_model_tags", action="store_false", help="Hide projected rigid tag outlines.")
    parser.add_argument("--draw-detected-corners", action="store_true", default=True, help="Draw detected ArUco corner dots. Default: on.")
    parser.add_argument("--no-draw-detected-corners", dest="draw_detected_corners", action="store_false", help="Hide detected corner dots.")
    parser.add_argument("--draw-corner-index", action="store_true", default=True, help="Draw marker-id:corner-index labels. Default: on.")
    parser.add_argument("--no-draw-corner-index", dest="draw_corner_index", action="store_false", help="Hide marker-id:corner-index labels.")
    parser.add_argument("--corner-radius", type=int, default=4, help="Detected corner dot radius in pixels.")
    parser.add_argument("--aruco3", action="store_true", help="Enable OpenCV ArUco3 detector path when supported.")
    parser.add_argument("--adaptive-min", type=int, default=3)
    parser.add_argument("--adaptive-max", type=int, default=53)
    parser.add_argument("--adaptive-step", type=int, default=10)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.015)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    parser.add_argument("--min-tags-for-pose", type=int, default=1, help="Minimum configured tags needed for fused object pose. Default: 1.")
    parser.add_argument("--lock-top-pose-to-top-tag", action="store_true", default=True, help="When top tag is visible, use it to disambiguate fused orientation candidates without hard-overriding the fused pose. Default: on.")
    parser.add_argument("--no-lock-top-pose-to-top-tag", dest="lock_top_pose_to_top_tag", action="store_false", help="Disable top-tag orientation disambiguation.")
    parser.add_argument("--top-disambiguation-weight", type=float, default=0.06, help="Score penalty per degree of fused-vs-top orientation mismatch when top tag is visible.")
    parser.add_argument("--top-disambiguation-retry-deg", type=float, default=35.0, help="If fused pose disagrees with the top-tag pose by at least this many degrees, retry fused solvePnP from the top-tag seed.")
    parser.add_argument("--pose-filter", choices=("none", "ema", "one_euro"), default="one_euro", help="Smooth the final fused rvec/tvec. Default: one_euro.")
    parser.add_argument("--ema-alpha", type=float, default=0.22, help="EMA pose smoothing alpha. Higher follows motion faster.")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.5, help="One Euro base cutoff. Lower is steadier.")
    parser.add_argument("--one-euro-beta", type=float, default=0.08, help="One Euro speed coefficient. Higher tracks fast motion more closely.")
    parser.add_argument("--one-euro-derivate-cutoff", type=float, default=1.0, help="One Euro derivative cutoff.")
    parser.add_argument("--min-pose-follow", type=float, default=0.18, help="Minimum fraction of each new pose update to keep under low confidence.")
    parser.add_argument("--low-confidence-extra-smoothing", type=float, default=2.2, help="Extra One Euro smoothing applied when visible tags are few or reprojection error is higher.")
    parser.add_argument("--hold-last-seconds", type=float, default=0.25, help="Keep the last stable pose this long after a miss.")
    parser.add_argument("--max-stable-reprojection-error", type=float, default=12.0, help="Reject noisy pose updates above this mean reprojection error.")
    parser.add_argument("--reject-outlier-tags", action="store_true", default=True, help="Drop visible tags whose reprojection error is inconsistent with the rigid model. Default: on.")
    parser.add_argument("--no-reject-outlier-tags", dest="reject_outlier_tags", action="store_false", help="Use all visible configured tags even when one has high reprojection error.")
    parser.add_argument("--tag-outlier-ratio", type=float, default=2.8, help="Reject a tag when its mean reprojection error is this many times above the median, capped by --max-tag-mean-reprojection-error.")
    parser.add_argument("--min-tag-mean-reprojection-error", type=float, default=4.0, help="Lower pixel threshold before tag outlier rejection can trigger.")
    parser.add_argument("--max-tag-mean-reprojection-error", type=float, default=12.0, help="Upper mean pixel error for keeping a tag during outlier rejection.")
    parser.add_argument("--roll-switch-penalty", type=float, default=0.75, help="Penalty added when dynamic ID roll selection changes from the previous frame.")
    parser.add_argument("--print-pose", action="store_true", help="Print pose every frame when available.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.vertical_gap is None:
        args.vertical_gap = args.tag_row_gap
    else:
        args.tag_row_gap = args.vertical_gap
    if args.cube_size is not None:
        args.prism_width = args.cube_size
        args.prism_depth = args.cube_size
        args.prism_height = 2.0 * args.cube_size + args.vertical_gap
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
