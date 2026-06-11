#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path


DEEPPARUCO_PYTHON = "/home/zsyy/anaconda3/envs/deeparuco39/bin/python"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "utils" / "camera_calibration.yml"
UPPER_CUBE_COLOR = (0, 255, 255)
LOWER_CUBE_COLOR = (255, 0, 0)
MODEL_TAG_COLOR = (0, 220, 0)
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
    # Object frame:
    # +X -> outward normal of ID0 face.
    # +Y -> outward normal of ID1 face.
    # +Z -> outward normal of ID8 face.
    # Origin stays at the center of the upper cube.
    # Each face stores: outward normal, marker horizontal axis, marker vertical-down axis.
    "front": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "right": (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "back": (np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "left": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "top": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
}
SIDE_FACE_NAMES = {"front", "right", "back", "left"}


@dataclass
class FramePacket:
    sequence: int
    frame: np.ndarray
    timestamp: float
    capture_fps: float


@dataclass
class PoseResult:
    sequence: int = 0
    timestamp: float = 0.0
    process_fps: float = 0.0
    capture_fps: float = 0.0
    marker_ids: list[int] = field(default_factory=list)
    marker_corners: list[np.ndarray] = field(default_factory=list)
    rejected_count: int = 0
    used_ids: list[int] = field(default_factory=list)
    dynamic_rolls: dict[int, int] = field(default_factory=dict)
    pose_status: str = "lost"
    inlier_count: int = 0
    mean_error: float | None = None
    max_error: float | None = None
    raw_rvec: np.ndarray | None = None
    raw_tvec: np.ndarray | None = None
    display_rvec: np.ndarray | None = None
    display_tvec: np.ndarray | None = None


class LatestSlot:
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

    def reset(self):
        self.rvec = None
        self.tvec = None

    def update(self, rvec, tvec, timestamp=None, confidence=1.0):
        del timestamp, confidence
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if self.rvec is None:
            self.rvec = rvec.copy()
            self.tvec = tvec.copy()
            return self.rvec.copy(), self.tvec.copy()
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


class PoseOneEuroFilter:
    def __init__(self, min_cutoff, beta, derivate_cutoff):
        self.r_filter = OneEuroVectorFilter(min_cutoff, beta, derivate_cutoff)
        self.t_filter = OneEuroVectorFilter(min_cutoff, beta, derivate_cutoff)

    def reset(self):
        self.r_filter.reset()
        self.t_filter.reset()

    def update(self, rvec, tvec, timestamp=None, confidence=1.0):
        cutoff_scale = 1.0 + (1.0 - float(np.clip(confidence, 0.0, 1.0)))
        rvec = self.r_filter.filter(rvec.reshape(-1), timestamp=timestamp, cutoff_scale=cutoff_scale).reshape(3, 1)
        tvec = self.t_filter.filter(tvec.reshape(-1), timestamp=timestamp, cutoff_scale=cutoff_scale).reshape(3, 1)
        return rvec, tvec


class PoseKalmanFilter:
    def __init__(self, process_noise=1e-3, measurement_noise=5e-3):
        self.filter = cv2.KalmanFilter(12, 6)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initialized = False
        self.last_time = None
        self._set_static_matrices()

    def _set_static_matrices(self):
        self.filter.measurementMatrix = np.zeros((6, 12), dtype=np.float32)
        self.filter.measurementMatrix[:6, :6] = np.eye(6, dtype=np.float32)
        self.filter.processNoiseCov = np.eye(12, dtype=np.float32) * self.process_noise
        self.filter.measurementNoiseCov = np.eye(6, dtype=np.float32) * self.measurement_noise
        self.filter.errorCovPost = np.eye(12, dtype=np.float32)

    def reset(self):
        self.initialized = False
        self.last_time = None
        self._set_static_matrices()

    def _update_transition(self, dt):
        transition = np.eye(12, dtype=np.float32)
        transition[:6, 6:] = np.eye(6, dtype=np.float32) * float(dt)
        self.filter.transitionMatrix = transition

    def update(self, rvec, tvec, timestamp=None, confidence=1.0):
        now = time.monotonic() if timestamp is None else float(timestamp)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        measurement = np.vstack((rvec, tvec)).astype(np.float32)
        if not self.initialized:
            self.filter.statePost = np.zeros((12, 1), dtype=np.float32)
            self.filter.statePost[:6] = measurement
            self.filter.statePre = self.filter.statePost.copy()
            self.filter.errorCovPost = np.eye(12, dtype=np.float32) * 0.1
            self.initialized = True
            self.last_time = now
            return rvec.copy(), tvec.copy()

        dt = max(now - self.last_time, 1e-4)
        self.last_time = now
        self._update_transition(dt)
        noise_scale = 1.0 / max(float(np.clip(confidence, 0.05, 1.0)), 0.05)
        self.filter.measurementNoiseCov = np.eye(6, dtype=np.float32) * (self.measurement_noise * noise_scale)
        self.filter.predict()
        corrected = self.filter.correct(measurement)
        corrected = np.asarray(corrected, dtype=np.float64).reshape(12, 1)
        return corrected[:3].copy(), corrected[3:6].copy()


class PoseTracker:
    def __init__(self, args):
        if args.pose_filter == "ema":
            self.pose_filter = PoseEmaFilter(args.ema_alpha)
        elif args.pose_filter == "one_euro":
            self.pose_filter = PoseOneEuroFilter(
                args.one_euro_min_cutoff,
                args.one_euro_beta,
                args.one_euro_derivate_cutoff,
            )
        elif args.pose_filter == "kalman":
            self.pose_filter = PoseKalmanFilter(args.kalman_process_noise, args.kalman_measurement_noise)
        else:
            self.pose_filter = None
        self.max_hold_time = float(args.hold_last_seconds)
        self.max_reprojection_error = float(args.max_stable_reprojection_error)
        self.min_follow = float(args.min_pose_follow)
        self.last_raw_rvec = None
        self.last_raw_tvec = None
        self.last_output_rvec = None
        self.last_output_tvec = None
        self.last_update_time = None
        self.last_rolls = {}

    def get_initial_guess(self):
        if self.last_output_rvec is None or self.last_output_tvec is None:
            return None
        return self.last_output_rvec.copy(), self.last_output_tvec.copy()

    def compute_confidence(self, num_tags, mean_error):
        tag_score = np.clip((float(num_tags) - 1.0) / 3.0, 0.0, 1.0)
        error_score = np.clip(1.0 - float(mean_error) / max(self.max_reprojection_error, 1e-6), 0.0, 1.0)
        return float(0.6 * tag_score + 0.4 * error_score)

    def update(self, rvec, tvec, mean_error, dynamic_rolls, num_tags, timestamp):
        if mean_error > self.max_reprojection_error:
            held = self.get_held_pose(timestamp)
            return held, "held_bad_reproj" if held is not None else "rejected_bad_reproj"

        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if self.last_raw_rvec is not None and float(np.dot(self.last_raw_rvec.reshape(3), rvec.reshape(3))) < 0.0:
            rvec = -rvec

        self.last_raw_rvec = rvec.copy()
        self.last_raw_tvec = tvec.copy()
        confidence = self.compute_confidence(num_tags, mean_error)
        if self.pose_filter is None:
            filtered_rvec, filtered_tvec = rvec.copy(), tvec.copy()
        else:
            filtered_rvec, filtered_tvec = self.pose_filter.update(rvec, tvec, timestamp=timestamp, confidence=confidence)

        if self.last_output_rvec is not None and self.last_output_tvec is not None:
            follow = self.min_follow + (1.0 - self.min_follow) * confidence
            filtered_rvec = (1.0 - follow) * self.last_output_rvec + follow * filtered_rvec
            filtered_tvec = (1.0 - follow) * self.last_output_tvec + follow * filtered_tvec

        self.last_output_rvec = filtered_rvec.copy()
        self.last_output_tvec = filtered_tvec.copy()
        self.last_update_time = float(timestamp)
        self.last_rolls = dict(dynamic_rolls)
        return (self.last_output_rvec.copy(), self.last_output_tvec.copy()), "measured"

    def get_held_pose(self, timestamp=None):
        if self.last_output_rvec is None or self.last_output_tvec is None or self.last_update_time is None:
            return None
        now = time.monotonic() if timestamp is None else float(timestamp)
        if now - self.last_update_time > self.max_hold_time:
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
            raise ValueError("--corner-rolls entries must look like id:roll")
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
            raise ValueError("--id-face-map entries must look like id:face")
        marker_id_text, face = part.split(":", 1)
        face = face.strip().lower()
        if face not in FACE_SPECS:
            raise ValueError("--id-face-map face must be one of: front,right,back,left,top")
        mapping[int(marker_id_text.strip())] = face
    return mapping


def roll_corners(corners, roll):
    roll = int(roll) % 4
    if roll == 0:
        return np.asarray(corners, dtype=np.float32).copy()
    return np.roll(np.asarray(corners, dtype=np.float32), -roll, axis=0).copy()


def marker_object_corners(face, center_z, cube_size, marker_length):
    normal, u_axis, v_axis = FACE_SPECS[face]
    center = normal * (cube_size * 0.5) + np.array([0.0, 0.0, center_z])
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
    return rotate_points_z(marker_object_corners(face, center_z, cube_size, marker_length), angle_deg)


def get_cube_center_z(args):
    upper_z = 0.0
    lower_z = -(args.cube_size + args.vertical_gap)
    return upper_z, lower_z


def build_marker_layout(args):
    face_order = parse_face_order(args.face_order)
    args.lower_ids = parse_id_list(args.lower_ids)
    args.corner_rolls = parse_corner_rolls(args.corner_rolls)
    args.auto_roll_ids = parse_id_list(args.auto_roll_ids)
    args.id_face_map = parse_id_face_map(args.id_face_map)
    upper_z, lower_z = get_cube_center_z(args)
    layout = {}
    for index, face in enumerate(face_order):
        marker_face = args.id_face_map.get(index, face)
        layout[index] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                upper_z,
                args.cube_size,
                args.marker_length,
                args.upper_rotation_deg,
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
    for corners, marker_id_value in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(marker_id_value)
        if marker_id not in layout:
            continue
        object_points.append(layout[marker_id])
        image_points.append(corners.reshape(4, 2).astype(np.float32))
        used_ids.append(marker_id)
    if not object_points:
        return None, None, []
    return np.vstack(object_points).astype(np.float32), np.vstack(image_points).astype(np.float32), used_ids


def solve_pnp_candidate(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    initial_guess=None,
    use_ransac=False,
    refine_pnp=False,
    reprojection_error=5.0,
    ransac_iterations=100,
):
    if use_ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            reprojectionError=float(reprojection_error),
            iterationsCount=int(ransac_iterations),
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
                np.asarray(guess_rvec, dtype=np.float64).reshape(3, 1).copy(),
                np.asarray(guess_tvec, dtype=np.float64).reshape(3, 1).copy(),
                True,
                cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                None,
                None,
                False,
                cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            return None
        inlier_count = len(object_points)

    if refine_pnp and hasattr(cv2, "solvePnPRefineLM") and len(object_points) >= 8:
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
    return rvec, tvec, inlier_count, float(np.mean(errors)), float(np.max(errors))


def estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, args, initial_guess=None):
    if object_points is None or len(object_points) < 4 or len(set(used_ids)) < args.min_tags_for_pose:
        return None
    candidates = []
    if initial_guess is not None:
        candidate = solve_pnp_candidate(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            initial_guess=initial_guess,
            use_ransac=False,
            refine_pnp=args.refine_pnp,
        )
        if candidate is not None:
            candidates.append(candidate)
    else:
        candidate = solve_pnp_candidate(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            initial_guess=None,
            use_ransac=False,
            refine_pnp=args.refine_pnp,
        )
        if candidate is not None:
            candidates.append(candidate)

    if args.ransac and len(object_points) >= 8:
        candidate = solve_pnp_candidate(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            initial_guess=None,
            use_ransac=True,
            refine_pnp=args.refine_pnp,
            reprojection_error=args.reprojection_error,
            ransac_iterations=args.ransac_iterations,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[3])


def estimate_best_fused_pose(
    marker_corners,
    marker_ids,
    layout,
    camera_matrix,
    dist_coeffs,
    args,
    initial_guess=None,
    previous_rolls=None,
):
    previous_rolls = {} if previous_rolls is None else previous_rolls
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
        pose = estimate_fused_pose(
            object_points,
            image_points,
            used_ids,
            camera_matrix,
            dist_coeffs,
            args,
            initial_guess=initial_guess,
        )
        if pose is None:
            continue
        switch_count = sum(
            1 for marker_id, roll in dynamic_rolls.items() if previous_rolls.get(marker_id, 0) != roll
        )
        score = pose[3] + args.roll_switch_penalty * float(switch_count)
        if best is None or score < best[5]:
            best = (object_points, image_points, used_ids, pose, dynamic_rolls, score)

    if best is None:
        object_points, image_points, used_ids = collect_correspondences(marker_corners, marker_ids, layout)
        return object_points, image_points, used_ids, None, {}
    return best[:5]


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


def point_tuple(point, max_abs=1000000):
    point = np.asarray(point, dtype=np.float64).reshape(-1)
    if len(point) < 2:
        return None
    if not np.isfinite(point[:2]).all():
        return None
    point = np.round(point).astype(np.int64)
    if abs(point[0]) > max_abs or abs(point[1]) > max_abs:
        return None
    return int(point[0]), int(point[1])


def draw_projected_polyline(frame, points, color, thickness=2, closed=True):
    pts = np.round(np.asarray(points).reshape(-1, 2)).astype(int)
    for idx in range(len(pts) - 1):
        pt0 = point_tuple(pts[idx])
        pt1 = point_tuple(pts[idx + 1])
        if pt0 is not None and pt1 is not None:
            cv2.line(frame, pt0, pt1, color, thickness, cv2.LINE_AA)
    if closed and len(pts) > 2:
        pt0 = point_tuple(pts[-1])
        pt1 = point_tuple(pts[0])
        if pt0 is not None and pt1 is not None:
            cv2.line(frame, pt0, pt1, color, thickness, cv2.LINE_AA)


def draw_rigid_tag_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs):
    for marker_id, object_corners in sorted(layout.items()):
        projected, _ = cv2.projectPoints(object_corners, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        draw_projected_polyline(frame, projected, MODEL_TAG_COLOR, thickness=2, closed=True)
        center = point_tuple(np.mean(projected, axis=0))
        if center is not None:
            cv2.putText(
                frame,
                str(marker_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                MODEL_TAG_COLOR,
                2,
                cv2.LINE_AA,
            )


def draw_fused_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs, args):
    if args.draw_cube_wireframe:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        upper_z, lower_z = get_cube_center_z(args)
        for vertices, color in (
            (cube_vertices(upper_z, args.cube_size, args.upper_rotation_deg), UPPER_CUBE_COLOR),
            (cube_vertices(lower_z, args.cube_size, args.lower_rotation_deg), LOWER_CUBE_COLOR),
        ):
            projected, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
            projected = projected.reshape(-1, 2)
            for i0, i1 in edges:
                pt0 = point_tuple(projected[i0])
                pt1 = point_tuple(projected[i1])
                if pt0 is not None and pt1 is not None:
                    cv2.line(frame, pt0, pt1, color, args.cube_line_thickness, cv2.LINE_AA)

    if args.draw_model_tags:
        draw_rigid_tag_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs)

    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, args.axis_length)


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
            cv2.circle(frame, center, args.corner_radius, DETECTED_CORNER_COLORS[corner_index % len(DETECTED_CORNER_COLORS)], -1, cv2.LINE_AA)
            if args.draw_corner_index:
                cv2.putText(
                    frame,
                    "{}:{}".format(marker_id, corner_index),
                    (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )


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


def process_packet(packet, dictionary, params, detector, layout, camera_matrix, dist_coeffs, args, pose_tracker, process_fps):
    gray = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, rejected = detect_markers(gray, dictionary, params, detector)
    ids = [] if marker_ids is None else marker_ids.reshape(-1).astype(int).tolist()

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

    result = PoseResult(
        sequence=packet.sequence,
        timestamp=packet.timestamp,
        process_fps=process_fps,
        capture_fps=packet.capture_fps,
        marker_ids=ids,
        marker_corners=[] if marker_corners is None else [corners.reshape(4, 2).astype(np.float32) for corners in marker_corners],
        rejected_count=0 if rejected is None else len(rejected),
        used_ids=used_ids,
        dynamic_rolls=dynamic_rolls,
    )

    if pose is not None:
        rvec, tvec, inlier_count, mean_error, max_error = pose
        result.raw_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        result.raw_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        result.inlier_count = int(inlier_count)
        result.mean_error = float(mean_error)
        result.max_error = float(max_error)
        if pose_tracker is not None:
            tracked_pose, pose_status = pose_tracker.update(
                result.raw_rvec,
                result.raw_tvec,
                result.mean_error,
                dynamic_rolls,
                num_tags=len(set(used_ids)),
                timestamp=packet.timestamp,
            )
            result.pose_status = pose_status
            if tracked_pose is not None:
                result.display_rvec, result.display_tvec = tracked_pose
        else:
            result.pose_status = "measured"
            result.display_rvec = result.raw_rvec.copy()
            result.display_tvec = result.raw_tvec.copy()
    elif pose_tracker is not None:
        held_pose = pose_tracker.get_held_pose(packet.timestamp)
        if held_pose is not None:
            result.display_rvec, result.display_tvec = held_pose
            result.dynamic_rolls = dict(pose_tracker.last_rolls)
            result.pose_status = "held_no_detection"

    if result.display_rvec is None or result.display_tvec is None:
        result.pose_status = "lost" if pose is None else result.pose_status
    return result


def render_frame(frame, result, layout, camera_matrix, dist_coeffs, args, render_fps=0.0, display_sequence=None):
    result = PoseResult() if result is None else result
    lag_frames = None if display_sequence is None else max(0, int(display_sequence) - int(result.sequence))
    lag_ms = max(0.0, (time.monotonic() - float(result.timestamp)) * 1000.0) if result.timestamp > 0.0 else 0.0
    overlay_is_fresh = lag_frames is None or args.max_overlay_lag_frames < 0 or lag_frames <= args.max_overlay_lag_frames

    marker_ids = None if not result.marker_ids else np.asarray(result.marker_ids, dtype=np.int32).reshape(-1, 1)
    marker_corners = None if not result.marker_corners else [corners.reshape(1, 4, 2) for corners in result.marker_corners]

    if overlay_is_fresh and marker_ids is not None and args.draw_detected_markers:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    if overlay_is_fresh and marker_ids is not None and args.draw_detected_corners:
        draw_detected_corners(frame, result.marker_corners, marker_ids, args)

    pose_text = "pose: need at least {} configured tags for fused rigid-object solvePnP".format(args.min_tags_for_pose)
    active_rolls = result.dynamic_rolls if result.dynamic_rolls else {}
    if overlay_is_fresh and result.display_rvec is not None and result.display_tvec is not None:
        draw_layout = make_candidate_layout(layout, active_rolls)
        draw_fused_model(frame, draw_layout, result.display_rvec, result.display_tvec, camera_matrix, dist_coeffs, args)
        tx, ty, tz = result.display_tvec.reshape(-1)
        roll, pitch, yaw = rotation_to_euler_xyz(result.display_rvec)
        pose_text = (
            "fused object pose t=({:.3f},{:.3f},{:.3f})m rpy=({:.1f},{:.1f},{:.1f})deg state={} err={} tags={}".format(
                tx,
                ty,
                tz,
                roll,
                pitch,
                yaw,
                result.pose_status,
                "n/a" if result.mean_error is None else "{:.2f}px".format(result.mean_error),
                len(set(result.used_ids)),
            )
        )
    elif result.display_rvec is not None and result.display_tvec is not None:
        pose_text = "fused object pose available but overlay stale lag={}f/{:.0f}ms".format(lag_frames, lag_ms)

    lines = [
        "ArUco 9-tag rigid cube fused pose  dict={} capture={:.1f}fps process={:.1f}fps render={:.1f}fps".format(
            args.dictionary,
            result.capture_fps,
            result.process_fps,
            render_fps,
        ),
        "fused pose ids: {}  used ids: {}  rejected: {}".format(result.marker_ids[:12], result.used_ids, result.rejected_count),
        pose_text,
        "rolls: {}  lag={}f/{:.0f}ms".format(active_rolls if active_rolls else "off/unchanged", lag_frames, lag_ms),
        "render: fused object axis + 4cm cube wireframe bound to the rigid 9-tag object",
        "axes: +X normal=ID0  +Y normal=ID1  +Z normal=ID8  origin=upper-cube center",
        "final pose = all visible tag corners -> one rigid-object solvePnP",
        "layout: upper[0:+X,1:+Y,2:-X,3:-Y,8:+Z] lower[4:+X,5:+Y,6:-X,7:-Y]".format(
        ),
        "cube_size={}m marker_length={}m top_id={} face_order={}".format(
            args.cube_size, args.marker_length, args.top_id, args.face_order
        ),
    ]
    draw_text_panel(frame, lines)


def print_pose_log(result):
    if result.display_rvec is None or result.display_tvec is None:
        return
    tx, ty, tz = result.display_tvec.reshape(-1)
    roll, pitch, yaw = rotation_to_euler_xyz(result.display_rvec)
    object_to_camera = pose_matrix_object_to_camera(result.display_rvec, result.display_tvec)
    print(
        "fused_object_pose ids={} t=({:.6f},{:.6f},{:.6f}) rvec=({:.6f},{:.6f},{:.6f}) "
        "rpy=({:.3f},{:.3f},{:.3f}) rolls={} object_to_camera={} mean_err={} max_err={} state={}".format(
            result.used_ids,
            tx,
            ty,
            tz,
            *result.display_rvec.reshape(-1),
            roll,
            pitch,
            yaw,
            result.dynamic_rolls if result.dynamic_rolls else {},
            format_matrix(object_to_camera),
            "n/a" if result.mean_error is None else "{:.6f}".format(result.mean_error),
            "n/a" if result.max_error is None else "{:.6f}".format(result.max_error),
            result.pose_status,
        ),
        flush=True,
    )


def open_capture(source):
    source_text = str(source).strip()
    if source_text.isdigit():
        return cv2.VideoCapture(int(source_text))
    return cv2.VideoCapture(source)


def run_image(args, dictionary, params, detector, layout, camera_matrix, dist_coeffs):
    frame = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Cannot read image: {}".format(args.image))
    if camera_matrix is None:
        camera_matrix, dist_coeffs = approximate_calibration(frame.shape[1], frame.shape[0])

    pose_tracker = PoseTracker(args)
    packet = FramePacket(sequence=1, frame=frame.copy(), timestamp=time.monotonic(), capture_fps=0.0)
    result = process_packet(packet, dictionary, params, detector, layout, camera_matrix, dist_coeffs, args, pose_tracker, process_fps=0.0)
    render_frame(frame, result, layout, camera_matrix, dist_coeffs, args, render_fps=0.0, display_sequence=packet.sequence)
    if args.print_pose:
        print_pose_log(result)

    print("image: {}".format(args.image), flush=True)
    print(
        "detected ids: {} used ids: {} pose: {}".format(
            result.marker_ids,
            result.used_ids,
            "ok" if result.display_rvec is not None else "failed",
        ),
        flush=True,
    )
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print("output: {}".format(output_path), flush=True)
    if args.show:
        cv2.imshow("ArUco 9-tag fused pose", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args, dictionary, params, detector, layout, camera_matrix, dist_coeffs):
    cap = open_capture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, args.autofocus)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source {}".format(args.camera))

    print("OpenCV: {}".format(cv2.__version__), flush=True)
    print(
        "Camera: {} requested={}x{} autofocus={} capture-thread=on process-thread=on".format(
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
    print("Object frame: +X normal=ID0, +Y normal=ID1, +Z normal=ID8, origin=upper cube center.", flush=True)
    print("Locked layout: upper[0:+X,1:+Y,2:-X,3:-Y,8:+Z], lower[4:+X,5:+Y,6:-X,7:-Y], face rolls=0.", flush=True)
    print("Final axis pose: one rigid-object pose solved jointly from all currently visible tag corners.", flush=True)
    if camera_matrix is None:
        print("Calibration: approximate from frame size. Use --calibration for metric pose.", flush=True)
    print("Press ESC or q to quit.", flush=True)

    stop_event = threading.Event()
    raw_frames = LatestSlot()
    pose_results = LatestSlot()
    capture_rate = RateCounter()
    process_rate = RateCounter()
    render_rate = RateCounter()
    pose_tracker = PoseTracker(args)
    window_name = "ArUco 9-tag Cube Dual Thread Fused Pose"

    def capture_loop():
        sequence = 0
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.002)
                    continue
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                sequence += 1
                packet = FramePacket(
                    sequence=sequence,
                    frame=frame,
                    timestamp=time.monotonic(),
                    capture_fps=capture_rate.tick(),
                )
                raw_frames.put(packet)
        except Exception:
            traceback.print_exc()
            stop_event.set()
        finally:
            raw_frames.close()

    def processing_loop():
        last_sequence = 0
        try:
            while not stop_event.is_set():
                packet, sequence = raw_frames.get_latest(last_sequence, timeout=0.05)
                if packet is None or sequence == last_sequence:
                    continue
                last_sequence = sequence
                frame = packet.frame
                if camera_matrix is None:
                    current_camera_matrix, current_dist_coeffs = approximate_calibration(frame.shape[1], frame.shape[0])
                else:
                    current_camera_matrix, current_dist_coeffs = camera_matrix, dist_coeffs
                result = process_packet(
                    packet,
                    dictionary,
                    params,
                    detector,
                    layout,
                    current_camera_matrix,
                    current_dist_coeffs,
                    args,
                    pose_tracker,
                    process_fps=process_rate.tick(),
                )
                if args.print_pose and result.display_rvec is not None and result.pose_status in ("measured", "held_no_detection", "held_bad_reproj"):
                    print_pose_log(result)
                pose_results.put(result)
        except Exception:
            traceback.print_exc()
            stop_event.set()
        finally:
            pose_results.close()

    capture_thread = threading.Thread(target=capture_loop, name="camera-capture", daemon=True)
    process_thread = threading.Thread(target=processing_loop, name="aruco-pose-processing", daemon=True)
    capture_thread.start()
    process_thread.start()

    last_frame = None
    last_frame_sequence = 0
    last_pose_result = None
    last_pose_sequence = 0
    try:
        while not stop_event.is_set():
            frame_changed = False
            pose_changed = False
            packet, frame_sequence = raw_frames.get_latest(last_frame_sequence, timeout=0.001)
            if packet is not None and frame_sequence != last_frame_sequence:
                last_frame = packet
                last_frame_sequence = frame_sequence
                frame_changed = True

            result, pose_sequence = pose_results.get_latest(last_pose_sequence, timeout=0.001)
            if result is not None and pose_sequence != last_pose_sequence:
                last_pose_result = result
                last_pose_sequence = pose_sequence
                pose_changed = True

            if last_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    stop_event.set()
                continue

            if camera_matrix is None:
                current_camera_matrix, current_dist_coeffs = approximate_calibration(
                    last_frame.frame.shape[1], last_frame.frame.shape[0]
                )
            else:
                current_camera_matrix, current_dist_coeffs = camera_matrix, dist_coeffs

            display_frame = last_frame.frame.copy()
            render_frame(
                display_frame,
                last_pose_result,
                layout,
                current_camera_matrix,
                current_dist_coeffs,
                args,
                render_fps=render_rate.tick(),
                display_sequence=last_frame.sequence,
            )
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                stop_event.set()
                break
    finally:
        stop_event.set()
        raw_frames.close()
        pose_results.close()
        capture_thread.join(timeout=1.0)
        process_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a rigid 9-tag ArUco cube stack, fuse all visible tag corners into one rigid-object solvePnP pose, and render only that final object axis asynchronously."
    )
    parser.add_argument("--dictionary", default="DICT_6X6_250", choices=sorted(DICTIONARIES))
    parser.add_argument("--camera", default="0", help="Camera index or video path. Default: 0.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width. Default: 1280.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height. Default: 720.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, open camera.")
    parser.add_argument("--output", default=None, help="Optional annotated output path for --image.")
    parser.add_argument("--show", action="store_true", help="Show annotated image when using --image.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file.")
    parser.add_argument("--cube-size", type=float, default=0.04, help="Single cube side length in meters. Default: 0.04.")
    parser.add_argument("--vertical-gap", type=float, default=0.0, help="Gap between lower and upper cube in meters.")
    parser.add_argument("--marker-length", type=float, default=0.032, help="ArUco marker side length in meters. Default: 0.032.")
    parser.add_argument(
        "--face-order",
        default="front,right,back,left",
        help="Face order for upper ids 0-3 and lower ids 4-7. Default: front,right,back,left.",
    )
    parser.add_argument("--lower-ids", default="4,5,6,7", help="Lower cube ArUco IDs actually present. Default: 4,5,6,7.")
    parser.add_argument("--top-id", type=int, default=8, help="Upper cube top-face ArUco ID. Use -1 to disable. Default: 8.")
    parser.add_argument("--upper-rotation-deg", type=float, default=45.0, help="Upper cube rotation around model Z axis. Default: 45.")
    parser.add_argument("--lower-rotation-deg", type=float, default=0.0, help="Lower cube rotation around model Z axis. Default: 0.")
    parser.add_argument(
        "--corner-rolls",
        default="",
        help="Static per-ID 3D corner order roll, e.g. 0:1,3:-1,8:2.",
    )
    parser.add_argument(
        "--auto-roll-ids",
        default="none",
        help="IDs whose corner order is searched each frame. Default: none because the 9-tag layout is fixed from the real object.",
    )
    parser.add_argument(
        "--auto-roll-max-candidates",
        type=int,
        default=64,
        help="Maximum dynamic corner-roll candidates tested per frame. Default: 64.",
    )
    parser.add_argument(
        "--roll-switch-penalty",
        type=float,
        default=0.15,
        help="Penalty added when dynamic corner rolls switch between frames. Default: 0.15 px.",
    )
    parser.add_argument(
        "--id-face-map",
        default="0:right,1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:top",
        help="Override ID to cube face assignment. Default aligns +X with ID0, +Y with ID1, +Z with ID8.",
    )
    parser.add_argument("--axis-length", type=float, default=0.04, help="Drawn object axis length in meters.")
    parser.add_argument("--draw-cube-wireframe", action="store_true", default=True, help="Draw the fused upper/lower 4cm cube wireframes. Default: on.")
    parser.add_argument("--no-draw-cube-wireframe", dest="draw_cube_wireframe", action="store_false", help="Hide cube wireframes.")
    parser.add_argument("--cube-line-thickness", type=int, default=3, help="Projected cube wireframe line thickness.")
    parser.add_argument("--draw-model-tags", action="store_true", default=False, help="Project configured rigid tag outlines. Default: off.")
    parser.add_argument("--no-draw-model-tags", dest="draw_model_tags", action="store_false", help="Hide projected rigid tag outlines.")
    parser.add_argument("--draw-detected-markers", action="store_true", default=True, help="Draw detected ArUco marker borders. Default: on.")
    parser.add_argument("--no-draw-detected-markers", dest="draw_detected_markers", action="store_false", help="Hide detected ArUco marker borders.")
    parser.add_argument("--draw-detected-corners", action="store_true", default=True, help="Draw detected ArUco corner dots. Default: on.")
    parser.add_argument("--no-draw-detected-corners", dest="draw_detected_corners", action="store_false", help="Hide detected corner dots.")
    parser.add_argument("--draw-corner-index", action="store_true", default=True, help="Draw marker-id:corner-index labels. Default: on.")
    parser.add_argument("--no-draw-corner-index", dest="draw_corner_index", action="store_false", help="Hide marker-id:corner-index labels.")
    parser.add_argument("--corner-radius", type=int, default=4, help="Detected corner dot radius in pixels.")
    parser.add_argument("--max-overlay-lag-frames", type=int, default=-1, help="Stop drawing stale geometry after this many frames. Negative keeps the last fused axis visible.")
    parser.add_argument("--aruco3", action="store_true", help="Enable OpenCV ArUco3 detector path when supported.")
    parser.add_argument("--adaptive-min", type=int, default=3)
    parser.add_argument("--adaptive-max", type=int, default=53)
    parser.add_argument("--adaptive-step", type=int, default=10)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.015)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    parser.add_argument("--ransac", action="store_true", default=False, help="Also test solvePnPRansac when enough corners exist. Default: off for smoother live view.")
    parser.add_argument("--no-ransac", dest="ransac", action="store_false", help="Use iterative solvePnP only.")
    parser.add_argument("--refine-pnp", action="store_true", default=False, help="Run solvePnPRefineLM after pose solve. Default: off for smoother live view.")
    parser.add_argument("--ransac-iterations", type=int, default=100)
    parser.add_argument("--reprojection-error", type=float, default=5.0)
    parser.add_argument("--min-tags-for-pose", type=int, default=1, help="Minimum configured visible tags needed before running the fused rigid-object pose solve.")
    parser.add_argument(
        "--pose-filter",
        choices=("none", "ema", "one_euro", "kalman"),
        default="none",
        help="Smooth final fused rvec/tvec. Default: none.",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="EMA pose smoothing alpha.")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.2, help="OneEuro min cutoff.")
    parser.add_argument("--one-euro-beta", type=float, default=0.04, help="OneEuro beta.")
    parser.add_argument("--one-euro-derivate-cutoff", type=float, default=1.0, help="OneEuro derivative cutoff.")
    parser.add_argument("--kalman-process-noise", type=float, default=1e-3, help="Kalman process noise.")
    parser.add_argument("--kalman-measurement-noise", type=float, default=5e-3, help="Kalman measurement noise.")
    parser.add_argument("--hold-last-seconds", type=float, default=0.35, help="Hold last pose this long when detections drop. Default: 0.35.")
    parser.add_argument("--max-stable-reprojection-error", type=float, default=8.0, help="Reject raw pose if mean reprojection error exceeds this.")
    parser.add_argument("--min-pose-follow", type=float, default=0.18, help="Extra output smoothing factor floor. Lower is smoother. Default: 0.18.")
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
