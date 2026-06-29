#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path


DEEPPARUCO_PYTHON = "/home/zsyy/anaconda3/envs/deeparuco39/bin/python"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "utils" / "camera_calibration.yml"
DEFAULT_DEEPPARUCO_REPO = "/home/zsyy/下载/deeparuco-main"
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


@dataclass
class PoseRenderState:
    marker_corners: object = field(default_factory=list)
    marker_ids: object = None
    rejected_count: int = 0
    pose: object = None
    ids: list = field(default_factory=list)
    used_ids: list = field(default_factory=list)
    render_anchor_ids: list = field(default_factory=list)
    dynamic_rolls: object = field(default_factory=dict)
    active_rolls: object = field(default_factory=dict)
    pose_text: str = "pose: waiting"
    pose_state_text: str = "lost"
    rolls_text: object = "off/unchanged"
    capture_fps: float = 0.0
    process_fps: float = 0.0
    tag_detect_ms: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    draw_pose: object = None
    render_alignment_offset: object = None


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


def vector3_list(vector):
    return [float(value) for value in np.asarray(vector, dtype=np.float64).reshape(3)]


def build_unity_pose_payload(valid, rvec, tvec, used_ids, mean_error, pose_state_text, timestamp=None):
    valid = bool(valid)
    payload = {
        "valid": valid,
        "timestamp": float(time.monotonic() if timestamp is None else timestamp),
        "rvec": vector3_list(rvec) if valid and rvec is not None else None,
        "tvec": vector3_list(tvec) if valid and tvec is not None else None,
        "used_ids": [int(marker_id) for marker_id in used_ids],
        "mean_error": None if mean_error is None else float(mean_error),
        "pose_state": str(pose_state_text),
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


class UnityPoseSender:
    def __init__(self, host, port):
        self.address = (str(host), int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self):
        self.socket.close()

    def send(self, valid, rvec=None, tvec=None, used_ids=None, mean_error=None, pose_state_text="lost"):
        payload = build_unity_pose_payload(
            valid,
            rvec,
            tvec,
            [] if used_ids is None else used_ids,
            mean_error,
            pose_state_text,
        )
        self.socket.sendto(payload.encode("utf-8"), self.address)


def create_unity_pose_sender(args):
    if not getattr(args, "send_unity_pose", False):
        return None
    return UnityPoseSender(args.unity_udp_host, args.unity_udp_port)


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
    (1, 2), (2, 3), (3, 4), (4, 1),
    (5, 6), (6, 7), (7, 8), (8, 5),
    (1, 5), (2, 6), (3, 7), (4, 8),
    (0, 1), (0, 2), (0, 3), (0, 4),
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


class FrameProcessScheduler:
    def __init__(self, process_every_n_frames=1):
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self.frame_index = 0

    def should_process(self):
        should_run = self.frame_index % self.process_every_n_frames == 0
        self.frame_index += 1
        return should_run


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
        self.last_candidate_ids = []
        self.last_candidate_score = None
        self.tag_quality_scores = {}
        self.last_render_anchor_ids = []
        self.last_render_alignment_offset = None
        self.last_render_state = PoseRenderState()

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
        cutoff_scale = 1.0 / (1.0 + (1.0 - confidence) * self.low_confidence_extra_smoothing)
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


def norm_image(image):
    image = np.asarray(image)
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-9)


def ensure_deeparuco_repo(repo):
    if not repo.exists():
        raise RuntimeError(
            "DeepArUco repo not found: {}. Pass --deeparuco-repo or clone/install the repo first.".format(repo)
        )
    for rel in ("impl/aruco.py", "impl/heatmaps.py", "impl/losses.py", "impl/utils.py"):
        if not (repo / rel).exists():
            raise RuntimeError("DeepArUco repo is missing expected file: {}".format(repo / rel))


def get_aruco_dictionary_bits(name):
    dictionary = get_dictionary(name)
    bytes_list = dictionary.bytesList
    if bytes_list.ndim < 2:
        raise RuntimeError("Unsupported ArUco bytesList shape: {}".format(bytes_list.shape))
    marker_size = int(round(np.sqrt(bytes_list.shape[1] * 8)))
    templates = []
    for marker_id in range(bytes_list.shape[0]):
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size + 2)
        else:
            marker = cv2.aruco.drawMarker(dictionary, marker_id, marker_size + 2)
        canonical = (marker[1:-1, 1:-1] > 127).astype(np.float32)
        rotations = []
        for rotation_index in range(4):
            rotations.append(np.rot90(canonical, rotation_index).astype(np.float32))
        templates.append(rotations)
    return templates


def find_best_aruco_template_match(bits, templates):
    bits = np.asarray(bits, dtype=np.float32).reshape(templates[0][0].shape)
    best = None
    for marker_id, rotations in enumerate(templates):
        for rotation_index, template in enumerate(rotations):
            distance = int(np.sum(np.abs(bits - template)))
            if best is None or distance < best[1]:
                best = (int(marker_id), distance, int(rotation_index))
    return best


def create_aruco_bit_decoder(name):
    templates = get_aruco_dictionary_bits(name)

    def find_id(bits):
        return find_best_aruco_template_match(bits, templates)

    return find_id


def create_deeparuco_bit_decoder(aruco_module):
    templates = []
    for marker_bits in aruco_module.ids_as_bits:
        flat = np.asarray(marker_bits, dtype=np.float32).reshape(-1)
        marker_size = int(round(np.sqrt(flat.size)))
        if marker_size * marker_size != flat.size:
            raise RuntimeError("Unsupported DeepArUco bit count: {}".format(flat.size))
        canonical = flat.reshape(marker_size, marker_size)
        templates.append([np.rot90(canonical, rotation_index) for rotation_index in range(4)])

    def find_id(bits):
        bits = np.asarray(bits, dtype=np.float32).reshape(templates[0][0].shape)
        best = None
        for marker_id, rotations in enumerate(templates):
            for rotation_index, template in enumerate(rotations):
                distance = int(np.sum(np.abs(bits - template)))
                if best is None or distance < best[1]:
                    best = (int(marker_id), distance, int(rotation_index))
        return best

    return find_id


def load_deeparuco_backend(repo, detector_name, regressor_name, dictionary_name):
    repo = Path(repo).expanduser().resolve()
    ensure_deeparuco_repo(repo)
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    try:
        import tensorflow as tf
        from impl import aruco as deeparuco_aruco
        from impl.heatmaps import pos_from_heatmap
        from impl.losses import weighted_loss
        from impl.utils import marker_from_corners, ordered_corners
        from tensorflow.keras.models import load_model
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Cannot import DeepArUco dependencies. Run this with the DeepArUco Python environment.\n"
            "Import error: {}".format(exc)
        )
    find_id = create_deeparuco_bit_decoder(deeparuco_aruco)

    model_dir = repo / "models"
    detector_path = model_dir / "{}.pt".format(detector_name)
    regressor_path = model_dir / "{}.h5".format(regressor_name)
    decoder_path = model_dir / "dec_new.h5"
    for path in (detector_path, regressor_path, decoder_path):
        if not path.exists():
            raise RuntimeError("DeepArUco model file not found: {}".format(path))

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
        "repo": repo,
        "detector_name": detector_name,
        "regressor_name": regressor_name,
        "dictionary_name": dictionary_name,
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


def detect_deeparuco(frame, backend, regressor_name, threshold, detector_conf, detector_iou):
    detections = backend["detector"](
        frame,
        verbose=False,
        iou=detector_iou,
        conf=detector_conf,
    )[0].cpu().boxes
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
    crops = crops_ori.copy() if regressor_name == "reg_baseline" else [norm_image(crop) for crop in crops_ori]

    corner_predictions = backend["refine_corners"](np.array(crops)).numpy()
    if regressor_name.split("_")[1] == "hmap":
        corners = hmap_to_corners(corner_predictions, backend["pos_from_heatmap"])
        keep = [len(cs) == 4 for cs in corners]
        reorg = [(det, crop, cs) for det, crop, cs, k in zip(xyxy, crops_ori, corners, keep) if k]
        if not reorg:
            return []
        xyxy, crops_ori, corners = zip(*reorg)
    else:
        corners = [[(pred[i], pred[i + 1]) for i in range(0, 8, 2)] for pred in corner_predictions]

    corners = [backend["ordered_corners"]([c[0] for c in cs], [c[1] for c in cs]) for cs in corners]

    markers = []
    for crop, cs in zip(crops_ori, corners):
        marker = backend["marker_from_corners"](crop, cs, 32)
        markers.append(norm_image(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)))
    decoder_out = np.round(backend["decode_markers"](np.array(markers)).numpy())
    decoded = [backend["find_id"](out) for out in decoder_out]
    ids, dists, rotations = zip(*decoded)

    results = []
    for det, cs, marker_id, dist, rotation in zip(xyxy, corners, ids, dists, rotations):
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
                "rotation": int(rotation),
                "accepted": float(dist) < threshold,
                "bbox": det,
                "corners": image_corners,
            }
        )
    return results


def detect_deeparuco_scaled(
    frame,
    backend,
    regressor_name,
    threshold,
    detector_conf,
    detector_iou,
    scale=1.0,
    detect_fn=detect_deeparuco,
):
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("--process-scale must be > 0")
    if abs(scale - 1.0) <= 1e-6:
        return detect_fn(frame, backend, regressor_name, threshold, detector_conf, detector_iou)

    scaled_frame = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    results = detect_fn(scaled_frame, backend, regressor_name, threshold, detector_conf, detector_iou)
    inverse_scale = 1.0 / scale
    scaled_results = []
    for result in results:
        item = dict(result)
        item["corners"] = np.asarray(result["corners"], dtype=np.float32) * inverse_scale
        if "bbox" in result:
            item["bbox"] = [int(round(float(value) * inverse_scale)) for value in result["bbox"]]
        scaled_results.append(item)
    return scaled_results


def parse_deeparuco_id_map(text):
    if text is None:
        return {}
    text = str(text).strip()
    if not text:
        return {}
    mapping = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        decoded_id_text, object_id_text = part.split(":", 1)
        mapping[int(decoded_id_text.strip())] = int(object_id_text.strip())
    return mapping


def deeparuco_results_to_aruco(results, include_rejected=False, id_map=None):
    id_map = {} if id_map is None else id_map
    marker_corners = []
    marker_ids = []
    rejected = []
    for result in results:
        corners = np.asarray(result["corners"], dtype=np.float32).reshape(4, 2)
        aruco_order = np.array([0, 3, 2, 1], dtype=np.int64)
        rotation = int(result.get("rotation", 0)) % 4
        aruco_corners = corners[np.roll(aruco_order, -rotation)].reshape(1, 4, 2)
        if result.get("accepted", False) or include_rejected:
            marker_corners.append(aruco_corners)
            decoded_id = int(result["id"])
            marker_ids.append([int(id_map.get(decoded_id, decoded_id))])
        if not result.get("accepted", False):
            rejected.append(aruco_corners)

    if not marker_ids:
        return marker_corners, None, rejected
    return marker_corners, np.asarray(marker_ids, dtype=np.int32), rejected


def detect_markers_with_backend(frame, gray, dictionary, params, detector, deeparuco_backend, args):
    if args.detector_backend == "opencv":
        return detect_markers(gray, dictionary, params, detector)
    results = detect_deeparuco_scaled(
        frame,
        deeparuco_backend,
        args.deeparuco_regressor,
        args.deeparuco_threshold,
        args.deeparuco_detector_conf,
        args.deeparuco_detector_iou,
        scale=args.process_scale,
    )
    return deeparuco_results_to_aruco(
        results,
        include_rejected=args.deeparuco_include_rejected,
        id_map=args.deeparuco_id_map,
    )


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


def configure_camera_values(cap, width, height, autofocus, fps=30.0, camera_buffer=1):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if float(fps) > 0.0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    if int(camera_buffer) > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(camera_buffer))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, int(autofocus))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)


def configure_camera(cap, args):
    configure_camera_values(
        cap,
        args.width,
        args.height,
        args.autofocus,
        fps=getattr(args, "fps", 30.0),
        camera_buffer=getattr(args, "camera_buffer", 1),
    )


def fourcc_to_text(value):
    value = int(value)
    if value <= 0:
        return "unknown"
    chars = [chr((value >> (8 * index)) & 0xFF) for index in range(4)]
    text = "".join(chars)
    if not text.strip() or any(ord(ch) < 32 or ord(ch) > 126 for ch in text):
        return str(value)
    return text


def describe_camera_actuals(cap):
    return "actual={:.0f}x{:.0f}@{:.1f}fps fourcc={} buffer={:.0f}".format(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FPS),
        fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC)),
        cap.get(cv2.CAP_PROP_BUFFERSIZE),
    )


def probe_camera_index(index, width, height, autofocus, fps=30.0, camera_buffer=1):
    cap = cv2.VideoCapture(int(index))
    configure_camera_values(cap, width, height, autofocus, fps=fps, camera_buffer=camera_buffer)
    if not cap.isOpened():
        cap.release()
        return False
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None)


def discover_camera_indices(max_index, width, height, autofocus, fps=30.0, camera_buffer=1):
    available = []
    for index in range(int(max_index)):
        if probe_camera_index(index, width, height, autofocus, fps=fps, camera_buffer=camera_buffer):
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


def parse_id_sequence(text, option_name, expected_count=None):
    if isinstance(text, (list, tuple)):
        ids = [int(marker_id) for marker_id in text]
    elif isinstance(text, set):
        ids = sorted(int(marker_id) for marker_id in text)
    elif text is None or str(text).strip().lower() in ("", "none", "off"):
        ids = []
    else:
        ids = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if expected_count is not None and len(ids) != int(expected_count):
        raise ValueError("{} must contain exactly {} IDs".format(option_name, expected_count))
    if len(set(ids)) != len(ids):
        raise ValueError("{} must not repeat IDs".format(option_name))
    return ids


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


def parse_vector3(text, option_name):
    if isinstance(text, np.ndarray):
        values = np.asarray(text, dtype=np.float32).reshape(-1).tolist()
    elif isinstance(text, (list, tuple)):
        values = [float(value) for value in text]
    else:
        values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("{} must contain exactly three comma-separated values".format(option_name))
    return np.array(values, dtype=np.float32)


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


def build_layout_for_tag_side_length(args, tag_side_length, use_id_rotation_map=True):
    face_order = parse_face_order(args.face_order)
    upper_ids = parse_id_sequence(args.upper_ids, "--upper-ids", expected_count=len(face_order))
    lower_ids = parse_id_sequence(args.lower_ids, "--lower-ids", expected_count=len(face_order))
    args.corner_rolls = parse_corner_rolls(args.corner_rolls)
    args.auto_roll_ids = parse_id_list(args.auto_roll_ids)
    args.id_face_map = parse_id_face_map(args.id_face_map)
    args.id_rotation_map = parse_id_rotation_map(args.id_rotation_map)
    id_rotation_map = args.id_rotation_map if use_id_rotation_map else {}
    cube_size = get_cube_size(args)
    tag_side_length = float(tag_side_length)
    upper_z, lower_z = get_cube_center_z(args)
    layout = {}
    side_ids = upper_ids + lower_ids
    if len(set(side_ids)) != len(side_ids):
        raise ValueError("--upper-ids and --lower-ids must not overlap")
    if args.top_id >= 0 and args.top_id in side_ids:
        raise ValueError("--top-id must not overlap side-face IDs")

    for marker_id, face in zip(upper_ids, face_order):
        marker_face = args.id_face_map.get(marker_id, face)
        layout[marker_id] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                upper_z,
                cube_size,
                tag_side_length,
                args.upper_rotation_deg,
                id_rotation_map.get(marker_id, 0.0),
            ),
            args.corner_rolls.get(marker_id, 0),
        )

    for marker_id, face in zip(lower_ids, face_order):
        marker_face = args.id_face_map.get(marker_id, face)
        layout[marker_id] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                lower_z,
                cube_size,
                tag_side_length,
                args.lower_rotation_deg,
                id_rotation_map.get(marker_id, 0.0),
            ),
            args.corner_rolls.get(marker_id, 0),
        )

    if args.top_id >= 0:
        marker_face = args.id_face_map.get(args.top_id, "top")
        layout[args.top_id] = roll_corners(
            rotated_marker_object_corners(
                marker_face,
                upper_z,
                cube_size,
                tag_side_length,
                args.upper_rotation_deg,
                id_rotation_map.get(args.top_id, 0.0),
            ),
            args.corner_rolls.get(args.top_id, 0),
        )
    return layout


def build_marker_layout(args):
    return build_layout_for_tag_side_length(args, args.marker_length)


def rotate_layout_z(layout, angle_deg):
    if abs(float(angle_deg)) < 1e-9:
        return layout
    return {
        marker_id: rotate_points_z(object_corners, angle_deg)
        for marker_id, object_corners in layout.items()
    }


def translate_layout(layout, offset):
    offset = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    if float(np.linalg.norm(offset)) <= 1e-9:
        return layout
    return {
        marker_id: np.asarray(object_corners, dtype=np.float32) + offset
        for marker_id, object_corners in layout.items()
    }


def build_physical_face_layout(args):
    layout = build_layout_for_tag_side_length(args, get_cube_size(args), use_id_rotation_map=True)
    layout = rotate_layout_z(layout, getattr(args, "render_model_rotation_deg", 0.0))
    return translate_layout(layout, parse_vector3(getattr(args, "render_model_translation", "0,0,0"), "--render-model-translation"))


def render_to_tag_corner_binding(marker_id):
    marker_id = int(marker_id)
    if marker_id == 0:
        return (2, 3, 0, 1)
    return (1, 0, 3, 2)


def compute_render_alignment_offset(render_layout, tag_layout, visible_ids, tag_quality_scores=None):
    raw_offsets = []
    weighted_offsets = []
    weights = []
    for marker_id in visible_ids:
        marker_id = int(marker_id)
        if marker_id not in render_layout or marker_id not in tag_layout:
            continue
        render_corners = np.asarray(render_layout[marker_id], dtype=np.float32).reshape(4, 3)
        tag_corners = np.asarray(tag_layout[marker_id], dtype=np.float32).reshape(4, 3)
        binding = render_to_tag_corner_binding(marker_id)
        render_edge_midpoint = (render_corners[0] + render_corners[1]) * 0.5
        tag_edge_midpoint = (tag_corners[binding[0]] + tag_corners[binding[1]]) * 0.5
        raw_offset = tag_edge_midpoint - render_edge_midpoint
        weight = 1.0
        if tag_quality_scores:
            weight = max(float(tag_quality_scores.get(marker_id, 0.0)), 0.0)
        raw_offsets.append(raw_offset)
        weighted_offsets.append(raw_offset * weight)
        weights.append(weight)
    if not weighted_offsets:
        return np.zeros(3, dtype=np.float32)
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-9:
        return np.mean(np.asarray(raw_offsets, dtype=np.float32), axis=0)
    return (np.sum(np.asarray(weighted_offsets, dtype=np.float32), axis=0) / total_weight).astype(np.float32)


def smooth_render_alignment_offset(current_offset, previous_offset=None, alpha=1.0):
    current_offset = np.asarray(current_offset, dtype=np.float32).reshape(3)
    if previous_offset is None:
        return current_offset
    previous_offset = np.asarray(previous_offset, dtype=np.float32).reshape(3)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (previous_offset * (1.0 - alpha) + current_offset * alpha).astype(np.float32)


def translate_layout_by_offset(layout, offset):
    offset = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    return {
        marker_id: np.asarray(object_corners, dtype=np.float32) + offset
        for marker_id, object_corners in layout.items()
    }


def align_render_layout_to_visible_tag_edges(render_layout, tag_layout, visible_ids, tag_quality_scores=None, alignment_offset=None):
    if alignment_offset is None:
        alignment_offset = compute_render_alignment_offset(render_layout, tag_layout, visible_ids, tag_quality_scores)
    return translate_layout_by_offset(render_layout, alignment_offset)


def outward_tag_normal(object_corners):
    corners = np.asarray(object_corners, dtype=np.float64).reshape(4, 3)
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    normal /= norm
    center = np.mean(corners, axis=0)
    if float(np.dot(normal, center)) < 0.0:
        normal = -normal
    return normal


def visible_adjacent_tag_pairs(visible_ids):
    visible = {int(marker_id) for marker_id in visible_ids}
    return [(a, b) for a, b in ADJACENT_TAG_PAIRS if a in visible and b in visible]


def build_stable_candidate_id_sets(used_ids, previous_candidate_ids=None, tag_quality_scores=None):
    ordered_ids = []
    for marker_id in used_ids or []:
        marker_id = int(marker_id)
        if marker_id not in ordered_ids:
            ordered_ids.append(marker_id)
    if not ordered_ids:
        return []

    candidate_sets = []

    def add_candidate(candidate_ids):
        candidate = tuple(int(marker_id) for marker_id in candidate_ids if int(marker_id) in ordered_ids)
        if not candidate or candidate in candidate_sets:
            return
        candidate_sets.append(candidate)

    previous = tuple(int(marker_id) for marker_id in previous_candidate_ids or ())
    if 1 <= len(previous) <= 2 and all(marker_id in ordered_ids for marker_id in previous):
        add_candidate(previous)

    if len(ordered_ids) <= 2:
        add_candidate(ordered_ids)
        return candidate_sets

    qualities = tag_quality_scores or {}
    best_id = max(ordered_ids, key=lambda marker_id: float(qualities.get(marker_id, 0.0)))
    add_candidate([best_id])

    adjacent_pairs = visible_adjacent_tag_pairs(ordered_ids)
    if adjacent_pairs:
        best_pair = max(
            adjacent_pairs,
            key=lambda pair: float(qualities.get(pair[0], 0.0)) + float(qualities.get(pair[1], 0.0)),
        )
        add_candidate(best_pair)
    return candidate_sets


def marker_polygon_area(corners):
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return float(abs(cv2.contourArea(points)))


def compute_marker_quality_scores(marker_corners, marker_ids, smoothing_state=None, alpha=0.35):
    if marker_ids is None or marker_corners is None:
        return {}
    if smoothing_state is None:
        smoothing_state = {}
    alpha = float(np.clip(alpha, 0.0, 1.0))
    qualities = {}
    for corners, marker_id_array in zip(marker_corners, marker_ids.reshape(-1)):
        marker_id = int(marker_id_array)
        area = marker_polygon_area(corners)
        previous = float(smoothing_state.get(marker_id, area))
        smoothed = previous * (1.0 - alpha) + area * alpha
        smoothing_state[marker_id] = smoothed
        qualities[marker_id] = smoothed
    return qualities


def candidate_quality_score(used_ids, tag_quality_scores):
    if not tag_quality_scores or not used_ids:
        return 0.0
    values = [float(tag_quality_scores.get(int(marker_id), 0.0)) for marker_id in used_ids]
    if not values:
        return 0.0
    return float(np.mean(values))


def should_switch_candidate_ids(previous_ids, previous_score, candidate_ids, candidate_score, args):
    previous_ids = tuple(int(marker_id) for marker_id in previous_ids or ())
    candidate_ids = tuple(int(marker_id) for marker_id in candidate_ids or ())
    if not previous_ids or previous_ids == candidate_ids:
        return True
    hysteresis = max(float(getattr(args, "candidate_switch_hysteresis", 0.0)), 0.0)
    return float(candidate_score) < float(previous_score) * (1.0 - hysteresis)


def select_stable_render_anchor_ids(used_ids, previous_anchor_ids, tag_quality_scores=None, switch_ratio=1.35):
    used = [int(marker_id) for marker_id in used_ids or []]
    if not used:
        return []
    previous = [int(marker_id) for marker_id in previous_anchor_ids or []]
    stable = [marker_id for marker_id in previous if marker_id in used]
    if not tag_quality_scores:
        return [stable[0]] if stable else [used[0]]

    best_id = max(used, key=lambda marker_id: float(tag_quality_scores.get(marker_id, 0.0)))
    if stable:
        current_id = stable[0]
        current_quality = float(tag_quality_scores.get(current_id, 0.0))
        best_quality = float(tag_quality_scores.get(best_id, 0.0))
        if best_id != current_id and best_quality > current_quality * float(switch_ratio):
            return [best_id]
        return [current_id]
    return [best_id]


def score_pose_candidate_for_selection(mean_error, used_ids, previous_pose=None, rvec=None, tvec=None, tag_quality_scores=None):
    score = float(mean_error)
    quality = candidate_quality_score(used_ids, tag_quality_scores)
    if quality > 0.0:
        score -= min(np.log1p(quality) * 0.04, 1.0)
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


def score_pose_candidate(pose, used_ids, dynamic_rolls, previous_rolls, top_pose, args, previous_pose=None, tag_quality_scores=None):
    rvec, _, _, mean_error, _ = pose
    switch_count = sum(1 for marker_id, roll in dynamic_rolls.items() if previous_rolls.get(marker_id, 0) != roll)
    score = score_pose_candidate_for_selection(
        mean_error,
        used_ids,
        previous_pose=previous_pose,
        rvec=pose[0],
        tvec=pose[1],
        tag_quality_scores=tag_quality_scores,
    )
    score += args.roll_switch_penalty * float(switch_count)
    if args.lock_top_pose_to_top_tag and top_pose is not None:
        score += args.top_disambiguation_weight * rotation_distance_degrees(rvec, top_pose[0])
    return float(score)


def estimate_best_fused_pose(
    marker_corners,
    marker_ids,
    layout,
    camera_matrix,
    dist_coeffs,
    args,
    initial_guess,
    previous_rolls,
    previous_candidate_ids=None,
    previous_candidate_score=None,
    tag_quality_scores=None,
):
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

        candidate_id_sets = build_stable_candidate_id_sets(
            used_ids,
            previous_candidate_ids=previous_candidate_ids,
            tag_quality_scores=tag_quality_scores,
        )

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
                            tag_quality_scores=tag_quality_scores,
                        )
                        top_seed_score = score_pose_candidate(
                            top_seed_pose,
                            candidate_used_ids,
                            dynamic_rolls,
                            previous_rolls,
                            top_pose,
                            args,
                            previous_pose=initial_guess,
                            tag_quality_scores=tag_quality_scores,
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
                tag_quality_scores=tag_quality_scores,
            )
            candidate_ids_tuple = tuple(int(marker_id) for marker_id in candidate_used_ids)
            previous_candidate_tuple = tuple(int(marker_id) for marker_id in previous_candidate_ids or ())
            if previous_candidate_tuple == candidate_ids_tuple and previous_candidate_score is not None:
                score = min(score, float(previous_candidate_score))
            if best is None:
                best = (candidate_object_points, candidate_image_points, candidate_used_ids, pose, dynamic_rolls, score)
                continue
            best_ids_tuple = tuple(int(marker_id) for marker_id in best[2])
            if previous_candidate_tuple and previous_candidate_tuple == best_ids_tuple:
                should_take_candidate = should_switch_candidate_ids(best_ids_tuple, best[5], candidate_ids_tuple, score, args)
            elif previous_candidate_tuple and previous_candidate_tuple == candidate_ids_tuple:
                should_take_candidate = score <= best[5] * (1.0 + max(float(getattr(args, "candidate_switch_hysteresis", 0.0)), 0.0))
            else:
                should_take_candidate = score < best[5]
            if should_take_candidate:
                best = (candidate_object_points, candidate_image_points, candidate_used_ids, pose, dynamic_rolls, score)

    if best is None:
        object_points, image_points, used_ids = collect_correspondences(marker_corners, marker_ids, layout)
        return object_points, image_points, used_ids, None, {}, None
    return best[:6]


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
        return axis_angle_rotation_matrix(axis, np.pi)

    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    return axis_angle_rotation_matrix(axis, angle)


def axis_angle_rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(np.linalg.norm(axis), 1e-9)
    x, y, z = axis
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1.0 - cos_a
    return np.array(
        [
            [cos_a + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_a, x * z * one_minus_cos + y * sin_a],
            [y * x * one_minus_cos + z * sin_a, cos_a + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_a],
            [z * x * one_minus_cos - y * sin_a, z * y * one_minus_cos + x * sin_a, cos_a + z * z * one_minus_cos],
        ],
        dtype=np.float64,
    )


def detected_tag_face_center_and_z_axis(layout, used_ids, prefer_side_face=False):
    if used_ids is None:
        return None, None
    fallback = (None, None)
    for marker_id in used_ids:
        marker_id = int(marker_id)
        if marker_id not in layout:
            continue
        object_corners = np.asarray(layout[marker_id], dtype=np.float64).reshape(4, 3)
        z_axis = outward_tag_normal(object_corners)
        if z_axis is None:
            continue
        center = np.mean(object_corners, axis=0)
        if not prefer_side_face or abs(float(z_axis[2])) < 0.9:
            return center, z_axis
        if fallback[0] is None:
            fallback = (center, z_axis)
    return fallback


def reflect_points_across_plane(points, plane_point, plane_normal):
    points = np.asarray(points, dtype=np.float32)
    plane_point = np.asarray(plane_point, dtype=np.float32).reshape(1, 3)
    plane_normal = np.asarray(plane_normal, dtype=np.float32).reshape(1, 3)
    plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-9)
    distances = (points.reshape(-1, 3) - plane_point) @ plane_normal.reshape(3, 1)
    reflected = points.reshape(-1, 3) - 2.0 * distances * plane_normal
    return reflected.reshape(points.shape).astype(np.float32)


def reflected_layout_across_plane(layout, plane_point, plane_normal):
    return {
        marker_id: reflect_points_across_plane(object_corners, plane_point, plane_normal)
        for marker_id, object_corners in layout.items()
    }


def reflected_layout_across_detected_tag_plane(layout, used_ids):
    plane = detected_tag_face_center_and_z_axis(layout, used_ids, prefer_side_face=True)
    pivot, z_axis = plane
    if pivot is None:
        return layout
    return reflected_layout_across_plane(layout, pivot, z_axis)


def render_pose_rotated_about_detected_tag_z(layout, used_ids, rvec, tvec, angle_deg=180.0):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    pivot, z_axis = detected_tag_face_center_and_z_axis(layout, used_ids)
    if pivot is None:
        return rvec.copy(), tvec.copy()

    pose_rotation, _ = cv2.Rodrigues(rvec)
    local_rotation = axis_angle_rotation_matrix(z_axis, np.deg2rad(angle_deg))
    render_rotation = pose_rotation @ local_rotation
    pivot = np.asarray(pivot, dtype=np.float64).reshape(3, 1)
    render_tvec = tvec + pose_rotation @ (pivot - local_rotation @ pivot)
    render_rvec, _ = cv2.Rodrigues(render_rotation)
    return render_rvec.reshape(3, 1), render_tvec.reshape(3, 1)


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


def cube_wireframe_vertices_from_layout(layout, marker_ids, atol=1e-7):
    unique_vertices = []
    for marker_id in marker_ids:
        marker_id = int(marker_id)
        if marker_id not in layout:
            continue
        for point in np.asarray(layout[marker_id], dtype=np.float32).reshape(4, 3):
            if not any(np.allclose(point, vertex, atol=atol) for vertex in unique_vertices):
                unique_vertices.append(point.copy())
    return np.asarray(unique_vertices, dtype=np.float32).reshape(-1, 3)


def cube_wireframe_edges_from_vertices(vertices, cube_size, atol=1e-5):
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    edges = []
    for i0 in range(len(vertices)):
        for i1 in range(i0 + 1, len(vertices)):
            if abs(float(np.linalg.norm(vertices[i1] - vertices[i0])) - float(cube_size)) <= float(atol):
                edges.append((i0, i1))
    return edges


def render_model_origin_from_layout(layout, args):
    upper_ids = [args.top_id] + parse_id_sequence(args.upper_ids, "--upper-ids", expected_count=4)
    lower_ids = parse_id_sequence(args.lower_ids, "--lower-ids", expected_count=4)
    upper_vertices = cube_wireframe_vertices_from_layout(layout, upper_ids)
    lower_vertices = cube_wireframe_vertices_from_layout(layout, lower_ids)
    centers = []
    if len(upper_vertices) > 0:
        centers.append(np.mean(upper_vertices, axis=0))
    if len(lower_vertices) > 0:
        centers.append(np.mean(lower_vertices, axis=0))
    if not centers:
        return np.zeros(3, dtype=np.float32)
    return np.mean(np.asarray(centers, dtype=np.float32), axis=0).astype(np.float32)


def tvec_for_object_space_axis_origin(rvec, tvec, object_origin):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    object_origin = np.asarray(object_origin, dtype=np.float64).reshape(3, 1)
    return np.asarray(tvec, dtype=np.float64).reshape(3, 1) + rotation @ object_origin


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


def corner_label_position(projected, corner_index, inward_fraction=0.30, min_inward_px=6.0, max_inward_px=14.0):
    points = np.asarray(projected, dtype=np.float32).reshape(-1, 2)
    point = points[int(corner_index)]
    center = np.mean(points, axis=0)
    inward = center - point
    distance = float(np.linalg.norm(inward))
    if distance <= 1e-6:
        label_point = point.copy()
    else:
        inward /= distance
        offset = float(np.clip(distance * float(inward_fraction), float(min_inward_px), float(max_inward_px)))
        label_point = point + inward * offset
    return int(round(float(label_point[0]))), int(round(float(label_point[1])))


def draw_rigid_tag_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs, draw_corner_labels=False):
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
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if draw_corner_labels:
            for corner_index, point in enumerate(projected):
                label_position = corner_label_position(projected, corner_index)
                if point_tuple(label_position) is None:
                    continue
                cv2.putText(
                    frame,
                    "a{}".format(corner_index),
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
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


def draw_fused_model(frame, layout, rvec, tvec, camera_matrix, dist_coeffs, args, reflection_plane=None):
    if args.draw_prism_wireframe:
        cube_size = get_cube_size(args)
        upper_z, lower_z = get_cube_center_z(args)
        upper_ids = [args.top_id] + parse_id_sequence(args.upper_ids, "--upper-ids", expected_count=4)
        lower_ids = parse_id_sequence(args.lower_ids, "--lower-ids", expected_count=4)
        for vertices, fallback_vertices, color in (
            (
                cube_wireframe_vertices_from_layout(layout, upper_ids),
                cube_vertices(upper_z, cube_size, args.upper_rotation_deg),
                UPPER_CUBE_COLOR,
            ),
            (
                cube_wireframe_vertices_from_layout(layout, lower_ids),
                cube_vertices(lower_z, cube_size, args.lower_rotation_deg),
                LOWER_CUBE_COLOR,
            ),
        ):
            edges = cube_wireframe_edges_from_vertices(vertices, cube_size)
            if len(vertices) != 8 or len(edges) != 12:
                vertices = fallback_vertices
                edges = cube_wireframe_edges_from_vertices(vertices, cube_size)
            if reflection_plane is not None:
                vertices = reflect_points_across_plane(vertices, reflection_plane[0], reflection_plane[1])
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
        draw_rigid_tag_model(
            frame,
            layout,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            draw_corner_labels=getattr(args, "draw_model_corner_labels", True),
        )

    if hasattr(cv2, "drawFrameAxes"):
        axis_origin = render_model_origin_from_layout(layout, args)
        axis_tvec = tvec_for_object_space_axis_origin(rvec, tvec, axis_origin)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, axis_tvec, args.axis_length)


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


def fps_to_ms(fps):
    fps = float(fps)
    if fps <= 1e-6:
        return 0.0
    return 1000.0 / fps


def text_pixel_width(text, font, scale, thickness):
    return cv2.getTextSize(str(text), font, scale, thickness)[0][0]


def split_text_to_fit(text, max_width, font, scale, thickness):
    text = str(text)
    if max_width <= 0 or text_pixel_width(text, font, scale, thickness) <= max_width:
        return [text]

    wrapped = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else "{} {}".format(current, word)
        if text_pixel_width(candidate, font, scale, thickness) <= max_width:
            current = candidate
            continue

        if current:
            wrapped.append(current)
            current = ""

        if text_pixel_width(word, font, scale, thickness) <= max_width:
            current = word
            continue

        chunk = ""
        for char in word:
            candidate = chunk + char
            if chunk and text_pixel_width(candidate, font, scale, thickness) > max_width:
                wrapped.append(chunk)
                chunk = char
            else:
                chunk = candidate
        current = chunk

    if current:
        wrapped.append(current)
    return wrapped or [""]


def draw_text_panel(image, lines):
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    left = 8
    top = 0
    line_step = 20
    baseline_y = 24
    panel_width = min(image.shape[1], 980)
    max_text_width = max(1, panel_width - left * 2)
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(split_text_to_fit(line, max_text_width, font, scale, thickness))
    max_visible_lines = max(1, (image.shape[0] - baseline_y) // line_step + 1)
    wrapped_lines = wrapped_lines[:max_visible_lines]
    height = min(image.shape[0], 18 + line_step * len(wrapped_lines))
    cv2.rectangle(overlay, (0, top), (panel_width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
    text_overlay = image.copy()
    for index, line in enumerate(wrapped_lines):
        cv2.putText(
            text_overlay,
            line,
            (left, baseline_y + line_step * index),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    cv2.addWeighted(text_overlay, 0.50, image, 0.50, 0, image)


def process_frame(
    frame,
    dictionary,
    params,
    detector,
    deeparuco_backend,
    layout,
    camera_matrix,
    dist_coeffs,
    args,
    pose_tracker=None,
    capture_fps=0.0,
    process_fps=0.0,
    render_layout=None,
    draw_output=True,
    unity_pose_sender=None,
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_start_time = time.perf_counter()
    marker_corners, marker_ids, rejected = detect_markers_with_backend(
        frame,
        gray,
        dictionary,
        params,
        detector,
        deeparuco_backend,
        args,
    )
    tag_detect_ms = (time.perf_counter() - detect_start_time) * 1000.0
    if render_layout is None:
        render_layout = layout
    frame_height, frame_width = frame.shape[:2]

    ids = [] if marker_ids is None else marker_ids.flatten().astype(int).tolist()
    tag_quality_scores = compute_marker_quality_scores(
        marker_corners,
        marker_ids,
        smoothing_state=None if pose_tracker is None else pose_tracker.tag_quality_scores,
        alpha=getattr(args, "tag_quality_smoothing_alpha", 0.35),
    )

    object_points, image_points, used_ids, pose, dynamic_rolls, candidate_score = estimate_best_fused_pose(
        marker_corners,
        marker_ids,
        layout,
        camera_matrix,
        dist_coeffs,
        args,
        initial_guess=None if pose_tracker is None else pose_tracker.get_initial_guess(),
        previous_rolls={} if pose_tracker is None else pose_tracker.last_rolls,
        previous_candidate_ids=[] if pose_tracker is None else pose_tracker.last_candidate_ids,
        previous_candidate_score=None if pose_tracker is None else pose_tracker.last_candidate_score,
        tag_quality_scores=tag_quality_scores,
    )

    pose_text = "pose: need at least {} configured tags".format(args.min_tags_for_pose)
    pose_state_text = "lost"
    active_rolls = {}
    render_anchor_ids = []
    render_alignment_offset = None
    draw_pose = None
    if pose is not None:
        rvec, tvec, inliers, mean_error, max_error = pose
        stable_pose = (rvec, tvec)
        raw_pose_drawn = False
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
                    raw_pose_drawn = True
        if stable_pose is not None:
            rvec, tvec = stable_pose
            active_rolls = (
                dynamic_rolls
                if pose_tracker is None or pose_state_text == "measured" or raw_pose_drawn
                else pose_tracker.last_rolls
            )
            draw_pose = (rvec.copy(), tvec.copy())
            previous_anchor_ids = [] if pose_tracker is None else pose_tracker.last_render_anchor_ids
            render_anchor_ids = select_stable_render_anchor_ids(
                used_ids,
                previous_anchor_ids,
                tag_quality_scores=tag_quality_scores,
                switch_ratio=getattr(args, "render_anchor_switch_ratio", 1.35),
            )
            current_alignment_offset = compute_render_alignment_offset(
                render_layout,
                layout,
                render_anchor_ids,
                tag_quality_scores=tag_quality_scores,
            )
            previous_alignment_offset = None if pose_tracker is None else pose_tracker.last_render_alignment_offset
            render_alignment_offset = smooth_render_alignment_offset(
                current_alignment_offset,
                previous_alignment_offset,
                alpha=getattr(args, "render_alignment_smoothing_alpha", 0.25),
            )
            if pose_tracker is not None:
                pose_tracker.last_candidate_ids = list(used_ids)
                pose_tracker.last_candidate_score = candidate_score
                pose_tracker.last_render_anchor_ids = list(render_anchor_ids)
                pose_tracker.last_render_alignment_offset = render_alignment_offset.copy()
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
            active_rolls = dict(pose_tracker.last_rolls)
            draw_pose = (rvec.copy(), tvec.copy())
            render_anchor_ids = list(
                pose_tracker.last_render_state.render_anchor_ids
                or pose_tracker.last_render_state.used_ids
            )
            render_alignment_offset = pose_tracker.last_render_alignment_offset
            tx, ty, tz = tvec.reshape(-1)
            roll, pitch, yaw = rotation_to_euler_xyz(rvec)
            pose_text = (
                "rigid object pose t=({:.3f},{:.3f},{:.3f})m rpy=({:.1f},{:.1f},{:.1f})deg state=held_no_detection"
                .format(tx, ty, tz, roll, pitch, yaw)
            )
            pose_state_text = "held_no_detection"

    rolls_text = dynamic_rolls if dynamic_rolls else (pose_tracker.last_rolls if pose_tracker is not None and pose_tracker.last_rolls else "off/unchanged")
    render_state = PoseRenderState(
        marker_corners=marker_corners,
        marker_ids=marker_ids,
        rejected_count=len(rejected),
        pose=pose,
        ids=ids,
        used_ids=used_ids,
        render_anchor_ids=render_anchor_ids,
        dynamic_rolls=dict(dynamic_rolls),
        active_rolls=dict(active_rolls),
        pose_text=pose_text,
        pose_state_text=pose_state_text,
        rolls_text=rolls_text,
        capture_fps=capture_fps,
        process_fps=process_fps,
        tag_detect_ms=tag_detect_ms,
        frame_width=frame_width,
        frame_height=frame_height,
        draw_pose=draw_pose,
        render_alignment_offset=render_alignment_offset,
    )
    if pose_tracker is not None:
        pose_tracker.last_render_state = render_state
    if unity_pose_sender is not None:
        try:
            if render_state.draw_pose is not None:
                send_rvec, send_tvec = render_state.draw_pose
                send_mean_error = None if render_state.pose is None else render_state.pose[3]
                unity_pose_sender.send(
                    True,
                    send_rvec,
                    send_tvec,
                    render_state.used_ids,
                    send_mean_error,
                    render_state.pose_state_text,
                )
            else:
                unity_pose_sender.send(
                    False,
                    used_ids=render_state.used_ids,
                    mean_error=None,
                    pose_state_text=render_state.pose_state_text,
                )
        except OSError:
            pass
    if draw_output:
        draw_render_state(
            frame,
            render_state,
            render_layout,
            camera_matrix,
            dist_coeffs,
            args,
            pose_layout=layout,
        )
    return pose, ids, used_ids


def draw_render_state(
    frame,
    render_state,
    render_layout,
    camera_matrix,
    dist_coeffs,
    args,
    display_fps=0.0,
    pose_layout=None,
):
    marker_corners = render_state.marker_corners
    marker_ids = render_state.marker_ids
    marker_count = 0 if marker_ids is None else len(marker_ids)
    if marker_count > 0:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        if args.draw_detected_corners:
            draw_detected_corners(frame, marker_corners, marker_ids, args)

    if render_state.draw_pose is not None:
        rvec, tvec = render_state.draw_pose
        anchor_ids = render_state.render_anchor_ids or render_state.used_ids
        draw_layout = dict(render_layout)
        if render_state.render_alignment_offset is not None:
            draw_layout = translate_layout_by_offset(draw_layout, render_state.render_alignment_offset)
        elif pose_layout is not None:
            draw_layout = align_render_layout_to_visible_tag_edges(draw_layout, pose_layout, anchor_ids)
        reflection_plane = None
        if getattr(args, "render_reflect_across_detected_tag_plane", True):
            reflection_plane = detected_tag_face_center_and_z_axis(draw_layout, anchor_ids, prefer_side_face=True)
            if reflection_plane[0] is None:
                reflection_plane = None
            else:
                draw_layout = reflected_layout_across_plane(draw_layout, reflection_plane[0], reflection_plane[1])
        if getattr(args, "render_flip_about_detected_tag_z", False):
            rvec, tvec = render_pose_rotated_about_detected_tag_z(draw_layout, anchor_ids, rvec, tvec)
        draw_fused_model(
            frame,
            draw_layout,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            args,
            reflection_plane=reflection_plane,
        )

    cube_size = get_cube_size(args)
    vertical_gap = get_vertical_gap(args)
    frame_width = int(getattr(render_state, "frame_width", 0) or 0)
    frame_height = int(getattr(render_state, "frame_height", 0) or 0)
    resolution_text = "{}x{}".format(frame_width, frame_height) if frame_width > 0 and frame_height > 0 else "unknown"
    backend_name = getattr(args, "detector_backend", "opencv")
    backend_display_name = "DeepArUco" if backend_name == "deeparuco" else "OpenCV ArUco"
    lines = [
        "DeepArUco rigid object fused pose  backend: {}  dictionary: {}  video_fps={:.1f} process_fps={:.1f} display_fps={:.1f} tag_detect={:.1f}ms resolution={}".format(
            backend_display_name,
            args.dictionary,
            float(render_state.capture_fps),
            float(render_state.process_fps),
            float(display_fps),
            float(getattr(render_state, "tag_detect_ms", 0.0)),
            resolution_text,
        ),
        "detected ids: {}  used ids: {}  rejected: {}".format(
            render_state.ids[:12], render_state.used_ids, render_state.rejected_count
        ),
        render_state.pose_text,
    ]
    draw_text_panel(frame, lines)
    return frame


def run_image(args, dictionary, params, detector, deeparuco_backend, layout, render_layout, camera_matrix, dist_coeffs):
    frame = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Cannot read image: {}".format(args.image))
    if camera_matrix is None:
        camera_matrix, dist_coeffs = approximate_calibration(frame.shape[1], frame.shape[0])

    unity_pose_sender = create_unity_pose_sender(args)
    try:
        pose, ids, used_ids = process_frame(
            frame,
            dictionary,
            params,
            detector,
            deeparuco_backend,
            layout,
            camera_matrix,
            dist_coeffs,
            args,
            render_layout=render_layout,
            unity_pose_sender=unity_pose_sender,
        )
    finally:
        if unity_pose_sender is not None:
            unity_pose_sender.close()
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


def run_camera(args, dictionary, params, detector, deeparuco_backend, layout, render_layout, camera_matrix, dist_coeffs):
    camera_text = str(args.camera).strip().lower()
    scanned = []
    if camera_text in ("auto", "any"):
        scanned = discover_camera_indices(
            args.camera_scan_max,
            args.width,
            args.height,
            args.autofocus,
            fps=args.fps,
            camera_buffer=args.camera_buffer,
        )
        if not scanned:
            raise RuntimeError(
                "No working camera found in 0..{}; check /dev/video* or pass --camera N".format(
                    args.camera_scan_max - 1
                )
            )
        camera_index = scanned[0]
    else:
        camera_index = int(args.camera)
        if not probe_camera_index(
            camera_index,
            args.width,
            args.height,
            args.autofocus,
            fps=args.fps,
            camera_buffer=args.camera_buffer,
        ):
            scanned = discover_camera_indices(
                args.camera_scan_max,
                args.width,
                args.height,
                args.autofocus,
                fps=args.fps,
                camera_buffer=args.camera_buffer,
            )
            if camera_index not in scanned and scanned:
                camera_index = scanned[0]
            elif not scanned:
                raise RuntimeError(
                    "Cannot open camera index {}. No working cameras found in 0..{}.".format(
                        args.camera, args.camera_scan_max - 1
                    )
                )

    cap = cv2.VideoCapture(camera_index)
    configure_camera(cap, args)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera index {}. Available candidates: {}".format(
                camera_index, scanned if scanned else "none"
            )
        )

    print("OpenCV: {}".format(cv2.__version__), flush=True)
    print(
        "Camera: live:{} requested={}x{}@{}fps autofocus={} buffer={} async_capture=on {}".format(
            camera_index,
            args.width,
            args.height,
            args.fps if args.fps > 0.0 else "default",
            args.autofocus,
            args.camera_buffer,
            describe_camera_actuals(cap),
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
    render_states = LatestFrameSlot()
    capture_rate = RateCounter()
    process_rate = RateCounter()
    display_rate = RateCounter()
    pose_tracker = PoseTracker(args)
    process_scheduler = FrameProcessScheduler(args.process_every_n_frames)
    unity_pose_sender = create_unity_pose_sender(args)
    window_name = "DeepArUco Cube Rigid Async Fused Pose"
    if unity_pose_sender is not None:
        print(
            "Unity pose UDP: {}:{}".format(args.unity_udp_host, args.unity_udp_port),
            flush=True,
        )

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
                if not process_scheduler.should_process():
                    continue
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
                        deeparuco_backend,
                        layout,
                        current_camera_matrix,
                        current_dist_coeffs,
                        args,
                        pose_tracker=pose_tracker,
                        capture_fps=current_capture_fps,
                        process_fps=process_fps,
                        render_layout=render_layout,
                        draw_output=False,
                        unity_pose_sender=unity_pose_sender,
                    )
                    render_states.put((pose_tracker.last_render_state, current_camera_matrix, current_dist_coeffs))
                except Exception:
                    traceback.print_exc()
                    render_states.put(
                        (
                            PoseRenderState(
                                capture_fps=current_capture_fps,
                                process_fps=process_fps,
                                pose_text="processing exception; see terminal traceback",
                                ids=[],
                                used_ids=[],
                            ),
                            current_camera_matrix,
                            current_dist_coeffs,
                        )
                    )
        finally:
            render_states.close()

    capture_thread = threading.Thread(target=capture_loop, name="camera-capture", daemon=True)
    process_thread = threading.Thread(target=processing_loop, name="aruco-pose-processing", daemon=True)
    capture_thread.start()
    process_thread.start()

    last_raw_sequence = 0
    last_render_state_sequence = 0
    latest_render_bundle = None
    try:
        while not stop_event.is_set():
            item, sequence = raw_frames.get_latest(last_raw_sequence, timeout=0.005)
            if item is not None and sequence != last_raw_sequence:
                last_raw_sequence = sequence
                raw_frame, _ = item
                render_item, render_sequence = render_states.get_latest(last_render_state_sequence, timeout=0.0)
                if render_item is not None:
                    latest_render_bundle = render_item
                    last_render_state_sequence = render_sequence
                frame = raw_frame.copy()
                if latest_render_bundle is None:
                    if camera_matrix is None:
                        current_camera_matrix, current_dist_coeffs = approximate_calibration(frame.shape[1], frame.shape[0])
                    else:
                        current_camera_matrix, current_dist_coeffs = camera_matrix, dist_coeffs
                    latest_render_bundle = (
                        PoseRenderState(capture_fps=capture_rate.rate, process_fps=process_rate.rate),
                        current_camera_matrix,
                        current_dist_coeffs,
                    )
                render_state, current_camera_matrix, current_dist_coeffs = latest_render_bundle
                display_fps = display_rate.tick()
                draw_render_state(
                    frame,
                    render_state,
                    render_layout,
                    current_camera_matrix,
                    current_dist_coeffs,
                    args,
                    display_fps=display_fps,
                    pose_layout=layout,
                )
                cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                stop_event.set()
                break
    finally:
        stop_event.set()
        raw_frames.close()
        render_states.close()
        capture_thread.join(timeout=1.0)
        process_thread.join(timeout=1.0)
        if unity_pose_sender is not None:
            unity_pose_sender.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a rigid 9-tag DeepArUco/ArUco cube stack and estimate one fused model pose asynchronously."
    )
    parser.add_argument("--dictionary", default="DICT_6X6_250", choices=sorted(DICTIONARIES))
    parser.add_argument("--camera", default="0", help="Camera index or 'auto'. Default: 0.")
    parser.add_argument("--camera-scan-max", type=int, default=10, help="Max camera index to probe when scanning. Default: 10.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width. Default: 640.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height. Default: 480.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS. Use 0 to keep the driver default. Default: 30.")
    parser.add_argument("--camera-buffer", type=int, default=1, help="Requested capture buffer size for lower latency. Use 0 to keep the driver default. Default: 1.")
    parser.add_argument("--autofocus", type=int, choices=(0, 1), default=1, help="Enable camera autofocus. Default: 1.")
    parser.add_argument("--process-scale", type=float, default=0.75, help="Scale frames before DeepArUco inference, then map corners back. Lower is faster. Default: 0.75.")
    parser.add_argument("--process-every-n-frames", type=int, default=2, help="Run marker inference every N captured frames. Higher improves display smoothness. Default: 2.")
    parser.add_argument("--send-unity-pose", action="store_true", help="Send the latest draw pose to Unity over UDP.")
    parser.add_argument("--unity-udp-host", default="127.0.0.1", help="Unity UDP receiver host. Default: 127.0.0.1.")
    parser.add_argument("--unity-udp-port", type=int, default=5055, help="Unity UDP receiver port. Default: 5055.")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, open camera.")
    parser.add_argument("--output", default=None, help="Optional annotated image output path for --image.")
    parser.add_argument("--show", action="store_true", help="Show the annotated image when using --image.")
    parser.add_argument("--calibration", default=None, help="Optional OpenCV camera calibration XML/YAML file.")
    parser.add_argument(
        "--detector-backend",
        choices=("deeparuco", "opencv"),
        default="deeparuco",
        help="Marker detection backend. Default: deeparuco.",
    )
    parser.add_argument(
        "--deeparuco-repo",
        default=DEFAULT_DEEPPARUCO_REPO,
        help="DeepArUco repository root. Default: {}.".format(DEFAULT_DEEPPARUCO_REPO),
    )
    parser.add_argument(
        "--deeparuco-detector",
        default="det_luma_bc_s",
        help="YOLO detector model name in deeparuco repo/models. Default: det_luma_bc_s.",
    )
    parser.add_argument(
        "--deeparuco-regressor",
        default="reg_hmap_8",
        help="Corner refinement model name in deeparuco repo/models. Default: reg_hmap_8.",
    )
    parser.add_argument(
        "--deeparuco-threshold",
        type=float,
        default=9.0,
        help="Reject decoded DeepArUco markers with distance >= threshold. Default: 9.",
    )
    parser.add_argument(
        "--deeparuco-detector-conf",
        type=float,
        default=0.03,
        help="DeepArUco YOLO confidence threshold. Default: 0.03.",
    )
    parser.add_argument(
        "--deeparuco-detector-iou",
        type=float,
        default=0.5,
        help="DeepArUco YOLO IoU threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--deeparuco-include-rejected",
        action="store_true",
        help="Also pass decoded markers rejected by --deeparuco-threshold into the fused pose path.",
    )
    parser.add_argument(
        "--deeparuco-id-map",
        default="",
        help="Map DeepArUco decoded IDs to physical object IDs, e.g. 23:0,16:1. Default: decoded IDs are used unchanged.",
    )
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
    parser.add_argument("--upper-ids", default="1,2,3,4", help="Upper cube side-face ArUco IDs in face-order. Default: 1,2,3,4.")
    parser.add_argument("--lower-ids", default="5,6,7,8", help="Lower cube side-face ArUco IDs in face-order. Default: 5,6,7,8.")
    parser.add_argument("--upper-rotation-deg", type=float, default=45.0, help="Upper cube rotation around object Z axis. Default: 45.")
    parser.add_argument("--lower-rotation-deg", type=float, default=0.0, help="Lower cube rotation around object Z axis. Default: 0.")
    parser.add_argument("--top-id", type=int, default=0, help="Top face ArUco ID. Set negative to disable top face tag. Default: 0.")
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
        default="1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:right,0:top",
        help="Override ID to prism face assignment. Faces: front,right,back,left,top.",
    )
    parser.add_argument(
        "--id-rotation-map",
        default="0:90",
        help="Override per-ID in-plane rotation in degrees, e.g. 0:0,1:0,2:0,3:0,8:0.",
    )
    parser.add_argument("--axis-length", type=float, default=0.04, help="Drawn model axis length in meters.")
    parser.add_argument("--cube-line-thickness", type=int, default=3, help="Projected cube wireframe line thickness.")
    parser.add_argument("--draw-prism-wireframe", action="store_true", default=False, help="Draw the upper/lower cube wireframes. Default: off.")
    parser.add_argument("--no-draw-prism-wireframe", dest="draw_prism_wireframe", action="store_false", help="Hide cube wireframes.")
    parser.add_argument("--draw-model-tags", action="store_true", default=True, help="Project all configured rigid tag outlines. Default: on.")
    parser.add_argument("--no-draw-model-tags", dest="draw_model_tags", action="store_false", help="Hide projected rigid tag outlines.")
    parser.add_argument("--draw-model-corner-labels", action="store_true", default=False, help="Draw small a0-a3 labels at each rendered model face corner. Default: off.")
    parser.add_argument("--no-draw-model-corner-labels", dest="draw_model_corner_labels", action="store_false", help="Hide rendered model face corner labels.")
    parser.add_argument("--render-model-rotation-deg", type=float, default=180.0, help="Rotate the rendered rigid model around object Z in degrees. Default: 180.")
    parser.add_argument("--render-model-translation", default="-0.04,0,-0.004", help="Translate the rendered rigid model in object coordinates, meters as x,y,z. Default: -0.04,0,-0.004.")
    parser.add_argument(
        "--render-flip-about-detected-tag-z",
        action="store_true",
        help="Flip the rendered model 180 degrees around the detected tag Z axis before drawing.",
    )
    parser.add_argument(
        "--render-reflect-across-detected-tag-plane",
        dest="render_reflect_across_detected_tag_plane",
        action="store_true",
        help="Reflect the rendered model through the detected tag plane before drawing.",
    )
    parser.add_argument(
        "--no-render-reflect-across-detected-tag-plane",
        dest="render_reflect_across_detected_tag_plane",
        action="store_false",
        help="Disable reflecting the rendered model through the detected tag plane.",
    )
    parser.set_defaults(render_reflect_across_detected_tag_plane=False)
    parser.add_argument("--draw-detected-corners", action="store_true", default=False, help="Draw detected ArUco corner dots. Default: off.")
    parser.add_argument("--no-draw-detected-corners", dest="draw_detected_corners", action="store_false", help="Hide detected corner dots.")
    parser.add_argument("--draw-corner-index", action="store_true", default=False, help="Draw marker-id:corner-index labels. Default: off.")
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
    parser.add_argument("--min-pose-follow", type=float, default=1.0, help="Minimum fraction of each new pose update to keep under low confidence. Default: 1.0 for fastest response.")
    parser.add_argument("--low-confidence-extra-smoothing", type=float, default=2.2, help="Extra One Euro smoothing applied when visible tags are few or reprojection error is higher.")
    parser.add_argument("--hold-last-seconds", type=float, default=0.25, help="Keep the last stable pose this long after a miss.")
    parser.add_argument("--max-stable-reprojection-error", type=float, default=8.0, help="Reject noisy pose updates above this mean reprojection error.")
    parser.add_argument("--reject-outlier-tags", action="store_true", default=True, help="Drop visible tags whose reprojection error is inconsistent with the rigid model. Default: on.")
    parser.add_argument("--no-reject-outlier-tags", dest="reject_outlier_tags", action="store_false", help="Use all visible configured tags even when one has high reprojection error.")
    parser.add_argument("--tag-outlier-ratio", type=float, default=2.8, help="Reject a tag when its mean reprojection error is this many times above the median, capped by --max-tag-mean-reprojection-error.")
    parser.add_argument("--min-tag-mean-reprojection-error", type=float, default=4.0, help="Lower pixel threshold before tag outlier rejection can trigger.")
    parser.add_argument("--max-tag-mean-reprojection-error", type=float, default=8.0, help="Upper mean pixel error for keeping a tag during outlier rejection.")
    parser.add_argument("--tag-quality-smoothing-alpha", type=float, default=0.35, help="EMA alpha for per-tag area quality scores used to stabilize adjacent-face transitions.")
    parser.add_argument("--candidate-switch-hysteresis", type=float, default=0.18, help="Relative score improvement required before switching away from the previous fused tag set.")
    parser.add_argument("--render-alignment-smoothing-alpha", type=float, default=0.25, help="EMA alpha for render-model alignment offset. Lower is smoother during adjacent-face transitions.")
    parser.add_argument("--render-anchor-switch-ratio", type=float, default=1.35, help="New render anchor tag quality must exceed current anchor by this ratio before switching.")
    parser.add_argument("--roll-switch-penalty", type=float, default=0.75, help="Penalty added when dynamic ID roll selection changes from the previous frame.")
    parser.add_argument("--print-pose", action="store_true", help="Print pose every frame when available.")
    args = parser.parse_args()
    args.upper_ids = parse_id_sequence(args.upper_ids, "--upper-ids", expected_count=4)
    args.lower_ids = parse_id_sequence(args.lower_ids, "--lower-ids", expected_count=4)
    return args


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
    args.deeparuco_id_map = parse_deeparuco_id_map(args.deeparuco_id_map)
    dictionary = get_dictionary(args.dictionary)
    params = create_detector_parameters(args)
    detector = create_detector(dictionary, params)
    deeparuco_backend = None
    if args.detector_backend == "deeparuco":
        deeparuco_backend = load_deeparuco_backend(
            args.deeparuco_repo,
            args.deeparuco_detector,
            args.deeparuco_regressor,
            args.dictionary,
        )
    layout = build_marker_layout(args)
    render_layout = build_physical_face_layout(args)

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
        run_image(args, dictionary, params, detector, deeparuco_backend, layout, render_layout, camera_matrix, dist_coeffs)
    else:
        run_camera(args, dictionary, params, detector, deeparuco_backend, layout, render_layout, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Exception:", exc, file=sys.stderr)
        sys.exit(1)
