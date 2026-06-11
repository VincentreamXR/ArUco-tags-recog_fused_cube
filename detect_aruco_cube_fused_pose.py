#!/usr/bin/env python3
import argparse
import os
import sys
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
    # normal, marker horizontal axis when viewed from outside the cube
    "front": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    "right": (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
    "back": (np.array([0.0, -1.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
    "left": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
}


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
    if len(faces) != 4 or any(face not in FACE_SPECS for face in faces):
        raise ValueError("--face-order must contain four faces from: front,right,back,left")
    if len(set(faces)) != 4:
        raise ValueError("--face-order must not repeat faces")
    return faces


def parse_id_list(text):
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not ids:
        raise ValueError("ID list must not be empty")
    return set(ids)


def marker_object_corners(face, cube_center_z, cube_size, marker_length):
    normal, u_axis = FACE_SPECS[face]
    v_axis = np.array([0.0, 0.0, -1.0])
    center = normal * (cube_size * 0.5) + np.array([0.0, 0.0, cube_center_z])
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


def rotated_marker_object_corners(face, cube_center_z, cube_size, marker_length, angle_deg):
    return rotate_points_z(marker_object_corners(face, cube_center_z, cube_size, marker_length), angle_deg)


def build_marker_layout(args):
    face_order = parse_face_order(args.face_order)
    args.lower_ids = parse_id_list(args.lower_ids)
    upper_z = (args.cube_size + args.vertical_gap) * 0.5
    lower_z = -(args.cube_size + args.vertical_gap) * 0.5
    layout = {}
    for index, face in enumerate(face_order):
        layout[index] = rotated_marker_object_corners(
            face,
            upper_z,
            args.cube_size,
            args.marker_length,
            args.upper_rotation_deg,
        )
        lower_id = index + 4
        if lower_id in args.lower_ids:
            layout[lower_id] = rotated_marker_object_corners(
                face,
                lower_z,
                args.cube_size,
                args.marker_length,
                args.lower_rotation_deg,
            )
    return layout


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


def estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, args):
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


def cube_vertices_from_tag_face(face, center_z, cube_size, marker_length, rotation_deg, inward_sign):
    normal, _ = FACE_SPECS[face]
    outer = rotated_marker_object_corners(face, center_z, cube_size, marker_length, rotation_deg)
    inner_normal = rotate_points_z(normal.reshape(1, 3), rotation_deg).reshape(3)
    inner = outer + float(inward_sign) * inner_normal * cube_size
    return np.vstack([outer, inner]).astype(np.float32)


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


def point_tuple(point, max_abs=1000000):
    point = np.round(point).reshape(-1)
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        return None
    if abs(point[0]) > max_abs or abs(point[1]) > max_abs:
        return None
    return int(point[0]), int(point[1])


def draw_fused_model(frame, rvec, tvec, camera_matrix, dist_coeffs, args):
    upper_z = (args.cube_size + args.vertical_gap) * 0.5
    lower_z = -(args.cube_size + args.vertical_gap) * 0.5
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    reference_face = parse_face_order(args.face_order)[0]

    for vertices, color in (
        (
            cube_vertices_from_tag_face(
                reference_face,
                upper_z,
                args.cube_size,
                args.marker_length,
                args.upper_rotation_deg,
                args.wireframe_inward_sign,
            ),
            UPPER_CUBE_COLOR,
        ),
        (
            cube_vertices_from_tag_face(
                reference_face,
                lower_z,
                args.cube_size,
                args.marker_length,
                args.lower_rotation_deg,
                args.wireframe_inward_sign,
            ),
            LOWER_CUBE_COLOR,
        ),
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


def process_frame(frame, dictionary, params, detector, layout, camera_matrix, dist_coeffs, args):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, rejected = detect_markers(gray, dictionary, params, detector)

    marker_count = 0 if marker_ids is None else len(marker_ids)
    ids = [] if marker_ids is None else marker_ids.flatten().astype(int).tolist()
    if marker_count > 0:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    object_points, image_points, used_ids = collect_correspondences(marker_corners, marker_ids, layout)
    pose = estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, args)

    pose_text = "pose: need at least {} configured tags".format(args.min_tags_for_pose)
    if pose is not None:
        rvec, tvec, inliers, mean_error, max_error = pose
        draw_fused_model(frame, rvec, tvec, camera_matrix, dist_coeffs, args)
        tx, ty, tz = tvec.reshape(-1)
        roll, pitch, yaw = rotation_to_euler_xyz(rvec)
        object_to_camera = pose_matrix_object_to_camera(rvec, tvec)
        pose_text = (
            "rigid object pose t=({:.3f},{:.3f},{:.3f})m rpy=({:.1f},{:.1f},{:.1f})deg inliers={} err={:.2f}px"
            .format(tx, ty, tz, roll, pitch, yaw, inliers, mean_error)
        )
        if args.print_pose:
            print(
                "object_pose ids={} t=({:.6f},{:.6f},{:.6f}) rvec=({:.6f},{:.6f},{:.6f}) "
                "rpy=({:.3f},{:.3f},{:.3f}) object_to_camera={} mean_err={:.3f} max_err={:.3f}".format(
                    used_ids,
                    tx,
                    ty,
                    tz,
                    *rvec.reshape(-1),
                    roll,
                    pitch,
                    yaw,
                    format_matrix(object_to_camera),
                    mean_error,
                    max_error,
                ),
                flush=True,
            )

    lines = [
        "ArUco rigid object fused pose  dictionary: {}".format(args.dictionary),
        "detected ids: {}  used ids: {}  rejected: {}".format(ids[:12], used_ids, len(rejected)),
        pose_text,
        "yellow=upper cube  blue=lower cube  cube_size={}m marker_length={}m".format(
            args.cube_size, args.marker_length
        ),
        "wireframe: extruded from tag face along inward normal",
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
        cv2.imshow("detect_aruco_9tags_cube2_fusedpose", frame)
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
    print("Camera: live:{} requested={}x{} autofocus={}".format(args.camera, args.width, args.height, args.autofocus), flush=True)
    print(
        "IDs: upper cube 0-3, lower cube {}; face order: {}; upper rotation={}deg".format(
            sorted(args.lower_ids), args.face_order, args.upper_rotation_deg
        ),
        flush=True,
    )
    if camera_matrix is None:
        print("Calibration: approximate from frame size. Use --calibration for metric pose.", flush=True)
    print("Press ESC or q to quit.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if camera_matrix is None:
            current_camera_matrix, current_dist_coeffs = approximate_calibration(frame.shape[1], frame.shape[0])
        else:
            current_camera_matrix, current_dist_coeffs = camera_matrix, dist_coeffs

        process_frame(frame, dictionary, params, detector, layout, current_camera_matrix, current_dist_coeffs, args)
        cv2.imshow("detect_aruco_9tags_cube2_fusedpose", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect ArUco tags on upper/lower cube faces and estimate one fused model pose."
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
    parser.add_argument("--upper-rotation-deg", type=float, default=45.0, help="Upper cube rotation around model Z axis. Default: 45.")
    parser.add_argument("--lower-rotation-deg", type=float, default=0.0, help="Lower cube rotation around model Z axis. Default: 0.")
    parser.add_argument("--axis-length", type=float, default=0.04, help="Drawn model axis length in meters.")
    parser.add_argument("--cube-line-thickness", type=int, default=3, help="Projected cube wireframe line thickness.")
    parser.add_argument(
        "--wireframe-inward-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help="Extrude wireframe from the tag face along normal sign. Default -1 builds into the cube.",
    )
    parser.add_argument("--aruco3", action="store_true", help="Enable OpenCV ArUco3 detector path when supported.")
    parser.add_argument("--adaptive-min", type=int, default=3)
    parser.add_argument("--adaptive-max", type=int, default=53)
    parser.add_argument("--adaptive-step", type=int, default=10)
    parser.add_argument("--min-marker-perimeter-rate", type=float, default=0.015)
    parser.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    parser.add_argument("--ransac", action="store_true", default=True, help="Use solvePnPRansac when enough points exist.")
    parser.add_argument("--no-ransac", dest="ransac", action="store_false", help="Use plain solvePnP.")
    parser.add_argument("--ransac-iterations", type=int, default=100)
    parser.add_argument("--reprojection-error", type=float, default=5.0)
    parser.add_argument("--min-tags-for-pose", type=int, default=2, help="Minimum configured tags needed for fused object pose.")
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
