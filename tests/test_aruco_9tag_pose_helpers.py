import argparse
import sys
import types

import numpy as np

cv2_stub = types.ModuleType("cv2")
cv2_stub.aruco = types.SimpleNamespace(
    DICT_4X4_50=0,
    DICT_4X4_100=1,
    DICT_4X4_250=2,
    DICT_4X4_1000=3,
    DICT_5X5_50=4,
    DICT_5X5_100=5,
    DICT_5X5_250=6,
    DICT_5X5_1000=7,
    DICT_6X6_50=8,
    DICT_6X6_100=9,
    DICT_6X6_250=10,
    DICT_6X6_1000=11,
    DICT_7X7_50=12,
    DICT_7X7_100=13,
    DICT_7X7_250=14,
    DICT_7X7_1000=15,
    DICT_ARUCO_ORIGINAL=16,
)
sys.modules.setdefault("cv2", cv2_stub)

import detect_aruco_cube_rigid_async_pose as pose


def make_args():
    return argparse.Namespace(
        cube_size=0.04,
        vertical_gap=0.0,
        marker_length=0.04,
        face_order="front,right,back,left",
        lower_ids="4,5,6,7",
        top_id=8,
        upper_rotation_deg=45.0,
        lower_rotation_deg=0.0,
        corner_rolls="",
        auto_roll_ids="none",
        id_face_map="0:right,1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:top",
        id_rotation_map="",
        min_tags_for_pose=1,
        roll_switch_penalty=0.0,
        lock_top_pose_to_top_tag=False,
        top_disambiguation_weight=0.0,
        top_disambiguation_retry_deg=35.0,
        auto_roll_max_candidates=64,
        reject_outlier_tags=True,
        tag_outlier_ratio=2.8,
        min_tag_mean_reprojection_error=4.0,
        max_tag_mean_reprojection_error=12.0,
    )


def test_default_layout_contains_all_nine_tags():
    layout = pose.build_marker_layout(make_args())

    assert sorted(layout) == list(range(9))
    for marker_id, corners in layout.items():
        assert corners.shape == (4, 3), marker_id
        assert corners.dtype == np.float32


def test_visible_adjacent_pairs_are_returned_in_stable_order():
    pairs = pose.visible_adjacent_tag_pairs([8, 1, 0, 4, 5])

    assert pairs == [(0, 1), (4, 5), (0, 4), (1, 5), (8, 0), (8, 1)]


def test_pose_candidate_score_prefers_more_tags_when_errors_are_close():
    low_count = pose.score_pose_candidate_for_selection(
        mean_error=2.0,
        used_ids=[0, 1],
        previous_pose=None,
        rvec=None,
        tvec=None,
    )
    high_count = pose.score_pose_candidate_for_selection(
        mean_error=2.1,
        used_ids=[0, 1, 2, 3],
        previous_pose=None,
        rvec=None,
        tvec=None,
    )

    assert high_count < low_count


def test_estimate_best_fused_pose_considers_adjacent_pair_candidates(monkeypatch):
    args = make_args()
    layout = pose.build_marker_layout(args)
    marker_ids = np.array([[0], [1], [2]], dtype=np.int32)
    marker_corners = [
        np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
        np.array([[[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]], dtype=np.float32),
        np.array([[[4.0, 0.0], [5.0, 0.0], [5.0, 1.0], [4.0, 1.0]]], dtype=np.float32),
    ]
    calls = []

    def fake_estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, call_args, initial_guess=None):
        calls.append(tuple(used_ids))
        mean_error = 1.0 if tuple(used_ids) == (0, 1) else 5.0
        return np.zeros((3, 1)), np.zeros((3, 1)), len(object_points), mean_error, mean_error

    monkeypatch.setattr(pose, "estimate_fused_pose", fake_estimate_fused_pose)

    _, _, used_ids, selected_pose, _ = pose.estimate_best_fused_pose(
        marker_corners,
        marker_ids,
        layout,
        camera_matrix=None,
        dist_coeffs=None,
        args=args,
        initial_guess=None,
        previous_rolls={},
    )

    assert tuple(used_ids) == (0, 1)
    assert selected_pose[3] == 1.0
    assert (0, 1) in calls


def test_parse_args_defaults_enable_nine_tag_overlay(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["detect_aruco_cube_rigid_async_pose.py"])

    args = pose.parse_args()

    assert args.cube_size == 0.04
    assert args.vertical_gap == 0.0
    assert args.lower_ids == "4,5,6,7"
    assert args.upper_rotation_deg == 45.0
    assert args.lower_rotation_deg == 0.0
    assert args.id_face_map == "0:right,1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:top"
    assert args.auto_roll_ids == "none"
    assert args.min_tags_for_pose == 1
    assert args.draw_prism_wireframe is True
    assert args.draw_model_tags is False
    assert args.draw_detected_corners is True
    assert args.draw_corner_index is True
    assert args.reject_outlier_tags is True
    assert args.tag_outlier_ratio == 2.8
    assert args.min_tag_mean_reprojection_error == 4.0
    assert args.max_tag_mean_reprojection_error == 12.0


def test_tag_outlier_selection_drops_only_high_error_tags():
    args = make_args()
    tag_errors = {
        0: (1.2, 1.6),
        1: (1.5, 1.9),
        2: (28.0, 34.0),
    }

    inlier_ids = pose.select_inlier_tag_ids(tag_errors, [0, 1, 2], args)

    assert inlier_ids == [0, 1]


def test_estimate_best_fused_pose_refines_without_high_error_tag(monkeypatch):
    args = make_args()
    layout = pose.build_marker_layout(args)
    marker_ids = np.array([[0], [1], [2]], dtype=np.int32)
    marker_corners = [
        np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
        np.array([[[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]]], dtype=np.float32),
        np.array([[[30.0, 0.0], [31.0, 0.0], [31.0, 1.0], [30.0, 1.0]]], dtype=np.float32),
    ]
    calls = []

    def fake_estimate_fused_pose(object_points, image_points, used_ids, camera_matrix, dist_coeffs, call_args, initial_guess=None):
        calls.append(tuple(used_ids))
        mean_error = 18.0 if tuple(used_ids) == (0, 1, 2) else 1.0
        return np.zeros((3, 1)), np.zeros((3, 1)), len(object_points), mean_error, mean_error

    monkeypatch.setattr(pose, "estimate_fused_pose", fake_estimate_fused_pose)
    monkeypatch.setattr(
        pose,
        "compute_tag_reprojection_errors",
        lambda *args, **kwargs: {0: (1.0, 1.0), 1: (1.2, 1.2), 2: (28.0, 34.0)},
    )

    _, _, used_ids, selected_pose, _ = pose.estimate_best_fused_pose(
        marker_corners,
        marker_ids,
        layout,
        camera_matrix=None,
        dist_coeffs=None,
        args=args,
        initial_guess=None,
        previous_rolls={},
    )

    assert tuple(used_ids) == (0, 1)
    assert selected_pose[3] == 1.0
    assert (0, 1, 2) in calls


def test_draw_detected_corners_labels_c0_to_c3(monkeypatch):
    labels = []
    circles = []

    monkeypatch.setattr(pose.cv2, "LINE_AA", 16, raising=False)
    monkeypatch.setattr(pose.cv2, "FONT_HERSHEY_SIMPLEX", 0, raising=False)
    monkeypatch.setattr(pose.cv2, "circle", lambda *args, **kwargs: circles.append(args), raising=False)
    monkeypatch.setattr(pose.cv2, "putText", lambda image, text, *args, **kwargs: labels.append(text), raising=False)

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    marker_ids = np.array([[7]], dtype=np.int32)
    marker_corners = [
        np.array([[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]]], dtype=np.float32)
    ]
    args = argparse.Namespace(corner_radius=4, draw_corner_index=True)

    pose.draw_detected_corners(frame, marker_corners, marker_ids, args)

    assert len(circles) == 4
    assert labels == ["7:c0", "7:c1", "7:c2", "7:c3"]


def test_process_frame_draws_raw_model_when_tracker_rejects_current_pose(monkeypatch):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    marker_ids = np.array([[0]], dtype=np.int32)
    marker_corners = [
        np.array([[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]]], dtype=np.float32)
    ]
    raw_rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
    raw_tvec = np.array([[0.01], [0.02], [0.30]], dtype=np.float64)
    draw_calls = []

    class RejectingTracker:
        last_rolls = {}

        def get_initial_guess(self):
            return None

        def update(self, *args, **kwargs):
            return None, "held_bad_reproj"

        def get_held_pose(self):
            return None

    monkeypatch.setattr(pose.cv2, "COLOR_BGR2GRAY", 0, raising=False)
    monkeypatch.setattr(pose.cv2, "cvtColor", lambda image, code: image, raising=False)
    monkeypatch.setattr(pose.cv2.aruco, "drawDetectedMarkers", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(pose, "detect_markers", lambda *args, **kwargs: (marker_corners, marker_ids, []))
    monkeypatch.setattr(
        pose,
        "estimate_best_fused_pose",
        lambda *args, **kwargs: (
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
            [0],
            (raw_rvec, raw_tvec, 4, 18.0, 20.0),
            {},
        ),
    )
    monkeypatch.setattr(
        pose,
        "draw_fused_model",
        lambda image, layout, rvec, tvec, camera_matrix, dist_coeffs, args: draw_calls.append((rvec.copy(), tvec.copy())),
    )
    monkeypatch.setattr(pose, "rotation_to_euler_xyz", lambda rvec: np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(pose, "pose_matrix_object_to_camera", lambda rvec, tvec: np.eye(4))
    monkeypatch.setattr(pose, "draw_text_panel", lambda image, lines: None)

    args = argparse.Namespace(
        min_tags_for_pose=1,
        draw_detected_corners=False,
        print_pose=False,
        dictionary="DICT_6X6_250",
        prism_width=0.04,
        prism_height=0.08,
        prism_depth=0.04,
        marker_length=0.04,
        hold_last_seconds=0.25,
    )

    pose.process_frame(
        frame,
        dictionary=None,
        params=None,
        detector=None,
        layout={0: np.zeros((4, 3), dtype=np.float32)},
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        args=args,
        pose_tracker=RejectingTracker(),
    )

    assert len(draw_calls) == 1
    np.testing.assert_allclose(draw_calls[0][0], raw_rvec)
    np.testing.assert_allclose(draw_calls[0][1], raw_tvec)
