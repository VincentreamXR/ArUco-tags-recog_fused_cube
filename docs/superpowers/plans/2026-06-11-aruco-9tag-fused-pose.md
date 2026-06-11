# ArUco 9-Tag Fused Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the main ArUco pose script so the full 9-tag rigid object contributes to one fused pose, detected tag corners are labeled `id:c0-c3`, and adjacent two-face observations improve pose robustness.

**Architecture:** Keep the single-script runtime path in `detect_aruco_cube_rigid_async_pose.py`. Add small pure helper functions for 9-tag geometry, adjacent pair selection, and candidate scoring so behavior can be tested without camera hardware. Reuse the existing `solvePnP`, tracker, and overlay flow.

**Tech Stack:** Python 3, OpenCV ArUco, NumPy, pytest-style pure tests plus Python syntax checks.

---

### Task 1: Add Geometry Tests

**Files:**
- Create: `tests/test_aruco_9tag_pose_helpers.py`
- Modify: none
- Test: `tests/test_aruco_9tag_pose_helpers.py`

- [ ] **Step 1: Write the failing tests**

```python
import argparse
import numpy as np

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
    low_count = pose.score_pose_candidate_for_selection(mean_error=2.0, used_ids=[0, 1], previous_pose=None, rvec=None, tvec=None)
    high_count = pose.score_pose_candidate_for_selection(mean_error=2.1, used_ids=[0, 1, 2, 3], previous_pose=None, rvec=None, tvec=None)

    assert high_count < low_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_aruco_9tag_pose_helpers.py -q`

Expected: tests fail because `visible_adjacent_tag_pairs` and `score_pose_candidate_for_selection` do not exist and the main layout only returns five tags.

### Task 2: Implement 9-Tag Layout Helpers

**Files:**
- Modify: `detect_aruco_cube_rigid_async_pose.py`
- Test: `tests/test_aruco_9tag_pose_helpers.py`

- [ ] **Step 1: Replace single-cube layout defaults**

Change `build_marker_layout(args)` so it:

```python
def get_cube_center_z(args):
    upper_z = 0.0
    lower_z = -(args.cube_size + args.vertical_gap)
    return upper_z, lower_z
```

and builds IDs `0-3`, configured lower IDs `4-7`, and `top_id`.

- [ ] **Step 2: Update parser defaults**

Set these defaults in `parse_args()`:

```python
parser.add_argument("--cube-size", type=float, default=0.04, help="Single cube side length in meters. Default: 0.04.")
parser.add_argument("--vertical-gap", type=float, default=0.0, help="Gap between lower and upper cube in meters.")
parser.add_argument("--marker-length", type=float, default=0.04, help="ArUco marker side length in meters. Default: 0.04.")
parser.add_argument("--lower-ids", default="4,5,6,7", help="Lower cube ArUco IDs actually present. Default: 4,5,6,7.")
parser.add_argument("--upper-rotation-deg", type=float, default=45.0, help="Upper cube rotation around model Z axis. Default: 45.")
parser.add_argument("--lower-rotation-deg", type=float, default=0.0, help="Lower cube rotation around model Z axis. Default: 0.")
parser.add_argument("--id-face-map", default="0:right,1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:top", help="Override ID to cube face assignment.")
```

- [ ] **Step 3: Run layout tests**

Run: `python3 -m pytest tests/test_aruco_9tag_pose_helpers.py::test_default_layout_contains_all_nine_tags -q`

Expected: the layout test passes.

### Task 3: Add Adjacent Pair Candidate Selection

**Files:**
- Modify: `detect_aruco_cube_rigid_async_pose.py`
- Test: `tests/test_aruco_9tag_pose_helpers.py`

- [ ] **Step 1: Add stable adjacent pair helper**

Add:

```python
ADJACENT_TAG_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (8, 0), (8, 1), (8, 2), (8, 3),
]


def visible_adjacent_tag_pairs(visible_ids):
    visible = {int(marker_id) for marker_id in visible_ids}
    return [(a, b) for a, b in ADJACENT_TAG_PAIRS if a in visible and b in visible]
```

- [ ] **Step 2: Add candidate scoring helper**

Add:

```python
def score_pose_candidate_for_selection(mean_error, used_ids, previous_pose=None, rvec=None, tvec=None):
    tag_bonus = 0.35 * max(0, len(set(used_ids)) - 1)
    return float(mean_error) - tag_bonus
```

- [ ] **Step 3: Extend `estimate_best_fused_pose`**

Keep the all-visible candidate and add adjacent-pair candidates from `visible_adjacent_tag_pairs(used_ids)`. For each pair, collect only those two tags' corners and call `estimate_fused_pose`. Select the candidate with the lowest score from `score_pose_candidate_for_selection`.

- [ ] **Step 4: Run candidate tests**

Run: `python3 -m pytest tests/test_aruco_9tag_pose_helpers.py -q`

Expected: all helper tests pass.

### Task 4: Add Corner Labels and Default Cube Wireframe

**Files:**
- Modify: `detect_aruco_cube_rigid_async_pose.py`
- Test: syntax check

- [ ] **Step 1: Add corner overlay constants and function**

Add colored corner dots and labels:

```python
DETECTED_CORNER_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 255, 0),
]


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
            cv2.circle(frame, center, args.corner_radius, DETECTED_CORNER_COLORS[corner_index % 4], -1, cv2.LINE_AA)
            if args.draw_corner_index:
                cv2.putText(frame, "{}:c{}".format(marker_id, corner_index), (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
```

- [ ] **Step 2: Call overlay in `process_frame`**

After `drawDetectedMarkers`, call:

```python
if marker_count > 0 and args.draw_detected_corners:
    draw_detected_corners(frame, marker_corners, marker_ids, args)
```

- [ ] **Step 3: Add overlay parser flags**

Add:

```python
parser.add_argument("--draw-detected-corners", action="store_true", default=True, help="Draw detected ArUco corner dots. Default: on.")
parser.add_argument("--no-draw-detected-corners", dest="draw_detected_corners", action="store_false", help="Hide detected corner dots.")
parser.add_argument("--draw-corner-index", action="store_true", default=True, help="Draw marker-id:corner-index labels. Default: on.")
parser.add_argument("--no-draw-corner-index", dest="draw_corner_index", action="store_false", help="Hide marker-id:corner-index labels.")
parser.add_argument("--corner-radius", type=int, default=4, help="Detected corner dot radius in pixels.")
```

- [ ] **Step 4: Make wireframe visible by default**

Change `--draw-prism-wireframe` to default true and add `--no-draw-prism-wireframe`.

- [ ] **Step 5: Run syntax check**

Run: `python3 -m py_compile detect_aruco_cube_rigid_async_pose.py`

Expected: no output and exit code 0.

### Task 5: Verification

**Files:**
- Modify: none
- Test: `tests/test_aruco_9tag_pose_helpers.py`, `detect_aruco_cube_rigid_async_pose.py`

- [ ] **Step 1: Run unit tests**

Run: `python3 -m pytest tests/test_aruco_9tag_pose_helpers.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run syntax check**

Run: `python3 -m py_compile detect_aruco_cube_rigid_async_pose.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Inspect git diff**

Run: `git diff -- detect_aruco_cube_rigid_async_pose.py tests/test_aruco_9tag_pose_helpers.py docs/superpowers`

Expected: diff only contains the 9-tag layout, adjacent candidate scoring, corner overlay, tests, and docs.
