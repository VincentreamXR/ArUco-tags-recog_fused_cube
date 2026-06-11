# ArUco 9-Tag Fused Pose Design

## Context

The current recommended entry point is `detect_aruco_cube_rigid_async_pose.py`, but its active layout only describes a single 4 cm cube with side IDs `0,1,2,3` and top ID `8`. The physical object and README describe a 9-tag rigid object: upper side IDs `0,1,2,3`, lower side IDs `4,5,6,7`, and top ID `8`. When IDs `4-7` are detected but absent from the model, their detected boxes cannot improve the fused pose.

## Goal

Update `detect_aruco_cube_rigid_async_pose.py` so the main script estimates and renders one stable rigid-object pose for the full 9-tag object. It should also show each detected tag corner as `id:c0` through `id:c3`, making corner order and face mapping errors visible during live testing.

## Non-Goals

This change does not add a new detector, change camera calibration, or average independent per-tag poses. It keeps one object-level pose produced by `solvePnP` from matched 3D object corners and 2D image corners.

## Object Layout

The object is modeled as two stacked 4 cm cubes:

- Upper cube center: `z = 0.0`.
- Lower cube center: `z = -(cube_size + vertical_gap)`.
- Upper side tags: IDs `0,1,2,3`.
- Lower side tags: IDs `4,5,6,7`.
- Top tag: ID `8`.
- Default face mapping: `0:right,1:front,2:left,3:back,4:right,5:front,6:left,7:back,8:top`.
- Default cube wireframe is visible.

Each tag contributes four 3D model corners in object coordinates. The OpenCV-detected image corners are paired with these model corners in the same `c0-c3` order after optional per-ID corner rolls.

## Pose Strategy

The pose solver keeps the existing global path: collect all visible configured tags and run one object-level `solvePnP`. This is the preferred output whenever it succeeds.

Adjacent two-face visibility improves robustness because the visible corners are no longer coplanar. The solver should explicitly build and score candidate poses from neighboring tag pairs when two adjacent faces are visible. Candidate pairs include the upper side ring, lower side ring, vertical same-face pairs, and top-to-upper-side pairs:

- Upper side ring: `(0,1)`, `(1,2)`, `(2,3)`, `(3,0)`.
- Lower side ring: `(4,5)`, `(5,6)`, `(6,7)`, `(7,4)`.
- Vertical pairs: `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`.
- Top pairs: `(8,0)`, `(8,1)`, `(8,2)`, `(8,3)`.

The solver scores all valid candidates by reprojection error, number of visible tags, and consistency with the previous pose. The final pose is the lowest-scoring valid candidate, with all-visible global solve favored when its reprojection error is acceptable.

## Overlay

The live/image overlay should draw:

- ArUco detected marker borders.
- Per-corner colored dots and labels formatted as `id:c0`, `id:c1`, `id:c2`, `id:c3`.
- Fused cube wireframes for the upper and lower 4 cm cubes.
- One fused object coordinate axis.
- A text line reporting detected IDs, used IDs, candidate source, reprojection error, and pose state.

## Error Handling

If no configured tag is visible, the tracker may hold the last stable pose for `hold_last_seconds`. If a candidate exceeds `max_stable_reprojection_error`, the tracker rejects it and holds the previous stable pose when available. Unconfigured detected IDs remain visible as detected boxes and corner labels but do not contribute to pose.

## Testing

Add focused tests around pure geometry and candidate selection:

- The default 9-tag layout contains IDs `0-8`.
- Each layout entry has exactly four 3D corners.
- Adjacent pair extraction returns expected pairs for visible ID sets.
- The pose candidate ranking prefers a lower reprojection error and favors candidates with more tags when errors are close.

Run syntax checks for the modified Python script after implementation.
