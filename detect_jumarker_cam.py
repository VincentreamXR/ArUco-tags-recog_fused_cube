#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path


def count_svg_bits(svg_path):
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r'id="bit_', text))


def resolve_path(path_text, base_dir=None):
    path = Path(path_text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def resolve_svg_paths(args, project_dir):
    if args.svg:
        return [resolve_path(args.svg, project_dir)]

    svg_dir = resolve_path(args.svg_dir, project_dir)
    if not svg_dir.exists():
        raise FileNotFoundError(f"Marker SVG directory not found: {svg_dir}")
    if not svg_dir.is_dir():
        raise NotADirectoryError(f"--svg-dir is not a directory: {svg_dir}")

    svg_paths = []
    for subdir in sorted(path for path in svg_dir.iterdir() if path.is_dir()):
        preferred = subdir / f"{subdir.name}_1.svg"
        if preferred.exists():
            svg_paths.append(preferred.resolve())
            continue

        candidates = sorted(subdir.glob("*.svg"))
        if candidates:
            svg_paths.append(candidates[0].resolve())

    if not svg_paths:
        raise FileNotFoundError(f"No marker SVG templates found under: {svg_dir}")
    return svg_paths


def build_command(args):
    project_dir = resolve_path(args.project)
    build_dir = resolve_path(args.build, project_dir)
    executable = resolve_path(args.executable, build_dir)
    svg_paths = resolve_svg_paths(args, project_dir)

    if not project_dir.exists():
        raise FileNotFoundError(f"JuMarker project directory not found: {project_dir}")
    if not executable.exists():
        raise FileNotFoundError(
            f"jumarker_test not found: {executable}\n"
            "Compile first from the JuMarker project root:\n"
            "  mkdir -p build\n"
            "  cd build\n"
            "  cmake ..\n"
            "  make"
        )
    for svg_path in svg_paths:
        if not svg_path.exists():
            raise FileNotFoundError(f"Marker SVG not found: {svg_path}")

    required_bits = args.id_bits + 16
    bit_counts = []
    for svg_path in svg_paths:
        bit_count = count_svg_bits(svg_path)
        bit_counts.append(bit_count)
        if bit_count < required_bits:
            raise RuntimeError(
                f"{svg_path} has only {bit_count} code bits. JuMarker needs {required_bits}: "
                "ID bits plus 16 CRC bits."
            )

    live_source = f"live:{args.camera}"
    cmd = [
        str(executable),
        ",".join(str(path) for path in svg_paths),
        str(args.id_bits),
        "-v",
        live_source,
        "-t",
        args.marker_type,
    ]

    if args.calibration:
        calib_path = resolve_path(args.calibration, project_dir)
        if not calib_path.exists():
            raise FileNotFoundError(f"Camera calibration file not found: {calib_path}")
        cmd.extend(["-c", str(calib_path)])

    if args.resize_factor is not None:
        cmd.extend(["-rf", str(args.resize_factor)])

    return project_dir, build_dir, svg_paths, bit_counts, cmd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run JuMarker camera detection on Linux. This script wraps the compiled "
            "JuMarker C++ detector because the release does not provide Python bindings."
        )
    )
    parser.add_argument(
        "--project",
        default="~/jumarker_release",
        help="JuMarker project root directory. Default: ~/jumarker_release",
    )
    parser.add_argument(
        "--build",
        default="build",
        help="Build directory, relative to --project unless absolute. Default: build",
    )
    parser.add_argument(
        "--executable",
        default="utils/jumarker_test",
        help="jumarker_test path, relative to --build unless absolute.",
    )
    parser.add_argument(
        "--svg",
        default=None,
        help="Single marker template SVG, relative to --project unless absolute. Overrides --svg-dir.",
    )
    parser.add_argument(
        "--svg-dir",
        default="output_marker_designs_8_id4",
        help=(
            "Directory containing generated marker template subdirectories. "
            "Default loads one SVG from each class in output_marker_designs_8_id4."
        ),
    )
    parser.add_argument(
        "--id-bits",
        type=int,
        default=4,
        help="ID bits used when generating markers. For IDs 1-8, use 4.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. JuMarker receives this as -v live:<index>.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional camera calibration XML/YML file, relative to --project unless absolute.",
    )
    parser.add_argument(
        "--marker-type",
        default="building",
        help="Value passed to jumarker_test -t. Default matches the release default: building.",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=None,
        help="Optional value passed to jumarker_test -rf, for example 0.5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running detection.",
    )

    args = parser.parse_args()
    project_dir, build_dir, svg_paths, bit_counts, cmd = build_command(args)

    print(f"JuMarker project: {project_dir}", flush=True)
    print(f"Build directory: {build_dir}", flush=True)
    print(f"Marker SVG templates: {len(svg_paths)}", flush=True)
    for svg_path, bit_count in zip(svg_paths, bit_counts):
        print(f"  {svg_path} ({bit_count} code bits)", flush=True)
    print(f"ID bits: {args.id_bits}", flush=True)
    print(f"Camera: live:{args.camera}", flush=True)
    print("Press ESC in the JuMarker window to quit.", flush=True)
    print("Command:", " ".join(cmd), flush=True)

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=str(build_dir), check=True)


if __name__ == "__main__":
    main()
