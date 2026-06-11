#!/usr/bin/env python3
import argparse
import re
import shutil
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


def build_command(args):
    project_dir = resolve_path(args.project)
    build_dir = resolve_path(args.build, project_dir)
    executable = resolve_path(args.executable, build_dir)
    svg_path = resolve_path(args.svg, project_dir)

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
    if not svg_path.exists():
        raise FileNotFoundError(f"Marker SVG not found: {svg_path}")

    bit_count = count_svg_bits(svg_path)
    if not args.vumark and 0 < bit_count < args.id_bits + 16:
        raise RuntimeError(
            f"SVG has only {bit_count} code bits. JuMarker needs ID bits plus 16 CRC bits."
        )

    cmd = [
        str(executable),
        str(svg_path),
        str(args.id_bits),
        "-v",
        f"live:{args.camera}",
        "-t",
        args.marker_type,
    ]

    if args.vumark:
        cmd.append("--vumark")

    if args.calibration:
        calib_path = resolve_path(args.calibration, project_dir)
        if not calib_path.exists():
            raise FileNotFoundError(f"Camera calibration file not found: {calib_path}")
        cmd.extend(["-c", str(calib_path)])

    if args.resize_factor is not None:
        cmd.extend(["-rf", str(args.resize_factor)])

    return project_dir, build_dir, svg_path, bit_count, cmd


def run_camera_controls(args):
    if args.no_camera_controls:
        return

    v4l2_ctl = shutil.which("v4l2-ctl")
    if not v4l2_ctl:
        print("v4l2-ctl not found; skipping camera control setup.", flush=True)
        return

    device = f"/dev/video{args.camera}"
    controls = []
    if args.disable_auto_focus:
        controls.append(("focus_auto", 0))
    if args.focus is not None:
        controls.append(("focus_absolute", args.focus))
    if args.manual_exposure:
        controls.append(("exposure_auto", 1))
    if args.exposure is not None:
        controls.append(("exposure_absolute", args.exposure))
    if args.gain is not None:
        controls.append(("gain", args.gain))

    for name, value in controls:
        cmd = [v4l2_ctl, "-d", device, f"--set-ctrl={name}={value}"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Set camera control: {name}={value}", flush=True)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            print(f"Could not set camera control {name}={value}: {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run JuMarker camera detection for marker_designs/seaberyMarker.svg on Linux. "
            "This wraps the compiled C++ jumarker_test binary."
        )
    )
    parser.add_argument("--project", default=".", help="JuMarker project root directory. Default: current directory.")
    parser.add_argument("--build", default="build", help="Build directory, relative to --project unless absolute.")
    parser.add_argument("--executable", default="utils/jumarker_test", help="jumarker_test path relative to --build.")
    parser.add_argument("--svg", default="marker_designs/seaberyMarker.svg", help="Seabery marker SVG.")
    parser.add_argument("--id-bits", type=int, default=1, help="ID bits. VuMark mode internally uses 1.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, passed as -v live:<index>.")
    parser.add_argument("--calibration", default="utils/camera_calibration.yml", help="Camera calibration XML/YML.")
    parser.add_argument("--marker-type", default="seabery", help="Value passed to jumarker_test -t.")
    parser.add_argument("--vumark", action="store_true", default=True, help="Enable JuMarker VuMark branch.")
    parser.add_argument(
        "--normal-jumarker",
        action="store_false",
        dest="vumark",
        help="Disable VuMark branch. Use this for 8 generated ID markers from create_marker.",
    )
    parser.add_argument("--resize-factor", type=float, default=None, help="Optional value passed as -rf.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running detection.")
    parser.add_argument("--no-camera-controls", action="store_true", help="Skip v4l2 camera controls.")
    parser.add_argument("--disable-auto-focus", action="store_true", default=True, help="Set focus_auto=0.")
    parser.add_argument("--focus", type=int, default=None, help="Optional v4l2 focus_absolute value.")
    parser.add_argument("--manual-exposure", action="store_true", default=True, help="Set exposure_auto=1.")
    parser.add_argument("--exposure", type=int, default=None, help="Optional v4l2 exposure_absolute value.")
    parser.add_argument("--gain", type=int, default=None, help="Optional v4l2 gain value.")

    args = parser.parse_args()
    project_dir, build_dir, svg_path, bit_count, cmd = build_command(args)

    print(f"JuMarker project: {project_dir}", flush=True)
    print(f"Build directory: {build_dir}", flush=True)
    print(f"Marker SVG: {svg_path}", flush=True)
    print(f"SVG bit_ elements: {bit_count}", flush=True)
    print(f"ID bits: {args.id_bits}", flush=True)
    print(f"Marker type: {args.marker_type}", flush=True)
    print(f"VuMark mode: {'on' if args.vumark else 'off'}", flush=True)
    print(f"Camera: live:{args.camera}", flush=True)
    print("Press ESC in the JuMarker window to quit.", flush=True)
    print("Command:", " ".join(cmd), flush=True)

    if args.dry_run:
        return

    run_camera_controls(args)
    subprocess.run(cmd, cwd=str(build_dir), check=True)


if __name__ == "__main__":
    main()
