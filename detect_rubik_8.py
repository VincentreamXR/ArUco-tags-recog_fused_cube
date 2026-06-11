#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
from pathlib import Path


ZERO_COLOR = "2a4ab7"
ONE_COLOR = "fff000"
CRC_BITS = 16


def resolve_path(path_text, base_dir=None):
    path = Path(path_text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def find_executable(build_dir, executable):
    exe = resolve_path(executable, build_dir)
    if exe.exists():
        return exe

    if exe.suffix.lower() != ".exe":
        exe_win = exe.with_suffix(exe.suffix + ".exe") if exe.suffix else Path(str(exe) + ".exe")
        if exe_win.exists():
            return exe_win

    return exe


def extract_code_layer(svg_text):
    match = re.search(r'(?s)<g\s+[^>]*id="code"[^>]*>.*?</g>', svg_text)
    if not match:
        raise RuntimeError('SVG code layer not found: expected <g id="code">')
    return match


def count_code_elements(svg_path):
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    code = extract_code_layer(text).group(0)
    return len(re.findall(r"<(path|rect|ellipse|circle)\b", code))


def crc16_bits(data):
    crc = 0
    for value in data.encode("ascii"):
        crc ^= value
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
            crc &= 0xFFFF
    return f"{crc:016b}"


def make_bits(marker_id, id_bits):
    if marker_id < 0 or marker_id >= (1 << id_bits):
        raise ValueError(f"marker_id {marker_id} does not fit in {id_bits} bits")
    id_code = f"{marker_id:0{id_bits}b}"
    return id_code, crc16_bits(id_code)


def generate_tags(template_svg, output_dir, count, id_bits):
    text = template_svg.read_text(encoding="utf-8", errors="replace")
    code_match = extract_code_layer(text)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for marker_id in range(1, count + 1):
        id_code, crc_code = make_bits(marker_id, id_bits)
        all_bits = id_code + crc_code

        bit_index = 0

        def replace_fill(match):
            nonlocal bit_index
            if bit_index >= len(all_bits):
                return match.group(0)
            color = ONE_COLOR if all_bits[bit_index] == "1" else ZERO_COLOR
            bit_index += 1
            return f"fill:#{color}"

        new_code = re.sub(r"fill:#[0-9a-fA-F]{6}", replace_fill, code_match.group(0))
        if bit_index != id_bits + CRC_BITS:
            raise RuntimeError(f"changed {bit_index} bits, expected {id_bits + CRC_BITS}")

        output = text[: code_match.start()] + new_code + text[code_match.end() :]
        output = re.sub(r'id="1"', f'id="{marker_id}"', output, count=1)
        output = re.sub(
            r'sodipodi:docname="[^"]+"',
            f'sodipodi:docname="rubikMarker_{marker_id}.svg"',
            output,
            count=1,
        )

        out_path = output_dir / f"rubikMarker_{marker_id}.svg"
        out_path.write_text(output, encoding="utf-8")
        generated.append((marker_id, id_code, crc_code, out_path))

    return generated


def build_command(args):
    script_dir = Path(__file__).resolve().parent
    project_dir = resolve_path(args.project, script_dir)
    build_dir = resolve_path(args.build, project_dir)
    executable = find_executable(build_dir, args.executable)
    svg_path = resolve_path(args.svg, project_dir)
    calibration = resolve_path(args.calibration, project_dir) if args.calibration else None

    if not project_dir.exists():
        raise FileNotFoundError(f"JuMarker project directory not found: {project_dir}")
    if not svg_path.exists():
        raise FileNotFoundError(f"Rubik marker SVG not found: {svg_path}")
    if not executable.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"jumarker_test not found: {executable}\n"
            "Build JuMarker first, then rerun this script."
        )
    if calibration and not calibration.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {calibration}")

    code_elements = count_code_elements(svg_path)
    required = args.id_bits + CRC_BITS
    if code_elements < required:
        raise RuntimeError(
            f"{svg_path.name} has {code_elements} code elements, but id-bits={args.id_bits} needs {required}."
        )

    video_input = args.input if args.input else f"live:{args.camera}"
    cmd = [
        str(executable),
        str(svg_path),
        str(args.id_bits),
        "-v",
        video_input,
        "-t",
        "rubik",
    ]

    if calibration:
        cmd.extend(["-c", str(calibration)])
    if args.resize_factor is not None:
        cmd.extend(["-rf", str(args.resize_factor)])
    if args.vumark:
        cmd.append("--vumark")

    return project_dir, build_dir, svg_path, code_elements, cmd


def run_camera_controls(args):
    if args.no_camera_controls:
        return
    if args.input and not args.input.startswith("live"):
        return

    v4l2_ctl = shutil.which("v4l2-ctl")
    if not v4l2_ctl:
        print("v4l2-ctl not found; skipping Ubuntu camera control setup.")
        return

    camera_index = args.camera
    if args.input and args.input.startswith("live:"):
        try:
            camera_index = int(args.input.split(":", 1)[1])
        except ValueError:
            camera_index = args.camera

    device = f"/dev/video{camera_index}"
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
            print(f"Set camera control: {name}={value}")
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            print(f"Could not set camera control {name}={value}: {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and run JuMarker detection for rubikMarker IDs 1-8."
    )
    parser.add_argument("--project", default=".", help="JuMarker project root. Default: script directory.")
    parser.add_argument("--build", default="build", help="Build directory relative to --project.")
    parser.add_argument("--executable", default="utils/jumarker_test", help="jumarker_test path relative to --build.")
    parser.add_argument("--svg", default="marker_designs/rubikMarker.svg", help="Base rubikMarker SVG template.")
    parser.add_argument("--id-bits", type=int, default=4, help="Use 4 for exact IDs 1-8.")
    parser.add_argument("--count", type=int, default=8, help="How many tag SVGs to generate.")
    parser.add_argument("--tag-dir", default="output_rubik_8_id4", help="Generated SVG tag output directory.")
    parser.add_argument("--no-generate", action="store_true", help="Do not generate SVG tags before detection.")
    parser.add_argument("--input", default=None, help="Input image/video path or live:N. Default: live:<camera>.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index used when --input is omitted.")
    parser.add_argument("--calibration", default="utils/camera_calibration.yml", help="Camera calibration file.")
    parser.add_argument("--resize-factor", type=float, default=None, help="Optional value passed as -rf.")
    parser.add_argument("--vumark", action="store_true", help="Enable legacy VuMark mode. Not for ID 1-8.")
    parser.add_argument("--no-camera-controls", action="store_true", help="Skip Ubuntu v4l2 camera controls.")
    parser.add_argument("--disable-auto-focus", action="store_true", default=True, help="Set focus_auto=0 on Ubuntu.")
    parser.add_argument("--focus", type=int, default=None, help="Optional v4l2 focus_absolute value.")
    parser.add_argument("--manual-exposure", action="store_true", default=True, help="Set exposure_auto=1 on Ubuntu.")
    parser.add_argument("--exposure", type=int, default=None, help="Optional v4l2 exposure_absolute value.")
    parser.add_argument("--gain", type=int, default=None, help="Optional v4l2 gain value.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running detection.")
    args = parser.parse_args()

    project_dir, build_dir, svg_path, code_elements, cmd = build_command(args)
    tag_dir = resolve_path(args.tag_dir, project_dir)

    print(f"JuMarker project: {project_dir}")
    print(f"Rubik template: {svg_path}")
    print(f"Code elements: {code_elements}")
    print(f"ID bits: {args.id_bits}")
    if not Path(cmd[0]).exists():
        print(f"jumarker_test not found yet: {cmd[0]}")

    if not args.no_generate:
        generated = generate_tags(svg_path, tag_dir, args.count, args.id_bits)
        print(f"Generated {len(generated)} SVG tags in: {tag_dir}")
        for marker_id, id_code, crc_code, out_path in generated:
            print(f"  ID {marker_id}: bits={id_code} crc={crc_code} file={out_path.name}")

    print("Command:", " ".join(cmd))
    if args.vumark:
        print("Warning: --vumark forces markerbitsData=1 in markerdetector.cpp; do not use it for IDs 1-8.")
    if args.dry_run:
        return

    run_camera_controls(args)
    subprocess.run(cmd, cwd=str(build_dir), check=True)


if __name__ == "__main__":
    main()
