#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PROJECT_DIR / "marker_designs"
OUTPUT_DIR = PROJECT_DIR / "output_marker_designs_8_id4"
ID_BITS = 4
CRC_BITS = 16
COUNT = 8

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
ET.register_namespace("cc", "http://creativecommons.org/ns#")
ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def crc16_bits(data):
    crc = 0
    for value in data.encode("ascii"):
        crc ^= value
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
            crc &= 0xFFFF
    return f"{crc:016b}"


def set_style_fill(style, color):
    if "fill:#" not in style:
        raise RuntimeError(f"code element style has no fill color: {style}")
    return re.sub(r"fill:#[0-9a-fA-F]{6}", f"fill:#{color}", style, count=1)


def first_code_layer(root):
    for elem in root.iter():
        if local_name(elem.tag) == "g" and elem.attrib.get("id") == "code":
            return elem
    raise RuntimeError('SVG code layer not found: expected <g id="code">')


def code_shapes(code_layer):
    allowed = {"path", "rect", "ellipse", "circle"}
    return [child for child in list(code_layer) if local_name(child.tag) in allowed]


def code_colors(shapes):
    colors = []
    for shape in shapes:
        style = shape.attrib.get("style", "")
        match = re.search(r"fill:#([0-9a-fA-F]{6})", style)
        if match and match.group(1).lower() not in colors:
            colors.append(match.group(1).lower())
    if len(colors) < 2:
        raise RuntimeError("code layer must contain at least two fill colors")
    return colors[0], colors[1]


def marker_bits(marker_id):
    if marker_id < 1 or marker_id >= (1 << ID_BITS):
        raise ValueError(f"marker ID {marker_id} does not fit in {ID_BITS} bits")
    id_code = f"{marker_id:0{ID_BITS}b}"
    return id_code, crc16_bits(id_code)


def generate_template(template_path):
    tree = ET.parse(template_path)
    root = tree.getroot()
    code_layer = first_code_layer(root)
    shapes = code_shapes(code_layer)
    required = ID_BITS + CRC_BITS
    if len(shapes) < required:
        raise RuntimeError(f"{template_path.name} has {len(shapes)} code shapes; {required} required")

    zero_color, one_color = code_colors(shapes)
    template_name = template_path.stem
    out_dir = OUTPUT_DIR / template_name
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for marker_id in range(1, COUNT + 1):
        out_root = deepcopy(root)
        out_root.attrib["id"] = str(marker_id)
        docname_key = "{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}docname"
        if docname_key in out_root.attrib:
            out_root.attrib[docname_key] = f"{template_name}_{marker_id}.svg"

        out_code_layer = first_code_layer(out_root)
        out_shapes = code_shapes(out_code_layer)
        id_code, crc_code = marker_bits(marker_id)
        all_bits = id_code + crc_code

        for bit_index, bit in enumerate(all_bits):
            shape = out_shapes[bit_index]
            shape.attrib["id"] = f"bit_{bit_index}"
            color = one_color if bit == "1" else zero_color
            shape.attrib["style"] = set_style_fill(shape.attrib.get("style", ""), color)

        out_path = out_dir / f"{template_name}_{marker_id}.svg"
        ET.ElementTree(out_root).write(out_path, encoding="utf-8", xml_declaration=True)
        generated.append((marker_id, id_code, crc_code, out_path))

    return template_name, len(shapes), generated


def main():
    templates = sorted(TEMPLATE_DIR.glob("*.svg"))
    if not templates:
        raise RuntimeError(f"No SVG templates found in {TEMPLATE_DIR}")

    for template_path in templates:
        name, shape_count, generated = generate_template(template_path)
        print(f"{name}: code_shapes={shape_count}, generated={len(generated)}")
        for marker_id, id_code, crc_code, out_path in generated:
            print(f"  ID={marker_id}: id_bits={id_code} crc={crc_code} file={out_path}")


if __name__ == "__main__":
    main()
