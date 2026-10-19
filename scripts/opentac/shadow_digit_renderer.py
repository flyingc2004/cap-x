"""CaP-X-only DIGIT subprocess: original TACTO optics, explicit piano config.

The native OpenTacBench DIGIT IPC exposes RGB only.  This bridge returns the
RGB and TACTO's simulated depth from the *same* renderer.render() call.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata, util
from pathlib import Path


def load_renderer_class():
    package = metadata.distribution("tacto")
    spec = util.spec_from_file_location("digit_tacto_renderer", package.locate_file("tacto/renderer.py"))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Renderer


def build_renderer(root: Path, config: Path, width: int):
    import cv2
    import trimesh

    background = cv2.imread(str(root / "assets/examples/conf/bg_digit_240_320.jpg"))
    if background is None:
        raise RuntimeError("Original DIGIT reference image is missing")
    renderer = load_renderer_class()(width, width * 3 // 4, background, str(config))
    case = trimesh.load(root / "assets/meshes/digit.STL")
    gel = renderer._generate_gel_trimesh()
    if not (.015 < case.extents.min() and case.extents.max() < .065):
        raise RuntimeError(f"Unexpected DIGIT mesh units: {case.extents}")
    return renderer, case, gel


def run_bridge(root: Path, config: Path, width: int):
    import numpy as np
    import trimesh

    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    renderer, case, gel = build_renderer(root, config, width)
    ready = dict(tacto_version=metadata.version("tacto"),
                 case_vertices=case.vertices.tolist(), case_faces=case.faces.tolist(),
                 gel_vertices=gel.vertices.tolist(), gel_faces=gel.faces.tolist(),
                 case_size_m=case.extents.tolist(), config_path=str(config))
    output.write(json.dumps(ready).encode() + b"\n")
    output.flush()
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "close":
                break
            if request["command"] == "setup":
                for name, size in request["keys"].items():
                    renderer.add_object(trimesh.creation.box(extents=size), name)
                output.write(b'{"ready":true}\n')
                output.flush()
                continue
            if request["command"] != "render":
                raise ValueError(f"Unknown DIGIT command: {request['command']}")
            renderer.update_camera_pose(*request["sensor"])
            poses = {name: value[:2] for name, value in request["objects"].items()}
            forces = {name: value[2] for name, value in request["objects"].items() if value[2] > .01}
            colors, depths = renderer.render(object_poses=poses, normal_forces=forces)
            rgb = np.ascontiguousarray(colors[0], dtype=np.uint8)
            depth = np.ascontiguousarray(depths[0], dtype=np.float32)
            reference = np.ascontiguousarray(renderer.depth0[0], dtype=np.float32)
            valid = np.isfinite(depth) & np.isfinite(reference) & (reference > 0) & (depth > 0)
            indentation = float(np.maximum(0., reference[valid] - depth[valid]).max(initial=0.))
            header = dict(rgb_bytes=rgb.nbytes, depth_bytes=depth.nbytes,
                          reference_bytes=reference.nbytes, indentation_m=indentation)
            output.write(json.dumps(header).encode() + b"\n")
            output.write(rgb.tobytes())
            output.write(depth.tobytes())
            output.write(reference.tobytes())
            output.flush()
    finally:
        renderer.r.delete()


def render_force_ramp(root: Path, config: Path, output_dir: Path, width: int = 320):
    """Fixed-geometry optical check; requires a working EGL GPU, no Isaac Kit."""
    import numpy as np
    import trimesh
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    renderer, _, _ = build_renderer(root, config, width)
    renderer.add_object(trimesh.creation.box(extents=(.02, .025, .015)), "key")
    renderer.update_camera_pose([0., 0., 0.], [0., 0., 0.])
    values = [0., .35, .5, .7, 1., 1.5, 2., 2.5]
    rows = []
    baseline_rgb = None
    try:
        for force in values:
            colors, depths = renderer.render(
                object_poses={"key": [[.035, 0., .015], [0., 0., 0.]]},
                normal_forces={"key": force} if force > .01 else {}, noise=False)
            rgb = np.asarray(colors[0], np.uint8)
            depth = np.asarray(depths[0], np.float32)
            reference = np.asarray(renderer.depth0[0], np.float32)
            valid = np.isfinite(depth) & np.isfinite(reference) & (depth > 0) & (reference > 0)
            if baseline_rgb is None:
                baseline_rgb = rgb.astype(np.float32)
            Image.fromarray(rgb).save(output_dir / f"force_{force:g}_N.png")
            rows.append(dict(force_N=force, valid_depth_fraction=float(valid.mean()),
                             rgb_delta=float(np.mean(np.abs(rgb.astype(np.float32) - baseline_rgb))),
                             depth_delta_m=float(np.mean(np.abs(depth[valid] - reference[valid]))) if valid.any() else None))
    finally:
        renderer.r.delete()
    (output_dir / "ramp.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_bridge(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 5 and sys.argv[1] == "--ramp":
        render_force_ramp(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit("Usage: shadow_digit_renderer.py ROOT CONFIG WIDTH | --ramp ROOT CONFIG OUTPUT")
