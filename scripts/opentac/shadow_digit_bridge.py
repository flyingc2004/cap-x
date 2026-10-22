"""Swap only the native DIGIT presentation backend during a CaP-X pilot.

Geometry, contact filtering, native Task.step(), score, and physics remain in
OpenTacBench.  The native module class binding is restored after task creation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np


def renderer_environment(source: dict[str, str]) -> dict[str, str]:
    environment = dict(source)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    environment.update(PYTHONNOUSERSITE="1", PYOPENGL_PLATFORM="egl")
    # CUDA_VISIBLE_DEVICES remaps CUDA ordinals; EGL indices remain physical.
    device = environment.get("EGL_DEVICE_ID") or environment.get("CAPX_GPU")
    if device is None or not device.isdecimal():
        raise RuntimeError("DIGIT EGL renderer needs a numeric EGL_DEVICE_ID or CAPX_GPU")
    environment["EGL_DEVICE_ID"] = device
    return environment


def make_digit_bridge_class(native_cls, *, opentac_root: Path,
                            piano_config: Path, episode_dir: Path):
    root = Path(opentac_root) / ".cache/piano_tactile/digit"
    python = root / "venv/bin/python"
    script = Path(__file__).with_name("shadow_digit_renderer.py")
    if not python.is_file() or not piano_config.is_file() or not script.is_file():
        raise FileNotFoundError("DIGIT Python, piano config, or CaP-X renderer is missing")

    class PilotDigitTactile(native_cls):
        def __init__(self, task):
            self.task = task
            self.image_shape = (task.cfg.tactile_image_width * 3 // 4,
                                task.cfg.tactile_image_width)
            self.peak_indentation = 0.
            self.depth_m = None
            self.reference_depth_m = None
            environment = renderer_environment(os.environ)
            egl_device = environment["EGL_DEVICE_ID"]
            log_path = episode_dir / "renderer.log"
            print(f"[capx-shadow] DIGIT EGL_DEVICE_ID={egl_device}", flush=True)
            self._log = log_path.open("w")
            self._process = subprocess.Popen(
                [str(python), str(script), str(root), str(piano_config),
                 str(task.cfg.tactile_image_width)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                env=environment)
            try:
                self._geometry = self._receive()
                if self._geometry.get("config_path") != str(piano_config):
                    raise RuntimeError("DIGIT renderer did not load the piano config")
                self._mount()
                self._send(dict(command="setup", keys={str(key.midi): key.size for key in task.layout}))
                self._receive()
            except BaseException as exc:
                print(
                    f"[capx-shadow] DIGIT renderer failed on EGL device {egl_device}; "
                    f"exit_code={self._process.poll()} log={log_path}: {exc}",
                    flush=True,
                )
                self.close()
                if isinstance(exc, Exception):
                    raise RuntimeError(f"DIGIT renderer initialization failed; see {log_path}") from exc
                raise

        def observation(self):
            from isaaclab.utils.math import matrix_from_quat
            from scipy.spatial.transform import Rotation

            task = self.task
            position = task.hand.data.body_pos_w[0, self._body].cpu().numpy()
            rotation = matrix_from_quat(task.hand.data.body_quat_w[0, self._body]).cpu().numpy()
            forces = self.key_forces()
            objects = {}
            for i, key in enumerate(task.layout):
                index = task._key_order[i]
                pose = task.keys.data.body_pos_w[index, self._key_body].cpu().numpy()
                orientation = matrix_from_quat(task.keys.data.body_quat_w[index, self._key_body]).cpu().numpy()
                objects[str(key.midi)] = [pose.tolist(),
                                          Rotation.from_matrix(orientation).as_euler("xyz").tolist(),
                                          float(forces[i])]
            self._send(dict(command="render", sensor=[position.tolist(),
                        Rotation.from_matrix(rotation).as_euler("xyz").tolist()],
                        objects=objects))
            header = self._receive()
            def read_exact(size):
                result = self._process.stdout.read(size)
                if len(result) != size:
                    raise RuntimeError("Incomplete CaP-X DIGIT RGB/depth frame")
                return result
            height, width = self.image_shape
            if (header.get("rgb_bytes") != height * width * 3 or
                    header.get("depth_bytes") != height * width * 4 or
                    header.get("reference_bytes") != height * width * 4):
                raise RuntimeError("Unexpected CaP-X DIGIT frame byte counts")
            rgb = np.frombuffer(read_exact(header["rgb_bytes"]), np.uint8).reshape(height, width, 3).copy()
            self.depth_m = np.frombuffer(read_exact(header["depth_bytes"]), np.float32).reshape(height, width).copy()
            self.reference_depth_m = np.frombuffer(read_exact(header["reference_bytes"]), np.float32).reshape(height, width).copy()
            self.peak_indentation = max(self.peak_indentation, float(header["indentation_m"]))
            return {"rgb": rgb}

        def metadata(self):
            value = super().metadata()
            value["backend"] = "Meta TACTO 0.0.3 Renderer; CaP-X piano force-range config"
            value["calibration"] = "TACTO DIGIT optics and CAD with piano-specific [0, 2] N visual range"
            value["observation_fields"] = ["rgb", "simulated_depth_m"]
            value["piano_config_path"] = str(piano_config)
            return value

    return PilotDigitTactile
