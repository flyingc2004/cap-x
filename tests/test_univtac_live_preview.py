from pathlib import Path

import numpy as np
from PIL import Image

from capx.envs.simulators.univtac import UniVTACLowLevelEnv


def _preview_env(path: Path, *, enabled: bool = True, stride: int = 5) -> UniVTACLowLevelEnv:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.live_preview_enabled = enabled
    env.live_preview_path = path
    env.live_preview_stride = stride
    env.live_preview_jpeg_quality = 80
    env._live_preview_write_failures = 0
    env._frame_buffer = []
    return env


def test_live_preview_writes_atomic_jpeg(tmp_path) -> None:
    preview = tmp_path / "latest_preview.jpg"
    env = _preview_env(preview)
    frame = np.full((12, 16, 3), 127, dtype=np.uint8)
    env._frame_buffer.append(frame)

    env._write_live_preview(frame, force=True)

    assert preview.exists()
    assert not (tmp_path / ".latest_preview.jpg.tmp").exists()
    image = Image.open(preview)
    assert image.size == (16, 12)
    assert image.mode == "RGB"


def test_live_preview_respects_stride_and_enabled_flag(tmp_path) -> None:
    preview = tmp_path / "latest_preview.jpg"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    env = _preview_env(preview, stride=5)
    env._frame_buffer.append(frame)
    env._write_live_preview(frame)
    assert not preview.exists()

    env._frame_buffer.extend([frame, frame, frame, frame])
    env._write_live_preview(frame)
    assert preview.exists()

    disabled_preview = tmp_path / "disabled.jpg"
    disabled_env = _preview_env(disabled_preview, enabled=False, stride=1)
    disabled_env._frame_buffer.append(frame)
    disabled_env._write_live_preview(frame, force=True)
    assert not disabled_preview.exists()
