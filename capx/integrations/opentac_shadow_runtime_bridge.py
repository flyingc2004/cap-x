"""Load the CaP-X pilot's matched RGB/depth DIGIT bridge inside Kit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def make_runtime_digit_bridge(native_class, *, opentac_root: Path,
                              piano_config: Path, output_dir: Path | None):
    if output_dir is None:
        raise ValueError("Shadow force smoke needs a run output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[2] / "scripts/opentac/shadow_digit_bridge.py"
    spec = importlib.util.spec_from_file_location("capx_shadow_digit_bridge", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the CaP-X DIGIT RGB/depth renderer bridge")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_digit_bridge_class(
        native_class, opentac_root=opentac_root,
        piano_config=piano_config, episode_dir=output_dir,
    )
