"""GPU-independent contracts for the Shadow DIGIT force collection pilot.

Public samples deliberately contain no native score or measured force.  The
private stream is aligned by physics_step and is never passed to CaP-X code.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import yaml


SCHEMA = "shadow_digit_piano_0_2.v1"
RANGE_FORCE_N = (0, 2)
PHASES = {"expert": 0, "hover": 1, "descend": 2, "flex": 3, "hold": 4,
          "release": 5, "lift": 6}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_piano_config(original: Path, destination: Path) -> dict:
    """Copy TACTO's installed config with exactly one force-range change."""
    original, destination = Path(original), Path(destination)
    source = original.read_text(encoding="utf-8")
    config = yaml.safe_load(source)
    if config["sensor"]["force"]["range_force"] != [0, 100]:
        raise ValueError("Expected the installed TACTO DIGIT [0, 100] N config")
    if float(config["sensor"]["force"]["max_deformation"]) != .005:
        raise ValueError("Unexpected installed TACTO DIGIT deformation limit")
    updated, count = re.subn(r"(?m)^(\s*range_force:\s*)\[0,\s*100\]",
                             r"\g<1>[0, 2]", source)
    if count != 1:
        raise ValueError("Cannot make a one-field TACTO config copy")
    revised = yaml.safe_load(updated)
    config["sensor"]["force"]["range_force"] = [0, 2]
    if revised != config:
        raise ValueError("TACTO config copy changed fields besides range_force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != updated:
            raise FileExistsError(f"Refusing to replace a different config: {destination}")
    else:
        destination.write_text(updated, encoding="utf-8")
    provenance = {"schema": SCHEMA, "source_config": str(original),
                  "source_sha256": sha256(original), "piano_config": str(destination),
                  "piano_sha256": sha256(destination), "range_force_N": list(RANGE_FORCE_N),
                  "max_deformation_m": .005}
    manifest = destination.with_suffix(".provenance.json")
    manifest_text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != manifest_text:
        raise FileExistsError(f"Refusing to replace a different config manifest: {manifest}")
    if not manifest.exists():
        manifest.write_text(manifest_text, encoding="utf-8")
    return provenance


def validate_sensor_frame(rgb: np.ndarray, depth: np.ndarray,
                          reference: np.ndarray) -> np.ndarray:
    rgb, depth, reference = map(np.asarray, (rgb, depth, reference))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("DIGIT RGB must be an HxWx3 uint8 image")
    if depth.shape != rgb.shape[:2] or reference.shape != depth.shape:
        raise ValueError("DIGIT depth/reference geometry does not match RGB")
    if depth.dtype != np.float32 or reference.dtype != np.float32:
        raise ValueError("DIGIT depth/reference must be float32 metres")
    return np.isfinite(depth) & np.isfinite(reference) & (depth > 0) & (reference > 0)


class PilotRecorder:
    """Keep public HDF5 frames and private per-physics-step labels separate."""

    def __init__(self, episode_dir: Path, *, metadata: dict):
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.public = h5py.File(self.episode_dir / "public.h5", "w")
        self.private = []
        self.public_count = 0
        self.last_private_step = -1
        self.metadata = dict(metadata)
        self.public.attrs["schema"] = SCHEMA
        self.public.attrs["metadata"] = json.dumps(metadata, sort_keys=True)

    def append_private(self, *, physics_step: int, forces: np.ndarray,
                       angles: np.ndarray, requested_midi: int, score_count: int,
                       score_armed: bool, score_success: bool, score_failure: str):
        forces, angles = np.asarray(forces, np.float32), np.asarray(angles, np.float32)
        if (forces.shape != (88,) or angles.shape != (88,) or
                not np.isfinite(forces).all() or not np.isfinite(angles).all() or
                (forces < 0).any()):
            raise ValueError("Private label must have 88 finite nonnegative key forces and angles")
        if physics_step != self.last_private_step + 1:
            raise ValueError("Private force labels must cover consecutive physics steps")
        self.private.append((int(physics_step), forces.copy(), angles.copy(),
                             int(requested_midi), int(score_count), bool(score_armed),
                             bool(score_success), str(score_failure)))
        self.last_private_step = physics_step

    def _append(self, name: str, value: np.ndarray):
        value = np.asarray(value)
        if name not in self.public:
            self.public.create_dataset(name, shape=(0, *value.shape),
                                       maxshape=(None, *value.shape), chunks=(1, *value.shape),
                                       dtype=value.dtype, compression="lzf", shuffle=True)
        dataset = self.public[name]
        if value.shape != dataset.shape[1:]:
            raise ValueError(f"Inconsistent public shape for {name}")
        dataset.resize(self.public_count + 1, axis=0)
        dataset[self.public_count] = value

    def append_public(self, *, physics_step: int, rgb: np.ndarray,
                      depth_m: np.ndarray, reference_depth_m: np.ndarray,
                      hand_state: np.ndarray, key_angles_rad: np.ndarray,
                      action_target: np.ndarray, requested_midi: int, phase: str):
        if physics_step != self.last_private_step:
            raise ValueError("Public frame has no exact-step private force label")
        if phase not in PHASES:
            raise ValueError(f"Unknown controller phase: {phase}")
        valid = validate_sensor_frame(rgb, depth_m, reference_depth_m)
        if not valid.any():
            raise ValueError("DIGIT frame has no valid depth pixels")
        hand_state = np.asarray(hand_state, np.float32)
        key_angles_rad = np.asarray(key_angles_rad, np.float32)
        action_target = np.asarray(action_target, np.float32)
        if (hand_state.shape != (21,) or key_angles_rad.shape != (88,) or
                action_target.shape != (4,) or not np.isfinite(hand_state).all() or
                not np.isfinite(key_angles_rad).all() or not np.isfinite(action_target).all()):
            raise ValueError("Invalid public proprioception or action shape")
        if "tactile/index/reference_depth_m" not in self.public:
            self.public.create_dataset("tactile/index/reference_depth_m",
                                       data=reference_depth_m, compression="lzf")
        elif not np.array_equal(self.public["tactile/index/reference_depth_m"][:], reference_depth_m):
            raise ValueError("DIGIT reference depth changed within an episode")
        values = {"physics_step": np.int64(physics_step),
                  "time_s": np.float64(physics_step * self.metadata["physics_dt_s"]),
                  "requested_midi": np.int16(requested_midi),
                  "controller_phase": np.uint8(PHASES[phase]),
                  "action_target": action_target, "hand_state": hand_state,
                  "key_angles_rad": key_angles_rad,
                  "tactile/index/rgb": rgb,
                  "tactile/index/depth_m": depth_m,
                  "tactile/index/valid_mask": valid.astype(np.uint8)}
        for name, value in values.items():
            self._append(name, value)
        self.public_count += 1
        if self.public_count % 30 == 0:
            self.public.flush()

    def close(self):
        if self.public:
            self.public.flush()
            self.public.close()
            self.public = None
        rows = self.private
        with h5py.File(self.episode_dir / "private_labels.h5", "w") as h5:
            h5.attrs["schema"] = SCHEMA
            h5.attrs["provenance"] = json.dumps(self.metadata, sort_keys=True)
            h5.attrs["access"] = "offline audit only; never pass to agent"
            h5.create_dataset("physics_step", data=np.asarray([r[0] for r in rows], np.int64))
            h5.create_dataset("key_contact_force_N", data=np.asarray([r[1] for r in rows], np.float32), compression="lzf")
            h5.create_dataset("key_angle_rad", data=np.asarray([r[2] for r in rows], np.float32), compression="lzf")
            h5.create_dataset("requested_midi", data=np.asarray([r[3] for r in rows], np.int16))
            h5.create_dataset("score_count", data=np.asarray([r[4] for r in rows], np.int16))
            h5.create_dataset("score_armed", data=np.asarray([r[5] for r in rows], np.uint8))
            h5.create_dataset("score_success", data=np.asarray([r[6] for r in rows], np.uint8))
            h5.create_dataset("score_failure", data=np.asarray([r[7] for r in rows], dtype=h5py.string_dtype()))


def audit_episode(episode_dir: Path) -> dict:
    """Align frames and native force labels without creating an estimator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode_dir = Path(episode_dir)
    with h5py.File(episode_dir / "public.h5") as pub, h5py.File(episode_dir / "private_labels.h5") as priv:
        steps = np.asarray(pub["physics_step"], np.int64)
        psteps = np.asarray(priv["physics_step"], np.int64)
        if not len(steps) or not len(psteps) or not np.all(np.diff(steps) > 0):
            raise ValueError("Pilot has no ordered public/private samples")
        indices = np.searchsorted(psteps, steps)
        if np.any(indices >= len(psteps)) or not np.array_equal(psteps[indices], steps):
            raise ValueError("Public frames do not have exact-step force labels")
        midi = np.asarray(pub["requested_midi"], int)
        force = np.asarray(priv["key_contact_force_N"][indices, :])[np.arange(len(steps)), midi - 21]
        angle = np.asarray(pub["key_angles_rad"])[np.arange(len(steps)), midi - 21]
        depth = np.asarray(pub["tactile/index/depth_m"], np.float32)
        reference = np.asarray(pub["tactile/index/reference_depth_m"], np.float32)
        valid = np.asarray(pub["tactile/index/valid_mask"], bool)
        rgb = np.asarray(pub["tactile/index/rgb"], np.uint8)
        rgb_delta = np.mean(np.abs(rgb.astype(np.float32) - rgb[0].astype(np.float32)), axis=(1, 2, 3))
        depth_delta = np.asarray([np.mean(np.abs((reference - frame)[mask])) if mask.any() else np.nan
                                  for frame, mask in zip(depth, valid)])
        time_s = np.asarray(pub["time_s"], float)
        result = {"public_frames": int(len(steps)), "private_physics_steps": int(len(psteps)),
                  "exact_step_alignment": True, "max_target_key_force_N": float(force.max()),
                  "frames_at_or_above_2_N": int(np.count_nonzero(force >= 2)),
                  "private_steps_any_key_at_or_above_2_N": int(np.count_nonzero(
                      np.max(priv["key_contact_force_N"], axis=1) >= 2)),
                  "valid_depth_fraction_min": float(np.mean(valid, axis=(1, 2)).min()),
                  "rgb_delta_max": float(rgb_delta.max()),
                  "depth_delta_m_max": float(np.nanmax(depth_delta))}
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(time_s, force); axes[0].axhline(2, color="red", linestyle="--"); axes[0].set_ylabel("target key N")
        axes[1].plot(time_s, angle); axes[1].set_ylabel("key rad")
        axes[2].plot(time_s, rgb_delta); axes[2].set_ylabel("RGB delta")
        axes[3].plot(time_s, depth_delta * 1000); axes[3].set_ylabel("depth delta mm")
        axes[3].set_xlabel("simulation time (s)")
        fig.tight_layout()
        fig.savefig(episode_dir / "alignment.png", dpi=140)
        plt.close(fig)
    (episode_dir / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
