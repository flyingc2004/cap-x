"""Small causal-window encoder and safe inference; no native task dependency."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from pathlib import Path

from .data import SCHEMA, WINDOW, causal_windows, read_json, reduce_depth, relative_motion, sha256


class TrajectoryEncoder(nn.Module):
    def __init__(self, use_motion=False):
        super().__init__()
        self.use_motion = bool(use_motion)
        self.spatial = nn.Sequential(nn.Conv2d(2, 16, 3, 2, 1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(32*3*4, 64), nn.ReLU())
        self.temporal = nn.GRU(64 + (8 if use_motion else 0), 64, batch_first=True)
        self.output = nn.Linear(64, 32)

    def forward(self, maps, motion=None):
        if maps.ndim != 5 or tuple(maps.shape[1:]) != (16, 2, 24, 32):
            raise ValueError("Expected B x 16 x 2 x 24 x 32 maps")
        b, t = maps.shape[:2]
        z = self.spatial(maps.reshape(b*t, 2, 24, 32)).reshape(b, t, 64)
        if self.use_motion:
            if motion is None or tuple(motion.shape) != (b, t, 8):
                raise ValueError("Motion model requires B x 16 x 8 public state")
            z = torch.cat([z, motion], -1)
        _, hidden = self.temporal(z)
        return self.output(hidden[-1])


def augment(maps, generator):
    x = maps.clone()
    present = x[:, :, 1] > 0
    noise = torch.randn(present.shape, generator=generator, device=x.device) * .01
    x[:, :, 0] += noise * present
    keep = torch.rand(present.shape, generator=generator, device=x.device) >= .02
    x *= keep.unsqueeze(2)
    return x


def vicreg(z1, z2):
    if z1.shape != z2.shape or len(z1) < 2:
        raise ValueError("VICReg requires matching batches of at least two windows")
    invariance = (z1-z2).square().mean()
    std1, std2 = (z1.var(0) + 1e-4).sqrt(), (z2.var(0) + 1e-4).sqrt()
    variance = (torch.relu(1-std1).mean() + torch.relu(1-std2).mean()) / 2
    a, b = z1-z1.mean(0), z2-z2.mean(0)
    ca, cb = a.T@a/(len(a)-1), b.T@b/(len(b)-1)
    off = ~torch.eye(ca.shape[0], dtype=torch.bool, device=z1.device)
    covariance = (ca[off].square().sum() + cb[off].square().sum()) / ca.shape[0]
    loss = 25*invariance + 25*variance + covariance
    return loss, {"invariance": invariance, "variance": variance, "covariance": covariance}


def unit_rows(x):
    x = np.asarray(x, np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms < 1e-12):
        raise ValueError("Nonfinite or zero embedding; no key can be fabricated")
    return x / norms


def load_checkpoint(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["schema"] != SCHEMA or bundle["window"] != WINDOW:
        raise ValueError("Incompatible encoder contract")
    model = TrajectoryEncoder(bundle["use_motion"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model.to(device).eval(), bundle


class FrozenTrajectoryEncoder:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model, self.bundle = load_checkpoint(checkpoint, device)
        self.encoder_version = sha256(checkpoint)

    def encode_window(self, depth, reference, valid_mask, times, *, sensor_sha256,
                      actions=None, actual_state=None):
        result = {"embedding": None, "quality": "unavailable", "encoder_version": self.encoder_version,
                  "sensor_sha256": sensor_sha256, "window": dict(WINDOW)}
        if sensor_sha256 != self.bundle["sensor_sha256"]:
            return {**result, "quality": "incompatible_sensor"}
        try:
            maps, _, good = reduce_depth(depth, reference, valid_mask)
            times = np.asarray(times, np.float64)
            if len(maps) != len(times):
                raise ValueError("Mismatched history length")
            windows = causal_windows(times, good)
            if not len(windows) or windows[-1, -1] != len(times)-1:
                return {**result, "quality": "insufficient_valid_history"}
            ids = windows[-1]
            norm = self.bundle["normalization"]
            x = maps[ids][None]
            x[:, :, 0] = np.where(x[:, :, 1] > 0, (x[:, :, 0]-norm["depth_mean"])/norm["depth_std"], 0)
            motion = None
            if self.model.use_motion:
                actions, actual_state = np.asarray(actions, np.float32), np.asarray(actual_state, np.float32)
                if actions.shape != (len(times), 4) or actual_state.shape != actions.shape:
                    raise ValueError("Missing or malformed public motion")
                motion = relative_motion(np.concatenate([actions[ids], actual_state[ids]], 1)[None])
                motion = (motion-np.asarray(norm["motion_mean"], np.float32))/np.asarray(norm["motion_std"], np.float32)
                if not np.isfinite(motion).all():
                    raise ValueError("Nonfinite motion")
                motion = torch.from_numpy(motion).to(self.device)
            with torch.inference_mode():
                z = self.model(torch.from_numpy(x).to(self.device), motion).cpu().numpy()
            key = unit_rows(z)[0]
        except (ValueError, TypeError, IndexError):
            return {**result, "quality": "invalid_input"}
        return {**result, "quality": "ok", "embedding": key.tolist(), "sample_time_s": float(times[-1])}


class EvidenceTrajectoryEncoder:
    """CPU adapter for the public evidence protocol; independent of force inference.

    All three spaces use the frozen training preprocessing. The episode supplies
    the actual sensor hash, timestamps and source references, never the expert index.
    """

    def __init__(self, run_dir, representation="tactile"):
        if representation not in ("window4", "tactile", "tactile_action"):
            raise ValueError("Unknown trajectory representation")
        root = Path(run_dir).expanduser().resolve()
        norm_path = root / "cache/normalization.json"
        self.normalization = read_json(norm_path)
        manifest = read_json(root / "cache/manifest.json")
        if manifest["schema"] != SCHEMA or manifest["window"] != WINDOW:
            raise ValueError("Incompatible trajectory preprocessing")
        self.sensor = manifest["sensor_sha256"]
        self.representation = representation
        self.frozen = None
        version = sha256(norm_path)
        if representation != "window4":
            self.frozen = FrozenTrajectoryEncoder(root / f"models/{representation}/seed_0/best.pt", device="cpu")
            if (self.frozen.bundle["normalization"] != self.normalization or
                    self.frozen.bundle["sensor_sha256"] != self.sensor or
                    self.frozen.model.use_motion != (representation == "tactile_action")):
                raise ValueError("Checkpoint and preprocessing contract differ")
            version = self.frozen.encoder_version
        self.contract = {"version": version, "representation": representation,
            "preprocessing": "digit_trajectory_key.v1/reduce_depth.v1/causal_previous.v1",
            "normalization_sha256": sha256(norm_path), "sensor_sha256": self.sensor,
            "window": dict(WINDOW), "dimension": 64 if representation == "window4" else 32,
            "modalities": ["depth_m", "reference_depth_m", "valid_mask"] +
                (["actions", "actual_state"] if representation == "tactile_action" else [])}

    def encode(self, window):
        unavailable = {"vector": None, "quality": "unavailable"}
        if window.get("sensor_sha256") != self.sensor:
            return {**unavailable, "quality": "incompatible_sensor"}
        try:
            times = np.asarray(window["timestamps_s"], np.float64)
            ids = causal_windows(times, np.ones(len(times), bool))
            if not len(ids) or ids[-1, -1] != len(times)-1 or len(set(ids[-1])) != 16:
                return {**unavailable, "quality": "insufficient_valid_history"}
            ids = ids[-1]
            frames = window["frames"]
            reference = np.asarray(frames[0]["reference_depth_m"])
            if any(not np.array_equal(f["reference_depth_m"], reference, equal_nan=True) for f in frames):
                return {**unavailable, "quality": "incompatible_reference"}
            depth = np.stack([f["depth_m"] for f in frames])
            if any(m is None for m in window["valid_masks"]):
                return {**unavailable, "quality": "missing_valid_mask"}
            mask = np.stack(window["valid_masks"])
            provenance = {"episode_id": window.get("episode_id"),
                "target_times_s": (times[-1] - np.arange(15, -1, -1)/30).tolist(),
                "samples": [window["sources"][i] for i in ids]}
            if self.frozen is None:
                _, features, good = reduce_depth(depth[ids], reference, mask[ids])
                if not good.all():
                    return {**unavailable, "quality": "insufficient_valid_history"}
                mean = np.asarray(self.normalization["feature_mean"])
                std = np.asarray(self.normalization["feature_std"])
                if mean.shape != (4,) or std.shape != (4,) or np.any(std <= 0):
                    raise ValueError("Invalid feature normalization")
                vector = unit_rows(((features-mean)/std).reshape(1, 64))[0].tolist()
            else:
                motion = {}
                if self.representation == "tactile_action":
                    for name in ("actions", "actual_state"):
                        value = window.get(name)
                        # Historical missing X remains unavailable, never zero-filled.
                        if value is None or any(value[i] is None for i in ids):
                            return {**unavailable, "quality": "missing_public_motion", "provenance": provenance}
                        array = np.asarray([value[i] for i in ids], np.float32)
                        if array.shape != (16, 4) or not np.isfinite(array).all():
                            return {**unavailable, "quality": "invalid_public_motion"}
                        motion[name] = array
                # Select by physical time first. This is 16 x 30Hz, not the last
                # 16 observations of a 60Hz controller. No interpolation/padding.
                result = self.frozen.encode_window(depth[ids], reference, mask[ids], times[ids],
                    sensor_sha256=window["sensor_sha256"], **motion)
                if result["quality"] != "ok":
                    return {**unavailable, "quality": result["quality"], "provenance": provenance}
                vector = result["embedding"]
            return {"vector": vector, "quality": "ok", "provenance": provenance}
        except (KeyError, ValueError, TypeError, IndexError):
            return {**unavailable, "quality": "invalid_input"}


def search_index(index_path, query, *, k=5):
    """No cross-encoder or cross-sensor comparisons, even for equal dimensions."""
    if query.get("quality") != "ok":
        return []
    meta = read_json(str(index_path) + ".json")
    for field in ("encoder_version", "sensor_sha256", "window"):
        if query.get(field) != meta.get(field):
            raise ValueError(f"Incompatible query/index {field}")
    with np.load(index_path, allow_pickle=False) as index:
        keys, ids = index["keys"], index["window_ids"]
        q = np.asarray(query["embedding"], np.float32)
        if q.shape != (meta.get("dimension",32),) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
            raise ValueError("Invalid query key")
        scores = keys @ (q / np.linalg.norm(q))
        selected = np.argsort(-scores, kind="stable")[:max(0, k)]
        return [{"window_id": int(ids[i]), "similarity": float(scores[i]), "skill_id": None} for i in selected]
