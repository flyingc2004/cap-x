"""Small causal-window encoder and safe inference; no native task dependency."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

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
