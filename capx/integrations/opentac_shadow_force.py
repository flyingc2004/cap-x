"""Public, simulated DIGIT depth to piano key-top force calibration.

This is an inverse of TACTO's PhysX-force-to-image renderer. It is not an
independent fingertip load cell. Private force labels are used only offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FEATURE_NAMES = ("mean_mm", "area_gt_0p01mm", "area_gt_0p05mm", "area_gt_0p2mm")
SCHEMA = "shadow_digit_key_top_force_4d.v1"


def public_depth_features(depth: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray | None, str]:
    depth, reference = np.asarray(depth), np.asarray(reference)
    if depth.ndim != 2 or depth.shape != reference.shape or depth.size == 0:
        return None, "invalid_shape"
    valid = np.isfinite(depth) & np.isfinite(reference) & (depth > 0) & (reference > 0)
    if valid.mean() < .60:
        return None, "invalid_depth"
    deformation_mm = np.abs(reference[valid] - depth[valid]) * 1000.
    if not np.isfinite(deformation_mm).all():
        return None, "nonfinite_depth"
    return np.array((deformation_mm.mean(), *(np.mean(deformation_mm > threshold)
                     for threshold in (.01, .05, .2))), dtype=float), "ok"


class PublicDigitKeyForce:
    def __init__(self, sidecar_path: str | Path, piano_config_path: str | Path):
        self.path = Path(sidecar_path)
        self.sidecar = json.loads(self.path.read_text(encoding="utf-8"))
        if self.sidecar.get("schema") != SCHEMA or self.sidecar.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Unsupported DIGIT key-force feature schema")
        digest = hashlib.sha256(Path(piano_config_path).read_bytes()).hexdigest()
        if digest != self.sidecar.get("sensor_sha256"):
            raise ValueError("DIGIT renderer configuration differs from force calibration")
        for color in ("white", "black"):
            group = self.sidecar["models"][color]
            for name, length in (("median", 4), ("iqr", 4), ("coef", 8), ("feature_min", 4), ("feature_max", 4)):
                value = np.asarray(group[name], float)
                if value.shape != (length,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid DIGIT {color} {name}")
            if not np.isfinite(group["intercept"]) or np.any(np.asarray(group["iqr"]) <= 0):
                raise ValueError("Invalid DIGIT force regression")

    def predict(self, depth: np.ndarray, reference: np.ndarray, *, color: str,
                sample_step: int, current_step: int) -> dict:
        result = {"estimated_key_force_N": None, "sample_step": int(sample_step),
                  "age_steps": int(current_step - sample_step), "quality": "unavailable",
                  "calibration_version": SCHEMA}
        if color not in ("white", "black") or result["age_steps"] != 0:
            result["quality"] = "invalid_color_or_stale"
            return result
        features, quality = public_depth_features(depth, reference)
        if features is None:
            result["quality"] = quality
            return result
        group = self.sidecar["models"][color]
        if np.any(features < np.asarray(group["feature_min"])) or np.any(features > np.asarray(group["feature_max"])):
            result["quality"] = "ood"
            return result
        if features[0] < .005 and features[1] == 0 and features[2] == 0 and features[3] == 0:
            result["quality"] = "no_contact"
            return result
        z = (features - np.asarray(group["median"])) / np.asarray(group["iqr"])
        force = float(np.dot(np.r_[z, z * z], group["coef"]) + group["intercept"])
        if not np.isfinite(force):
            result["quality"] = "invalid_prediction"
            return result
        if force < .05:
            result["quality"] = "no_contact"
            return result
        # Deliberately do not clip at 2 N: saturated/out-of-distribution
        # images must remain detectable rather than appearing safely in range.
        result.update(estimated_key_force_N=force, quality="ok")
        return result
