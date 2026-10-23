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
AREA_RATE_FEATURE_NAMES = FEATURE_NAMES + ("area_gt_0p05mm_rate_per_s",)
AREA_RATE_SCHEMA = "shadow_digit_key_top_force_5d_area_rate.v1"
AREA_RATE_HISTORY = {"physics_dt_seconds": 1 / 240, "window_physics_steps": 8,
                     "window_seconds": 1 / 30, "area_threshold_mm": .05}


def public_depth_features(depth: np.ndarray, reference: np.ndarray, *,
                          diagnostics: dict | None = None) -> tuple[np.ndarray | None, str]:
    depth, reference = np.asarray(depth), np.asarray(reference)
    if diagnostics is not None:
        diagnostics.update(depth_shape=list(depth.shape), reference_shape=list(reference.shape),
                           valid_depth_fraction=None, features=None)
    if depth.ndim != 2 or depth.shape != reference.shape or depth.size == 0:
        return None, "invalid_shape"
    valid = np.isfinite(depth) & np.isfinite(reference) & (depth > 0) & (reference > 0)
    if diagnostics is not None:
        diagnostics["valid_depth_fraction"] = float(valid.mean())
    if valid.mean() < .60:
        return None, "invalid_depth"
    deformation_mm = np.abs(reference[valid] - depth[valid]) * 1000.
    if not np.isfinite(deformation_mm).all():
        return None, "nonfinite_depth"
    features = np.array((deformation_mm.mean(), *(np.mean(deformation_mm > threshold)
                         for threshold in (.01, .05, .2))), dtype=float)
    if diagnostics is not None:
        diagnostics["features"] = features.tolist()
        signed_mm = (reference[valid] - depth[valid]) * 1000.
        diagnostics.update(positive_mean_mm=float(np.maximum(signed_mm, 0).mean()),
                           negative_mean_mm=float(np.maximum(-signed_mm, 0).mean()),
                           max_absolute_delta_mm=float(deformation_mm.max()))
    return features, "ok"


class PublicDigitKeyForce:
    def __init__(self, sidecar_path: str | Path, piano_config_path: str | Path):
        self.path = Path(sidecar_path)
        self.sidecar = json.loads(self.path.read_text(encoding="utf-8"))
        names = AREA_RATE_FEATURE_NAMES if self.uses_area_rate else FEATURE_NAMES
        if (self.sidecar.get("schema") not in (SCHEMA, AREA_RATE_SCHEMA) or
                self.sidecar.get("feature_names") != list(names)):
            raise ValueError("Unsupported DIGIT key-force feature schema")
        if self.uses_area_rate and self.sidecar.get("history_contract") != AREA_RATE_HISTORY:
            raise ValueError("Unsupported DIGIT causal area-rate history contract")
        digest = hashlib.sha256(Path(piano_config_path).read_bytes()).hexdigest()
        if digest != self.sidecar.get("sensor_sha256"):
            raise ValueError("DIGIT renderer configuration differs from force calibration")
        for color in ("white", "black"):
            group = self.sidecar["models"][color]
            size = len(names)
            for name, length in (("median", size), ("iqr", size), ("coef", 2 * size),
                                 ("feature_min", size), ("feature_max", size)):
                value = np.asarray(group[name], float)
                if value.shape != (length,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid DIGIT {color} {name}")
            if not np.isfinite(group["intercept"]) or np.any(np.asarray(group["iqr"]) <= 0):
                raise ValueError("Invalid DIGIT force regression")
        self.reset_history()

    @property
    def uses_area_rate(self) -> bool:
        return self.sidecar.get("schema") == AREA_RATE_SCHEMA

    def reset_history(self) -> None:
        self._depth_history = {}

    def observe_frame(self, depth: np.ndarray, reference: np.ndarray, *, physics_step: int) -> None:
        """Sample EVERY native action, even when generated code does not query force.

        This buffer contains only public depth features and their physics time.
        Repeated queries do not advance time; reset clears it between episodes.
        """
        if not self.uses_area_rate:
            return
        if isinstance(physics_step, bool) or not isinstance(physics_step, (int, np.integer)) or physics_step < 0:
            raise ValueError("DIGIT history needs a nonnegative physical step")
        history = self._depth_history
        if history and physics_step < max(history):
            raise ValueError("DIGIT physical step moved backwards without reset")
        features, quality = public_depth_features(depth, reference)
        history[int(physics_step)] = (features, quality)
        for step in list(history):
            if step < physics_step - AREA_RATE_HISTORY["window_physics_steps"]:
                del history[step]

    def predict(self, depth: np.ndarray, reference: np.ndarray, *, color: str,
                sample_step: int, current_step: int,
                diagnostics_out: dict | None = None, physics_step: int | None = None) -> dict:
        """Estimate force; optional diagnostics are for a separate audit stream."""
        result = {"estimated_key_force_N": None, "sample_step": int(sample_step),
                  "age_steps": int(current_step - sample_step), "quality": "unavailable",
                  "calibration_version": self.sidecar.get("schema", SCHEMA)}
        names = AREA_RATE_FEATURE_NAMES if self.uses_area_rate else FEATURE_NAMES
        if diagnostics_out is not None:
            diagnostics_out.clear()
            diagnostics_out.update(schema="shadow_digit_force_feature_audit.v1", color=color,
                                   feature_names=list(names),
                                   sensor_sha256=self.sidecar["sensor_sha256"])

        def finish(quality: str, force: float | None = None):
            result.update(quality=quality, estimated_key_force_N=force)
            if diagnostics_out is not None:
                diagnostics_out["quality"] = quality
            return result

        if color not in ("white", "black") or result["age_steps"] != 0:
            return finish("invalid_color_or_stale")
        features, quality = public_depth_features(depth, reference, diagnostics=diagnostics_out)
        if features is None:
            return finish(quality)
        if self.uses_area_rate:
            if physics_step is None:
                return finish("insufficient_history")
            self.observe_frame(depth, reference, physics_step=physics_step)
            past_step = int(physics_step) - AREA_RATE_HISTORY["window_physics_steps"]
            past_features, past_quality = self._depth_history.get(past_step, (None, "missing"))
            if diagnostics_out is not None:
                diagnostics_out.update(base_features=features.tolist(), features=None,
                    physics_step=int(physics_step), history_physics_step=past_step,
                    history_quality=past_quality, history_contract=dict(AREA_RATE_HISTORY))
            if past_features is None:
                return finish("insufficient_history")
            rate = (features[2] - past_features[2]) / AREA_RATE_HISTORY["window_seconds"]
            features = np.r_[features, rate]
            if diagnostics_out is not None:
                diagnostics_out.update(features=features.tolist(),
                                       current_area_fraction=float(features[2]),
                                       previous_area_fraction=float(past_features[2]))
        group = self.sidecar["models"][color]
        lower, upper = np.asarray(group["feature_min"]), np.asarray(group["feature_max"])
        below, above = features < lower, features > upper
        if diagnostics_out is not None:
            diagnostics_out.update(feature_min=lower.tolist(), feature_max=upper.tolist(),
                exceeded_bounds=[dict(feature=names[i], value=float(features[i]),
                                      lower=float(lower[i]), upper=float(upper[i]),
                                      direction="below" if below[i] else "above")
                                 for i in np.flatnonzero(below | above)])
        if np.any(below | above):
            return finish("ood")
        if features[0] < .005 and features[1] == 0 and features[2] == 0 and features[3] == 0:
            return finish("no_contact")
        z = (features - np.asarray(group["median"])) / np.asarray(group["iqr"])
        force = float(np.dot(np.r_[z, z * z], group["coef"]) + group["intercept"])
        if not np.isfinite(force):
            return finish("invalid_prediction")
        if force < .05:
            return finish("no_contact")
        # Deliberately do not clip at 2 N: saturated/out-of-distribution
        # images must remain detectable rather than appearing safely in range.
        return finish("ok", force)
