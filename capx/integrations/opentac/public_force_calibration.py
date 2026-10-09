"""Public depth/marker force-calibration utilities for the tension-strap task.

The frozen model is trained offline with private labels, but its runtime input
is deliberately limited to the current public GelSight depth and marker
coordinates.  This module is Isaac-free so its feature contract can be tested
without a simulator.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


CALIBRATION_SCHEMA = "tension_strap_tactile_force_calibration.v1"
FEATURE_SCHEMA = "tension_strap_marker_depth_force_features.v1"
ESTIMATOR_MODE = "public_depth_marker_regression_v1"
FEATURE_NAMES = tuple(
    name
    for hand in ("left", "right")
    for name in (
        f"{hand}_indentation_p98_mm",
        f"{hand}_marker_mean_dx_px",
        f"{hand}_marker_mean_dy_px",
        f"{hand}_marker_median_dx_px",
        f"{hand}_marker_median_dy_px",
        f"{hand}_marker_mean_displacement_px",
        f"{hand}_marker_p90_displacement_px",
        f"{hand}_marker_coherence",
        f"{hand}_marker_row_gradient_dx_px",
        f"{hand}_marker_row_gradient_dy_px",
        f"{hand}_marker_column_gradient_dx_px",
        f"{hand}_marker_column_gradient_dy_px",
        f"{hand}_marker_log_anisotropy",
    )
)


def as_numpy(value: Any) -> np.ndarray:
    """Convert a public simulator observation to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_public_tactile_force_features(
    depths: Iterable[Any],
    markers: Iterable[Any],
    *,
    far_plane_mm: float,
) -> np.ndarray:
    """Match the offline 26D depth/marker feature contract exactly."""
    values: list[float] = []
    for raw_depth, raw_marker in zip(depths, markers, strict=True):
        depth = as_numpy(raw_depth).astype(np.float64, copy=False)
        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size == 0:
            raise ValueError("depth frame has no finite values")
        indentation = float(
            np.percentile(np.clip(float(far_plane_mm) - finite_depth, 0.0, None), 98.0)
        )

        marker = _normalize_marker(raw_marker)
        initial, current = marker
        valid = np.isfinite(initial).all(axis=1) & np.isfinite(current).all(axis=1)
        if int(valid.sum()) < 16:
            raise ValueError("marker frame has fewer than 16 finite correspondences")
        coordinates = initial[valid]
        delta = current[valid] - coordinates
        magnitude = np.linalg.norm(delta, axis=1)
        mean = delta.mean(axis=0)
        median = np.median(delta, axis=0)
        mean_magnitude = float(magnitude.mean())
        coherence = float(np.linalg.norm(mean) / max(mean_magnitude, 1e-9))

        def gradient(axis: int) -> np.ndarray:
            coordinate = coordinates[:, axis]
            split = float(np.median(coordinate))
            lower, upper = delta[coordinate <= split], delta[coordinate > split]
            if not len(lower) or not len(upper):
                return np.zeros(2, dtype=np.float64)
            return upper.mean(axis=0) - lower.mean(axis=0)

        row_gradient = gradient(1)
        column_gradient = gradient(0)
        centered = delta - mean
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / max(len(centered), 1))
        log_anisotropy = float(
            np.log((eigenvalues[-1] + 1e-6) / (eigenvalues[0] + 1e-6))
        )
        values.extend(
            (
                indentation,
                float(mean[0]),
                float(mean[1]),
                float(median[0]),
                float(median[1]),
                mean_magnitude,
                float(np.percentile(magnitude, 90.0)),
                float(np.clip(coherence, 0.0, 1.0)),
                float(row_gradient[0]),
                float(row_gradient[1]),
                float(column_gradient[0]),
                float(column_gradient[1]),
                log_anisotropy,
            )
        )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("public tactile force feature vector is invalid")
    return result


def validate_public_force_calibration(payload: Any) -> dict[str, Any]:
    """Validate a frozen public-only 26D calibration sidecar."""
    if not isinstance(payload, dict):
        raise ValueError("public tactile force calibration must be an object")
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected public tactile force calibration schema")
    contract = payload.get("input_contract")
    if not isinstance(contract, dict) or contract.get("feature_schema_version") != FEATURE_SCHEMA:
        raise ValueError("public tactile force calibration has an invalid feature schema")
    if contract.get("feature_transform") != "robust_median_iqr_then_linear_and_square":
        raise ValueError("public tactile force calibration has an unsupported feature transform")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("public tactile force calibration feature order does not match runtime")
    far_plane = _finite_float(contract.get("far_plane_mm"))
    if far_plane is None or far_plane <= 0.0:
        raise ValueError("public tactile force calibration has an invalid far plane")

    scaler = payload.get("scaler")
    regression = payload.get("regression")
    training = payload.get("training")
    if not isinstance(scaler, dict) or not isinstance(regression, dict) or not isinstance(training, dict):
        raise ValueError("public tactile force calibration is missing model fields")
    median = _finite_vector(scaler.get("median"), len(FEATURE_NAMES), "scaler median")
    iqr = _finite_vector(scaler.get("iqr"), len(FEATURE_NAMES), "scaler iqr")
    if np.any(iqr <= 0.0):
        raise ValueError("public tactile force calibration has nonpositive IQR")
    weights = _finite_vector(regression.get("weights"), len(FEATURE_NAMES) * 2, "ridge weights")
    bias = _finite_float(regression.get("bias"))
    if bias is None:
        raise ValueError("public tactile force calibration has an invalid bias")
    sampling = training.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("public tactile force calibration is missing training range")
    force_range = _finite_vector(sampling.get("force_range_N"), 2, "training force range")
    if force_range[0] >= force_range[1]:
        raise ValueError("public tactile force calibration has an invalid training force range")
    if "true_tension" in repr(payload).lower() or "private_force_labels" in repr(payload).lower():
        raise ValueError("public tactile force calibration must not embed private labels")

    return {
        "schema_version": CALIBRATION_SCHEMA,
        "calibration_id": str(payload.get("calibration_id", CALIBRATION_SCHEMA)),
        "far_plane_mm": float(far_plane),
        "median": median,
        "iqr": iqr,
        "weights": weights,
        "bias": float(bias),
        "force_range_N": force_range,
    }


def predict_public_tactile_force(calibration: dict[str, Any], features: Any) -> float:
    """Run the frozen robust-scaler + linear/square ridge model."""
    values = _finite_vector(features, len(FEATURE_NAMES), "public tactile features")
    normalized = (values - calibration["median"]) / calibration["iqr"]
    design = np.concatenate((normalized, normalized * normalized))
    estimate = float(design @ calibration["weights"] + calibration["bias"])
    if not np.isfinite(estimate):
        raise ValueError("public tactile force calibration returned a non-finite value")
    minimum, maximum = calibration["force_range_N"]
    return float(np.clip(estimate, minimum, maximum))


def _normalize_marker(value: Any) -> np.ndarray:
    marker = as_numpy(value).astype(np.float64, copy=False)
    if marker.ndim == 4 and marker.shape[0] == 1:
        marker = marker[0]
    if marker.ndim == 4 and marker.shape[0] == 2 and marker.shape[-1] == 2:
        marker = marker.reshape(2, -1, 2)
    if marker.shape != (2, 64, 2):
        raise ValueError(f"expected marker shape (2, 64, 2), got {marker.shape}")
    return marker


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"public tactile force calibration has invalid {name}") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"public tactile force calibration has invalid {name}")
    return result
