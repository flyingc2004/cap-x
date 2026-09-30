"""ViTaForge-specific public tactile APIs used by CaP-X force tasks.

This module deliberately owns force-task calibration and stage-memory access.
It does not read task-private physical measurements or modify UniVTAC's
general tactile API surface.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.univtac.native_tactile import summarize_tactile_stage_response


_STAGE_MEMORY_SCHEMA = "tactile_stage_memory.v1"
_STAGE_RESPONSE_SCHEMA = "tactile_stage_response.v1"
_ESTIMATE_SCHEMA = "opentac_tension_estimate.v1"
_CONTROL_SCHEMA = "opentac_tension_control_contract.v1"


class OpenTacApi(ApiBase):
    """Public tactile sensing and frozen response memory for ViTaForge.

    ``estimated_tension_N`` is computed only from public GelSight marker-RGB
    observations and the configured calibration sidecar.  The API never reads
    the task's true tension, physical scorer, reward, success, actor state, or
    private metadata.
    """

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        config = self._runtime_config()
        visible = config.get("llm_visible_functions")
        if visible is not None and not isinstance(visible, (list, tuple)):
            raise ValueError("llm_visible_functions must be a list when configured")
        self._visible_functions = None if visible is None else {str(item) for item in visible}
        self._capture_count = 0
        self._estimate_count = 0
        self._tracker: Any | None = None
        self._reference_images: dict[str, np.ndarray] | None = None
        self._calibration: dict[str, Any] | None = None

    def functions(self) -> dict[str, Any]:
        full = {
            "get_tactile_tension_control_contract": self.get_tactile_tension_control_contract,
            "begin_tactile_tension_estimator": self.begin_tactile_tension_estimator,
            "get_tactile_tension_estimate": self.get_tactile_tension_estimate,
            "get_tactile_stage_memory": self.get_tactile_stage_memory,
            "capture_tactile_stage_response": self.capture_tactile_stage_response,
        }
        if self._visible_functions is None:
            return full
        unknown = self._visible_functions.difference(full)
        if unknown:
            raise ValueError(f"unsupported OpenTacApi functions: {sorted(unknown)}")
        return {name: function for name, function in full.items() if name in self._visible_functions}

    def reset_episode(self) -> None:
        """Discard the per-episode marker reference and capture counters."""
        self._capture_count = 0
        self._estimate_count = 0
        self._tracker = None
        self._reference_images = None
        self._calibration = None

    def get_tactile_tension_control_contract(self) -> dict[str, Any]:
        """Return the public local-control contract for the force task.

        It exposes only action safety and sampling cadence. Target force values
        remain part of the task instruction; the runtime physical scorer is
        never exposed.
        """
        config = self._runtime_config()
        contract = {
            "schema_version": _CONTROL_SCHEMA,
            "control_frame": "world",
            "allowed_translation_axes": ["z"],
            "max_delta_z_m": float(config.get("max_delta_z_m", 0.002)),
            "observation_wait_steps": int(config.get("observation_wait_steps", 2)),
            "observation_dt_s": float(config.get("observation_dt_s", 1.0 / 60.0)),
            "stage_hold_seconds": float(config.get("stage_hold_seconds", 3.0)),
            "max_control_actions_per_stage": int(config.get("max_control_actions_per_stage", 120)),
        }
        if contract["max_delta_z_m"] <= 0.0:
            raise RuntimeError("OpenTac max_delta_z_m must be positive")
        if (
            contract["observation_wait_steps"] < 1
            or contract["observation_dt_s"] <= 0.0
            or contract["max_control_actions_per_stage"] < 1
        ):
            raise RuntimeError("OpenTac control step limits must be positive")
        return _jsonable(contract)

    def begin_tactile_tension_estimator(self) -> dict[str, Any]:
        """Capture a marker-RGB baseline after the agent has secured the strap."""
        images = self._read_marker_images()
        calibration = self._load_tension_calibration()
        utilities = self._force_task_utilities()
        try:
            tracker = utilities.MarkerFlowTracker(images)
        except Exception as exc:
            raise RuntimeError(f"OpenTac could not initialize marker tracking: {exc}") from exc
        self._reference_images = images
        self._calibration = calibration
        self._tracker = tracker
        record = {
            "schema_version": _ESTIMATE_SCHEMA,
            "ok": True,
            "status": "baseline_ready",
            "sensor_keys": sorted(images),
            "step": self._step_count(),
        }
        self._trace("tension_estimator_begin", record)
        return _jsonable(record)

    def get_tactile_tension_estimate(self) -> dict[str, Any]:
        """Estimate strap tension from public marker-RGB flow and calibration.

        Call :meth:`begin_tactile_tension_estimator` once after a bilateral
        grasp.  A returned estimate is an observation-model output, not the
        simulator's physical tension value.
        """
        if self._tracker is None or self._calibration is None or self._reference_images is None:
            raise RuntimeError("call begin_tactile_tension_estimator() after grasping first")
        images = self._read_marker_images()
        utilities = self._force_task_utilities()
        try:
            features, tracking = utilities.tracked_flow_rgb_features(
                self._reference_images,
                images,
                self._tracker,
            )
            estimate = float(utilities.predict_calibrated(self._calibration, features))
        except Exception as exc:
            return {
                "schema_version": _ESTIMATE_SCHEMA,
                "ok": False,
                "status": "tracking_unavailable",
                "message": str(exc),
                "step": self._step_count(),
            }
        self._estimate_count += 1
        record = {
            "schema_version": _ESTIMATE_SCHEMA,
            "ok": bool(np.isfinite(estimate)),
            "status": "ok" if np.isfinite(estimate) else "invalid_estimate",
            "estimated_tension_N": float(estimate) if np.isfinite(estimate) else None,
            "tracking": _public_tracking_summary(tracking),
            "estimate_id": f"estimate_{self._estimate_count:03d}",
            "step": self._step_count(),
        }
        self._trace("tension_estimate", record)
        return _jsonable(record)

    def get_tactile_stage_memory(self) -> dict[str, Any]:
        """Return the frozen 12N/18N public tactile-response memory.

        The caller owns response distance, confidence, and stage-transition
        decisions. The returned data contains no current-stage label.
        """
        memory = self._load_stage_memory()
        _validate_stage_memory(memory)
        record = _jsonable(memory)
        self._trace(
            "stage_memory",
            {
                "schema_version": record["schema_version"],
                "protocol_id": record["protocol_id"],
                "memory_ids": [stage["memory_id"] for stage in record["stages"]],
            },
        )
        return record

    def capture_tactile_stage_response(self) -> dict[str, Any]:
        """Reduce the newest public hold window into a 10D-compatible response."""
        memory = self._load_stage_memory()
        _validate_stage_memory(memory)
        capture = memory["capture"]
        _refresh_tactile(self._env)
        calibration = self._native_tactile_calibration()
        response = summarize_tactile_stage_response(
            self._env.tactile_buffer.recent(int(capture["window_steps"])),
            window_steps=int(capture["window_steps"]),
            edge_window_steps=int(capture["edge_window_steps"]),
            min_bilateral_contact_ratio=float(capture["min_bilateral_contact_ratio"]),
            contact_area_threshold=float(capture["contact_area_threshold"]),
            depth_far_plane_mm=calibration.get("depth_far_plane_mm"),
            depth_contact_margin_mm=float(calibration.get("depth_contact_margin_mm", 0.5)),
        )
        self._capture_count += 1
        record = {
            **response,
            "capture_id": f"stage_capture_{self._capture_count:03d}",
            "protocol_id": str(memory["protocol_id"]),
            "step": self._step_count(),
        }
        if record.get("schema_version") != _STAGE_RESPONSE_SCHEMA:
            raise RuntimeError("OpenTac stage reducer returned an invalid schema")
        self._trace("stage_response_capture", {"record": record})
        return _jsonable(record)

    def _runtime_config(self) -> dict[str, Any]:
        configs = getattr(self._env, "api_configs", {})
        config = configs.get("opentac_api", {}) if isinstance(configs, dict) else {}
        return config if isinstance(config, dict) else {}

    def _read_marker_images(self) -> dict[str, np.ndarray]:
        _refresh_tactile(self._env, data_types=["rgb_marker"])
        raw_fn = getattr(self._env, "current_raw_observation", None)
        raw = raw_fn() if callable(raw_fn) else {}
        tactile = raw.get("tactile", {}) if isinstance(raw, dict) else {}
        images = {
            str(name): _as_uint8_rgb(record["rgb_marker"])
            for name, record in tactile.items()
            if isinstance(record, dict) and record.get("rgb_marker") is not None
        }
        if len(images) < 2:
            raise RuntimeError("OpenTac requires marker-RGB observations from both tactile sensors")
        return dict(sorted(images.items()))

    def _load_tension_calibration(self) -> dict[str, Any]:
        value = self._runtime_config().get("tension_calibration_path")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("OpenTac requires tension_calibration_path")
        path = _resolve_path(self._env, value)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not load OpenTac calibration from {path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("weights"), list):
            raise RuntimeError("OpenTac tension calibration has no public linear weights")
        return payload

    def _load_stage_memory(self) -> dict[str, Any]:
        config = self._runtime_config()
        source = config.get("tactile_stage_memory")
        path_value = config.get("tactile_stage_memory_path")
        if source is None and isinstance(path_value, str) and path_value.strip():
            path = _resolve_path(self._env, path_value)
            try:
                source = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"could not load OpenTac stage memory from {path}: {exc}") from exc
        if not isinstance(source, dict):
            raise RuntimeError("OpenTac requires a tactile_stage_memory.v1 sidecar")
        return source

    def _force_task_utilities(self) -> Any:
        try:
            return importlib.import_module("envs._force_task_utils")
        except Exception as exc:
            raise RuntimeError("OpenTac must run with a ViTaForge-compatible task root") from exc

    def _native_tactile_calibration(self) -> dict[str, float | None]:
        getter = getattr(self._env, "get_native_tactile_calibration", None)
        raw = getter() if callable(getter) else {}
        raw = raw if isinstance(raw, dict) else {}
        far_plane = raw.get("depth_far_plane_mm")
        return {
            "depth_far_plane_mm": float(far_plane) if _finite(far_plane) else None,
            "depth_contact_margin_mm": float(raw.get("depth_contact_margin_mm", 0.5)),
        }

    def _step_count(self) -> int | None:
        getter = getattr(self._env, "get_step_count", None)
        try:
            return int(getter()) if callable(getter) else None
        except Exception:
            return None

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        append = getattr(self._env, "append_tactile_working_memory_trace", None)
        if callable(append):
            append({"schema_version": "opentac_event.v1", "event": event, "step": self._step_count(), "payload": _jsonable(payload)})


def _refresh_tactile(env: BaseEnv, data_types: list[str] | None = None) -> None:
    refresh = getattr(env, "refresh_native_observation", None)
    if callable(refresh):
        refresh(include_camera=False, include_tactile=True, tactile_data_types=data_types)
    else:
        env.get_observation()


def _resolve_path(env: BaseEnv, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(getattr(env, "univtac_root", Path.cwd()))
    capx_root = Path(__file__).resolve().parents[3]
    for candidate in (root / path, capx_root / path):
        if candidate.is_file():
            return candidate
    return root / path


def _validate_stage_memory(source: dict[str, Any]) -> None:
    if source.get("schema_version") != _STAGE_MEMORY_SCHEMA:
        raise RuntimeError("OpenTac stage memory must use tactile_stage_memory.v1")
    if source.get("response_schema_version") != _STAGE_RESPONSE_SCHEMA:
        raise RuntimeError("OpenTac stage memory must target tactile_stage_response.v1")
    capture = source.get("capture")
    if not isinstance(capture, dict):
        raise RuntimeError("OpenTac stage memory must define a capture protocol")
    window, edge = capture.get("window_steps"), capture.get("edge_window_steps")
    if not isinstance(window, int) or not isinstance(edge, int) or edge < 1 or edge * 2 > window:
        raise RuntimeError("OpenTac stage memory has an invalid response window")
    stages = source.get("stages")
    if not isinstance(stages, list) or len(stages) < 2:
        raise RuntimeError("OpenTac stage memory requires at least two frozen stages")
    forbidden = {"actor", "density", "friction", "metadata", "pose", "reward", "success", "tension"}
    layouts: list[tuple[str, tuple[str, ...]]] = []
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("memory_id"), str):
            raise RuntimeError("OpenTac stage memory has an invalid stage record")
        blocks = stage.get("response_blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError("OpenTac stage memory has no response blocks")
        layout: list[tuple[str, tuple[str, ...]]] = []
        for block in blocks:
            fields = block.get("fields") if isinstance(block, dict) else None
            scaler = block.get("scaler") if isinstance(block, dict) else None
            if not isinstance(block, dict) or not isinstance(block.get("name"), str) or not isinstance(fields, list) or not isinstance(scaler, dict):
                raise RuntimeError("OpenTac stage memory has an invalid response block")
            paths = tuple(str(field.get("path", "")) for field in fields if isinstance(field, dict))
            if len(paths) != len(fields) or any(not path.startswith(("hold.", "end_minus_start.")) or forbidden.intersection(path.lower().split(".")) for path in paths):
                raise RuntimeError("OpenTac stage memory contains a non-public response field")
            for name in ("median", "iqr"):
                values = scaler.get(name)
                if not isinstance(values, list) or len(values) != len(paths) or not np.isfinite(np.asarray(values, dtype=float)).all():
                    raise RuntimeError("OpenTac stage memory has an invalid block scaler")
            if np.any(np.asarray(scaler["iqr"], dtype=float) <= 0.0):
                raise RuntimeError("OpenTac stage memory has a nonpositive block IQR")
            layout.append((block["name"], paths))
        layouts.append((stage["memory_id"], tuple(name for name, _ in layout)))
    if len({layout for _, layout in layouts}) != 1:
        raise RuntimeError("OpenTac stage memories must share one response layout")
    retrieval = source.get("retrieval_scaler")
    if not isinstance(retrieval, dict) or retrieval.get("method") != "pooled_calibration_iqr":
        raise RuntimeError("OpenTac retrieval must use pooled_calibration_iqr")


def _as_uint8_rgb(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[0]
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * (255.0 if np.nanmax(image) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _public_tracking_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): {
            "matched": int(item.get("matched", 0)),
            "p90_px": float(item.get("p90_px", 0.0)),
        }
        for key, item in value.items()
        if isinstance(item, dict)
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
