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
        self._observer_name = f"opentac_tension_estimator_{id(self)}"
        self._observer_registered = False
        self._latest_estimate: dict[str, Any] | None = None
        self._last_observed_step: int | None = None
        self._stage_memory_cache: dict[str, Any] | None = None

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
        self._unregister_observers()
        self._capture_count = 0
        self._estimate_count = 0
        self._tracker = None
        self._reference_images = None
        self._calibration = None
        self._latest_estimate = None
        self._last_observed_step = None
        self._stage_memory_cache = None
        self._publish_diagnostics()

    def get_tactile_tension_control_contract(self) -> dict[str, Any]:
        """Return the public local-control contract for the force task.

        It exposes only public action safety, sampling cadence, and target
        bands. The runtime physical scorer is never exposed.
        """
        config = self._runtime_config()
        stage_targets = _float_list(config.get("stage_targets_N", [12.0, 18.0]))
        stage_tolerance = float(config.get("stage_tolerance_N", 0.5))
        contract = {
            "schema_version": _CONTROL_SCHEMA,
            "control_frame": "world",
            "allowed_translation_axes": ["z"],
            "max_delta_z_m": float(config.get("max_delta_z_m", 0.002)),
            "observation_wait_steps": int(config.get("observation_wait_steps", 2)),
            "observation_dt_s": float(config.get("observation_dt_s", 1.0 / 60.0)),
            "stage_hold_seconds": float(config.get("stage_hold_seconds", 3.0)),
            "estimator_settle_steps": int(config.get("estimator_settle_steps", 30)),
            "proportional_delta_gain_m_per_N": float(
                config.get("proportional_delta_gain_m_per_N", 1.0 / 120000.0)
            ),
            "stage_targets_N": stage_targets,
            "stage_bands_N": [
                [float(target - stage_tolerance), float(target + stage_tolerance)]
                for target in stage_targets
            ],
            "estimator_update_stride": int(config.get("estimator_update_stride", 1)),
            "max_estimate_age_steps": int(config.get("max_estimate_age_steps", 3)),
        }
        if contract["max_delta_z_m"] <= 0.0:
            raise RuntimeError("OpenTac max_delta_z_m must be positive")
        if (
            contract["observation_wait_steps"] < 1
            or contract["estimator_settle_steps"] < 1
            or contract["proportional_delta_gain_m_per_N"] <= 0.0
            or contract["observation_dt_s"] <= 0.0
            or stage_tolerance <= 0.0
            or contract["estimator_update_stride"] < 1
            or contract["max_estimate_age_steps"] < 0
            or len(contract["stage_targets_N"]) != 2
            or any(target <= 0.0 for target in contract["stage_targets_N"])
        ):
            raise RuntimeError("OpenTac control step limits must be positive")
        return _jsonable(contract)

    def begin_tactile_tension_estimator(self) -> dict[str, Any]:
        """Capture a marker-RGB baseline after the agent has secured the strap."""
        # Repeated calls preserve the marker baseline.  The API has no notion
        # of control-stage progress; whether to resume, regrasp, or stop is
        # owned entirely by generated agent code.
        if self._tracker is not None:
            return _jsonable(
                {
                    "schema_version": _ESTIMATE_SCHEMA,
                    "started": True,
                    "step": self._step_count(),
                }
            )
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
        # A baseline establishes marker identities; it is not itself a flow
        # sample. Updating a temporal tracker against the identical baseline
        # image would consume its first coarse-flow update and make the first
        # physical motion use only the fragile local correspondence search.
        self._latest_estimate = self._baseline_pending_record()
        self._last_observed_step = self._step_count()
        self._register_observers()
        record = {
            "schema_version": _ESTIMATE_SCHEMA,
            "started": True,
            "sensor_keys": sorted(images),
            "step": self._step_count(),
        }
        self._trace("tension_estimator_begin", record)
        self._publish_diagnostics()
        return _jsonable(record)

    def get_tactile_tension_estimate(self) -> dict[str, Any]:
        """Estimate strap tension from public marker-RGB flow and calibration.

        Call :meth:`begin_tactile_tension_estimator` once after a bilateral
        grasp.  A returned estimate is an observation-model output, not the
        simulator's physical tension value.  If tracking is temporarily not
        usable, ``available`` is false and ``estimated_tension_N`` is null;
        the API does not count failures or prescribe a recovery policy.
        """
        if self._tracker is None or self._calibration is None or self._reference_images is None:
            raise RuntimeError("call begin_tactile_tension_estimator() after grasping first")
        record = dict(self._latest_estimate or self._unavailable_record())
        age_steps = self._estimate_age_steps(record)
        record["observation_age_steps"] = age_steps
        if record.get("available") is True and age_steps > int(
            self.get_tactile_tension_control_contract()["max_estimate_age_steps"]
        ):
            record = {
                "schema_version": _ESTIMATE_SCHEMA,
                "available": False,
                "estimated_tension_N": None,
                "sample_step": record.get("sample_step"),
                "step": self._step_count(),
                "observation_age_steps": age_steps,
            }
        return _jsonable(record)

    def get_tactile_stage_memory(self) -> dict[str, Any]:
        """Return the frozen 12N/18N public tactile-response memory.

        The caller owns response distance, confidence, and stage-transition
        decisions. The returned data contains no current-stage label.
        """
        memory = self._load_stage_memory()
        _validate_stage_memory(memory)
        record = _jsonable(memory)
        publish_memory = getattr(
            self._env, "set_tension_stage_memory_visualization", None
        )
        if callable(publish_memory):
            # Frozen medians are useful video context even before the first
            # completed 3-second capture exists. This is display-only.
            publish_memory(record)
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
        # The video renderer receives the public capture and frozen medians as
        # a read-only diagnostic. This never feeds into OpenTac control or the
        # LLM request context.
        publish_visualization = getattr(
            self._env, "set_tension_response_visualization", None
        )
        if callable(publish_visualization):
            publish_visualization(record, memory)
        self._trace("stage_response_capture", {"record": record})
        return _jsonable(record)

    def runtime_memory_context(self, max_chars: int = 4000) -> str:
        """Return only the latest public numeric estimate for recovery context."""
        latest = self.get_tactile_tension_estimate() if self._tracker is not None else None
        context = {
            "schema_version": _ESTIMATE_SCHEMA,
            "estimator_active": self._tracker is not None,
            "latest_estimate": latest,
        }
        text = json.dumps(_jsonable(context), ensure_ascii=False, separators=(",", ":"))
        return text[: max(0, int(max_chars))]

    def _register_observers(self) -> None:
        if self._observer_registered:
            return
        register_step = getattr(self._env, "register_post_step_observer", None)
        if callable(register_step):
            register_step(self._observer_name, self._on_post_task_step)
        self._observer_registered = callable(register_step)

    def _unregister_observers(self) -> None:
        unregister_step = getattr(self._env, "unregister_post_step_observer", None)
        if callable(unregister_step):
            unregister_step(self._observer_name)
        self._observer_registered = False

    def _on_post_task_step(self) -> None:
        if self._tracker is None:
            return
        self._sample_estimator()

    def _sample_estimator(self, *, force: bool = False) -> None:
        if self._tracker is None or self._calibration is None or self._reference_images is None:
            return
        step = self._step_count()
        if step is not None and self._last_observed_step == step and not force:
            return
        if step is not None and not force:
            stride = int(self.get_tactile_tension_control_contract()["estimator_update_stride"])
            if self._last_observed_step is not None and step - self._last_observed_step < stride:
                return
        if step is not None:
            self._last_observed_step = step
        try:
            images = self._read_marker_images()
            utilities = self._force_task_utilities()
            features, tracking = utilities.tracked_flow_rgb_features(
                self._reference_images,
                images,
                self._tracker,
            )
            estimate = float(utilities.predict_calibrated(self._calibration, features))
        except Exception as exc:
            self._record_invalid_estimate("tracking_unavailable", str(exc), step)
            return

        if not np.isfinite(estimate):
            self._record_invalid_estimate("invalid_estimate", "calibration returned non-finite value", step)
            return

        self._estimate_count += 1
        record = {
            "schema_version": _ESTIMATE_SCHEMA,
            "available": True,
            "estimated_tension_N": float(estimate),
            "tracking": _public_tracking_summary(tracking),
            "estimate_id": f"estimate_{self._estimate_count:03d}",
            "sample_step": step,
            "step": step,
        }
        self._latest_estimate = record
        self._publish_diagnostics()
        self._publish_live_stage_response()

    def _publish_live_stage_response(self) -> None:
        """Refresh the video-only rolling 10D preview from public frames.

        This does not create a capture artifact, advance a stage, or enter the
        LLM context. It is only the dashboard's view of the recent public
        tactile window that a later explicit capture will reduce.
        """
        publish = getattr(self._env, "set_tension_response_preview", None)
        tactile_buffer = getattr(self._env, "tactile_buffer", None)
        if not callable(publish) or tactile_buffer is None:
            return
        try:
            memory = self._load_stage_memory()
            capture = memory["capture"]
            calibration = self._native_tactile_calibration()
            response = summarize_tactile_stage_response(
                tactile_buffer.recent(int(capture["window_steps"])),
                window_steps=int(capture["window_steps"]),
                edge_window_steps=int(capture["edge_window_steps"]),
                min_bilateral_contact_ratio=float(capture["min_bilateral_contact_ratio"]),
                contact_area_threshold=float(capture["contact_area_threshold"]),
                depth_far_plane_mm=calibration.get("depth_far_plane_mm"),
                depth_contact_margin_mm=float(
                    calibration.get("depth_contact_margin_mm", 0.5)
                ),
            )
            step = self._step_count()
            response.update(
                {
                    "capture_id": f"live_step_{step}" if step is not None else "live",
                    "protocol_id": str(memory["protocol_id"]),
                    "step": step,
                }
            )
            publish(response, memory)
        except Exception:
            # Preview must never interrupt control or turn a rendering hiccup
            # into a false environment/control failure.
            return

    def _record_invalid_estimate(self, _status: str, _message: str, step: int | None) -> None:
        """Cache an unavailable sample without imposing a recovery policy."""
        self._latest_estimate = {
            "schema_version": _ESTIMATE_SCHEMA,
            "available": False,
            "estimated_tension_N": None,
            "sample_step": step,
            "step": step,
        }
        self._publish_diagnostics()

    def _estimate_age_steps(self, estimate: dict[str, Any]) -> int | None:
        current_step = self._step_count()
        sample_step = estimate.get("sample_step")
        if not isinstance(current_step, int) or not isinstance(sample_step, int):
            return None
        return max(0, int(current_step - sample_step))

    def _unavailable_record(self) -> dict[str, Any]:
        return {
            "schema_version": _ESTIMATE_SCHEMA,
            "available": False,
            "estimated_tension_N": None,
            "sample_step": self._step_count(),
            "step": self._step_count(),
        }

    def _baseline_pending_record(self) -> dict[str, Any]:
        step = self._step_count()
        return {
            "schema_version": _ESTIMATE_SCHEMA,
            "available": False,
            "estimated_tension_N": None,
            "sample_step": step,
            "step": step,
        }

    def _publish_diagnostics(self) -> None:
        setter = getattr(self._env, "set_opentac_tension_estimator_diagnostics", None)
        if not callable(setter):
            return
        latest = self._latest_estimate or {}
        setter(
            {
                "schema_version": _ESTIMATE_SCHEMA,
                "estimator_active": self._tracker is not None,
                "latest_estimate": latest,
            }
        )

    def _runtime_config(self) -> dict[str, Any]:
        configs = getattr(self._env, "api_configs", {})
        config = configs.get("opentac_api", {}) if isinstance(configs, dict) else {}
        return config if isinstance(config, dict) else {}

    def _read_marker_images(self) -> dict[str, np.ndarray]:
        # The tracker consumes marker RGB; the co-sampled public depth and
        # marker coordinates are retained for the runtime 10D panel/capture.
        _refresh_tactile(self._env, data_types=["rgb_marker", "depth", "marker"])
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
        if self._stage_memory_cache is not None:
            return self._stage_memory_cache
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
        self._stage_memory_cache = source
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


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("OpenTac stage_targets_N must be a list")
    values = [float(item) for item in value]
    if not all(np.isfinite(item) for item in values):
        raise RuntimeError("OpenTac stage_targets_N must be finite")
    return values


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
