"""Native tactile API for CaP-X on UniVTAC."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase

from .native_tactile import summarize_native_tactile, tactile_event_sequence


_TRIAL_MEMORY_MAX_ENTRY_BYTES = 8 * 1024
_TRIAL_MEMORY_MAX_TOTAL_BYTES = 32 * 1024
_TACTILE_OBSERVATION_SCHEMA = "tactile_observation.v1"
_TRIAL_MEMORY_RECORD_SCHEMA = "trial_memory.v1"
_TRIAL_MEMORY_SNAPSHOT_SCHEMA = "trial_memory_snapshot.v1"
_TRIAL_MEMORY_EVENT_SCHEMA = "trial_memory_event.v1"
_TRIAL_MEMORY_OPERATION_SCHEMA = "trial_memory_operation.v1"
_TRIAL_MEMORY_KINDS = {"evidence", "hypothesis", "decision", "state"}
_TACTILE_MEASUREMENT_PROTOCOL_SCHEMA = "tactile_measurement_protocol.v1"
_DEFAULT_TACTILE_MEASUREMENT_PROTOCOL = {
    "capture_window": 20,
    "settle_steps": 10,
    "close_target_force": 0.82,
    "close_max_steps": 120,
    "probe_lift_m": 0.018,
    "probe_hold_steps": 10,
    "max_attempts_per_object": 2,
    "adaptive_close": True,
}


class UniVTACTactileApi(ApiBase):
    """Expose UniVTAC native tactile observations to CaP-generated code."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self._trial_memory: dict[str, dict[str, Any]] = {}
        self._working_memory_trace: list[dict[str, Any]] = []
        self._capture_count = 0
        cfg = self._runtime_config()
        visible_functions = cfg.get("llm_visible_functions")
        if visible_functions is not None and not isinstance(visible_functions, (list, tuple)):
            raise ValueError("llm_visible_functions must be a list when configured")
        self.llm_visible_functions = (
            {str(name).strip() for name in visible_functions}
            if visible_functions is not None
            else None
        )

    def functions(self) -> dict[str, Any]:
        full = {
            "get_tactile_summary": self.get_tactile_summary,
            "is_contacting": self.is_contacting,
            "is_slipping": self.is_slipping,
            "is_grasp_stable": self.is_grasp_stable,
            "wait_until_contact": self.wait_until_contact,
            "get_tactile_image": self.get_tactile_image,
            "get_tactile_depth": self.get_tactile_depth,
            "get_marker_motion_summary": self.get_marker_motion_summary,
            "get_recent_tactile_events": self.get_recent_tactile_events,
            "get_tactile_measurement_protocol": self.get_tactile_measurement_protocol,
            "get_public_probe_spec": self.get_public_probe_spec,
            "begin_public_probe_capture": self.begin_public_probe_capture,
            "begin_public_probe_segment": self.begin_public_probe_segment,
            "end_public_probe_segment": self.end_public_probe_segment,
            "finalize_public_probe_capture": self.finalize_public_probe_capture,
            "get_provisional_slot_expression": self.get_provisional_slot_expression,
            "get_tactile_response_expression": self.get_tactile_response_expression,
            "capture_tactile_observation": self.capture_tactile_observation,
            "write_trial_memory": self.write_trial_memory,
            "read_trial_memory": self.read_trial_memory,
            "list_trial_memory": self.list_trial_memory,
            "clear_trial_memory": self.clear_trial_memory,
        }
        visible_functions = getattr(self, "llm_visible_functions", None)
        if visible_functions is None:
            return full
        unknown = visible_functions.difference(full)
        if unknown:
            raise ValueError(
                "llm_visible_functions contains unsupported UniVTACTactileApi functions: "
                f"{sorted(unknown)}"
            )
        return {name: full[name] for name in full if name in visible_functions}

    def reset_episode(self) -> None:
        """Clear native tactile and agent-owned memory at the start of a trial."""
        if hasattr(self._env, "reset_tactile_buffer"):
            self._env.reset_tactile_buffer()
        elif hasattr(self._env, "tactile_buffer"):
            self._env.tactile_buffer.clear()
        self._working_memory_trace = []
        self._capture_count = 0
        self.clear_trial_memory()

    def get_tactile_summary(self, window: int = 20, hand: str = "both") -> dict:
        """Return native UniVTAC tactile summary over recent frames.

        Args:
            window: Number of recent native tactile frames to aggregate.
            hand: Which tactile sensor to inspect: left, right, or both.

        Returns:
            Dictionary with contact, left_contact, right_contact, normal_force,
            contact_area, depth_delta_mm, shear_magnitude, slip_score,
            contact_balance, stable, per-hand metrics, and event. The summary
            is based only on UniVTAC tactile depth and marker outputs.
        """
        _refresh_native_tactile(self._env, data_types=["rgb", "rgb_marker", "marker", "depth", "pose"])
        frames = self._env.tactile_buffer.recent(window)
        summary = summarize_native_tactile(
            frames,
            hand=hand,
            **_native_tactile_calibration(self._env),
        )
        print(
            "[univtac-tactile] "
            f"hand={hand} contact={summary['contact']} "
            f"left={summary['left_contact']} right={summary['right_contact']} "
            f"force={summary['normal_force']:.3f} depth={summary['depth_delta_mm']:.3f}mm "
            f"shear={summary['shear_magnitude']:.3f} "
            f"centroid={summary['marker_centroid_displacement']:.3f} "
            f"slip={summary['slip_score']:.3f} stable={summary['stable']} "
            f"event={summary['event']}"
        )
        return summary

    def is_contacting(self, hand: str = "both", threshold: float = 0.2) -> bool:
        """Return whether native tactile sensing indicates contact.

        Args:
            hand: Which tactile sensor to inspect: left, right, or both.
            threshold: Minimum normal_force proxy for reliable contact.

        Returns:
            True if contact is detected and normal_force exceeds threshold.
        """
        summary = self.get_tactile_summary(hand=hand)
        return bool(summary["contact"] and summary["normal_force"] >= threshold)

    def is_slipping(self, hand: str = "both", threshold: float = 0.6) -> bool:
        """Return whether native tactile marker/depth history indicates slip.

        Args:
            hand: Which tactile sensor to inspect: left, right, or both.
            threshold: Slip score threshold in [0, 1].

        Returns:
            True when slip_score is greater than or equal to threshold.
        """
        summary = self.get_tactile_summary(hand=hand)
        return bool(summary["slip_score"] >= threshold)

    def is_grasp_stable(self, threshold: float = 0.5) -> bool:
        """Return whether both native tactile sensors indicate a stable grasp.

        Args:
            threshold: Minimum normal_force proxy for a stable two-hand grasp.

        Returns:
            True when both sensors have contact, force is sufficient, contact is
            balanced, and slip_score is low.
        """
        summary = self.get_tactile_summary(hand="both")
        min_area = float(summary.get("stable_contact_area_threshold", 0.01))
        return bool(
            summary["event"] == "stable_grasp"
            and summary["left_contact"]
            and summary["right_contact"]
            and summary["normal_force"] >= threshold
            and float(summary["left"].get("contact_area", 0.0)) >= min_area
            and float(summary["right"].get("contact_area", 0.0)) >= min_area
            and abs(summary["contact_balance"]) <= 0.45
            and summary["slip_score"] < 0.6
        )

    def wait_until_contact(
        self,
        timeout: float = 2.0,
        hand: str = "both",
        threshold: float = 0.2,
    ) -> dict:
        """Wait until native tactile contact is detected or timeout expires.

        Args:
            timeout: Maximum wall-clock seconds to monitor.
            hand: Which tactile sensor to inspect: left, right, or both.
            threshold: Minimum normal_force proxy for reliable contact.

        Returns:
            Last tactile summary at contact or timeout.
        """
        deadline = time.time() + max(0.0, float(timeout))
        summary = self.get_tactile_summary(hand=hand)
        while time.time() < deadline:
            if summary["contact"] and summary["normal_force"] >= threshold:
                return summary
            self._env.wait_steps(1)
            summary = self.get_tactile_summary(hand=hand)
        return summary

    def get_tactile_image(self, hand: str = "left", image_type: str = "rgb_marker") -> np.ndarray:
        """Return a native UniVTAC tactile image.

        Args:
            hand: left or right.
            image_type: rgb or rgb_marker.

        Returns:
            Tactile image as a numpy array from UniVTAC observation.
        """
        if image_type not in {"rgb", "rgb_marker"}:
            raise ValueError("image_type must be 'rgb' or 'rgb_marker'")

        candidates = [image_type]
        candidates.append("rgb" if image_type == "rgb_marker" else "rgb_marker")
        errors: list[str] = []
        for candidate in candidates:
            try:
                hand_obs = self._hand_observation(
                    hand,
                    data_types=[candidate, "marker", "depth", "pose"],
                )
            except KeyError as exc:
                errors.append(str(exc))
                continue
            if candidate in hand_obs:
                if candidate != image_type:
                    print(
                        "[univtac-tactile] "
                        f"image_type={image_type} unavailable for {hand}; using {candidate}",
                        flush=True,
                    )
                return _to_numpy(hand_obs[candidate])
            errors.append(f"{candidate} absent; available={sorted(hand_obs.keys())}")

        raise KeyError(
            f"{image_type} not available for {hand}; tried={candidates}; details={errors}"
        )

    def get_tactile_depth(self, hand: str = "left") -> np.ndarray:
        """Return a native UniVTAC tactile depth / height map.

        Args:
            hand: left or right.

        Returns:
            Depth or height map as a numpy array from UniVTAC observation.
        """
        hand_obs = self._hand_observation(hand, data_types=["depth", "marker", "pose"])
        if "depth" not in hand_obs:
            raise KeyError(f"depth not available for {hand}; available={sorted(hand_obs.keys())}")
        return _to_numpy(hand_obs["depth"])

    def get_marker_motion_summary(self, hand: str = "both", window: int = 20) -> dict:
        """Return compact marker motion statistics from native UniVTAC tactile data.

        Args:
            hand: Which tactile sensor to inspect: left, right, or both.
            window: Number of recent tactile frames to aggregate.

        Returns:
            Dictionary with marker displacement statistics and shear magnitude.
        """
        summary = self.get_tactile_summary(window=window, hand=hand)
        left_mean = summary["left"]["marker_mean_displacement"]
        right_mean = summary["right"]["marker_mean_displacement"]
        left_max = summary["left"]["marker_max_displacement"]
        right_max = summary["right"]["marker_max_displacement"]
        left_centroid = summary["left"]["marker_centroid_displacement"]
        right_centroid = summary["right"]["marker_centroid_displacement"]
        return {
            "hand": hand,
            "shear_magnitude": summary["shear_magnitude"],
            "marker_centroid_displacement": summary["marker_centroid_displacement"],
            "mean_displacement": max(left_mean, right_mean),
            "max_displacement": max(left_max, right_max),
            "left_marker_mean_displacement": summary["left"]["marker_mean_displacement"],
            "right_marker_mean_displacement": summary["right"]["marker_mean_displacement"],
            "left_marker_max_displacement": summary["left"]["marker_max_displacement"],
            "right_marker_max_displacement": summary["right"]["marker_max_displacement"],
            "left_marker_centroid_displacement": left_centroid,
            "right_marker_centroid_displacement": right_centroid,
            "left_marker_displacement_px": summary["left"]["marker_displacement_px"],
            "right_marker_displacement_px": summary["right"]["marker_displacement_px"],
            "left_marker_coherence": summary["left"]["marker_coherence"],
            "right_marker_coherence": summary["right"]["marker_coherence"],
        }

    def get_recent_tactile_events(self, window: int = 50) -> list[str]:
        """Return recent native tactile event sequence.

        Args:
            window: Number of recent native tactile frames to inspect.

        Returns:
            Deduplicated event list such as no_contact, one_hand_contact,
            stable_grasp, slip_detected, or contact_lost.
        """
        _refresh_native_tactile(self._env, data_types=["rgb", "rgb_marker", "marker", "depth", "pose"])
        events = tactile_event_sequence(
            self._env.tactile_buffer.recent(window),
            **_native_tactile_calibration(self._env),
        )
        print(f"[univtac-tactile] recent_events={events}")
        return events

    def get_tactile_measurement_protocol(self) -> dict:
        """Return the public task-neutral calibration for one tactile probe.

        The caller must persist this dictionary as its ``probe_spec`` and reuse
        its close target force, close limit, capture window, lift, and hold
        unchanged for the reference and both candidates. It intentionally
        contains no object pose, identity, physical label, or task outcome.
        """
        protocol = _tactile_measurement_protocol(self._env)
        self._append_working_memory_trace("protocol", {"protocol": protocol})
        return protocol

    def get_public_probe_spec(self) -> dict:
        """Return the benchmark-owned standard tactile probe specification.

        This is the public action protocol shared by the official expert and
        external agents. It contains only measurement timing and quality gates;
        it contains no pose, hidden physical class, reward, or match identity.
        Call it once and reuse the unchanged result for reference and both
        candidates.
        """
        getter = getattr(self._env, "get_public_probe_spec", None)
        if not callable(getter):
            raise RuntimeError("the active environment does not expose a public probe spec")
        spec = getter()
        if not isinstance(spec, dict) or spec.get("schema_version") != "public_probe_spec.v2":
            raise RuntimeError("environment returned an invalid public_probe_spec.v2 record")
        normalized = _jsonable(spec)
        self._append_working_memory_trace("public_probe_spec", {"spec": normalized})
        return normalized

    def begin_public_probe_capture(self, object_name: str) -> dict:
        """Start recording one externally executed standard probe.

        This function does not move the robot. Use
        :meth:`begin_public_probe_segment` and
        :meth:`end_public_probe_segment` around the public protocol's preload,
        lift-motion, and hold actions; then call
        :meth:`finalize_public_probe_capture`.
        """
        begin = getattr(self._env, "begin_public_probe_capture", None)
        if not callable(begin):
            raise RuntimeError("the active environment does not support public probe recording")
        result = _jsonable(begin(object_name))
        self._append_working_memory_trace("public_probe_begin", {"record": result})
        return result

    def begin_public_probe_segment(self, capture_id: str, segment: str) -> dict:
        """Begin recording a public ``preload``, ``lift_motion``, or ``hold`` segment."""
        begin = getattr(self._env, "begin_public_probe_segment", None)
        if not callable(begin):
            raise RuntimeError("the active environment does not support public probe segments")
        result = _jsonable(begin(capture_id, segment))
        self._append_working_memory_trace("public_probe_segment_begin", {"record": result})
        return result

    def end_public_probe_segment(self, capture_id: str, segment: str) -> dict:
        """Stop and aggregate a public probe segment using the expert v4 reducer."""
        end = getattr(self._env, "end_public_probe_segment", None)
        if not callable(end):
            raise RuntimeError("the active environment does not support public probe segments")
        result = _jsonable(end(capture_id, segment))
        self._append_working_memory_trace("public_probe_segment_end", {"record": result})
        return result

    def finalize_public_probe_capture(self, capture_id: str, execution: dict) -> dict:
        """Build a public ``tactile_probe.v4`` record from agent-owned motion.

        ``execution`` reports only public action outcomes: ``approach_ok``,
        ``close_ok``, ``bilateral_gate``, ``lift_ok``, ``lower_ok``,
        ``release_ok``, and ``clearance_ok``. The environment aggregates raw
        tactile frames but does not construct an identity score or choice.
        """
        finalize = getattr(self._env, "finalize_public_probe_capture", None)
        if not callable(finalize):
            raise RuntimeError("the active environment does not support public probe finalization")
        result = _jsonable(finalize(capture_id, execution))
        self._append_working_memory_trace("public_probe_finalize", {"record": result})
        return result

    def get_provisional_slot_expression(self) -> dict:
        """Return the read-only public feature rule for the engineering demo.

        It only names paths in ``tactile_probe.v3`` and confidence inputs. The
        caller computes vectors, distances, fusion, and the candidate choice.
        """
        source = self._runtime_config().get("tactile_slot_expression", {})
        if not isinstance(source, dict) or source.get("schema_version") != "provisional_slot_expression.v0":
            raise RuntimeError("no provisional_slot_expression.v0 is configured")
        expression = _jsonable(source)
        self._append_working_memory_trace("slot_expression", {"expression": expression})
        return expression

    def get_tactile_response_expression(self) -> dict:
        """Return the read-only public tactile response expression.

        The expression is a frozen protocol/sensor-local normalizer.  It names
        public ``tactile_probe.v4`` feature paths, their transforms and Median/
        IQR scales, plus a fixed quality-weighted block-RMS comparison rule.
        It does not create a memory record, compute a distance, choose a
        candidate, or contain hidden object properties.

        Configure either ``tactile_response_expression`` directly or
        ``tactile_response_expression_path``.  Relative paths are resolved
        against the configured UniVTAC root so the same public sidecar can be
        loaded by other model integrations.
        """
        source = self._runtime_config().get("tactile_response_expression")
        path_value = self._runtime_config().get("tactile_response_expression_path")
        if source is None and isinstance(path_value, str) and path_value.strip():
            path = Path(path_value).expanduser()
            if not path.is_absolute():
                root = Path(getattr(self._env, "univtac_root", Path.cwd()))
                path = root / path
            try:
                source = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"could not load tactile response expression from {path}: {exc}") from exc
        if not isinstance(source, dict) or source.get("schema_version") != "tactile_response_expression.v1":
            raise RuntimeError("no tactile_response_expression.v1 is configured")
        if source.get("public_probe_schema_version") != "tactile_probe.v4":
            raise RuntimeError("tactile response expression must target tactile_probe.v4")
        _validate_tactile_response_expression(source)
        protocol_id = source.get("protocol_id")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise RuntimeError("tactile response expression is missing its public probe protocol_id")
        get_probe_spec = getattr(self._env, "get_public_probe_spec", None)
        if callable(get_probe_spec):
            active_spec = get_probe_spec()
            if isinstance(active_spec, dict):
                active_protocol_id = active_spec.get("protocol_id")
                if isinstance(active_protocol_id, str) and active_protocol_id != protocol_id:
                    raise RuntimeError(
                        "tactile response expression protocol_id does not match the active public probe"
                    )
        expression = _jsonable(source)
        self._append_working_memory_trace("response_expression", {"expression": expression})
        return expression

    def capture_tactile_observation(self, window: int = 20) -> dict:
        """Capture one canonical public tactile observation for this trial.

        The returned ``tactile_observation.v1`` record contains
        ``record[\"tactile\"]`` with whole-gripper and explicit
        ``left_*``/``right_*`` fields, ``record[\"marker_motion\"]``, and
        ``record[\"embodiment\"]``. For example, use
        ``record[\"tactile\"][\"left_normal_force\"]`` rather than guessing
        a nested ``summary`` field. For tactile matching, use
        ``record["tactile"]["left_depth_mm"]`` and
        ``record["tactile"]["right_depth_mm"]`` with
        ``record["marker_motion"]["left_marker_displacement_px"]``,
        ``record["marker_motion"]["right_marker_displacement_px"]``,
        ``record["marker_motion"]["left_marker_coherence"]``, and
        ``record["marker_motion"]["right_marker_coherence"]``. The depth
        fields are robust indentation depths in mm; marker displacement stays
        in raw native pixel coordinates.

        The API never creates an identity profile, compares candidates, or
        returns a hidden label. Generated code chooses which public fields to
        retain and records its own evidence with :meth:`write_trial_memory`.
        """
        normalized_window = max(1, int(window))
        summary = self.get_tactile_summary(window=normalized_window, hand="both")
        marker_motion = self.get_marker_motion_summary(hand="both", window=normalized_window)
        robot_state = self._safe_robot_state()
        self._capture_count += 1
        record = {
            "schema_version": _TACTILE_OBSERVATION_SCHEMA,
            "capture_id": f"capture_{self._capture_count:03d}",
            "step": _safe_int(_maybe_call(self._env, "get_step_count"), default=None),
            "window": normalized_window,
            "tactile": _canonical_tactile_metrics(summary),
            "marker_motion": _compact_marker_motion(marker_motion),
            "embodiment": {
                "gripper_qpos": _safe_float(robot_state.get("gripper_qpos")),
                "control_frame": _public_control_frame(robot_state.get("control_frame")),
            },
        }
        self._append_working_memory_trace("capture", {"record": record})
        print(
            "[univtac-tactile-memory] capture "
            f"window={normalized_window} contact={summary['contact']} "
            f"force={float(summary.get('normal_force', 0.0)):.3f} "
            f"depth={float(summary.get('depth_delta_mm', 0.0)):.3f}mm",
            flush=True,
        )
        return _jsonable(record)

    def write_trial_memory(self, key: str, record: dict) -> dict:
        """Persist one agent-authored ``trial_memory.v1`` record.

        ``record`` must contain ``schema_version``, ``kind``, ``phase``,
        ``data``, and ``provenance``. For example,
        ``{"schema_version": "trial_memory.v1", "kind": "evidence",
        "phase": "probe", "data": {}, "provenance": {"capture_ids": [],
        "memory_keys": []}}``. ``kind`` is ``evidence``, ``hypothesis``,
        ``decision``, or ``state``. ``data`` remains agent-owned, while
        ``provenance`` supplies JSON-safe ``capture_ids`` and ``memory_keys``.
        Raw sensor arrays are rejected. A key is capped at 8 KiB and a trial at
        32 KiB.
        """
        memory_key = _memory_key(key)
        if not memory_key:
            raise ValueError("trial memory key must not be empty")
        normalized = _validate_trial_memory_record(record)
        encoded = _json_bytes(normalized)
        if len(encoded) > _TRIAL_MEMORY_MAX_ENTRY_BYTES:
            raise ValueError(
                f"trial memory entry exceeds {_TRIAL_MEMORY_MAX_ENTRY_BYTES} byte limit"
            )

        old_size = (
            len(_json_bytes(self._trial_memory[memory_key]))
            if memory_key in self._trial_memory
            else 0
        )
        total_bytes = _trial_memory_size(self._trial_memory) - old_size + len(encoded)
        if total_bytes > _TRIAL_MEMORY_MAX_TOTAL_BYTES:
            raise ValueError(
                f"trial memory exceeds {_TRIAL_MEMORY_MAX_TOTAL_BYTES} byte total limit"
            )

        self._trial_memory[memory_key] = normalized
        self._sync_trial_memory_snapshot()
        result = {
            "schema_version": _TRIAL_MEMORY_OPERATION_SCHEMA,
            "operation": "write",
            "ok": True,
            "key": memory_key,
            "entry_bytes": len(encoded),
            "total_bytes": total_bytes,
        }
        self._append_working_memory_trace(
            "write", {"key": memory_key, "record": normalized, "result": result}
        )
        return result

    def read_trial_memory(self, key: str) -> dict | None:
        """Return one ``trial_memory.v1`` record or ``None`` when missing."""
        memory_key = _memory_key(key)
        value = self._trial_memory.get(memory_key)
        self._append_working_memory_trace(
            "read", {"key": memory_key, "found": value is not None}
        )
        return _jsonable(value) if value is not None else None

    def list_trial_memory(self) -> dict:
        """Return the complete versioned agent-authored memory snapshot."""
        return _trial_memory_snapshot(self._trial_memory)

    def clear_trial_memory(self) -> dict:
        """Clear agent-authored trial memory while retaining its audit history."""
        self._trial_memory = {}
        self._sync_trial_memory_snapshot()
        result = {
            "schema_version": _TRIAL_MEMORY_OPERATION_SCHEMA,
            "operation": "clear",
            "ok": True,
            "message": "trial tactile memory cleared",
        }
        self._append_working_memory_trace("clear", {"result": result})
        return result

    def runtime_memory_context(self, max_chars: int = 4000) -> str:
        """Return a bounded public snapshot for a multi-turn continuation."""
        return json.dumps(
            _bounded_trial_memory_snapshot(self._trial_memory, max(256, int(max_chars))),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _runtime_config(self) -> dict[str, Any]:
        configs = getattr(self._env, "api_configs", None)
        if not isinstance(configs, dict):
            return {}
        config = configs.get("univtac_tactile_api", {})
        return config if isinstance(config, dict) else {}

    def _safe_robot_state(self) -> dict[str, Any]:
        state_fn = getattr(self._env, "get_robot_state", None)
        if not callable(state_fn):
            return {}
        try:
            state = state_fn()
        except Exception:
            return {}
        return state if isinstance(state, dict) else {}

    def _append_working_memory_trace(self, event: str, payload: dict[str, Any]) -> None:
        trace_record = {
            "schema_version": _TRIAL_MEMORY_EVENT_SCHEMA,
            "event": str(event),
            "step": _safe_int(_maybe_call(self._env, "get_step_count"), default=None),
            "payload": _jsonable(payload),
        }
        self._working_memory_trace.append(_jsonable(trace_record))
        append_fn = getattr(self._env, "append_tactile_working_memory_trace", None)
        if callable(append_fn):
            append_fn(trace_record)

    def _sync_trial_memory_snapshot(self) -> None:
        set_snapshot = getattr(self._env, "set_tactile_trial_memory_snapshot", None)
        if callable(set_snapshot):
            set_snapshot(_trial_memory_snapshot(self._trial_memory))

    def _hand_observation(
        self,
        hand: str,
        *,
        data_types: list[str] | None = None,
    ) -> dict[str, Any]:
        _refresh_native_tactile(self._env, data_types=data_types or ["marker", "depth", "pose"])
        raw = self._env.current_raw_observation()
        key = _hand_key(hand)
        tactile = raw.get("tactile", {})
        if key not in tactile:
            alternate = _find_hand_key(tactile, hand)
            if alternate is None:
                available = sorted(tactile.keys()) if isinstance(tactile, dict) else []
                raise KeyError(
                    f"{key} not available in UniVTAC tactile observation; available={available}"
                )
            key = alternate
        return tactile[key]


def _native_tactile_calibration(env: BaseEnv) -> dict[str, float]:
    calibration_fn = getattr(env, "get_native_tactile_calibration", None)
    if not callable(calibration_fn):
        return {}
    return dict(calibration_fn())


def _tactile_measurement_protocol(env: BaseEnv) -> dict[str, Any]:
    """Normalize public probe calibration without reading task-private state."""
    api_configs = getattr(env, "api_configs", {})
    source = (
        api_configs.get("tactile_measurement_protocol", {})
        if isinstance(api_configs, dict)
        else {}
    )
    if not isinstance(source, dict):
        source = {}

    protocol = dict(_DEFAULT_TACTILE_MEASUREMENT_PROTOCOL)
    protocol.update({key: source[key] for key in protocol if key in source})
    return _jsonable(
        {
            "schema_version": _TACTILE_MEASUREMENT_PROTOCOL_SCHEMA,
            "capture_window": max(1, int(protocol["capture_window"])),
            "settle_steps": max(0, int(protocol["settle_steps"])),
            "close_target_force": float(
                np.clip(float(protocol["close_target_force"]), 0.0, 1.0)
            ),
            "close_max_steps": max(1, int(protocol["close_max_steps"])),
            "probe_lift_m": max(0.0, float(protocol["probe_lift_m"])),
            "probe_hold_steps": max(0, int(protocol["probe_hold_steps"])),
            "max_attempts_per_object": max(1, int(protocol["max_attempts_per_object"])),
            "adaptive_close": bool(protocol["adaptive_close"]),
        }
    )


def _hand_key(hand: str) -> str:
    value = str(hand).strip().lower()
    if value in {"left", "left_tactile"}:
        return "left_tactile"
    if value in {"right", "right_tactile"}:
        return "right_tactile"
    raise ValueError("hand must be 'left' or 'right'")


def _find_hand_key(tactile: Any, hand: str) -> str | None:
    if not isinstance(tactile, dict):
        return None
    prefix = str(hand).strip().lower().split("_", maxsplit=1)[0]
    for key in sorted(tactile.keys()):
        if str(key).lower().startswith(prefix):
            return str(key)
    return None


def _memory_key(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _public_control_frame(value: Any) -> str | None:
    """Keep the capture contract compact and independent of private task state."""
    return str(value) if isinstance(value, str) else None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _trial_memory_size(memory: dict[str, dict[str, Any]]) -> int:
    return sum(len(_json_bytes(value)) for value in memory.values())


def _validate_trial_memory_record(record: Any) -> dict[str, Any]:
    """Validate the stable agent-owned trial-memory envelope."""
    if not isinstance(record, dict):
        raise TypeError("trial memory record must be a JSON object (dict)")
    normalized = _validate_trial_memory_value(record, path="record")
    required = {"schema_version", "kind", "phase", "data", "provenance"}
    missing = sorted(required.difference(normalized))
    extra = sorted(set(normalized).difference(required))
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing required fields: {', '.join(missing)}")
        if extra:
            detail.append(f"unsupported fields: {', '.join(extra)}")
        raise ValueError("trial memory record envelope " + "; ".join(detail))
    if normalized["schema_version"] != _TRIAL_MEMORY_RECORD_SCHEMA:
        raise ValueError(
            f"trial memory record schema_version must be {_TRIAL_MEMORY_RECORD_SCHEMA!r}"
        )
    if normalized["kind"] not in _TRIAL_MEMORY_KINDS:
        allowed = ", ".join(sorted(_TRIAL_MEMORY_KINDS))
        raise ValueError(f"trial memory record kind must be one of: {allowed}")
    if not isinstance(normalized["phase"], str) or not normalized["phase"].strip():
        raise ValueError("trial memory record phase must be a non-empty string")
    if not isinstance(normalized["data"], dict):
        raise TypeError("trial memory record data must be a JSON object (dict)")
    provenance = normalized["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"capture_ids", "memory_keys"}:
        raise ValueError(
            "trial memory record provenance must contain only capture_ids and memory_keys"
        )
    for field in ("capture_ids", "memory_keys"):
        values = provenance[field]
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise TypeError(f"trial memory provenance.{field} must be a list of strings")
    return normalized


def _validate_trial_memory_value(value: Any, path: str = "value") -> Any:
    """Return a JSON-safe memory payload while rejecting raw sensor arrays."""
    if isinstance(value, np.ndarray):
        raise TypeError(f"{path} must not contain numpy.ndarray or raw images")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [
            _validate_trial_memory_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return [
            _validate_trial_memory_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string JSON key {key!r}")
            normalized[key] = _validate_trial_memory_value(item, f"{path}.{key}")
        return normalized
    raise TypeError(f"{path} must contain only JSON-safe scalars, lists, and dicts")


def _bounded_trial_memory_snapshot(
    memory: dict[str, dict[str, Any]],
    max_chars: int,
) -> dict[str, Any]:
    """Keep multi-turn context bounded without emitting invalid partial JSON."""
    snapshot: dict[str, Any] = {
        "schema_version": _TRIAL_MEMORY_SNAPSHOT_SCHEMA,
        "records": {},
        "omitted_keys": [],
    }
    for key in sorted(memory):
        candidate = {
            "schema_version": _TRIAL_MEMORY_SNAPSHOT_SCHEMA,
            "records": {**snapshot["records"], key: memory[key]},
            "omitted_keys": snapshot["omitted_keys"],
        }
        if len(_json_bytes(candidate)) <= max_chars:
            snapshot = candidate
        else:
            snapshot["omitted_keys"].append(key)
    return snapshot


def _trial_memory_snapshot(memory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": _TRIAL_MEMORY_SNAPSHOT_SCHEMA,
        "records": _jsonable(memory),
    }


def _maybe_call(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _canonical_tactile_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten public bilateral tactile metrics into the v1 capture contract."""
    left = dict(summary.get("left", {}))
    right = dict(summary.get("right", {}))
    metrics = {
        "contact": summary.get("contact"),
        "left_contact": summary.get("left_contact"),
        "right_contact": summary.get("right_contact"),
        "stable": summary.get("stable"),
        "normal_force": summary.get("normal_force"),
        "contact_area": summary.get("contact_area"),
        "depth_delta_mm": summary.get("depth_delta_mm"),
        "shear_magnitude": summary.get("shear_magnitude"),
        "marker_centroid_displacement": summary.get("marker_centroid_displacement"),
        "slip_score": summary.get("slip_score"),
        "contact_balance": summary.get("contact_balance"),
        "event": summary.get("event"),
    }
    for side, hand in (("left", left), ("right", right)):
        metrics.update(
            {
                f"{side}_normal_force": hand.get("normal_force"),
                f"{side}_contact_area": hand.get("contact_area"),
                f"{side}_depth_delta_mm": hand.get("depth_delta_mm"),
                # ``depth_mm`` is robust indentation depth, not a raw map.
                f"{side}_depth_mm": hand.get("depth_delta_mm"),
                f"{side}_depth_min_mm": hand.get("depth_min_mm"),
                f"{side}_shear_magnitude": hand.get("shear_magnitude"),
                f"{side}_marker_mean_displacement": hand.get("marker_mean_displacement"),
                f"{side}_marker_max_displacement": hand.get("marker_max_displacement"),
                f"{side}_marker_centroid_displacement": hand.get(
                    "marker_centroid_displacement"
                ),
            }
        )
    return _jsonable(metrics)


def _compact_marker_motion(marker_motion: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "hand",
        "shear_magnitude",
        "marker_centroid_displacement",
        "mean_displacement",
        "max_displacement",
        "left_marker_mean_displacement",
        "right_marker_mean_displacement",
        "left_marker_max_displacement",
        "right_marker_max_displacement",
        "left_marker_centroid_displacement",
        "right_marker_centroid_displacement",
        "left_marker_displacement_px",
        "right_marker_displacement_px",
        "left_marker_coherence",
        "right_marker_coherence",
    ]
    return _jsonable({key: marker_motion.get(key) for key in fields if key in marker_motion})


def _validate_tactile_response_expression(source: dict[str, Any]) -> None:
    """Reject malformed or non-public fields before exposing an expression."""
    blocks = source.get("response_blocks")
    if not isinstance(blocks, list) or not blocks:
        raise RuntimeError("tactile response expression must contain response_blocks")
    forbidden = {
        "actor",
        "class",
        "density",
        "friction",
        "hardness",
        "label",
        "metadata",
        "pose",
        "reward",
        "success",
    }
    valid_prefixes = ("preload.", "lift_motion.", "lift_minus_preload.")
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("name"), str):
            raise RuntimeError("each tactile response block must have a string name")
        fields = block.get("fields")
        if not isinstance(fields, list) or not fields:
            raise RuntimeError(f"response block {block.get('name')!r} must contain fields")
        scaler = block.get("scaler")
        if not isinstance(scaler, dict):
            raise RuntimeError(f"response block {block['name']!r} is missing a scaler")
        median, iqr = scaler.get("median"), scaler.get("iqr")
        if not isinstance(median, list) or not isinstance(iqr, list) or len(median) != len(fields) or len(iqr) != len(fields):
            raise RuntimeError(f"response block {block['name']!r} has incompatible scaler dimensions")
        if not np.isfinite(np.asarray(median, dtype=np.float64)).all() or np.any(np.asarray(iqr, dtype=np.float64) <= 0.0):
            raise RuntimeError(f"response block {block['name']!r} has an invalid scaler")
        for field in fields:
            if not isinstance(field, dict):
                raise RuntimeError(f"response block {block['name']!r} has a non-object field")
            path = field.get("path")
            if not isinstance(path, str) or not path.startswith(valid_prefixes):
                raise RuntimeError(f"response block {block['name']!r} has a non-public field path")
            if forbidden.intersection(path.lower().split(".")):
                raise RuntimeError(f"response block {block['name']!r} contains a private field path")
            if field.get("transform", "identity") not in {"identity", "log"}:
                raise RuntimeError(f"response block {block['name']!r} has an unsupported field transform")
            snr_paths = field.get("snr_paths")
            if not isinstance(snr_paths, list) or not snr_paths or not all(
                isinstance(item, str) and item.startswith(("preload.noise.", "lift_motion.noise."))
                for item in snr_paths
            ):
                raise RuntimeError(f"response block {block['name']!r} has invalid public SNR paths")


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


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _refresh_native_tactile(env: BaseEnv, data_types: list[str] | None = None) -> None:
    if hasattr(env, "refresh_native_observation"):
        env.refresh_native_observation(
            include_camera=False,
            include_tactile=True,
            tactile_data_types=data_types,
        )
    else:
        env.get_observation()
