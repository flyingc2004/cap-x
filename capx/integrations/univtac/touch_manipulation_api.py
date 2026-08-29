"""Generic tactile manipulation primitives for UniVTAC tasks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase

from .native_tactile import summarize_native_tactile


class UniVTACTouchManipulationApi(ApiBase):
    """Expose region-level tactile manipulation primitives to CaP code."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        cfg = dict(getattr(env, "api_configs", {}).get("touch_manipulation_api", {}))
        self.max_delta_xyz = float(cfg.get("max_delta_xyz", 0.012))
        self.search_xy_step = float(cfg.get("search_xy_step", 0.012))
        self.search_z_step = float(cfg.get("search_z_step", 0.006))
        self.center_xy_step = float(cfg.get("center_xy_step", 0.006))
        self.transport_xy_step = float(cfg.get("transport_xy_step", 0.018))
        self.lift_step = float(cfg.get("lift_step", 0.010))
        self.release_open_width = float(cfg.get("release_open_width", 1.0))
        self.contact_threshold = float(cfg.get("contact_threshold", 0.15))
        self.default_target_force = float(cfg.get("target_force", 0.70))
        self.stable_debounce_frames = int(cfg.get("stable_debounce_frames", 3))
        self.gripper_coarse_qpos_step = float(cfg.get("coarse_qpos_step", 0.0010))
        self.gripper_fine_qpos_step = float(cfg.get("fine_qpos_step", 0.0002))
        self.gripper_settle_steps = int(cfg.get("gripper_settle_steps", 2))
        self.max_open_steps = int(cfg.get("max_open_steps", 120))

    def functions(self) -> dict[str, Any]:
        return {
            "get_region": self.get_region,
            "list_regions": self.list_regions,
            "move_to_region": self.move_to_region,
            "search_contact": self.search_contact,
            "center_by_tactile": self.center_by_tactile,
            "close_until_stable": self.close_until_stable,
            "guarded_lift": self.guarded_lift,
            "transport_to_region": self.transport_to_region,
            "release_when_supported": self.release_when_supported,
            "wait_steps": self.wait_steps,
        }

    def list_regions(self) -> list[str]:
        """Return public region names available for the current task.

        Returns:
            Names such as pickup_a_region, pickup_b_region, slot_a_region, and
            slot_b_region. Regions are coarse public task geometry, not live
            object tracking.
        """
        lister = getattr(self._env, "list_public_regions", None)
        if callable(lister):
            return list(lister())
        return sorted(self._public_regions())

    def get_region(self, name: str) -> dict[str, Any]:
        """Return a public coarse region.

        Args:
            name: Region name from list_regions().

        Returns:
            Dictionary with region geometry such as center_xy, half_extents,
            hover_z, search_z_range, and release_z. It never contains live
            actor pose, reward, success, material, density, friction, or seed.
        """
        getter = getattr(self._env, "get_public_region", None)
        if callable(getter):
            region = getter(name)
        else:
            region = self._public_regions().get(str(name), {})
        if not region or not bool(region.get("ok", True)):
            raise KeyError(f"public region {name!r} is not available")
        return _clean_region(region)

    def move_to_region(self, region: str, z: str = "hover") -> dict[str, Any]:
        """Move the gripper/tool center to a public region.

        Args:
            region: Region name from list_regions().
            z: Z target selector: hover, high, low, search_low, or release.

        Returns:
            Motion result with ok, region, target_position, and step metadata.
        """
        start = self._primitive_start("move_to_region", {"region": region, "z": z})
        target = self._region_target(region, z)
        result = self._move_tool_to(target)
        result.update({"region": region, "target_position": target.tolist()})
        return self._finish_primitive(start, result)

    def search_contact(
        self,
        region: str | None = None,
        pattern: str = "local_raster",
    ) -> dict[str, Any]:
        """Search locally for tactile contact without using object pose.

        Args:
            region: Optional public pickup region. If omitted, search around the
                current gripper pose.
            pattern: local_raster or center_only.

        Returns:
            Result with ok, summary, contact_position, and reason.
        """
        start = self._primitive_start("search_contact", {"region": region, "pattern": pattern})
        if region is not None:
            self.move_to_region(region, z="hover")
            reg = self.get_region(region)
            z_low, z_high = self._search_z_range(reg)
            center_xy = np.asarray(reg["center_xy"], dtype=np.float32).reshape(2)
            half_extents = np.asarray(reg.get("half_extents", [self.search_xy_step, self.search_xy_step]), dtype=np.float32)
        else:
            pos, _quat = self._current_tool_pose()
            z_low, z_high = float(pos[2]) - 0.035, float(pos[2])
            center_xy = pos[:2]
            half_extents = np.array([self.search_xy_step, self.search_xy_step], dtype=np.float32)

        offsets = self._search_offsets(pattern)
        best_summary: dict[str, Any] | None = None
        best_force = -1.0
        for offset in offsets:
            self._open_to_width(1.0, max_steps=30)
            xy = center_xy + np.clip(offset, -half_extents, half_extents)
            self._move_tool_to(np.array([xy[0], xy[1], z_high], dtype=np.float32))
            steps = max(1, int(np.ceil(max(0.0, z_high - z_low) / max(self.search_z_step, 1e-6))))
            for idx in range(steps + 1):
                z_value = z_high - (z_high - z_low) * (idx / steps)
                self._move_tool_to(np.array([xy[0], xy[1], z_value], dtype=np.float32))
                self.wait_steps(1)
                summary = self._summary(window=6)
                force = float(summary.get("normal_force", 0.0))
                if force > best_force:
                    best_summary = summary
                    best_force = force
                if bool(summary.get("contact")) and force >= self.contact_threshold:
                    pos, _quat = self._current_tool_pose()
                    result = {
                        "ok": True,
                        "reason": "contact_found",
                        "summary": _jsonable(summary),
                        "contact_position": pos.tolist(),
                    }
                    return self._finish_primitive(start, result)
            probe_summary = self._probe_close_until_contact(max_steps=45)
            force = float(probe_summary.get("normal_force", 0.0))
            if force > best_force:
                best_summary = probe_summary
                best_force = force
            if bool(probe_summary.get("contact")) and force >= self.contact_threshold:
                pos, _quat = self._current_tool_pose()
                result = {
                    "ok": True,
                    "reason": "contact_found_by_probe_close",
                    "summary": _jsonable(probe_summary),
                    "contact_position": pos.tolist(),
                }
                return self._finish_primitive(start, result)

        result = {
            "ok": False,
            "reason": "contact_not_found",
            "summary": _jsonable(best_summary or self._summary(window=6)),
        }
        return self._finish_primitive(start, result)

    def center_by_tactile(self, max_iters: int = 8) -> dict[str, Any]:
        """Make small local motions to improve two-sided tactile balance.

        Args:
            max_iters: Maximum local adjustment attempts.

        Returns:
            Result with ok, reason, best_summary, and iteration count.
        """
        start = self._primitive_start("center_by_tactile", {"max_iters": max_iters})
        max_iters = max(1, int(max_iters))
        best = self._summary(window=6)
        best_score = self._grasp_quality(best)
        last_result: dict[str, Any] = {"ok": True}
        for idx in range(max_iters):
            if self._is_stable(best, self.default_target_force):
                return self._finish_primitive(
                    start,
                    {
                        "ok": True,
                        "reason": "already_centered",
                        "iterations": idx,
                        "summary": _jsonable(best),
                    },
                )

            balance = float(best.get("contact_balance", 0.0))
            candidates = self._centering_offsets(balance, idx)
            improved = False
            for delta_xy in candidates:
                last_result = self._move_by(np.array([delta_xy[0], delta_xy[1], 0.0], dtype=np.float32))
                if not bool(last_result.get("ok", False)):
                    continue
                self.wait_steps(1)
                summary = self._summary(window=6)
                score = self._grasp_quality(summary)
                if score > best_score:
                    best = summary
                    best_score = score
                    improved = True
                    break
            if not improved and not bool(best.get("contact")):
                break

        result = {
            "ok": self._is_stable(best, self.default_target_force),
            "reason": "centered" if self._is_stable(best, self.default_target_force) else "not_centered",
            "iterations": max_iters,
            "summary": _jsonable(best),
            "last_motion": _jsonable(last_result),
        }
        return self._finish_primitive(start, result)

    def close_until_stable(
        self,
        target_force: float = 0.7,
        max_steps: int = 180,
    ) -> dict[str, Any]:
        """Close the gripper until tactile evidence supports a stable grasp.

        Args:
            target_force: Normalized tactile force target.
            max_steps: Maximum gripper servo iterations.

        Returns:
            Result with ok, stable, reason, width, steps, and final summary.
        """
        start = self._primitive_start(
            "close_until_stable",
            {"target_force": target_force, "max_steps": max_steps},
        )
        begin_fn = getattr(self._env, "begin_high_level_action", None)
        if callable(begin_fn) and not bool(begin_fn()):
            return self._finish_primitive(start, {"ok": False, "stable": False, "reason": "protocol_blocked"})

        target_force = float(np.clip(target_force, 0.0, 1.0))
        max_steps = max(1, int(max_steps))
        stable_frames = 0
        best_summary: dict[str, Any] | None = None
        best_score = -1.0
        result: dict[str, Any] = {}

        for idx in range(max_steps):
            summary = self._summary(window=8)
            score = self._grasp_quality(summary)
            if score > best_score:
                best_summary = summary
                best_score = score
            if self._is_stable(summary, target_force):
                stable_frames += 1
            else:
                stable_frames = 0
            if stable_frames >= self.stable_debounce_frames:
                result = self._close_result(True, "stable_grasp", idx + 1, summary)
                break

            if self._is_one_sided(summary) and idx % 12 == 6:
                self.center_by_tactile(max_iters=2)

            width = self._width()
            if width <= 1e-6:
                reason = "unstable_at_min_width" if bool(summary.get("contact")) else "missed_grasp"
                result = self._close_result(False, reason, idx + 1, summary)
                break
            step = self.gripper_fine_qpos_step if bool(summary.get("contact")) else self.gripper_coarse_qpos_step
            self._command_width(max(0.0, width - self._qpos_to_width_step(step)))

        if not result:
            summary = self._summary(window=8)
            result = self._close_result(
                False,
                "max_steps",
                max_steps,
                summary if summary else (best_summary or {}),
            )
        finalize_fn = getattr(self._env, "finalize_high_level_action", None)
        if callable(finalize_fn):
            finalize_fn()
        return self._finish_primitive(start, result)

    def guarded_lift(self, height: float = 0.06, abort_on_slip: bool = True) -> dict[str, Any]:
        """Lift in small increments while monitoring tactile stability.

        Args:
            height: Desired lift height in meters.
            abort_on_slip: Stop early if slip or contact loss is detected.

        Returns:
            Result with ok, lifted_height, reason, and final summary.
        """
        start = self._primitive_start("guarded_lift", {"height": height, "abort_on_slip": abort_on_slip})
        height = max(0.0, float(height))
        steps = max(1, int(np.ceil(height / max(self.lift_step, 1e-6))))
        lifted = 0.0
        summary = self._summary(window=8)
        for _idx in range(steps):
            if abort_on_slip and self._lift_should_abort(summary):
                return self._finish_primitive(
                    start,
                    {
                        "ok": False,
                        "reason": "tactile_unstable_before_lift",
                        "lifted_height": lifted,
                        "summary": _jsonable(summary),
                    },
                )
            dz = min(self.lift_step, height - lifted)
            if dz <= 0.0:
                break
            result = self._move_by(np.array([0.0, 0.0, dz], dtype=np.float32))
            if not bool(result.get("ok", False)):
                result.update({"lifted_height": lifted, "reason": "lift_motion_failed"})
                return self._finish_primitive(start, result)
            lifted += dz
            self.wait_steps(1)
            summary = self._summary(window=8)

        result = {
            "ok": True,
            "reason": "lift_completed",
            "lifted_height": lifted,
            "summary": _jsonable(summary),
        }
        return self._finish_primitive(start, result)

    def transport_to_region(self, region: str, monitor_tactile: bool = True) -> dict[str, Any]:
        """Move horizontally to a public region while monitoring slip.

        Args:
            region: Destination region name.
            monitor_tactile: Abort if tactile contact is lost or slipping.

        Returns:
            Motion result with ok, reason, and destination region.
        """
        start = self._primitive_start(
            "transport_to_region",
            {"region": region, "monitor_tactile": monitor_tactile},
        )
        reg = self.get_region(region)
        current, _quat = self._current_tool_pose()
        target_xy = np.asarray(reg["center_xy"], dtype=np.float32).reshape(2)
        target_z = max(float(current[2]), float(reg.get("hover_z", current[2])))
        target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
        steps = max(
            1,
            int(np.ceil(float(np.linalg.norm(target - current)) / max(self.transport_xy_step, 1e-6))),
        )
        last_result: dict[str, Any] = {"ok": True}
        for idx in range(1, steps + 1):
            waypoint = current + (target - current) * (idx / steps)
            last_result = self._move_tool_to(waypoint)
            if not bool(last_result.get("ok", False)):
                last_result.update({"reason": "transport_motion_failed", "region": region})
                return self._finish_primitive(start, last_result)
            if monitor_tactile:
                summary = self._summary(window=8)
                if self._lift_should_abort(summary):
                    return self._finish_primitive(
                        start,
                        {
                            "ok": False,
                            "reason": "tactile_unstable_during_transport",
                            "region": region,
                            "summary": _jsonable(summary),
                        },
                    )
        last_result.update({"region": region, "reason": "transport_completed"})
        return self._finish_primitive(start, last_result)

    def release_when_supported(self, region: str) -> dict[str, Any]:
        """Descend to a public placement region and open the gripper.

        Args:
            region: Slot region name.

        Returns:
            Result with ok, released, region, and final summary.
        """
        start = self._primitive_start("release_when_supported", {"region": region})
        reg = self.get_region(region)
        release_target = self._region_target(region, "release")
        move_result = self._move_tool_to(release_target)
        if not bool(move_result.get("ok", False)):
            move_result.update({"released": False, "region": region, "reason": "release_descend_failed"})
            return self._finish_primitive(start, move_result)
        open_result = self._open_until_released()
        open_result.update({"region": region})
        retreat = self._move_tool_to(
            np.array([release_target[0], release_target[1], float(reg.get("hover_z", release_target[2] + 0.08))], dtype=np.float32)
        )
        open_result["retreat"] = _jsonable(retreat)
        return self._finish_primitive(start, open_result)

    def wait_steps(self, n: int = 1) -> dict[str, Any]:
        """Advance the simulation without issuing a task-level decision."""
        waiter = getattr(self._env, "wait_steps", None)
        if callable(waiter):
            return waiter(n)
        return {"ok": False, "message": "environment does not provide wait_steps"}

    def reset_episode(self) -> None:
        """Clear per-trial primitive trace when supported."""
        reset_fn = getattr(self._env, "reset_primitive_trace", None)
        if callable(reset_fn):
            reset_fn()

    def _public_regions(self) -> dict[str, dict[str, Any]]:
        getter = getattr(self._env, "get_public_regions", None)
        if callable(getter):
            return dict(getter())
        return {}

    def _region_target(self, name: str, z: str) -> np.ndarray:
        region = self.get_region(name)
        center_xy = np.asarray(region["center_xy"], dtype=np.float32).reshape(2)
        z_key = str(z).strip().lower()
        if z_key in {"hover", "high"}:
            target_z = float(region.get("hover_z", 0.16))
        elif z_key in {"low", "search_low"}:
            target_z = self._search_z_range(region)[0]
        elif z_key == "release":
            target_z = float(region.get("release_z", region.get("hover_z", 0.16)))
        else:
            target_z = float(z)
        return np.array([center_xy[0], center_xy[1], target_z], dtype=np.float32)

    @staticmethod
    def _search_z_range(region: dict[str, Any]) -> tuple[float, float]:
        raw = region.get("search_z_range", [0.035, region.get("hover_z", 0.16)])
        values = np.asarray(raw, dtype=np.float32).reshape(2)
        low, high = float(np.min(values)), float(np.max(values))
        return low, high

    def _move_tool_to(self, target_pos: np.ndarray) -> dict[str, Any]:
        target = np.asarray(target_pos, dtype=np.float32).reshape(3)
        current, _quat = self._current_tool_pose()
        dist = float(np.linalg.norm(target - current))
        steps = max(1, int(np.ceil(dist / max(self.max_delta_xyz, 1e-6))))
        last: dict[str, Any] = {"ok": True, "message": "no movement requested"}
        for idx in range(1, steps + 1):
            pos, _quat = self._current_tool_pose()
            waypoint = current + (target - current) * (idx / steps)
            delta = np.clip(waypoint - pos, -self.max_delta_xyz, self.max_delta_xyz)
            last = self._move_by(delta)
            if not bool(last.get("ok", False)):
                return {**last, "completed_steps": idx - 1, "steps": steps}
        return {**last, "completed_steps": steps, "steps": steps}

    def _move_by(self, delta_xyz: np.ndarray) -> dict[str, Any]:
        action = np.concatenate(
            [np.asarray(delta_xyz, dtype=np.float32).reshape(3), np.zeros(4, dtype=np.float32)]
        )
        return self._env.take_action(action, action_type="delta_ee")

    def _current_tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        task = getattr(self._env, "task", None)
        robot_manager = getattr(task, "_robot_manager", None)
        if robot_manager is not None:
            try:
                pose = robot_manager.get_gripper_center_pose()
                return (
                    np.asarray(pose.p, dtype=np.float32).reshape(3),
                    np.asarray(pose.q, dtype=np.float32).reshape(4),
                )
            except Exception:
                pass
        state = self._env.get_robot_state()
        return (
            np.asarray(state.get("ee_pos", [0.0, 0.0, 0.2]), dtype=np.float32).reshape(3),
            np.asarray(state.get("ee_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4),
        )

    def _summary(self, window: int = 8) -> dict[str, Any]:
        refresh = getattr(self._env, "refresh_native_observation", None)
        if callable(refresh):
            refresh(
                include_camera=False,
                include_tactile=True,
                include_embodiment=False,
                include_actor=False,
                tactile_data_types=["rgb", "rgb_marker", "marker", "depth", "pose"],
            )
        frames = self._env.tactile_buffer.recent(window)
        calibration_fn = getattr(self._env, "get_native_tactile_calibration", None)
        calibration = calibration_fn() if callable(calibration_fn) else {}
        return summarize_native_tactile(frames, hand="both", **calibration)

    def _is_stable(self, summary: dict[str, Any], target_force: float) -> bool:
        min_area = float(summary.get("stable_contact_area_threshold", 0.01))
        left = dict(summary.get("left") or {})
        right = dict(summary.get("right") or {})
        return bool(
            summary.get("event") == "stable_grasp"
            and summary.get("left_contact")
            and summary.get("right_contact")
            and float(summary.get("normal_force", 0.0)) >= float(target_force)
            and float(left.get("contact_area", 0.0)) >= min_area
            and float(right.get("contact_area", 0.0)) >= min_area
            and abs(float(summary.get("contact_balance", 0.0))) <= 0.45
            and float(summary.get("slip_score", 0.0)) < 0.6
        )

    @staticmethod
    def _is_one_sided(summary: dict[str, Any]) -> bool:
        return bool(summary.get("left_contact")) ^ bool(summary.get("right_contact"))

    def _lift_should_abort(self, summary: dict[str, Any]) -> bool:
        return bool(
            summary.get("event") in {"contact_lost", "slip_detected"}
            or not summary.get("contact")
            or float(summary.get("slip_score", 0.0)) >= 0.6
        )

    def _grasp_quality(self, summary: dict[str, Any]) -> float:
        if not summary:
            return 0.0
        left = dict(summary.get("left") or {})
        right = dict(summary.get("right") or {})
        area = min(float(left.get("contact_area", 0.0)), float(right.get("contact_area", 0.0)))
        force = min(float(left.get("normal_force", 0.0)), float(right.get("normal_force", 0.0)))
        balance = 1.0 - min(1.0, abs(float(summary.get("contact_balance", 0.0))))
        slip = 1.0 - min(1.0, float(summary.get("slip_score", 0.0)))
        return 0.45 * force + 0.30 * min(1.0, area / 0.03) + 0.15 * balance + 0.10 * slip

    def _width(self) -> float:
        calibration = self._env.get_gripper_calibration()
        return float(np.clip(calibration["current_width"], 0.0, 1.0))

    def _qpos_to_width_step(self, qpos_step: float) -> float:
        calibration = self._env.get_gripper_calibration()
        return float(qpos_step) / max(float(calibration["gripper_max_qpos"]), 1e-6)

    def _command_width(self, width: float) -> dict[str, Any]:
        command = getattr(self._env, "command_gripper_width_step", None)
        if not callable(command):
            raise RuntimeError("environment does not provide command_gripper_width_step")
        return command(float(np.clip(width, 0.0, 1.0)), settle_steps=self.gripper_settle_steps)

    def _open_until_released(self) -> dict[str, Any]:
        begin_fn = getattr(self._env, "begin_high_level_action", None)
        if callable(begin_fn) and not bool(begin_fn()):
            return {"ok": False, "released": False, "reason": "protocol_blocked"}
        for idx in range(max(1, self.max_open_steps)):
            summary = self._summary(window=6)
            width = self._width()
            if width >= self.release_open_width - 1e-6 and not bool(summary.get("contact")):
                self._finalize_gripper_action()
                return {
                    "ok": True,
                    "released": True,
                    "reason": "released",
                    "steps": idx + 1,
                    "width": width,
                    "summary": _jsonable(summary),
                }
            self._command_width(min(self.release_open_width, width + self._qpos_to_width_step(self.gripper_coarse_qpos_step)))
        summary = self._summary(window=6)
        self._finalize_gripper_action()
        return {
            "ok": False,
            "released": False,
            "reason": "max_steps",
            "steps": self.max_open_steps,
            "width": self._width(),
            "summary": _jsonable(summary),
        }

    def _open_to_width(self, target_width: float, *, max_steps: int) -> None:
        target = float(np.clip(target_width, 0.0, 1.0))
        for _idx in range(max(1, int(max_steps))):
            width = self._width()
            if width >= target - 1e-6:
                return
            self._command_width(min(target, width + self._qpos_to_width_step(self.gripper_coarse_qpos_step)))

    def _probe_close_until_contact(self, *, max_steps: int) -> dict[str, Any]:
        min_width = 0.15
        summary = self._summary(window=6)
        for _idx in range(max(1, int(max_steps))):
            if bool(summary.get("contact")) and float(summary.get("normal_force", 0.0)) >= self.contact_threshold:
                return summary
            width = self._width()
            if width <= min_width:
                return summary
            self._command_width(max(min_width, width - self._qpos_to_width_step(self.gripper_coarse_qpos_step)))
            summary = self._summary(window=6)
        return summary

    def _finalize_gripper_action(self) -> None:
        finalize_fn = getattr(self._env, "finalize_high_level_action", None)
        if callable(finalize_fn):
            finalize_fn()

    def _close_result(self, stable: bool, reason: str, steps: int, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": bool(stable),
            "stable": bool(stable),
            "reason": reason,
            "steps": int(steps),
            "width": self._width(),
            "normal_force": float(summary.get("normal_force", 0.0)),
            "contact": bool(summary.get("contact", False)),
            "left_contact": bool(summary.get("left_contact", False)),
            "right_contact": bool(summary.get("right_contact", False)),
            "slip_score": float(summary.get("slip_score", 0.0)),
            "summary": _jsonable(summary),
        }

    def _primitive_start(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "primitive": name,
            "args": _jsonable(args),
            "start_step": self._step_count(),
            "start_time": time.time(),
            "start_summary": _jsonable(self._summary(window=4)),
        }

    def _finish_primitive(self, start: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        record = {
            **start,
            "end_step": self._step_count(),
            "elapsed_s": float(time.time() - float(start["start_time"])),
            "result": _jsonable(result),
            "end_summary": _jsonable(self._summary(window=4)),
        }
        append = getattr(self._env, "append_primitive_trace", None)
        if callable(append):
            append(record)
        print(
            "[univtac-touch] "
            f"{record['primitive']} ok={bool(result.get('ok', False))} "
            f"reason={result.get('reason', result.get('message', ''))}",
            flush=True,
        )
        return result

    def _step_count(self) -> int:
        getter = getattr(self._env, "get_step_count", None)
        return int(getter()) if callable(getter) else 0

    def _search_offsets(self, pattern: str) -> list[np.ndarray]:
        if pattern == "center_only":
            return [np.zeros(2, dtype=np.float32)]
        s = self.search_xy_step
        return [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([s, 0.0], dtype=np.float32),
            np.array([-s, 0.0], dtype=np.float32),
            np.array([0.0, s], dtype=np.float32),
            np.array([0.0, -s], dtype=np.float32),
            np.array([s, s], dtype=np.float32),
            np.array([s, -s], dtype=np.float32),
            np.array([-s, s], dtype=np.float32),
            np.array([-s, -s], dtype=np.float32),
        ]

    def _centering_offsets(self, balance: float, iteration: int) -> list[np.ndarray]:
        s = self.center_xy_step
        preferred_y = -s if balance > 0.0 else s
        patterns = [
            [np.array([0.0, preferred_y], dtype=np.float32), np.array([0.0, -preferred_y], dtype=np.float32)],
            [np.array([s, 0.0], dtype=np.float32), np.array([-s, 0.0], dtype=np.float32)],
        ]
        return patterns[iteration % len(patterns)]


def _clean_region(region: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "ok",
        "name",
        "kind",
        "center_xy",
        "half_extents",
        "radius",
        "hover_z",
        "search_z_range",
        "release_z",
        "description",
        "source",
    }
    return {key: _jsonable(value) for key, value in region.items() if key in allowed}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
