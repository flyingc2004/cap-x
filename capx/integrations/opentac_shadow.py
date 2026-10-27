"""Public, bounded controls for the OpenTacBench Shadow piano task."""

from __future__ import annotations

import numpy as np

from capx.integrations.base_api import ApiBase


class ShadowPianoControlApi(ApiBase):
    def functions(self):
        return {
            "get_note_target": self.get_note_target,
            "get_hand_state": self.get_hand_state,
            "get_key_angles": self.get_key_angles,
            "move_shadow_index": self.move_shadow_index,
            "wait_steps": self.wait_steps,
            "get_index_tactile_rgb": self.get_index_tactile_rgb,
        }

    def get_note_target(self, note: str | int) -> dict:
        """Get a static piano-key wrist target for a note name or MIDI number.

        The returned X/Y and press/hover heights are nominal geometry, not
        feedback that a note sounded. Black keys need the returned X offset.
        """
        return self._env.get_note_target(note)

    def get_hand_state(self) -> dict:
        """Read public wrist, index-finger and Franka joint state."""
        return self._env.get_observation()["hand_state"]

    def get_key_angles(self, notes: list[str | int] | None = None) -> dict[str, float]:
        """Read physical key angles in radians; omitted notes return all 88 keys.

        This is public key travel, not contact force or the private score.
        """
        angles = self._env.get_observation()["key_angles_rad"]
        if notes is None:
            from envs._piano_sequence import note_name
            return {note_name(midi): float(angles[midi - 21]) for midi in range(21, 109)}
        if not isinstance(notes, (list, tuple)):
            raise TypeError("notes must be a list of note names or MIDI numbers")
        return {self._env.get_note_target(note)["note"]: float(angles[self._env.get_note_target(note)["midi"] - 21]) for note in notes}

    def move_shadow_index(self, wrist_x_offset_m: float, wrist_y_m: float,
                          lift_z_m: float, index_flexion_rad: float,
                          steps: int = 1) -> dict:
        """Command the native four-dimensional wrist/index target for 1..240 steps.

        Targets are absolute in the task's local control frame. The adapter
        rejects values outside the native action bounds. Native Task.step()
        applies its own rate filter and advances PhysX. Stop if native_done.
        """
        action = np.asarray([wrist_x_offset_m, wrist_y_m, lift_z_m, index_flexion_rad], dtype=np.float32)
        return self._env.command(action, steps)

    def wait_steps(self, steps: int) -> dict:
        """Advance 1..240 native action steps while holding the last target."""
        return self._env.wait_steps(steps)

    def get_index_tactile_rgb(self) -> np.ndarray:
        """Return the latest single-index DIGIT RGB image as uint8 HxWx3.

        RGB is tactile appearance only; it is not a calibrated force reading.
        """
        return self._env.get_index_tactile_rgb()


class ShadowPianoVisualForceApi(ApiBase):
    """Piano controls using visual key picks and public DIGIT appearance only."""

    def functions(self):
        functions = {"get_keyboard_endpoints": self.get_keyboard_endpoints,
                "locate_visual_keys": self.locate_visual_keys,
                "get_hand_state": self.get_hand_state,
                "move_shadow_index": self.move_shadow_index,
                "wait_steps": self.wait_steps,
                "get_index_tactile_rgb": self.get_index_tactile_rgb,
                "estimate_key_force": self.estimate_key_force,
                "finish_piano_note_attempt": self.finish_piano_note_attempt}
        if getattr(self._env, 'tactile_skill_memory_path', None) is not None:
            functions.update(query_tactile_skill=self.query_tactile_skill,
                             record_tactile_skill=self.record_tactile_skill)
        return functions

    def query_tactile_skill(self, note: str, dynamics: str, probe_z_m: float) -> dict:
        """Match the first three valid public 5D contact frames to a prior skill.

        Returns collecting/empty/miss/matched/unavailable. A match supplies a
        bounded z_correction_m, to approach gradually within existing controls.
        It does not move or certify scoring. Query after fresh estimate_key_force.
        """
        return self._env.query_tactile_skill(note, dynamics, probe_z_m)

    def record_tactile_skill(self, note: str, dynamics: str, probe_z_m: float,
                             function_name: str) -> dict:
        """Record a controller's public contact/hold evidence for a later episode.

        Call before release, after at least nine consecutive fresh qualified
        estimates spanning 0.12 s. The host independently checks the public
        window. No true force or score is used. function_name identifies the
        generated press/hold helper. A reuse snapshot cannot be overwritten.
        """
        return self._env.record_tactile_skill(note, dynamics, probe_z_m, function_name)

    def get_keyboard_endpoints(self) -> dict:
        """Return public fixed A0/C8 white-key centers in world metres and HEAD pixels.

        display_image_center_px uses the currently mirrored 960x662 labeled
        HEAD image. The value is static scene geometry, not a force, score,
        requested-note target or proof that a pictured key is unobstructed.
        In endpoint-calibration mode, mapped_keys contains the current requested
        wrist targets when calibration_ready is true. Reuse these without images.
        """
        return self._env.get_keyboard_endpoints()

    def locate_visual_keys(self, pixels: dict[str, dict]) -> dict[str, dict]:
        """Project all requested keys from normalized pixels in the HEAD tile.

        Example: {'C4': {'u': .4, 'v': .6, 'color': 'white'}}.
        The mapping must cover the complete set of distinct requested keys.
        Returns bounded wrist targets from camera/robot calibration, with no
        lookup of the task's requested-key layout. After successful private
        save, get_hand_state()['visual_key_map_ready'] becomes 1 for this episode.
        In endpoint-calibration mode you may instead pass exactly A0 and C8,
        both color='white', after checking the supplied anchors in the image.
        This returns all current requested targets computed from public endpoints,
        musical white-key intervals and generic black/white geometry. It saves
        the reusable endpoint calibration for subsequent episodes.
        """
        return self._env.locate_visual_keys(pixels)

    def get_hand_state(self) -> dict:
        """Return measured wrist X/Y/Z, finger and arm state, and map-ready flag.

        wrist_x_offset_m and wrist_y_m are measured positions, not the last
        commanded target. wrist_and_index_velocity[0] is wrist Y speed in m/s.
        Compare measured X/Y with locate_visual_keys() targets before pressing.
        """
        return self._env.get_observation()["hand_state"]

    def move_shadow_index(self, wrist_x_offset_m: float, wrist_y_m: float,
                          lift_z_m: float, index_flexion_rad: float,
                          steps: int = 1) -> dict:
        """Send one bounded native four-dimensional action for 1..240 steps."""
        action = np.asarray([wrist_x_offset_m, wrist_y_m, lift_z_m, index_flexion_rad], dtype=np.float32)
        return self._env.command(action, steps)

    def wait_steps(self, steps: int) -> dict:
        """Hold the last native action target for a bounded number of steps."""
        return self._env.wait_steps(steps)

    def get_index_tactile_rgb(self) -> np.ndarray:
        """Return one public DIGIT RGB image, without native force labels."""
        return self._env.get_index_tactile_rgb()

    def estimate_key_force(self, note: str | int) -> dict:
        """Estimate key-top contact force from public simulated DIGIT depth.

        quality values are lowercase and case-sensitive: 'ok', 'no_contact',
        'ood', 'insufficient_history', or another unavailable/invalid state.
        Only fresh 'ok' has a usable estimated_key_force_N. Branch on 'ood',
        never 'OOD'; release and use the configured re-probe policy.
        'no_contact' intentionally has None force and is normal before contact.
        Handle it before checking for a finite force: keep the bounded initial
        curl and reset hold counters; it is not an invalid estimate by itself.
        """
        return self._env.estimate_key_force(note)

    def finish_piano_note_attempt(self, index: int) -> dict:
        """After safely releasing note index (zero based), consume that attempt.

        Returns only whether the native episode has ended. Native per-note
        success, failure and measured force remain private.
        """
        return self._env.finish_piano_note_attempt(index)
