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
