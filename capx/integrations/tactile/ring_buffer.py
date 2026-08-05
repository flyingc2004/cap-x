"""Small in-memory buffer for tactile proxy frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class TactileFrame:
    """One compact tactile-proxy sample.

    The frame intentionally stores summarized simulator state, not task reward or
    success labels. That keeps the tactile API usable as a policy signal without
    leaking evaluation state into generated code.
    """

    sim_step: int
    timestamp: float
    target: str
    left_contact: bool
    right_contact: bool
    contact_count: int
    penetration_depth: float
    gripper_width: float | None
    gripper_velocity: float | None
    gripper_pos: np.ndarray | None
    target_pos: np.ndarray | None

    @property
    def contact(self) -> bool:
        return self.left_contact or self.right_contact

    @property
    def relative_pos(self) -> np.ndarray | None:
        if self.gripper_pos is None or self.target_pos is None:
            return None
        return np.asarray(self.target_pos) - np.asarray(self.gripper_pos)


class TactileRingBuffer:
    """Fixed-size FIFO buffer for tactile frames."""

    def __init__(self, maxlen: int = 500) -> None:
        self._frames: deque[TactileFrame] = deque(maxlen=maxlen)

    def append(self, frame: TactileFrame) -> None:
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def frames(self) -> list[TactileFrame]:
        return list(self._frames)

    def recent(self, window: int | None = None, target: str | None = None) -> list[TactileFrame]:
        frames: Iterable[TactileFrame] = self._frames
        if target is not None:
            frames = (frame for frame in frames if frame.target == target)
        selected = list(frames)
        if window is None or window <= 0:
            return selected
        return selected[-window:]
