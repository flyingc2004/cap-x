from __future__ import annotations

import numpy as np

from capx.integrations.tactile.ring_buffer import TactileFrame
from capx.integrations.tactile.summarizer import summarize_tactile_frames


def frame(
    *,
    left: bool = False,
    right: bool = False,
    rel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    step: int = 0,
) -> TactileFrame:
    contact_count = int(left) + int(right)
    gripper_pos = np.zeros(3, dtype=np.float64)
    target_pos = np.asarray(rel, dtype=np.float64)
    return TactileFrame(
        sim_step=step,
        timestamp=float(step),
        target="cubeA",
        left_contact=left,
        right_contact=right,
        contact_count=contact_count,
        penetration_depth=0.002 * contact_count,
        gripper_width=0.0,
        gripper_velocity=0.0,
        gripper_pos=gripper_pos,
        target_pos=target_pos,
    )


def test_no_contact_summary() -> None:
    summary = summarize_tactile_frames([frame()])
    assert summary["event"] == "no_contact"
    assert summary["contact"] is False
    assert summary["slip_score"] == 0.0


def test_one_finger_contact_summary() -> None:
    summary = summarize_tactile_frames([frame(left=True)])
    assert summary["event"] == "one_finger_contact"
    assert summary["contact"] is True
    assert summary["left_contact"] is True
    assert summary["right_contact"] is False


def test_stable_grasp_summary() -> None:
    summary = summarize_tactile_frames([frame(left=True, right=True)])
    assert summary["event"] == "stable_grasp"
    assert summary["contact"] is True
    assert summary["normal_force"] > 0.0
    assert abs(summary["contact_balance"]) <= 0.35


def test_slip_summary_from_relative_motion() -> None:
    frames = [
        frame(left=True, right=True, rel=(0.0, 0.0, 0.0), step=0),
        frame(left=True, right=True, rel=(0.05, 0.0, 0.0), step=1),
    ]
    summary = summarize_tactile_frames(frames)
    assert summary["event"] == "slip_detected"
    assert summary["slip_score"] >= 0.6
    assert summary["max_marker_displacement"] > 0.04


def test_contact_lost_summary() -> None:
    frames = [
        frame(left=True, right=True, step=0),
        frame(left=False, right=False, step=1),
    ]
    summary = summarize_tactile_frames(frames)
    assert summary["event"] == "contact_lost"
    assert summary["contact"] is False
    assert summary["slip_score"] >= 0.6
