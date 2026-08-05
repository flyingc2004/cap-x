from __future__ import annotations

import json

import numpy as np

from capx.integrations.tactile.visualization import (
    build_tactile_timeline,
    draw_tactile_overlay,
    save_tactile_artifacts,
)


def record(
    *,
    left: bool = False,
    right: bool = False,
    rel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    step: int = 0,
    target: str = "cubeA",
) -> dict:
    return {
        "sim_step": step,
        "timestamp": float(step),
        "target": target,
        "left_contact": left,
        "right_contact": right,
        "contact_count": int(left) + int(right),
        "penetration_depth": 0.002 * (int(left) + int(right)),
        "gripper_width": 0.0,
        "gripper_velocity": 0.0,
        "gripper_pos": [0.0, 0.0, 0.0],
        "target_pos": list(rel),
    }


def test_build_tactile_timeline_contains_events() -> None:
    timeline = build_tactile_timeline(
        [
            record(step=0),
            record(left=True, right=True, step=1),
            record(left=True, right=True, rel=(0.05, 0.0, 0.0), step=2),
            record(step=3),
        ],
        window=3,
    )

    assert [row["event"] for row in timeline] == [
        "no_contact",
        "stable_grasp",
        "slip_detected",
        "contact_lost",
    ]
    assert timeline[-1]["contact"] is False
    assert timeline[-1]["slip_score"] >= 0.6


def test_draw_tactile_overlay_changes_frame_pixels() -> None:
    frame = np.zeros((180, 260, 3), dtype=np.uint8)
    row = build_tactile_timeline([record(left=True, right=True, step=5)])[0]

    overlay = draw_tactile_overlay(frame, row)

    assert overlay.shape == frame.shape
    assert overlay.dtype == frame.dtype
    assert np.count_nonzero(overlay) > 0


def test_save_tactile_artifacts_writes_timeline_files(tmp_path) -> None:
    records = [
        record(step=0),
        record(left=True, right=True, step=1),
        record(left=True, right=True, target="cubeB", step=1),
    ]

    save_tactile_artifacts(records, tmp_path, target="cubeA")

    json_path = tmp_path / "tactile_timeline.json"
    csv_path = tmp_path / "tactile_timeline.csv"
    png_path = tmp_path / "tactile_timeline.png"
    assert json_path.exists()
    assert csv_path.exists()
    assert png_path.exists()
    assert json.loads(json_path.read_text())[0]["target"] == "cubeA"
