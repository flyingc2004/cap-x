"""Visualization helpers for tactile proxy traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import imageio
import matplotlib
import numpy as np

from .ring_buffer import TactileFrame
from .summarizer import summarize_tactile_frames

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVENT_COLORS = {
    "no_contact": "#8c8c8c",
    "one_finger_contact": "#d97904",
    "stable_grasp": "#148a45",
    "slip_detected": "#d62728",
    "contact_lost": "#7b2cbf",
    "unknown": "#444444",
}


def save_tactile_artifacts(
    records: list[dict[str, Any]],
    trial_dir: str | Path,
    *,
    target: str = "cubeA",
    video_path: str | Path | None = None,
) -> None:
    """Save tactile timeline CSV/JSON/PNG and optional overlay video."""
    if not records:
        return

    trial_path = Path(trial_dir)
    trial_path.mkdir(parents=True, exist_ok=True)

    target_records = [record for record in records if record.get("target") == target]
    if not target_records:
        return

    timeline = build_tactile_timeline(target_records)
    if not timeline:
        return

    json_path = trial_path / "tactile_timeline.json"
    csv_path = trial_path / "tactile_timeline.csv"
    png_path = trial_path / "tactile_timeline.png"
    json_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    _write_timeline_csv(timeline, csv_path)
    plot_tactile_timeline(timeline, png_path)

    if video_path is not None and Path(video_path).exists():
        overlay_tactile_video(
            video_path,
            timeline,
            trial_path / "video_tactile_overlay.mp4",
        )


def build_tactile_timeline(records: list[dict[str, Any]], window: int = 20) -> list[dict[str, Any]]:
    """Build per-frame tactile summaries from raw exported records."""
    frames = [_record_to_frame(record) for record in records]
    timeline: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        start = max(0, idx - max(1, window) + 1)
        summary = summarize_tactile_frames(frames[start : idx + 1])
        timeline.append(
            {
                "index": idx,
                "sim_step": frame.sim_step,
                "timestamp": frame.timestamp,
                "target": frame.target,
                **summary,
            }
        )
    return timeline


def plot_tactile_timeline(timeline: list[dict[str, Any]], path: str | Path) -> None:
    """Render a static tactile timeline plot."""
    steps = np.asarray([row["sim_step"] for row in timeline], dtype=np.float64)
    if steps.size == 0:
        return
    steps = steps - steps[0]
    left = np.asarray([float(row["left_contact"]) for row in timeline], dtype=np.float64)
    right = np.asarray([float(row["right_contact"]) for row in timeline], dtype=np.float64)
    force = np.asarray([float(row["normal_force"]) for row in timeline], dtype=np.float64)
    slip = np.asarray([float(row["slip_score"]) for row in timeline], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(10, 5.8), sharex=True)
    axes[0].step(steps, left, where="post", label="left pad", color="#1f77b4")
    axes[0].step(steps, right, where="post", label="right pad", color="#ff7f0e")
    axes[0].set_ylim(-0.08, 1.08)
    axes[0].set_ylabel("contact")
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, force, label="normal force proxy", color="#148a45")
    axes[1].plot(steps, slip, label="slip score", color="#d62728")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper right")

    event_colors = [EVENT_COLORS.get(str(row["event"]), EVENT_COLORS["unknown"]) for row in timeline]
    axes[2].bar(steps, np.ones_like(steps), color=event_colors, width=1.0, align="edge")
    axes[2].set_yticks([])
    axes[2].set_ylabel("event")
    axes[2].set_xlabel("sim step")

    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=event)
        for event, color in EVENT_COLORS.items()
        if event in {str(row["event"]) for row in timeline}
    ]
    if handles:
        axes[2].legend(handles=handles, loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def overlay_tactile_video(
    video_path: str | Path,
    timeline: list[dict[str, Any]],
    out_path: str | Path,
    *,
    fps: int | None = None,
) -> None:
    """Write an MP4 with tactile state overlayed on each frame."""
    if not timeline:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        source_fps = meta.get("fps")
        frame_count = 0
        if hasattr(reader, "count_frames"):
            try:
                frame_count = int(reader.count_frames())
            except Exception:
                frame_count = 0
        if frame_count <= 0:
            frame_count = int(meta.get("nframes", 0) or 0)
        if frame_count <= 0:
            frame_count = len(list(reader))
            reader.close()
            reader = imageio.get_reader(video_path)

        writer_fps = int(fps or source_fps or 30)
        written = 0
        with imageio.get_writer(out_path, fps=writer_fps, format="FFMPEG", codec="libx264") as writer:
            for idx, frame in enumerate(reader):
                row = timeline[min(int(idx * len(timeline) / max(frame_count, 1)), len(timeline) - 1)]
                overlay = draw_tactile_overlay(np.asarray(frame), row)
                writer.append_data(np.ascontiguousarray(overlay))
                written += 1
        print(f"Saved tactile overlay video to {out_path} ({written} frames)")
    finally:
        reader.close()


def draw_tactile_overlay(frame: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    """Draw tactile contact indicators and score bars on an RGB frame."""
    img = frame.copy()
    _, w = img.shape[:2]
    panel_w = min(260, max(210, w // 2))
    panel_h = 118
    x0, y0 = 12, 12

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.68, img, 0.32, 0)

    event = str(row.get("event", "unknown"))
    event_color = _hex_to_rgb(EVENT_COLORS.get(event, EVENT_COLORS["unknown"]))
    left_on = bool(row.get("left_contact", False))
    right_on = bool(row.get("right_contact", False))
    force = float(row.get("normal_force", 0.0))
    slip = float(row.get("slip_score", 0.0))

    cv2.putText(
        img,
        f"Tactile: {event}",
        (x0 + 10, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        event_color,
        2,
        cv2.LINE_AA,
    )
    _draw_contact_dot(img, x0 + 24, y0 + 52, left_on, "L")
    _draw_contact_dot(img, x0 + 74, y0 + 52, right_on, "R")
    _draw_bar(img, x0 + 110, y0 + 43, panel_w - 125, 12, force, (20, 150, 70), "force")
    _draw_bar(img, x0 + 110, y0 + 73, panel_w - 125, 12, slip, (210, 35, 35), "slip")

    cv2.putText(
        img,
        f"step {int(row.get('sim_step', 0))}",
        (x0 + 10, y0 + panel_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return img


def _write_timeline_csv(timeline: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "index",
        "sim_step",
        "timestamp",
        "target",
        "contact",
        "left_contact",
        "right_contact",
        "normal_force",
        "shear_magnitude",
        "slip_score",
        "contact_balance",
        "max_marker_displacement",
        "mean_marker_displacement",
        "event",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in timeline:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _record_to_frame(record: dict[str, Any]) -> TactileFrame:
    return TactileFrame(
        sim_step=int(record["sim_step"]),
        timestamp=float(record["timestamp"]),
        target=str(record["target"]),
        left_contact=bool(record["left_contact"]),
        right_contact=bool(record["right_contact"]),
        contact_count=int(record["contact_count"]),
        penetration_depth=float(record["penetration_depth"]),
        gripper_width=_maybe_float(record.get("gripper_width")),
        gripper_velocity=_maybe_float(record.get("gripper_velocity")),
        gripper_pos=_maybe_array(record.get("gripper_pos")),
        target_pos=_maybe_array(record.get("target_pos")),
    )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _draw_contact_dot(img: np.ndarray, x: int, y: int, on: bool, label: str) -> None:
    color = (40, 200, 90) if on else (90, 90, 90)
    cv2.circle(img, (x, y), 12, color, -1)
    cv2.putText(
        img,
        label,
        (x - 5, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_bar(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: tuple[int, int, int],
    label: str,
) -> None:
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + width, y + height), (95, 95, 95), 1)
    cv2.rectangle(img, (x, y), (x + int(width * value), y + height), color, -1)
    cv2.putText(
        img,
        f"{label} {value:.2f}",
        (x, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return r, g, b
