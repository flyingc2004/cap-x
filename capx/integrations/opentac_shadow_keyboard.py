"""Public endpoint calibration, independent of requested-key task geometry."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from capx.integrations.opentac_shadow_vision import (
    HEAD_EYE, HEAD_LOOK_AT, HEAD_FOCAL_PX, HEAD_WIDTH, HEAD_HEIGHT,
    head_world_to_display_pixel, parse_note,
)

SCHEMA = "shadow_piano_public_endpoint_calibration.v1"


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def signature(endpoints: dict) -> dict:
    return {"schema": SCHEMA, "frame": endpoints["frame"],
            "mirrored": endpoints["mirrored"], "image_size": endpoints["display_image_size_px"],
            "A0": endpoints["A0"], "C8": endpoints["C8"],
            "spacing_m": endpoints["white_center_spacing_m"],
            "camera": {"eye": HEAD_EYE.tolist(), "look_at": HEAD_LOOK_AT.tolist(),
                       "focal_px": HEAD_FOCAL_PX, "size": [HEAD_WIDTH, HEAD_HEIGHT]},
            "robot_control_frame": "shadow_wrist_offset_v1", "fingertip_y_offset_m": .033,
            "black_x_offset_m": .07, "black_top_z_m": .085,
            "observed_camera": {k: endpoints[k] for k in
                ("observed_camera_intrinsics", "camera_pos_w", "camera_quat_w_world") if k in endpoints}}


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def make_calibration(pixels: dict, endpoints: dict, source_run: str) -> dict:
    if not isinstance(pixels, dict) or set(pixels) != {"A0", "C8"}:
        raise ValueError("Endpoint calibration requires only A0 and C8")
    checked = {}
    for note in ("A0", "C8"):
        row = pixels[note]
        if not isinstance(row, dict) or row.get("color") != "white":
            raise ValueError("Both endpoints must be white keys")
        uv = np.asarray([row["u"], row["v"]], float)
        actual = np.array([uv[0] * HEAD_WIDTH, 22 + uv[1] * HEAD_HEIGHT])
        expected = np.asarray(endpoints[note]["display_image_center_px"], float)
        if not np.isfinite(uv).all() or (uv < 0).any() or (uv > 1).any():
            raise ValueError("Endpoint pixels must be finite normalized coordinates")
        if np.linalg.norm(actual - expected) > 8.:
            raise ValueError(f"Selected {note} disagrees with its public projected center (>8 px)")
        checked[note] = {"u": float(uv[0]), "v": float(uv[1]), "color": "white"}
    if not np.isclose(endpoints["white_center_spacing_m"], .0235, atol=1.e-8):
        raise ValueError("Unsupported public white-key spacing")
    record = {"schema": SCHEMA, "signature": signature(endpoints), "pixels": checked,
              "source_run": source_run, "localization_mode": "public_endpoint_geometry"}
    record["calibration_id"] = _digest(record)
    return record


def validate_calibration(record: dict, endpoints: dict) -> dict:
    rebuilt = make_calibration(record["pixels"], endpoints, record["source_run"])
    if rebuilt != record:
        raise ValueError("Keyboard calibration signature or checksum differs")
    return rebuilt


def targets_from_calibration(record: dict, endpoints: dict, notes: list) -> dict:
    validate_calibration(record, endpoints)
    start = np.asarray(endpoints["A0"]["world_key_top_center_m"], float)
    spacing = float(endpoints["white_center_spacing_m"])
    result = {}
    for note in dict.fromkeys(notes):
        midi, black = parse_note(note)
        # Count musical white-key intervals from A0; black keys sit halfway
        # between their neighbouring white centers. No native layout lookup.
        preceding = sum(m % 12 not in (1, 3, 6, 8, 10) for m in range(21, midi))
        index = preceding - .5 if black else preceding
        point = [float(start[0] + (.03 if black else 0)),
                 float(start[1] + index * spacing), .085 if black else float(start[2])]
        wrist_y = point[1] - .033
        if not -.32 <= wrist_y <= .26:
            raise ValueError(f"Requested {note} exceeds the public wrist workspace")
        px = head_world_to_display_pixel(point, mirrored=endpoints["mirrored"])
        result[str(note)] = {"midi": midi, "color": "black" if black else "white",
            "head_pixel": [px[0] / HEAD_WIDTH, (px[1] - 22) / HEAD_HEIGHT],
            "wrist_x_offset_m": .07 if black else 0., "wrist_y_m": wrist_y,
            "press_lift_z_m": .0155 if black else 0., "hover_lift_z_m": .04,
            "projected_world_xyz_m": point, "calibration_id": record["calibration_id"]}
    return result
