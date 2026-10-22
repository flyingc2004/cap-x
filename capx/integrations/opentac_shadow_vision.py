"""Convert visually selected head-camera key pixels to bounded Shadow actions.

No requested-key layout lookup occurs here. The only physical calibration is
the fixed camera pose, pinhole intrinsics, key-top heights and fingertip offset.
"""

from __future__ import annotations

import re

import numpy as np

# Elevated keyboard overview.  The front-side offset avoids looking through
# the tall cabinet behind the keys while keeping both keyboard ends in frame.
HEAD_EYE = np.array([.20, 0., 1.15], dtype=float)
HEAD_LOOK_AT = np.array([.525, 0., .072], dtype=float)
HEAD_WIDTH, HEAD_HEIGHT = 960, 640
HEAD_FOCAL_PX = 24. / 36. * HEAD_WIDTH
PITCHES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def head_world_to_display_pixel(point_xyz_m: tuple[float, float, float], *, mirrored: bool) -> list[float]:
    """Project a static world point into the labeled 960x662 HEAD image."""
    point = np.asarray(point_xyz_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("HEAD projection needs one finite world point")
    forward = HEAD_LOOK_AT - HEAD_EYE
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ray = point - HEAD_EYE
    depth = float(ray @ forward)
    if depth <= 0:
        raise ValueError("World point is behind the HEAD camera")
    u = .5 + float(ray @ right) / depth * HEAD_FOCAL_PX / HEAD_WIDTH
    v = .5 - float(ray @ up) / depth * HEAD_FOCAL_PX / HEAD_HEIGHT
    if mirrored:
        u = 1. - u
    return [u * HEAD_WIDTH, 22. + v * HEAD_HEIGHT]


def _dark_runs(profile: np.ndarray, threshold: float) -> list[tuple[float, int]]:
    """Return centers and widths of dark horizontal runs in a camera row."""
    dark = np.asarray(profile < threshold, dtype=bool)
    edges = np.diff(np.r_[False, dark, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [((float(start) + float(end - 1)) / 2., int(end - start))
            for start, end in zip(starts, ends)]


def snap_key_pixels_to_visible_centers(
    pixels: dict[str, dict], head_rgb: np.ndarray,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Snap a visually chosen point within its own white cell or black cap.

    This checks local image boundaries only. It cannot identify a note or
    repair a whole-key/octave error in the model's endpoint calibration.
    """
    image = np.asarray(head_rgb)
    if image.shape != (HEAD_HEIGHT, HEAD_WIDTH, 3) or not np.isfinite(image).all():
        raise ValueError("Current HEAD RGB has an invalid shape or pixels")
    grey = image.astype(np.float32).mean(axis=2)
    snapped: dict[str, dict] = {}
    audit: dict[str, dict] = {}
    for note, specification in pixels.items():
        x = float(specification["u"]) * HEAD_WIDTH
        y = float(specification["v"]) * HEAD_HEIGHT
        if not np.isfinite([x, y]).all() or not 25 <= x < HEAD_WIDTH - 25 or not 5 <= y < HEAD_HEIGHT - 5:
            raise ValueError(f"Visual key point for {note} is outside the clear HEAD picture")
        row = int(round(y))
        profile = grey[row - 2:row + 3].mean(axis=0)
        color = specification.get("color")
        if color == "white":
            # The white-key front has narrow grey seams and a bright interior.
            runs = [(center, width) for center, width in _dark_runs(profile, 170.)
                    if 1 <= width <= 5 and abs(center - x) <= 22]
            left = [center for center, _ in runs if center < x - .5]
            right = [center for center, _ in runs if center > x + .5]
            if not left or not right:
                raise ValueError(f"Cannot bracket the selected white key {note} with visible seams")
            lo, hi = max(left), min(right)
            if not 9 <= hi - lo <= 18:
                raise ValueError(f"Selected white key {note} has an implausible visible width")
            centered_x = (lo + hi) / 2.
            if abs(centered_x - x) > 5.5:
                raise ValueError(f"Selected point for {note} is too far from its visible white key center")
        elif color == "black":
            # At the chosen black-key row, the cap is a compact dark run.
            runs = [(center, width) for center, width in _dark_runs(profile, 70.)
                    if 3 <= width <= 10 and abs(center - x) <= 6.5]
            if len(runs) != 1:
                raise ValueError(f"Cannot uniquely identify the selected black key {note}")
            centered_x = runs[0][0]
        else:
            raise ValueError(f"Invalid key color for {note}")
        snapped[note] = {**specification, "u": centered_x / HEAD_WIDTH}
        audit[note] = {"selected_x_px": x, "snapped_x_px": centered_x,
                       "delta_x_px": centered_x - x, "camera_y_px": y}
    return snapped, audit


def parse_note(note: str | int) -> tuple[int, bool]:
    if isinstance(note, bool):
        raise ValueError("Boolean is not a piano note")
    if isinstance(note, int):
        midi = note
    else:
        match = re.fullmatch(r"([A-G])(#?)(-?\d+)", str(note))
        if match is None:
            raise ValueError(f"Invalid note: {note!r}")
        midi = (int(match[3]) + 1) * 12 + PITCHES[match[1]] + bool(match[2])
    if not 21 <= midi <= 108:
        raise ValueError("Note outside piano MIDI range")
    return midi, midi % 12 in (1, 3, 6, 8, 10)


def head_pixel_to_world(u: float, v: float, *, black: bool) -> np.ndarray:
    """Intersect a normalized native head RGB pixel with a generic key-top plane."""
    if not np.isfinite([u, v]).all() or not .02 <= u <= .98 or not .02 <= v <= .98:
        raise ValueError("Key pixel must be finite and inside the head view")
    forward = HEAD_LOOK_AT - HEAD_EYE
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ray = forward + ((u - .5) * HEAD_WIDTH / HEAD_FOCAL_PX) * right \
        + ((.5 - v) * HEAD_HEIGHT / HEAD_FOCAL_PX) * up
    if ray[2] >= -.05:
        raise ValueError("Visual ray does not meet the keyboard from above")
    key_top_z = .085 if black else .0725
    distance = (key_top_z - HEAD_EYE[2]) / ray[2]
    if distance <= 0:
        raise ValueError("Key pixel projects behind the camera")
    return HEAD_EYE + distance * ray


def locate_key_pixels(
    pixels: dict[str, dict], *, expected_notes: list[str], mirrored: bool = False,
) -> dict[str, dict]:
    """Batch-check displayed key pixels before any press action.

    A mirrored HEAD display reverses only image u. Projection still uses the
    unchanged native camera geometry, while returned head pixels stay in the
    displayed coordinate system used by the agent.
    """
    if not isinstance(pixels, dict) or not isinstance(expected_notes, list):
        raise TypeError("Pass a note to {u,v,color} mapping and expected note list")
    unique = list(dict.fromkeys(expected_notes))
    if set(pixels) != set(unique):
        raise ValueError("First key map must cover every distinct requested note")
    targets = {}
    for note in unique:
        specification = pixels[note]
        if not isinstance(specification, dict):
            raise ValueError(f"Missing visual key selection for {note}")
        midi, black = parse_note(note)
        color = "black" if black else "white"
        if specification.get("color") != color:
            raise ValueError(f"Key color disagrees with visible selection for {note}")
        displayed_u = float(specification["u"])
        displayed_v = float(specification["v"])
        native_u = 1.0 - displayed_u if mirrored else displayed_u
        point = head_pixel_to_world(native_u, displayed_v, black=black)
        x_limits = (.47, .61) if black else (.40, .58)
        if not x_limits[0] <= point[0] <= x_limits[1]:
            raise ValueError(f"Pixel for {note} is outside the generic {color} key top")
        wrist_y = float(point[1] - .033)
        if not -.32 <= wrist_y <= .26:
            raise ValueError(f"Pixel for {note} is outside native wrist travel")
        targets[note] = {"midi": midi, "color": color, "head_pixel": [displayed_u, displayed_v],
                         "wrist_x_offset_m": .07 if black else 0., "wrist_y_m": wrist_y,
                         "press_lift_z_m": .0155 if black else 0., "hover_lift_z_m": .04,
                         "projected_world_xyz_m": point.tolist()}
    ordered = sorted(targets.items(), key=lambda item: item[1]["midi"])
    for (left_note, left), (right_note, right) in zip(ordered, ordered[1:]):
        if not left["wrist_y_m"] < right["wrist_y_m"]:
            raise ValueError(
                "Visual key positions contradict piano pitch order: "
                f"{left_note} at u={left['head_pixel'][0]:.4f} projects to "
                f"y={left['wrist_y_m']:.4f} m, but {right_note} at "
                f"u={right['head_pixel'][0]:.4f} projects to "
                f"y={right['wrist_y_m']:.4f} m. "
                "Recheck these note identities in the clear HEAD image."
            )
        spacing = right["wrist_y_m"] - left["wrist_y_m"]
        if spacing > .15:
            raise ValueError("Implausible visual gap between neighbouring requested keys")
    return targets
