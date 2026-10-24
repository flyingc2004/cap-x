"""Public-state-only probes for a new, control-frequency-matched dataset.

The numerical priors match the current CaP-X piano prompt. These are collection
commands, not force labels. No score, measured force or force estimator is read.
"""
from __future__ import annotations

import math

import numpy as np


PROFILES = {
    "runtime": (0., .45),
    "higher": (.0003, .45),
    "lower": (-.0003, .45),
    "slower": (0., .75),
}
TRIM_M = {False: {"light": -.00069, "heavy": -.00286},
          True: {"light": -.00090, "heavy": -.00209}}


def probe_plan(notes, dynamics, *, seed: int, profile: str) -> dict:
    """Freeze command variations before any physics/label sampling."""
    if profile not in (*PROFILES, "varied"):
        raise ValueError(f"Unknown collection profile: {profile}")
    if len(notes) != len(dynamics) or not notes:
        raise ValueError("Probe notes/dynamics must be nonempty and aligned")
    # Same profile for the paired light/heavy trials on a given key. Across
    # seeds each key sees the four profiles, instead of confounding dynamics.
    profiles = {}
    for index, note in enumerate(dict.fromkeys(notes)):
        profiles[note] = tuple(PROFILES)[(seed + index) % len(PROFILES)] if profile == "varied" else profile
    records = []
    for note, dynamic in zip(notes, dynamics):
        if dynamic not in ("light", "heavy"):
            raise ValueError("Expected light/heavy dynamics")
        black = "#" in note
        selected = profiles[note]
        offset, curl = PROFILES[selected]
        records.append(dict(note=note, dynamic=dynamic, profile=selected,
                            press_z_m=(.0155 if black else 0.) + TRIM_M[black][dynamic] + offset,
                            curl_seconds=curl, full_flexion_rad=.50,
                            hold_seconds=.75, release_seconds=.30,
                            x_offset_m=.07 if black else 0.))
    return dict(schema="shadow_digit_matched_probe.v1", seed=seed, profile=profile,
                policy_inputs=["specified_note", "public_hand_position"],
                uses_private_force_for_control=False,
                stops_on_old_estimator_ood=False, notes=records)


def matched_probe_actions(task, plan):
    """Yield (phase, target, MIDI, attempt index, finish flag).

    Callers finalize each attempt only after the released/lifted sample. Failed
    notes stay in the dataset. Native hard termination still ends collection.
    """
    dt = float(task.cfg.physics_dt) * int(task.cfg.action_repeat)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid native action duration")

    def transition(start, target, seconds):
        count = max(1, math.ceil(seconds / dt))
        for index in range(1, count + 1):
            u = index / count
            yield np.asarray(start + (target - start) * (u * u * (3 - 2 * u)), np.float32)

    for index, record in enumerate(plan["notes"]):
        midi = int(task.cfg.melody[index])
        key = task.layout[midi - 21]
        x, y = record["x_offset_m"], float(key.position[1] - .033)
        z, flex = record["press_z_m"], record["full_flexion_rad"]
        hover = np.array([x, y, .04, 0.])
        # A released physical joint can undershoot zero slightly. Start the
        # commanded interpolation within the documented action bounds.
        start = np.clip(np.asarray(task._action_state(), float),
                        task.action_space.low, task.action_space.high)
        # Lift before lateral travel. Every decision uses public position only.
        raised = start.copy(); raised[2:] = [.04, 0.]
        for target in transition(start, raised, .55):
            yield "lift", target, midi, index, False
        duration = max(.10, float(np.linalg.norm(raised[:2] - hover[:2])) / .10)
        for target in transition(raised, hover, duration):
            yield "hover", target, midi, index, False
        settled = 0
        for _ in range(math.ceil(4. / dt)):
            actual = np.asarray(task._action_state(), float)
            settled = settled + 1 if (np.linalg.norm(actual[:2] - hover[:2]) < .001 and
                                      abs(actual[2] - .04) < .001 and abs(actual[3]) < .02) else 0
            if settled >= 3:
                break
            yield "hover", hover.astype(np.float32), midi, index, False
        else:
            raise RuntimeError(f"Public position did not reach hover for {record['note']}")
        down = np.array([x, y, z, 0.])
        for target in transition(hover, down, .50):
            yield "descend", target, midi, index, False
        # Match the native .65 s lower/settle stage before starting the curl.
        for _ in range(math.ceil(.15 / dt)):
            yield "descend", down.astype(np.float32), midi, index, False
        pressed = np.array([x, y, z, flex])
        for target in transition(down, pressed, record["curl_seconds"]):
            yield "flex", target, midi, index, False
        for _ in range(math.ceil(record["hold_seconds"] / dt)):
            yield "hold", pressed.astype(np.float32), midi, index, False
        for target in transition(pressed, down, record["release_seconds"]):
            yield "release", target, midi, index, False
        for target in transition(down, hover, .55):
            yield "lift", target, midi, index, False
        for _ in range(math.ceil(.20 / dt)):
            yield "lift", hover.astype(np.float32), midi, index, False
        yield "lift", hover.astype(np.float32), midi, index, True
