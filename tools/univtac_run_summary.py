#!/usr/bin/env python3
"""Summarize a CaP-X UniVTAC run.log into a compact stage/action table.

Usage:
    python tools/univtac_run_summary.py <path-to-run.log> [--tactile]

Columns in the action table (one row per move_relative call):
    a       action_count (physical actions consumed so far)
    req_xyz requested translation [x, y, z] -- what the LLM asked for
    req_rpy requested rotation [r, p, y]   -- the angle/direction the LLM probed
    exe_rpy executed rotation -- actual motion after guard clipping
    rot     executed rotation magnitude (rad); >0 means a rotation probe ran
    guard   guard_stop_phase: none/rotation/translation
    imp     direction_improved flag
    reason  adapter result reason (direction_not_improving, completed,
            interrupted_by_slip_warning, preempted_by_slip_risk, ...)

Stage lines are the LLM's own prints (Probing/Stage/finds succeed/stop_reason),
plus inserted_depth_world progression. With --tactile, slip/centroid/risk of
each tactile read is printed as an extra column.
"""

import re
import sys

FRANKA_RE = re.compile(
    r"\[univtac-franka\] move_relative ok=\S+ reason=(\S+) .*?"
    r"requested_xyz=\[([-0-9., ]+)\] executed_xyz=\[([-0-9., ]+)\] "
    r"requested_rpy=\[([-0-9., ]+)\] executed_rpy=\[([-0-9., ]+)\] "
    r"rotation=([0-9.]+) guard_stop_phase=(\S+) direction_improved=(\S+) .*?"
    r"action_count=(\d+) .*?remaining_actions=(\d+)"
)
DEPTH_RE = re.compile(
    r"\[univtac-insert-depth\] move_relative_depth inserted_depth_world=([0-9.]+) "
    r"reference_z=([0-9.]+) current_z=([0-9.]+) executed_depth=([0-9.]+) "
    r"action_count=(\d+) remaining_actions=(\d+) reason=(\S+)"
)
TACTILE_RE = re.compile(
    r"\[univtac-tactile\] .*?slip=([0-9.]+) risk=(\S+) .*?centroid=([0-9.]+)"
)
STAGE_RE = re.compile(
    r"Probing|Stage:|finds succeed|stop_reason|Adjusting|Retract|Insert cycle|"
    r"No clear|Exploration|Preload"
)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_tactile = "--tactile" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(1)
    path = args[0]

    with open(path, errors="ignore") as fh:
        content = fh.read()

    prev_tactile = ""
    for line in content.replace("\r", "\n").split("\n"):
        if line.startswith("Step") or "FPS" in line:
            continue
        m = FRANKA_RE.search(line)
        if m:
            depth = f" d={prev_tactile}" if show_tactile else ""
            print(
                f"a={int(m.group(9)):>3} req_xyz=[{m.group(2)}] req_rpy=[{m.group(4)}] "
                f"exe_rpy=[{m.group(5)}] rot={m.group(6)} guard={m.group(7)} "
                f"imp={m.group(8)} reason={m.group(1)}{depth}"
            )
            continue
        m = DEPTH_RE.search(line)
        if m:
            print(
                f"   depth_world={m.group(1)} current_z={m.group(3)} "
                f"exec={m.group(4)} a={int(m.group(5)):>3} rem={m.group(6)} "
                f"reason={m.group(7)}"
            )
            continue
        m = TACTILE_RE.search(line)
        if m and show_tactile:
            prev_tactile = f"slip={m.group(1)} risk={m.group(2)} cent={m.group(3)}"
            continue
        if STAGE_RE.search(line):
            # skip the code.py echo section (contains print("...") source lines)
            if "print(" in line or "Stdout:" in line or "Code block" in line:
                continue
            print(f"   <stage> {line.strip()[:120]}")


if __name__ == "__main__":
    main()
