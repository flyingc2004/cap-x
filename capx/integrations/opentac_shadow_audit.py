"""Private, post-trial plot of public DIGIT estimates against native key force."""

from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_native_piano_summary(output_dir: str | Path) -> Path | None:
    """Evaluate native pitch, dynamics and release events using private records."""
    output_dir = Path(output_dir)
    native = _read_jsonl(output_dir / "shadow_native_events.jsonl")
    trace = _read_jsonl(output_dir / "shadow_piano_physics_trace.jsonl")
    events = _read_jsonl(output_dir / "shadow_piano_score_events.jsonl")
    if not native:
        return None
    info = native[-1].get("native_info", {})
    sequence = info.get("piano_sequence", {})
    requested_midi = sequence.get("midi", [])
    requested_dynamics = sequence.get("dynamics", [])
    by_step = {int(row["physics_step"]): row for row in trace}
    presses = [row for row in events if row.get("kind") == "press"]
    releases = [row for row in events if row.get("kind") == "release"]
    misses = [row for row in events if row.get("kind") == "miss"]
    press_by_attempt = {int(row.get("attempt_index", index)): row
                        for index, row in enumerate(presses)}
    release_by_attempt = {int(row.get("attempt_index", index)): row
                          for index, row in enumerate(releases)}
    miss_by_attempt = {int(row["attempt_index"]): row for row in misses}
    notes = []
    for index, (midi, dynamic) in enumerate(zip(requested_midi, requested_dynamics)):
        press = press_by_attempt.get(index)
        release = release_by_attempt.get(index)
        miss = miss_by_attempt.get(index)
        press_trace = by_step.get(int(press["step"])) if press is not None else None
        native_press_matches = bool(press is not None and int(press["midi"]) == int(midi))
        native_release_matches = bool(release is not None and int(release["midi"]) == int(midi)
                                      and int(release.get("repetition", -1)) == index + 1)
        dynamic_at_press = ({1: "light", 2: "heavy"}.get(int(press_trace["actual_dynamic"]))
                            if press_trace is not None else None)
        notes.append({"index": index + 1, "requested_midi": int(midi),
                      "requested_dynamic": dynamic,
                      "native_press_step": int(press["step"]) if press is not None else None,
                      "native_release_step": int(release["step"]) if release is not None else None,
                      "native_press_midi_matches": native_press_matches,
                      "native_release_midi_matches": native_release_matches,
                      "native_scored_note": native_press_matches and native_release_matches,
                      "native_failed_reason": miss.get("reason") if miss else None,
                      "observed_dynamic_at_press": dynamic_at_press,
                      "observed_dynamic_matches": dynamic_at_press == dynamic if press_trace is not None else None,
                      "measured_key_top_force_at_press_N": float(press["force_N"]) if press is not None else None,
                      "native_trace_at_press": press_trace})
    steps = [int(row["physics_step"]) for row in trace]
    gaps = sum(max(0, right - left - 1) for left, right in zip(steps, steps[1:]))
    report = {"native_success": bool(info.get("success", False)),
              "native_failure": info.get("failure", ""),
              "native_notes_completed": int(info.get("notes_completed", 0)),
              "native_notes_attempted": int(info.get("notes_attempted", info.get("notes_completed", 0))),
              "native_notes_failed": int(info.get("notes_failed", 0)),
              "end_submitted": any(row.get("kind") == "END" for row in native),
              "require_dynamics": bool(info.get("require_dynamics", False)),
              "physics_trace_rows": len(trace), "physics_trace_missing_steps": gaps,
              "score_event_count": len(events), "notes": notes}
    target = output_dir / "shadow_piano_native_audit_summary.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target


def save_force_alignment(output_dir: str | Path) -> Path | None:
    output_dir = Path(output_dir)
    source = output_dir / "shadow_force_estimates_private_audit.jsonl"
    if not source.exists():
        return None
    rows = _read_jsonl(source)
    if not rows:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    steps = [row["action_step"] for row in rows]
    truth = [row["private_key_top_force_N"] for row in rows]
    estimated = [row["estimate"]["estimated_key_force_N"] for row in rows]
    ax.plot(steps, truth, label="Native key-top force (private audit)", linewidth=1.4)
    ax.scatter(steps, estimated, label="Public DIGIT depth estimate", s=8, alpha=.7)
    for row in rows:
        if row["estimate"]["quality"] != "ok":
            ax.axvline(row["action_step"], color="orange", alpha=.04)
    ax.set(xlabel="Native action step (30 Hz)", ylabel="Key-top contact force (N)",
           title="Shadow piano: estimate vs native force; private labels are not sent to the agent")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    target = output_dir / "shadow_force_estimate_vs_native.png"
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target
