#!/usr/bin/env python3
"""PRIVATE, post-freeze force audit. Never imported by training or key inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from capx.memory.trajectory.data import Corpus, read_json, write_json
from capx.memory.trajectory.evaluate import check_freeze


def label_frames(data):
    """Measured force on the publicly specified key at each exact observed step."""
    force = np.full(len(data.steps), np.nan, np.float32)
    requested = np.full(len(data.steps), -1, np.int16)
    for ep in data.episodes:
        with h5py.File(ep["source"], "r") as public:
            midi = np.asarray(public["requested_midi"], int)
            psteps = np.asarray(public["physics_step"], np.int64)
        if np.any((midi < 21) | (midi > 108)):
            raise ValueError("Invalid specified key in private audit")
        offset, n = ep["offset"], ep["frames"]
        if not np.array_equal(psteps, data.steps[offset:offset+n]):
            raise ValueError("Public source steps changed after preprocessing")
        with h5py.File(Path(ep["source"]).with_name("private_labels.h5"), "r") as private:
            steps = np.asarray(private["physics_step"], np.int64)
            rows = np.searchsorted(steps, psteps)
            if np.any(rows >= len(steps)) or not np.array_equal(steps[rows], psteps):
                raise ValueError("Private labels are not exactly aligned")
            values = np.asarray(private["key_contact_force_N"][rows], np.float32)
            if values.shape != (n,88): raise ValueError("Malformed private force matrix")
            force[offset:offset+n] = values[np.arange(n), midi-21]
            requested[offset:offset+n] = midi
    if not np.isfinite(force).all() or np.any(force < 0):
        raise ValueError("Invalid private actual forces")
    return force, requested


def metrics(q, n, selection):
    if not selection.any(): return {"windows": 0}
    q, n = q[selection], n[selection]
    error = n-q
    per_window = np.abs(error).mean(1)
    return dict(windows=len(q), force_curve_mae_N=float(per_window.mean()),
                force_curve_rmse_N=float(np.sqrt(np.square(error).mean())),
                p95_window_mae_N=float(np.percentile(per_window,95)), bias_N=float(error.mean()),
                peak_mae_N=float(np.abs(n.max(1)-q.max(1)).mean()),
                current_force_mae_N=float(np.abs(error[:,-1]).mean()))


def plot_curves(q, n, mask, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    choices=np.flatnonzero(mask)
    if not len(choices): return
    choices=choices[np.linspace(0,len(choices)-1,min(4,len(choices))).astype(int)]
    fig,axes=plt.subplots(len(choices),1,figsize=(8,2.3*len(choices)),squeeze=False)
    for ax,i in zip(axes[:,0],choices):
        ax.plot(np.arange(16)/30-.5,q[i],label="query actual key force")
        ax.plot(np.arange(16)/30-.5,n[i],label="retrieved top1 actual key force")
        ax.set_ylabel("N"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[-1,0].set_xlabel("relative sample grid (seconds); labels use exact preceding observed steps")
    fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)


def audit(run):
    run=Path(run)
    check_freeze(run)
    if not (run/"evaluation/summary.json").exists():
        raise ValueError("Complete public frozen evaluation before opening private labels")
    out=run/"private_audit"
    out.mkdir(exist_ok=False)
    data=Corpus(run/"cache")
    force,requested=label_frames(data)
    wf=force[data.frames]
    stable=(requested[data.frames] == requested[data.frames[:,:1]]).all(1)
    reports=[]
    for method in read_json(run/"evaluation/summary.json"):
        folder=run/"evaluation"/method["name"]
        result={"method":method["name"],"splits":{}}
        for split in ("test","external"):
            with np.load(folder/f"neighbors_{split}.npz",allow_pickle=False) as item:
                ids=item["query_window_ids"]; neighbor=item["neighbor_window_ids"][:,0]
            q,n=wf[ids],wf[neighbor]
            contact=q.max(1)>=.05
            over=q.max(1)>2
            valid=np.ones(len(q),bool)
            no_contact=~contact
            per_episode={}
            for e in np.unique(data.episode[ids]):
                selection=(data.episode[ids]==e)&contact&~over
                per_episode[data.episodes[int(e)]["id"]]=metrics(q,n,selection)
            result["splits"][split]=dict(all=metrics(q,n,valid), contact=metrics(q,n,contact&~over),
                over_range=metrics(q,n,over), stable_requested_key_contact=metrics(q,n,contact&~over&stable[ids]),
                no_contact_queries=int(no_contact.sum()),
                no_contact_retrieved_contact_fraction=float((n[no_contact].max(1)>=.05).mean()) if no_contact.any() else None,
                retrieved_windows_with_requested_key_change=int((~stable[neighbor]).sum()),
                per_episode_contact=per_episode)
            plot_curves(q,n,contact&~over,out/f"{method['name']}_{split}_force.png")
        reports.append(result)
    write_json(out/"metrics.json",dict(private_labels_not_exported_to_index=True,
        selection_feedback_to_training=False,
        force_definition="PhysX contact force on each frame's publicly specified key; exact observed physics_step",
        limitation="Physical similarity audit, not a skill usefulness label; low-rate samples cannot resolve intervening physics steps.",methods=reports))
    aggregate={}
    for prefix in ("tactile_seed_","tactile_action_seed_"):
        members=[r for r in reports if r["method"].startswith(prefix)]
        aggregate[prefix.rstrip('_')]={}
        for split in ("test","external"):
            values=[r["splits"][split]["contact"]["force_curve_mae_N"] for r in members if r["splits"][split]["contact"]["windows"]]
            aggregate[prefix.rstrip('_')][split]={"contact_curve_mae_N_mean":float(np.mean(values)) if values else None,
                "contact_curve_mae_N_std":float(np.std(values)) if values else None,"training_seeds":len(values)}
    write_json(out/"three_seed_aggregate.json",aggregate)
    lines=["# DIGIT trajectory encoder: frozen offline report", "",
           "This run evaluates representation retrieval, not executable skill usefulness. No control rollout was started.", "",
           "| Method | Test contact curve MAE (N) | External contact curve MAE (N) |",
           "|---|---:|---:|"]
    for r in reports:
        vals=[r["splits"][s]["contact"].get("force_curve_mae_N") for s in ("test","external")]
        lines.append("| "+r["method"]+" | "+" | ".join("n/a" if v is None else f"{v:.6f}" for v in vals)+" |")
    lines += ["", "All methods use the same training gallery and query windows. Native force labels were opened only after checkpoint freezing and public evaluation.",
              "Exact-window reidentification is an augmentation diagnostic; it is not cross-seed skill retrieval accuracy.",
              "The 30 Hz test split and the separate 60 Hz expert/probe split are reported separately. This is one simulated DIGIT configuration, not cross-sensor or real-world validation.",
              "", "See evaluation/summary.json, evaluation/three_seed_aggregate.json, training_summary.json and private_audit/metrics.json for timing, dimensions, variability and coverage."]
    (run/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    check_freeze(run)
    print(json.dumps(aggregate,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    audit(args.run)


if __name__=="__main__": main()
