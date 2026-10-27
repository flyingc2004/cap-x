#!/usr/bin/env python3
"""Compare frozen 16-frame 5D retrieval; native labels are opened only in audit stage."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from capx.memory.trajectory.data import Corpus, read_json, sha256, write_json
from capx.memory.trajectory.evaluate import check_freeze
from capx.memory.trajectory.rate_baseline import public_evaluation


def private_audit(run, output):
    run, output = Path(run), Path(output)
    check_freeze(run)
    public = read_json(output/"public_manifest.json")
    if Path(public["base_run"]).resolve() != run.resolve():
        raise ValueError("Base run mismatch")
    for path, expected in public["inputs_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Changed frozen input: {path}")
    for relative, expected in public["neighbors_sha256"].items():
        if sha256(output/relative) != expected:
            raise ValueError(f"Changed retrieval: {relative}")
    out = output/"private_audit"
    out.mkdir(exist_ok=False)
    # This is the first point at which private label reading code is imported/called.
    spec = importlib.util.spec_from_file_location("private_force_audit", Path(__file__).with_name("audit_digit_trajectory_neighbors.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    data = Corpus(run/"cache")
    force, requested = audit.label_frames(data)
    wf = force[data.frames]
    stable = (requested[data.frames] == requested[data.frames[:, :1]]).all(1)
    reports = []
    for method in public["methods"]:
        name = method["name"]
        result = dict(method=name, splits={})
        for split in ("test", "external"):
            with np.load(output/name/f"neighbors_{split}.npz", allow_pickle=False) as f:
                ids, nn = f["query_window_ids"], f["neighbor_window_ids"]
            q, n = wf[ids], wf[nn]
            contact, over = q.max(1) >= .05, q.max(1) > 2
            per_episode = {}
            for e in np.unique(data.episode[ids]):
                selection = (data.episode[ids] == e) & contact & ~over
                per_episode[data.episodes[int(e)]["id"]] = audit.metrics(q, n, selection)
            result["splits"][split] = dict(
                all=audit.metrics(q, n, np.ones(len(q), bool)),
                contact=audit.metrics(q, n, contact & ~over),
                stable_requested_key_contact=audit.metrics(q, n, contact & ~over & stable[ids]),
                over_range=audit.metrics(q, n, over),
                no_contact_queries=int((~contact).sum()),
                no_contact_retrieved_contact_fraction=float((n[~contact].max(1) >= .05).mean()) if (~contact).any() else None,
                per_episode_contact=per_episode)
            if name in ("window16x4", "window16x5", "window17x4"):
                audit.plot_curves(q, n, contact & ~over, out/f"{name}_{split}.png")
        reports.append(result)
    report = dict(methods=reports, force_definition="Exact observed step's specified-key PhysX contact force",
                  private_labels_not_exported_to_index=True, selection_feedback_to_training=False,
                  limitation="Retrieved-curve similarity, not force-estimator error or skill relevance")
    write_json(out/"metrics.json", report)
    base = {r["method"]: r for r in reports}
    comparisons = {}
    for split in ("test", "external"):
        a = base["window16x4"]["splits"][split]
        b = base["window16x5"]["splits"][split]
        pairs = [(v["force_curve_mae_N"], b["per_episode_contact"][e]["force_curve_mae_N"])
                 for e, v in a["per_episode_contact"].items() if v["windows"]]
        comparisons[split] = dict(contact_windows=b["contact"]["windows"],
            relative_mae_reduction_5d=1-b["contact"]["force_curve_mae_N"]/a["contact"]["force_curve_mae_N"],
            episodes_with_contact=len(pairs), episodes_5d_better=sum(y<x for x,y in pairs))
    write_json(out/"paired_comparison.json", comparisons)
    lines = ["# 16 帧 5D 检索补充对照", "", "权重保持冻结；仅补充检索与私有真实力审计。", "",
             "| 表示 | 留出接触力曲线 MAE (N) | 额外轨迹 MAE (N) |", "|---|---:|---:|"]
    for r in reports:
        lines.append(f"| {r['method']} | {r['splits']['test']['contact']['force_curve_mae_N']:.6f} | {r['splits']['external']['contact']['force_curve_mae_N']:.6f} |")
    lines += ["", "所有方法在同一有效训练 gallery 与留出 query 上重算；以 float64 余弦检索，精确同分按窗口编号排序。",
              "因此本表与旧报告的数值可能略有差异，应在本表内比较。", "",
              "第五维是面积比例在精确前 8 个物理 step（1/30 秒）的变化率。缺少历史时丢弃窗口，不补零。",
              "16x5 的原始历史支持为约 0.533 秒；17x4 使用相同额外历史，辅助区分导数与额外历史的收益。",
              "尺度只从训练集计算，没有拟合力标签或按审计结果调权重。",
              "误差是查询与检索结果的实测力曲线之差，不是预测力误差，也不是技能适用性准确率。", "",
              "## 覆盖率", ""]
    for split, c in public["coverage"].items():
        lines.append(f"- {split}: {c}")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    check_freeze(run)
    print("AREA_RATE_AUDIT_COMPLETE", comparisons, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("all", "public", "audit"), default="all")
    args = p.parse_args()
    started = time.monotonic()
    if args.stage in ("all", "public"):
        public_evaluation(args.run, args.output)
        paths = [Path(__file__), REPO/"capx/memory/trajectory/rate_baseline.py",
                 Path(__file__).with_name("audit_digit_trajectory_neighbors.py")]
        (args.output/"source").mkdir()
        for path in paths:
            shutil.copyfile(path, args.output/"source"/path.name)
        write_json(args.output/"code_provenance.json", {str(p.relative_to(REPO)): sha256(p) for p in paths})
        (args.output/"command.txt").write_text(f"{sys.executable} scripts/opentac/evaluate_digit_area_rate_baseline.py --run {args.run} --output {args.output}\n", encoding="utf-8")
    if args.stage in ("all", "audit"):
        private_audit(args.run, args.output)
    write_json(args.output/f"status_{args.stage}.json", dict(state="complete", exit_code=0, seconds=time.monotonic()-started))


if __name__ == "__main__":
    main()
