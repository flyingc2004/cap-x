#!/usr/bin/env python3
"""Build a paper-facing Shadow piano run ledger from private audit summaries."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


TIMESTAMP = re.compile(r"_(\d{8}T\d{6}Z)$")
AUDIT_NAME = re.compile(r"(.+_\d{8}T\d{6}Z)_[0-9a-f]{32}$")
AUDIT_LOG = re.compile(r"\[capx-shadow\] native piano audit=([^\s]+)")
CLEAR_LOG = re.compile(r"\[capx-shadow\] keyboard clear at action_step=(\d+)")
EXIT_LOG = re.compile(r"\[run\] rep 1 exit_code=(\d+)")


@dataclass
class Run:
    run_id: str
    log: Path | None
    audit: Path | None
    summary: dict | None
    action_steps: int | None
    code_action_steps: int | None
    exit_code: int | None
    audit_problem: str | None = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    return int(matches[-1]) if matches else None


def _audit_run_id(path: Path) -> str | None:
    match = AUDIT_NAME.fullmatch(path.name)
    return match.group(1) if match else None


def collect(runs_root: Path, audit_root: Path) -> list[Run]:
    logs = {path.parent.name: path for path in runs_root.glob("shadow_*/run.log")}
    candidates: dict[str, list[Path]] = {}
    for directory in audit_root.iterdir():
        if directory.is_dir() and (run_id := _audit_run_id(directory)):
            candidates.setdefault(run_id, []).append(directory)

    result = []
    for run_id in sorted(set(logs) | set(candidates)):
        log = logs.get(run_id)
        log_text = _read_text(log) if log else ""
        declared = AUDIT_LOG.findall(log_text)
        audit = None
        problem = None
        if declared:
            declared_path = Path(declared[-1])
            if declared_path.is_file():
                audit = declared_path.parent
        if audit is None and len(candidates.get(run_id, [])) == 1:
            audit = candidates[run_id][0]
        elif audit is None and len(candidates.get(run_id, [])) > 1:
            problem = "审计目录不唯一"

        summary = None
        actions = None
        if audit:
            summary_path = audit / "shadow_piano_native_audit_summary.json"
            events_path = audit / "shadow_native_events.jsonl"
            try:
                summary = json.loads(_read_text(summary_path))
            except (OSError, ValueError) as exc:
                problem = "审计无权限或缺失" if isinstance(exc, OSError) else "审计格式错误"
            try:
                with events_path.open(encoding="utf-8") as stream:
                    actions = sum(json.loads(line).get("kind") == "action" for line in stream)
            except (OSError, ValueError):
                problem = problem or "动作审计无权限或缺失"
        else:
            problem = problem or "原生审计缺失"

        clear_steps = _last_int(CLEAR_LOG, log_text)
        if actions is None:
            code_actions = None
        elif clear_steps is not None and 0 <= clear_steps <= actions:
            code_actions = actions - clear_steps
        elif actions == 0:
            code_actions = 0
        else:
            # Audit-only historical runs do not record when automatic view
            # clearing ends. Never infer the generated-code count from a pose.
            code_actions = None
        result.append(Run(
            run_id=run_id, log=log, audit=audit, summary=summary,
            action_steps=actions, code_action_steps=code_actions,
            exit_code=_last_int(EXIT_LOG, log_text), audit_problem=problem,
        ))
    return sorted(result, key=lambda row: (TIMESTAMP.search(row.run_id).group(1), row.run_id))


def _link(label: str, path: Path | None) -> str:
    return f"[{label}]({path})" if path and path.is_file() else "—"


def render(rows: list[Run], runs_root: Path, audit_root: Path) -> str:
    auditable = sum(row.summary is not None and row.action_steps is not None for row in rows)
    lines = [
        "# Shadow Hand 钢琴实验：成功音数与动作步数",
        "",
        f"当前索引：{len(rows)} 次运行；{auditable} 次有可读原生计分和动作审计。其余保留在表中，缺失值用“—”表示，不按 0 处理。",
        "",
        "## 统计口径",
        "",
        "- **成功音数**取 `shadow_piano_native_audit_summary.json` 的 `native_notes_completed`，分母取该摘要中的请求音数；它是原生任务的逐音计分，不是生成代码打印的信心值，也不是 `Task Completed`。启用失败后继续尝试的运行会将成功音数和尝试数分开记录。",
        "- **总原生动作步**精确计数 `shadow_native_events.jsonl` 中 `kind=action` 的行数。每行是一次 `Task.step()`；旧配置推进 8 个 240 Hz 物理步，`4phys` 配置推进 4 个。批量 `steps=n` 计为 n 步。包括 reset 后的自动视觉清场动作，不包括 reset、键位映射、尝试结束或 END。",
        "- **生成代码动作步**＝总原生动作步减去同次 `run.log` 记录的 `keyboard clear at action_step`。没有清场记录但总数为 0 时记 0；其余缺少清场记录的运行记“—”，避免把适配器清场算成模型动作。",
        "- **END**取私有摘要 `end_submitted`。终端退出码只在 `run.log` 存在时记录；退出码 0 本身不表示钢琴成功。若审计无权限、缺失或运行在启动阶段中断，不虚构成功音数或动作数。",
        "- 历史运行的控制配置和故障类型不同。本表是完整实验账本；论文计算成功率或平均动作数时，须先固定纳入标准、配置版本及是否包含未完成运行，再从本表筛选，不能直接平均全部行。",
        "",
        "## 逐次记录（UTC）",
        "",
        "| 时间 | 实验版本 | Seed | 原生成功音数 | 尝试数 | 生成代码动作步 | 总原生动作步 | END | 退出码 | 原始证据 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |",
    ]
    for row in rows:
        match = TIMESTAMP.search(row.run_id)
        stamp = match.group(1)
        time = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}"
        version = re.sub(r"_seed\d+", "", row.run_id.removeprefix("shadow_").split("_rep1_")[0])
        seed_match = re.search(r"_seed(\d+)", row.run_id)
        seed = seed_match.group(1) if seed_match else "—"
        summary = row.summary
        successes = f"{summary['native_notes_completed']}/{len(summary['notes'])}" if summary else "—"
        attempts = str(summary["native_notes_attempted"]) if summary and summary.get("native_notes_attempted") is not None else "—"
        end = "是" if summary and summary.get("end_submitted") else "否" if summary else "—"
        code_actions = str(row.code_action_steps) if row.code_action_steps is not None else "—"
        total_actions = str(row.action_steps) if row.action_steps is not None else "—"
        exit_code = str(row.exit_code) if row.exit_code is not None else "—"
        evidence = f"{_link('日志', row.log)} · {_link('审计', row.audit / 'shadow_piano_native_audit_summary.json' if row.audit else None)}"
        if row.audit_problem:
            evidence += f"（{row.audit_problem}）"
        lines.append(f"| {time} | `{version}` | {seed} | {successes} | {attempts} | {code_actions} | {total_actions} | {end} | {exit_code} | {evidence} |")
    lines += [
        "",
        "## 更新方式",
        "",
        "在现有 CaP-X 工作区运行：",
        "",
        "```bash",
        "python3 scripts/opentac/build_shadow_piano_experiment_ledger.py",
        "```",
        "",
        f"默认读取 `{runs_root}/shadow_*/run.log` 和 `{audit_root}` 下的审计目录，重新生成本文件。此脚本只读实验产物，不运行仿真，不把私有逐帧力标签写入文档或运行时提示词。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/mnt/data/ljz/opentacbench/capx-runs"))
    parser.add_argument("--audit-root", type=Path, default=Path("/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "docs/opentac-shadow-piano-experiment-ledger.md")
    args = parser.parse_args()
    rows = collect(args.runs_root, args.audit_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, args.runs_root, args.audit_root), encoding="utf-8")
    print(f"wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
