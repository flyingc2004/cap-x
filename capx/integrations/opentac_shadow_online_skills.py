"""Episode-boundary skill reflection from a strictly public evidence file.

This module never opens native audits, run.log, rewards or trial summaries.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

import yaml

from capx.integrations.opentac_shadow_keyboard import atomic_json
from capx.llm.deadline import request_deadline
from capx.integrations.opentac_shadow_skills import (
    SCHEMA, MAX_FILE_BYTES, MAX_SKILLS, _PRIVATE_MARKERS, inject_shadow_piano_public_skills,
)

REFLECTION_TIMEOUT_SECONDS = 120


@contextmanager
def _reflection_deadline(seconds):
    """The text-library organizer retains its independent 120-second budget."""
    with request_deadline(seconds, phase='skill_reflection'):
        yield


def library_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_library(raw: bytes) -> dict:
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("Library exceeds the fixed 8 KB budget")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or set(data) != {"schema", "skills"} or data["schema"] != SCHEMA:
        raise ValueError("Unsupported library schema")
    skills = data["skills"]
    if not isinstance(skills, list) or not 0 <= len(skills) <= MAX_SKILLS:
        raise ValueError("Library requires 0..16 skills")
    ids = set()
    for row in skills:
        if not isinstance(row, dict) or set(row) != {"id", "trigger", "practice"}:
            raise ValueError("Skill fields must be id/trigger/practice")
        if not isinstance(row["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", row["id"]) or row["id"] in ids:
            raise ValueError("Duplicate or invalid skill id")
        ids.add(row["id"])
        if not isinstance(row["trigger"], str) or not 1 <= len(row["trigger"]) <= 240:
            raise ValueError("Invalid trigger length")
        if not isinstance(row["practice"], str) or not 1 <= len(row["practice"]) <= 600:
            raise ValueError("Invalid practice length")
    text = raw.decode("utf-8").lower()
    if any(marker in text for marker in _PRIVATE_MARKERS):
        raise ValueError("Private audit markers are prohibited")
    if re.search(r"(?:ignore|disable|bypass|override|remove)\s+(?:the\s+)?(?:hard|safety|force|action|native|fixed|task)", text):
        raise ValueError("Skill attempts to override a fixed constraint")
    return data


def initialize_library(source: Path | None, destination: Path) -> None:
    """None starts with no historical entries; task constraints remain fixed."""
    data = {"schema": SCHEMA, "skills": []} if source is None else validate_library(source.read_bytes())
    for row in data["skills"]:
        if row["id"] == "visual_key_map":
            row["practice"] = ("Read get_keyboard_endpoints(). When calibration_ready is true, reuse mapped_keys without an image. "
                "Otherwise confirm the supplied A0/C8 endpoints in the clear HEAD image and pass only those two normalized "
                "white-key pixels to locate_visual_keys. Save the returned requested-key targets in piano_state['keys']. "
                "The public endpoint calibration works for new notes across compatible seeds; fresh hand/DIGIT feedback is still required.")
    raw = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")
    validate_library(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)


def public_evidence(run_dir: Path) -> dict:
    episode_path = run_dir / "shadow_public_episode.json"
    if not episode_path.is_file():
        raise ValueError("No completed public reset record; retain the previous library")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    # Explicit field selection: do not pass arbitrary artifact fields onward.
    request = {k: episode[k] for k in ("seed", "sequence", "keyboard_cache_hit", "clear_action_steps")}
    source = run_dir / "shadow_public_observations.jsonl"
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines() if source.is_file() else []:
        row = json.loads(line)
        kind = row.get("kind")
        safe = {"kind": kind, "action_step": row["action_step"]}
        if kind == "force_estimate":
            hand = {k: row["hand_state"][k] for k in ("wrist_x_offset_m", "wrist_y_m", "wrist_z_m", "index_flexion_rad", "wrist_and_index_velocity")}
            safe.update(note=row["note"], target=row["target"], hand_state=hand,
                estimate={k: row["estimate"].get(k) for k in
                    ("estimated_key_force_N", "sample_step", "age_steps", "quality", "calibration_version")})
        elif kind == "action":
            safe.update(target=row["target"], hand_state=row["hand_state"], native_done=row["native_done"])
        elif kind == "attempt_finalized":
            safe["index"] = row["index"]
        elif kind == 'release_rejected':
            safe.update(index=row['index'], hand_state=row['hand_state'],
                continuous_release_seconds=row['continuous_release_seconds'])
        elif kind == "code_execution":
            safe["public_messages"] = row["public_messages"]
        else:
            continue
        safe["event_id"] = len(rows)
        rows.append(safe)
    forces = [r for r in rows if r["kind"] == "force_estimate"]
    quality = Counter(r["estimate"]["quality"] for r in forces)
    # Keep representative contact dynamics and all rare invalid states within a
    # bounded prompt. Full public telemetry remains available for reproduction.
    selected = {r["event_id"] for r in rows if r["kind"] in ("attempt_finalized", "code_execution")}
    mandatory = set(selected)
    # Preserve measured release trajectories, not only sparse press estimates.
    for event in rows:
        if event['kind'] in ('attempt_finalized', 'release_rejected'):
            mandatory.add(event['event_id'])
            prior_actions = [r for r in rows if r['kind']=='action' and r['event_id']<event['event_id']]
            mandatory.update(r['event_id'] for r in prior_actions[-8:])
    for note in dict.fromkeys(r["note"] for r in forces):
        group = [r for r in forces if r["note"] == note]
        stride = max(1, len(group) // 5)
        selected.update(r["event_id"] for r in group[::stride][:6])
        selected.update(r["event_id"] for r in group if r["estimate"]["quality"] not in ("ok", "no_contact"))
        selected.add(group[-1]["event_id"])
    selected.update(mandatory)
    keep = set(sorted(mandatory)[-128:])
    for i in sorted(selected - keep):
        if len(keep)>=128:break
        keep.add(i)
    selected_rows = [r for r in rows if r['event_id'] in keep]
    code_paths = sorted(run_dir.glob("trial_*/code.py")) + sorted(run_dir.glob("**/in_progress_code.py"))
    code = code_paths[0].read_text(encoding="utf-8", errors="replace") if code_paths else ""
    if any(marker in code.lower() for marker in _PRIVATE_MARKERS):
        code = "[Code omitted: contained a prohibited private marker.]"
    if len(code) > 16000:
        code = code[:10000] + "\n# [middle omitted]\n" + code[-6000:]
    result = {"schema": "shadow_piano_skill_public_evidence.v1", "run_id": run_dir.name,
              "request": request, "quality_counts": dict(quality), "public_events": selected_rows,
              "generated_code": code,
              "limitations": "No native score/true force is available. Public hold confidence is not native success."}
    # This verifies the whitelist output, rather than relying on prompts alone.
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if any(marker in encoded.lower() for marker in _PRIVATE_MARKERS):
        raise ValueError("Public evidence contains a prohibited field")
    return result


def validate_proposal(proposal: dict, previous: dict, evidence: dict) -> tuple[bytes, list]:
    if not isinstance(proposal, dict) or set(proposal) != {"skills", "evidence"}:
        raise ValueError("Reflection must return skills and evidence")
    raw = yaml.safe_dump({"schema": SCHEMA, "skills": proposal["skills"]}, sort_keys=False, allow_unicode=True).encode("utf-8")
    data = validate_library(raw)
    if not isinstance(proposal["evidence"], list):
        raise ValueError("Evidence must be a list")
    event_ids = {r["event_id"] for r in evidence["public_events"]}
    supported = {}
    for row in proposal["evidence"]:
        if not isinstance(row, dict) or set(row) != {"skill_id", "event_ids", "basis", "reason"}:
            raise ValueError("Invalid evidence record")
        if row["basis"] not in ("observed", "hypothesis") or not isinstance(row["event_ids"], list) or not row["event_ids"]:
            raise ValueError("Each update needs public events and an evidence basis")
        if any(isinstance(i, bool) or not isinstance(i, int) or i not in event_ids for i in row["event_ids"]):
            raise ValueError("Reflection cites an unavailable public event")
        if not isinstance(row["reason"], str) or not 1 <= len(row["reason"]) <= 500:
            raise ValueError("Invalid evidence explanation")
        supported[row["skill_id"]] = row
    old = {r["id"]: r for r in previous["skills"]}
    for row in data["skills"]:
        if old.get(row["id"]) != row and row["id"] not in supported:
            raise ValueError("Every changed skill requires traceable public evidence")
    if len({r["id"] for r in data["skills"]} & old.keys()) < min(4, len(old)):
        raise ValueError("Reflection discarded the fixed safety foundation")
    return raw, proposal["evidence"]


def reflect_once(library: Path, run_dir: Path, output: Path, query_args) -> dict:
    from capx.llm.client import collect_llm_usage, query_model, summarize_llm_usage
    output.mkdir(parents=True, exist_ok=False)
    before = library.read_bytes()
    (output / "library_input.yaml").write_bytes(before)
    previous = validate_library(before)
    events = []
    result = {"input_sha256": hashlib.sha256(before).hexdigest(), "accepted": False}
    try:
        evidence = public_evidence(run_dir)
        atomic_json(output / "public_evidence.json", evidence)
        instructions = (
            "整理钢琴公开技能库，输出一个 JSON 对象，仅有 skills 和 evidence。skills 是完整的下一版条目列表，"
            "每条仅 id/trigger/practice，保持英文便于后续代码生成，每条 trigger<=240字符、practice<=600字符，最多16条、总YAML<=8192字节。"
            "合并去重，不为更新而更新。维持现有安全基础，不修改固定动作边界、力阈值、估计器、原生判分或任务时限。"
            "可建议更好的规划、接近、等待、稳定、微调与释放，但不能读私有数据、绕过保护或声称已获官方成功。"
            "仅依据给出的公开反馈；每条新增或修改建议必须在 evidence 给出 {skill_id,event_ids:[整数],basis:'observed'或'hypothesis',reason:简短解释}。"
            "没有确证的控制改善必须标为 hypothesis，并在 practice 明确需要新触觉验证。未改条目无须 evidence。"
            "只有公开估计，没有真实力或原生得分；不能推断哪一音官方成功。端点标定跨兼容seed复用，缓存命中无需图片；"
            "get_hand_state没有native_done，终止标记来自motion返回。输出纯JSON，不要代码围栏。")
        prompt = [{"role": "system", "content": instructions},
                  {"role": "user", "content": json.dumps({"previous_library": previous, "public_evidence": evidence}, ensure_ascii=False)}]
        atomic_json(output / "reflection_prompt.json", {"messages": prompt})
        with collect_llm_usage(events, "skill_reflection"), _reflection_deadline(REFLECTION_TIMEOUT_SECONDS):
            response = query_model(query_args, prompt)
        (output / "response.txt").write_text(response["content"], encoding="utf-8")
        text = response["content"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        proposal = json.loads(text)
        atomic_json(output / "proposal.json", proposal)
        raw, references = validate_proposal(proposal, previous, evidence)
        atomic_json(output / "provenance.json", {"run_id": run_dir.name, "evidence": references})
        temp = library.with_suffix(".next.yaml")
        temp.write_bytes(raw)
        temp.replace(library)
        result.update(accepted=True, changed=raw != before)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    (output / "library_output.yaml").write_bytes(library.read_bytes())
    result["output_sha256"] = library_digest(library)
    result["usage"] = summarize_llm_usage(events)
    atomic_json(output / "result.json", result)
    return result
