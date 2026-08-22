from __future__ import annotations

import json

from capx.envs.launch import LaunchArgs
from capx.envs.trial import _extract_configured_code_blocks
from capx.memory.tactile_code import (
    TactileCodeMemoryBank,
    build_candidates_from_examples,
    build_signature_from_trace,
    classify_signature,
    code_strategy_tags,
    format_memory_for_prompt,
    format_runtime_memory_for_prompt,
    parse_trial_dir,
    promote_candidates_with_validation,
    repair_template,
    scan_trial_examples,
)
from capx.utils.launch_utils import (
    _load_config,
    _extract_code,
    _normalize_extracted_code,
    _parse_multi_turn_decision,
)


def trace_row(
    event: str,
    idx: int,
    *,
    left: bool = False,
    right: bool = False,
    force: float = 0.0,
    balance: float = 0.0,
    slip: float = 0.0,
) -> dict:
    return {
        "operation": "close",
        "phase": "fine_close",
        "iteration": idx,
        "event": event,
        "left_contact": left,
        "right_contact": right,
        "normal_force": force,
        "left_normal_force": force if left else 0.0,
        "right_normal_force": force if right else 0.0,
        "contact_balance": balance,
        "slip_score": slip,
        "width": 0.5,
    }


def write_trial(
    root,
    name: str,
    trace: list[dict],
    *,
    success: bool,
    sandbox_rc: int = 0,
    code: str = "close_gripper(adaptive=True)\n",
    summary_extra: str = "",
):
    trial = root / name
    trial.mkdir(parents=True)
    reward = 1.0 if success else 0.0
    (trial / "tactile_gripper_trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (trial / "summary.txt").write_text(
        (
            f"Sandbox failed: {sandbox_rc}\nReward: {reward}\n"
            f"Task Completed: {success}\n{summary_extra}"
        ),
        encoding="utf-8",
    )
    (trial / "code.py").write_text(code, encoding="utf-8")
    return trial


def test_signature_classifier_one_sided_contact_lost() -> None:
    trace = [
        trace_row("no_contact", 0),
        trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
        trace_row("contact_lost", 2),
        trace_row("no_contact", 3),
    ]

    signature = build_signature_from_trace(trace)

    assert signature["event_sequence"] == [
        "no_contact",
        "one_hand_contact",
        "contact_lost",
        "no_contact",
    ]
    assert classify_signature(signature) == "contact_lost_after_one_hand_contact"
    assert signature["feature_trends"]["bilateral_contact_ratio"] == 0.0


def test_build_candidates_filters_non_tactile_failures(tmp_path) -> None:
    tactile_failure = write_trial(
        tmp_path,
        "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
        [
            trace_row("no_contact", 0),
            trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
            trace_row("contact_lost", 2),
        ],
        success=False,
    )
    write_trial(
        tmp_path,
        "trial_02_sandboxrc_1_reward_0.000_taskcompleted_0",
        [trace_row("no_contact", 0)],
        success=False,
        sandbox_rc=1,
    )
    write_trial(
        tmp_path,
        "trial_03_sandboxrc_0_reward_1.000_taskcompleted_1",
        [
            trace_row("no_contact", 0),
            trace_row("one_hand_contact", 1, left=True, force=0.25, balance=1.0),
            trace_row("stable_grasp", 2, left=True, right=True, force=0.9),
        ],
        success=True,
    )

    examples = scan_trial_examples(tmp_path, start=1, end=80)
    records = build_candidates_from_examples(examples)

    diagnoses = {record.diagnosis for record in records}
    assert parse_trial_dir(tactile_failure) is not None
    assert "contact_lost_after_one_hand_contact" in diagnoses
    assert "maintain_stable_grasp" in diagnoses
    assert all("trial_01" not in json.dumps(record.to_dict()) for record in records)


def test_code_strategy_tags_extract_only_compact_axes() -> None:
    guarded = """
try:
    grasp_pos, grasp_quat = sample_grasp_pose("can")
except RuntimeError:
    grasp_pos = None
result = close_gripper(adaptive=True)
if is_grasp_stable():
    home_pose()
"""
    close_only = """
result = close_gripper(adaptive=True)
if result["stable"]:
    home_pose()
"""

    assert code_strategy_tags(guarded) == {
        "grasp_verification": "tactile_recheck",
        "recovery_source": "vision_regrasp",
        "perception_guard": "guarded",
    }
    assert code_strategy_tags(close_only) == {
        "grasp_verification": "close_stable_only",
        "recovery_source": "none",
        "perception_guard": "none",
    }


def test_rgbd_depth_failure_is_not_tactile_candidate(tmp_path) -> None:
    write_trial(
        tmp_path,
        "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
        [
            trace_row("no_contact", 0),
            trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
            trace_row("contact_lost", 2),
        ],
        success=False,
        summary_extra="RuntimeError: RGB-D detection has only 0 valid depth points\n",
    )

    assert build_candidates_from_examples(scan_trial_examples(tmp_path)) == []


def test_strategy_variants_do_not_split_same_tactile_diagnosis(tmp_path) -> None:
    trace = [
        trace_row("no_contact", 0),
        trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
        trace_row("contact_lost", 2),
    ]
    write_trial(
        tmp_path,
        "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
        trace,
        success=False,
        code='result = close_gripper(adaptive=True)\nif result["stable"]:\n    home_pose()\n',
    )
    write_trial(
        tmp_path,
        "trial_02_sandboxrc_0_reward_0.000_taskcompleted_0",
        trace,
        success=False,
        code=(
            "try:\n"
            '    sample_grasp_pose("can")\n'
            "except RuntimeError:\n"
            "    pass\n"
            "close_gripper(adaptive=True)\n"
            "if is_grasp_stable():\n"
            "    home_pose()\n"
        ),
    )

    records = build_candidates_from_examples(scan_trial_examples(tmp_path))

    assert len(records) == 1
    assert records[0].diagnosis == "contact_lost_after_one_hand_contact"
    assert records[0].strategy_tags["recovery_source"] == "pregrasp_local_adjustment"
    observed = records[0].trace_summary["observed_strategy_counts"]
    assert observed["recovery_source"]["vision_regrasp"] == 1
    assert observed["grasp_verification"]["tactile_recheck"] == 1


def test_pregrasp_repair_template_uses_local_anchor_not_visual_regrasp() -> None:
    _strategy, sketch = repair_template("contact_lost_after_one_hand_contact")

    assert "tactile_pregrasp_repair" in sketch
    assert "public official grasp anchor" in sketch
    assert "goto_pose(np.asarray(anchor_pos" in sketch
    assert "z_approach=0.0" in sketch
    assert "home_pose()" not in sketch


def test_bank_retrieval_prefers_validated_matching_signature(tmp_path) -> None:
    trace = [
        trace_row("no_contact", 0),
        trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
        trace_row("contact_lost", 2),
    ]
    write_trial(
        tmp_path,
        "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
        trace,
        success=False,
    )
    records = build_candidates_from_examples(scan_trial_examples(tmp_path))
    records[0].status = "validated"
    records[0].evidence["recovery_count"] = 2
    records[0].evidence["validation_success_rate"] = 0.67
    bank = TactileCodeMemoryBank(tmp_path / "bank.jsonl")
    bank.write(records)

    matches = bank.retrieve(
        current_signature=build_signature_from_trace(trace),
        required_apis=["FrankaControlApi", "UniVTACTactileApi"],
        statuses=["validated"],
        include_scores=True,
    )

    assert len(matches) == 1
    assert matches[0]["status"] == "validated"
    assert matches[0]["_score"] > 0
    initial_prompt = format_memory_for_prompt(matches)
    runtime_prompt = format_runtime_memory_for_prompt(
        build_signature_from_trace(trace),
        matches,
    )
    assert "Tactile code memory (compact skill cards):" in initial_prompt
    assert "Runtime tactile code memory" in runtime_prompt
    for prompt in (initial_prompt, runtime_prompt):
        assert "code_hint:" in prompt
        assert "evidence:" not in prompt
        assert "failure_signature:" not in prompt
        assert "strategy_tags:" not in prompt


def test_runtime_retrieval_ignores_release_no_contact(tmp_path) -> None:
    failure_trace = [
        trace_row("no_contact", 0),
        trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
        trace_row("contact_lost", 2),
    ]
    write_trial(
        tmp_path,
        "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
        failure_trace,
        success=False,
    )
    records = build_candidates_from_examples(scan_trial_examples(tmp_path))
    records[0].status = "validated"
    bank = TactileCodeMemoryBank(tmp_path / "bank.jsonl")
    bank.write(records)

    release_signature = build_signature_from_trace(
        [trace_row("no_contact", idx) for idx in range(20)]
    )
    matches = bank.retrieve(
        current_signature=release_signature,
        required_apis=["FrankaControlApi", "UniVTACTactileApi"],
        statuses=["validated"],
        include_scores=True,
    )

    assert matches == []


def test_promotion_requires_runtime_memory_hit(tmp_path) -> None:
    trace = [
        trace_row("no_contact", 0),
        trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
        trace_row("contact_lost", 2),
    ]
    write_trial(
        tmp_path / "train",
        "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
        trace,
        success=False,
    )
    candidates = build_candidates_from_examples(scan_trial_examples(tmp_path / "train"))
    target = candidates[0]

    natural_success = write_trial(
        tmp_path / "validation_no_hit",
        "trial_81_sandboxrc_0_reward_1.000_taskcompleted_1",
        trace,
        success=True,
    )
    (natural_success / "tactile_code_memory_trace.json").write_text("[]", encoding="utf-8")
    assert (
        promote_candidates_with_validation(
            [target],
            scan_trial_examples(tmp_path / "validation_no_hit"),
            min_applicable=1,
            min_recovery=1,
        )
        == []
    )

    recovered = write_trial(
        tmp_path / "validation_hit",
        "trial_82_sandboxrc_0_reward_1.000_taskcompleted_1",
        trace,
        success=True,
    )
    (recovered / "tactile_code_memory_trace.json").write_text(
        json.dumps(
            [
                {
                    "kind": "runtime_retrieval",
                    "matches": [{"id": target.id, "score": 99.0}],
                }
            ]
        ),
        encoding="utf-8",
    )
    promoted = promote_candidates_with_validation(
        [target],
        scan_trial_examples(tmp_path / "validation_hit"),
        min_applicable=1,
        min_recovery=1,
    )
    assert promoted[0].status == "validated"
    assert promoted[0].evidence["recovery_count"] == 1


def test_load_config_injects_tactile_code_memory(tmp_path, monkeypatch) -> None:
    trace = [
        trace_row("no_contact", 0),
        trace_row("one_hand_contact", 1, right=True, force=0.45, balance=-1.0),
        trace_row("contact_lost", 2),
    ]
    write_trial(
        tmp_path,
        "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0",
        trace,
        success=False,
    )
    records = build_candidates_from_examples(scan_trial_examples(tmp_path))
    for record in records:
        record.status = "validated"
    bank = tmp_path / "bank.jsonl"
    TactileCodeMemoryBank(bank).write(records)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
env:
  _target_: capx.envs.tasks.base.CodeExecutionEnvBase
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: fake_low_level
    apis:
      - FrankaControlApi
      - UniVTACTactileApi
    prompt: minimal prompt
tactile_code_memory:
  enabled: true
  mode: read
  path: {bank}
  top_k_initial: 1
  top_k_runtime: 1
trials: 1
num_workers: 1
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    env_factory, config, _ = _load_config(LaunchArgs(config_path=str(cfg)))

    assert config["tactile_code_memory"]["enabled"] is True
    assert config["max_regenerations"] == 2
    assert "Tactile code memory (compact skill cards):" in env_factory["cfg"]["prompt"]
    assert "evidence:" not in env_factory["cfg"]["prompt"]
    assert "failure_signature:" not in env_factory["cfg"]["prompt"]


def test_configured_code_block_split() -> None:
    content = """```python
close_gripper(adaptive=True)
breakpoint_code_block()
home_pose()
```"""

    blocks = _extract_configured_code_blocks(
        content,
        {"tactile_code_memory": {"split_code_blocks_on_breakpoint": True}},
    )

    assert blocks == ["close_gripper(adaptive=True)", "home_pose()"]


def test_configured_code_block_split_ignores_indented_breakpoint() -> None:
    content = """```python
def run():
    anchor_pos, anchor_quat = sample_grasp_pose("can")
    breakpoint_code_block()
    goto_pose(anchor_pos, anchor_quat)

run()
```"""

    blocks = _extract_configured_code_blocks(
        content,
        {"tactile_code_memory": {"split_code_blocks_on_breakpoint": True}},
    )

    assert len(blocks) == 1
    assert "anchor_pos, anchor_quat" in blocks[0]
    assert "breakpoint_code_block()" in blocks[0]
    assert blocks[0].endswith("run()")


def test_parse_multi_turn_continue_decision() -> None:
    decision, payload = _parse_multi_turn_decision("CONTINUE\nrun the next block")

    assert decision == "continue"
    assert payload == "CONTINUE\nrun the next block"


def test_parse_multi_turn_regenerate_extracts_first_python_block_only() -> None:
    content = """
I will repair the remaining program.

REGENERATE
```python
home_pose()
open_gripper(adaptive=True)
```

This explanatory text must not be executed.

```python
close_gripper(adaptive=True)
```
"""

    decision, payload = _parse_multi_turn_decision(content)

    assert decision == "regenerate"
    assert payload == "home_pose()\nopen_gripper(adaptive=True)"


def test_extract_code_uses_first_python_fence_only() -> None:
    content = """
```python
print("first")
```
not python
```python
print("second")
```
"""

    assert _extract_code(content) == ['print("first")']


def test_normalize_extracted_code_removes_model_block_marker_and_dedents() -> None:
    content = "# Code block 2\n    lift_height = 0.15\n    print(lift_height)"

    assert _normalize_extracted_code(content) == "lift_height = 0.15\nprint(lift_height)"
