from __future__ import annotations

import json

from capx.envs.launch import LaunchArgs
from capx.integrations.tactile.memory_api import TactileMemoryApi
from capx.integrations.tactile.strategy_memory import (
    TactileStrategyMemory,
    build_strategy_record,
    build_strategy_record_from_trial_dir,
    classify_failure,
    compile_memory_from_run_dir,
    event_counts_from_timeline,
    event_sequence_from_timeline,
    format_strategies_for_prompt,
)
from capx.utils.launch_utils import _load_config


def row(event: str, idx: int = 0) -> dict:
    return {
        "index": idx,
        "sim_step": idx,
        "timestamp": float(idx),
        "target": "cubeA",
        "contact": event != "no_contact",
        "left_contact": event in {"one_finger_contact", "stable_grasp", "slip_detected"},
        "right_contact": event in {"stable_grasp", "slip_detected"},
        "normal_force": 0.8 if event != "no_contact" else 0.0,
        "shear_magnitude": 0.7 if event == "slip_detected" else 0.0,
        "slip_score": 0.8 if event in {"slip_detected", "contact_lost"} else 0.0,
        "contact_balance": 0.0,
        "max_marker_displacement": 0.0,
        "mean_marker_displacement": 0.0,
        "event": event,
    }


def timeline(events: list[str]) -> list[dict]:
    return [row(event, idx) for idx, event in enumerate(events)]


def test_event_sequence_and_counts() -> None:
    data = timeline(["no_contact", "no_contact", "stable_grasp", "stable_grasp", "contact_lost"])

    assert event_sequence_from_timeline(data) == ["no_contact", "stable_grasp", "contact_lost"]
    assert event_counts_from_timeline(data) == {
        "no_contact": 2,
        "stable_grasp": 2,
        "contact_lost": 1,
    }


def test_failure_classifier_rules() -> None:
    assert classify_failure(timeline(["no_contact"] * 8), reward=0.0, task_completed=False) == "missed_grasp"
    assert (
        classify_failure(
            timeline(["no_contact", "one_finger_contact", "no_contact"]),
            reward=0.0,
            task_completed=False,
        )
        == "off_center_grasp"
    )
    assert (
        classify_failure(
            timeline(["no_contact", "stable_grasp", "slip_detected", "contact_lost"]),
            reward=0.0,
            task_completed=False,
        )
        == "slip_during_lift"
    )
    assert (
        classify_failure(
            timeline(["no_contact", "stable_grasp", "stable_grasp"]),
            reward=0.0,
            task_completed=False,
        )
        == "placement_error"
    )
    assert (
        classify_failure(timeline(["stable_grasp"]), reward=1.0, task_completed=True)
        == "successful_execution"
    )


def test_memory_append_dedup_and_retrieve(tmp_path) -> None:
    memory = TactileStrategyMemory(tmp_path / "memory.jsonl")
    failure_record = build_strategy_record(
        timeline(["stable_grasp", "slip_detected", "contact_lost"]),
        task="cube_stack",
        target="red cube",
        source_trial_dir=tmp_path / "failure_trial",
        reward=0.0,
        task_completed=False,
    )
    success_record = build_strategy_record(
        timeline(["no_contact", "stable_grasp"]),
        task="cube_stack",
        target="red cube",
        source_trial_dir=tmp_path / "success_trial",
        reward=1.0,
        task_completed=True,
    )

    assert memory.append(failure_record) is True
    assert memory.append(failure_record) is False
    assert memory.append(success_record) is True

    records = memory.retrieve(target="red cube", top_k=2)
    assert [record["outcome"] for record in records] == ["success", "failure"]

    slip_records = memory.retrieve(failure_type="slip_during_lift", target="red cube", top_k=1)
    assert slip_records[0]["failure_type"] == "slip_during_lift"
    assert "lower 5mm" in format_strategies_for_prompt(slip_records)


def test_default_retrieval_can_read_frozen_snapshot(tmp_path, monkeypatch) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    work_path = tmp_path / "work.jsonl"
    snapshot_record = build_strategy_record(
        timeline(["no_contact", "stable_grasp"]),
        task="cube_stack",
        target="red cube",
        source_trial_dir=tmp_path / "snapshot_trial",
        reward=1.0,
        task_completed=True,
    )
    work_record = build_strategy_record(
        timeline(["stable_grasp", "slip_detected", "contact_lost"]),
        task="cube_stack",
        target="red cube",
        source_trial_dir=tmp_path / "work_trial",
        reward=0.0,
        task_completed=False,
    )
    assert TactileStrategyMemory(snapshot_path).append(snapshot_record)
    assert TactileStrategyMemory(work_path).append(work_record)

    monkeypatch.setenv("CAPX_TACTILE_STRATEGY_MEMORY_READ_PATH", str(snapshot_path))
    monkeypatch.setenv("CAPX_TACTILE_STRATEGY_MEMORY_PATH", str(work_path))

    records = TactileStrategyMemory().retrieve(target="red cube", top_k=2)
    assert [record["source_trial_dir"] for record in records] == [
        snapshot_record.source_trial_dir
    ]


def test_build_record_from_trial_dir_and_compile(tmp_path) -> None:
    trial_dir = tmp_path / "run" / "outputs" / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "tactile_timeline.json").write_text(
        json.dumps(timeline(["stable_grasp", "slip_detected", "contact_lost"])),
        encoding="utf-8",
    )
    (trial_dir / "summary.txt").write_text(
        "Reward: 0.0\nTask Completed: False\n",
        encoding="utf-8",
    )
    (trial_dir / "code.py").write_text("close_gripper()\n", encoding="utf-8")

    record = build_strategy_record_from_trial_dir(trial_dir)
    assert record is not None
    assert record.failure_type == "slip_during_lift"

    added = compile_memory_from_run_dir(tmp_path / "run", memory_path=tmp_path / "memory.jsonl")
    assert len(added) == 1
    assert compile_memory_from_run_dir(tmp_path / "run", memory_path=tmp_path / "memory.jsonl") == []


def test_tactile_memory_api_exposes_strategy_retrieval(monkeypatch) -> None:
    api = TactileMemoryApi.__new__(TactileMemoryApi)

    monkeypatch.setenv("CAPX_TACTILE_STRATEGY_MEMORY_ENABLED", "0")
    assert "retrieve_tactile_strategies" not in api.functions()

    monkeypatch.setenv("CAPX_TACTILE_STRATEGY_MEMORY_ENABLED", "1")
    assert "retrieve_tactile_strategies" in api.functions()


def test_load_config_reads_tactile_strategy_fields(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
env:
  _target_: capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cubes_low_level
    apis:
      - TactileMemoryApi
    prompt: test prompt
tactile_strategy_memory: true
tactile_strategy_memory_path: test_memory.jsonl
tactile_strategy_memory_read_path: snapshot_memory.jsonl
tactile_strategy_top_k: 2
trials: 1
num_workers: 1
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    env_factory, config, _ = _load_config(LaunchArgs(config_path=str(cfg)))

    assert config["tactile_strategy_memory"] is True
    assert config["tactile_strategy_memory_path"] == "test_memory.jsonl"
    assert config["tactile_strategy_memory_read_path"] == "snapshot_memory.jsonl"
    assert config["tactile_strategy_top_k"] == 2
    assert "Tactile strategy memory:" in env_factory["cfg"]["prompt"]
