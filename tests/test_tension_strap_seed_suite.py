"""The fixed expert/CaP-X instructions stay outside OpenTacBench source."""

from pathlib import Path

from capx.integrations.opentac.seed_suite import load_seed_sequence
from scripts.run_tension_strap_seed_suite_expert import build_expert_config


def test_expert_config_uses_frozen_instruction_without_mutating_base(tmp_path: Path) -> None:
    base = {"task_parameters": {"force_time_sequence": [12.0, 3.0, 18.0, 3.0]}}
    sequence = load_seed_sequence("env_configs/univtac/tension_strap_seed_suite_1_30.v1.json", 2)
    expert = build_expert_config(base, sequence, tmp_path / "seed_02")
    assert expert["task_parameters"]["force_time_sequence"] == [17.2, 1.5, 15.2, 1.6]
    assert expert["save_dir_exact"] == str(tmp_path / "seed_02")
    assert base["task_parameters"]["force_time_sequence"] == [12.0, 3.0, 18.0, 3.0]
