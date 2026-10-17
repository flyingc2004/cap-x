"""Run the stock OpenTacBench expert against CaP-X's frozen seed suite."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys

import yaml

from capx.integrations.opentac.seed_suite import load_seed_sequence


def build_expert_config(base: dict, sequence: list[float], output: Path) -> dict:
    config = copy.deepcopy(base)
    config.setdefault("task_parameters", {})["force_time_sequence"] = list(sequence)
    config["save_dir_exact"] = str(output)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentac-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", default="env_configs/univtac/tension_strap_seed_suite_1_30.v1.json")
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--end-seed", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.start_seed < 0 or args.end_seed < args.start_seed:
        parser.error("seed range must be nonnegative and ordered")
    root = args.opentac_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    source_config = root / "task_config" / "tension_strap.yml"
    collector = root / "scripts" / "collect_data.py"
    if not source_config.is_file() or not collector.is_file():
        parser.error("--opentac-root must contain the stock tension_strap task and collector")
    base = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    sequences = {
        seed: load_seed_sequence(args.manifest, seed)
        for seed in range(args.start_seed, args.end_seed + 1)
    }
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    tacex = root / "third_party" / "TacEx" / "source"
    env = os.environ.copy()
    paths = [str(tacex / name) for name in ("tacex", "tacex_uipc", "tacex_assets", "tacex_tasks")]
    env["PYTHONPATH"] = os.pathsep.join([*paths, env.get("PYTHONPATH", "")])

    failed = []
    for seed, sequence in sequences.items():
        trial_output = output / f"seed_{seed:02d}"
        config_path = config_dir / f"seed_{seed:02d}.yml"
        config = build_expert_config(base, sequence, trial_output)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [
            sys.executable, "-u", str(collector), "tension_strap", str(config_path),
            "--episode_num", "1", "--start_seed", str(seed),
            "--max_seed", str(seed), "--gpu", str(args.gpu),
        ]
        print(f"expert seed={seed} instruction={sequence} output={trial_output}", flush=True)
        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode:
            failed.append(seed)
    if failed:
        print(f"expert collection failed for seeds: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
