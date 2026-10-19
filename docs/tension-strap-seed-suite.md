# Frozen Tension-Strap Seed Suite

The CaP-X-owned [manifest](../env_configs/univtac/tension_strap_seed_suite_1_30.v1.json)
maps physics seed 0 to a smoke instruction (10 N for 1.3 s, then 15 N for
2.0 s). Seeds 1-30 define 30 distinct evaluation instructions: 15 increasing
and 15 decreasing, with force targets within 10-18 N. Keep the manifest
unchanged between paired expert and CaP-X runs.

## Evaluation Instructions

| Seed | First hold | Second hold |
| ---: | ---: | ---: |
| 1 | 12.1 N, 1.6 s | 15.3 N, 1.4 s |
| 2 | 17.2 N, 1.5 s | 15.2 N, 1.6 s |
| 3 | 12.4 N, 1.9 s | 15.0 N, 1.9 s |
| 4 | 16.6 N, 1.8 s | 10.4 N, 1.7 s |
| 5 | 13.6 N, 1.5 s | 17.1 N, 1.9 s |
| 6 | 16.1 N, 1.5 s | 13.9 N, 1.3 s |
| 7 | 13.7 N, 2.0 s | 17.8 N, 1.9 s |
| 8 | 14.6 N, 1.6 s | 11.5 N, 1.9 s |
| 9 | 10.9 N, 1.6 s | 14.9 N, 1.9 s |
| 10 | 17.3 N, 1.4 s | 12.3 N, 1.6 s |
| 11 | 10.1 N, 1.8 s | 12.6 N, 1.4 s |
| 12 | 17.5 N, 1.6 s | 13.4 N, 1.6 s |
| 13 | 11.9 N, 1.4 s | 16.1 N, 1.9 s |
| 14 | 16.1 N, 1.9 s | 11.8 N, 1.8 s |
| 15 | 10.3 N, 1.6 s | 14.5 N, 1.6 s |
| 16 | 17.3 N, 1.9 s | 11.4 N, 1.6 s |
| 17 | 10.6 N, 1.6 s | 17.5 N, 1.6 s |
| 18 | 17.9 N, 1.3 s | 12.9 N, 1.8 s |
| 19 | 10.6 N, 1.5 s | 14.2 N, 1.7 s |
| 20 | 14.9 N, 1.6 s | 12.8 N, 1.4 s |
| 21 | 14.2 N, 1.4 s | 17.2 N, 1.7 s |
| 22 | 17.7 N, 1.5 s | 14.0 N, 1.5 s |
| 23 | 15.1 N, 1.4 s | 17.9 N, 1.8 s |
| 24 | 17.8 N, 1.8 s | 12.5 N, 1.4 s |
| 25 | 11.3 N, 1.7 s | 15.2 N, 1.6 s |
| 26 | 14.6 N, 1.7 s | 11.8 N, 1.6 s |
| 27 | 13.5 N, 1.6 s | 15.6 N, 1.3 s |
| 28 | 15.1 N, 1.9 s | 11.7 N, 1.4 s |
| 29 | 10.3 N, 1.8 s | 16.0 N, 1.7 s |
| 30 | 14.7 N, 1.5 s | 12.7 N, 1.9 s |

OpenTacBench provides the task and its existing arbitrary-force interface;
this suite does not patch OpenTacBench code. Use its `tactile-memory-tasks`
branch in the existing checkout. The expert runner writes one temporary task
YAML per seed under the chosen output directory, then calls the stock
`collect_data.py` for each seed:

```bash
cd /mnt/sdc/ljz/t-cap/cap-x
/mnt/sdc/ljz/miniforge3/envs/UniVTAC/bin/python -m scripts.run_tension_strap_seed_suite_expert \
  --opentac-root /mnt/sdc/ljz/OpenTacBench_capx \
  --output-dir /mnt/sdc/ljz/t-cap/capx-runs/tension_suite_expert_r1 \
  --start-seed 1 --end-seed 30 --gpu 0
```

For an independent expert repeat, use a fresh `--output-dir`. The stock
collector records each seed's result in its own output directory.

CaP-X reads the same manifest at environment initialization and passes that
seed's force targets and hold durations to both the OpenTacBench task config
and the public OpenTac control contract. For seed 0 smoke, from the `t-cap`
root with Responses API credentials configured:

```bash
CONFIG_PATH=env_configs/univtac/tension_strap_stage_memory_control.yaml \
UNIVTAC_ROOT=/mnt/sdc/ljz/OpenTacBench_capx \
TACEX_RUNTIME_ROOT=/mnt/sdc/ljz/OpenTacBench_capx/third_party/TacEx \
CAPX_FORCE_SEQUENCE_MANIFEST=env_configs/univtac/tension_strap_seed_suite_1_30.v1.json \
CAPX_FORCE_TASK_SEED=0 GPU=0 MODE=smoke TRIALS=1 REPEATS=1 \
TRIAL_TIMEOUT_SECONDS=3600 PROCESS_TIMEOUT_SECONDS=4200 \
RUN_LABEL=tension_suite_seed0 ./run_responses
```

For evaluation, set `CAPX_FORCE_TASK_SEED` to one of 1-30 and use a unique
`RUN_LABEL` for each run. The current visual-assist YAML is not part of this
suite. Neither the manifest nor the public control contract supplies the
task's measured true tension to CaP-X.
