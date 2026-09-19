# VitaForge Tactile Suite

This branch adds a small CaP-X adapter extension plus configurations for three
VitaForge tasks. The VitaForge task, physics, and scripted expert are not
modified.

| Task | Purpose | Easy route | Hard route |
| --- | --- | --- | --- |
| `place_cube_on_colored_area` | API, RGB-D, stable grasp-and-place baseline | public reset anchors | SAM/RGB-D |
| `roughness_regrasp` | primary tactile evidence and selection task | public reset anchors | fixed left/right coarse regions; dark blocks remain visually uninformative |
| `can_empty_select` | longer weight-sensitive probe, decision, and basket placement | public reset anchors | SAM/RGB-D for localization |

All six configurations expose only `FrankaControlApi` and
`UniVTACTactileApi`. Actor observations, private metadata, reward/success
fields, and persistent code memory are disabled. Easy configurations are
engineering checks and are not formal non-privileged results.

## Runtime Protocol

`roughness_regrasp` and `can_empty_select` use agent-owned trial memory. The
generated code gets one protocol from `get_tactile_measurement_protocol()` and
must reuse it for every object:

1. Adaptive close and bilateral-contact validation.
2. Static tactile capture.
3. Short lift/hold dynamic capture.
4. Lower, release, and return to table clearance.

The code writes evidence, hypothesis, and decision records through
`write_trial_memory()`. It computes the selection itself from bilateral depth,
marker displacement/coherence, slip, and dynamic changes. The adapter does not
produce a tactile profile, score, or selected object. A final transport always
uses a fresh stable grasp, low lift, horizontal move above the target, descent,
and release.

## Official Expert Gate

Run the official expert before CaP-X. Replace `<GPU>` with an available GPU.

```bash
cd /mnt/sdc/ljz/ViTaForge
python scripts/collect_data.py place_cube_on_colored_area gelsight \
  --episode_num 20 --start_seed 0 --max_seed 19 --gpu <GPU>
python scripts/collect_data.py roughness_regrasp gelsight \
  --episode_num 20 --start_seed 0 --max_seed 19 --gpu <GPU>
python scripts/collect_data.py can_empty_select gelsight \
  --episode_num 20 --start_seed 0 --max_seed 19 --gpu <GPU>
```

Each task must reach at least 80% expert success before CaP-X runs. For the two
selection tasks, the expert stops as soon as it finds the target. Consequently,
a 20-seed smoke validates task health but normally does not yield 20 paired
two-object probes. Collect roughly 160 seeds when the offline calibration needs
80 valid paired episodes, then report rejected/incomplete probes separately.

## CaP-X Smoke

```bash
cd /mnt/sdc/ljz/t-cap

CONFIG_PATH=env_configs/vitaforge/place_cube_on_colored_area_easy_gt.yaml \
GPU=<GPU> MODE=smoke TRIALS=1 REPEATS=1 RECORD_VIDEO=True ./run

CONFIG_PATH=env_configs/vitaforge/place_cube_on_colored_area_hard_sam.yaml \
GPU=<GPU> MODE=smoke TRIALS=1 REPEATS=1 RECORD_VIDEO=True ./run

CONFIG_PATH=env_configs/vitaforge/roughness_regrasp_easy_gt.yaml \
GPU=<GPU> MODE=smoke TRIALS=1 REPEATS=1 RECORD_VIDEO=True ./run

CONFIG_PATH=env_configs/vitaforge/roughness_regrasp_hard_touch.yaml \
GPU=<GPU> MODE=smoke TRIALS=1 REPEATS=1 RECORD_VIDEO=True ./run

CONFIG_PATH=env_configs/vitaforge/can_empty_select_easy_gt.yaml \
GPU=<GPU> MODE=smoke TRIALS=1 REPEATS=1 RECORD_VIDEO=True ./run

CONFIG_PATH=env_configs/vitaforge/can_empty_select_hard_sam.yaml \
GPU=<GPU> MODE=smoke TRIALS=1 REPEATS=1 RECORD_VIDEO=True ./run
```

Hard SAM configurations cause `./run` to start the configured SAM3 and
Contact-GraspNet services. They require the local model checkpoint cache. Use
the Easy smoke first to isolate motion/tactile failures from perception-service
failures.

## Evaluation Order

1. `place_cube_on_colored_area`: Easy then Hard. Verify RGB-D artifacts, video,
   task success, and release handling.
2. `roughness_regrasp`: Easy then Hard. Track selected-side accuracy, valid
   bilateral probe rate, memory completeness, margin, slip/drop count, timeout,
   and final placement success.
3. `can_empty_select`: Easy then Hard with the same metrics, plus empty-can
   selection accuracy and basket containment.

Do not enter a memory comparison for a task until its offline held-out probe
calibration reaches 80% macro accuracy. That calibration is a signal
separability gate; it is separate from CaP-X planning performance.
