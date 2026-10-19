# play_piano public tactile key-force audit (2026-10-17)

## Provenance and boundary

- Data mirror: `/mnt/data/ljz/opentacbench/datasets/OpenTacBench-tactile-memory`, commit `346933c248e90fc730311d5b5a8463d80eb316c9`.
- The episode configuration records the collector-side source directory as `/root/gpufree-data/OpenTacBench/data/tactile_memory/play_piano/dataset/episodes/<seed>`; that source directory is not mounted on this host. The ModelScope mirror is the available copy.
- OpenTacBench code remained at `tactile-memory-tasks`, commit `e72773be2588949eb580f8b5d584157929d34a22`, without edits. CaP-X `cap-x` is on `feat/opentac-play-piano`, commit `f5fd5041cb415922315354f2761778ef9b406261`, with pre-existing worktree changes preserved.
- The prior untracked `ViTaForge/tools/calibrate_tension_strap_force_labels.py` is absent here. The offline audit/trainer is `scripts/opentac/train_play_piano_public_key_force.py` in CaP-X.

## Data audit

There are 24 episode directories (seed 2000 and 2100–2122), all with native `result: success`, native `tool_grasped: true`, and successful marker-reference resets on both sides. All 24 collection configs request only `observations.tactile: [rgb_marker]`. Every sequence has six notes and `hold_seconds: 0.12`; there are 86 light and 58 heavy note requests, with no varied hold duration.

The 24 `piano_steps/*.jsonl` files have 62,218 rows, consecutive physics steps at 120 Hz. The metadata contain 19,263 `piano_pressure_trace` rows at 60 Hz within pressing intervals. These report the force on the current key, but the trace is recorded only in the expert's press loop. The HDF5 `piano/actual_force_N` is the maximum among 88 keys, not an unconditional label for the specified key. It is a valid target-key label only when the recorded actual MIDI matches the specified MIDI, or a zero label when the maximum of all keys is zero. All 24 successful episodes have zero rows marked as wrong-key contact. Pressure-trace force ranges from 0 to 2.944 N; no positive sample exceeds the native 6 N overload limit. The target 1.5 N and 3.0 N values are requests, never measured labels.

Six HDF5 files were physically checked out and opened (2000, 2100–2103, 2122); the other 18 new HDF5 files remain LFS pointers on this host. These six have 60 Hz saved frames and exactly aligned `phase/sim_step == piano/physics_step` labels. For seed 2100, all 824 pressure-trace rows lie one 120 Hz physics step from the saved HDF5 frames; the others checked are exact. The six files contain only `tactile/left_tactile/rgb_marker` and `tactile/right_tactile/rgb_marker`; all four numeric datasets required for the committed 26D feature contract (`depth` and `marker` for each side) are absent. The newest sample, seed 2122, has the same schema. Every JPEG in both sensors in all six checked files decoded at 240×320, with no failures.

Where the pressure trace and HDF5 frame have the same physics step, their measured force values agree exactly in these six checked episodes. Seed 2100 must not be joined to a neighbouring frame without acknowledging the offset: nearest-frame force differs by 0.027 N MAE and up to 2.773 N at a transient. The offline trainer uses exact per-frame labels or an exactly observed specified-key maximum and never substitutes the requested force.

The native results and finger-joint observations support stable expert tool grasp: the two finger joints were about 0.0086 m each during key contact in all six opened episodes. The previous CaP-X Franka smoke reached approximately 0.00856 m in its gripper trace, but it failed with `note_too_short` and has no co-sampled numeric depth/marker or target-key force. One similar finger-width number does not establish that the complete grasp/tactile distribution transfers to CaP-X.

The existing CaP-X adapter has a plausible public runtime path: `UniVTACLowLevelEnv.refresh_native_observation(..., tactile_data_types=["depth", "marker"])` can request these sensor outputs, and `OpenTacApi._read_public_tactile_observations()` already uses that mechanism for the strap model. This is code inspection, not a verified piano force calibration or a claim that the saved expert RGB can be converted back to the numeric 26D features.

## Result and decision

No valid 26D depth/marker training or episode-held-out estimate can be computed from this snapshot. Depth-only, marker-only, 26D, temporal-feature MAE/RMSE/P95/bias, short-vs-long results, contact/hold/release reliability, overforce miss rate, and no-contact false-positive rate are **unavailable**, not zero. The force-only alignment plots deliberately leave the missing tactile panels blank. No piano sidecar or runtime estimator was produced, and the piano task, physics, expert, scorer, termination, and YAML were not changed.

To proceed without changing task behavior, obtain a collection-side export with both sensors' numeric depth frames and `(2,64,2)` initial/current marker coordinates at saved observation steps, the physics-step clock, and the specified-key top contact force through approach, hold, and release. Collect at least two hold durations, some safe varied grasp postures including CaP-X grasps, no-contact frames, and examples approaching the overload boundary; the current success-only set contains no overload positives. Preserve whole-episode seed splits. The private force labels must stay outside any runtime sidecar or LLM context.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/mpl-opentac-force \
  /mnt/data/ljz/opentacbench/conda-envs/UniVTAC/bin/python \
  scripts/opentac/train_play_piano_public_key_force.py \
  --output-dir /mnt/data/ljz/opentacbench/force-audit/play_piano_346933c_20261017

# With this snapshot, --train exits 2 with a missing-depth/marker diagnosis:
MPLCONFIGDIR=/tmp/mpl-opentac-force \
  /mnt/data/ljz/opentacbench/conda-envs/UniVTAC/bin/python \
  scripts/opentac/train_play_piano_public_key_force.py \
  --output-dir /mnt/data/ljz/opentacbench/force-audit/play_piano_346933c_20261017 \
  --train --far-plane-mm 34.0
```

Machine-readable audit and private plots are in `/mnt/data/ljz/opentacbench/force-audit/play_piano_346933c_20261017/`. The 34 mm far plane comes from the piano GelSight camera clipping range in `envs/sensors/tactile.py`; it is not imported from the strap model. The trainer's successful branch was exercised only with synthetic temporary data to check episode splitting and model-variant execution; those synthetic metrics are not evidence about the real task.
