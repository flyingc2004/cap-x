#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ljz/opentacbench/shadow_digit_force_v1/balanced_50seed_v1}"
START_SEED="${START_SEED:-3100}"
END_SEED="${END_SEED:-3149}"
PER_SEED_TIMEOUT_SECONDS="${PER_SEED_TIMEOUT_SECONDS:-1200}"

if [[ -z "${GPU:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Set GPU or CUDA_VISIBLE_DEVICES explicitly' >&2
  exit 2
fi
if [[ "${OUTPUT_ROOT}" != /* || ! "${START_SEED}" =~ ^[0-9]+$ ||
      ! "${END_SEED}" =~ ^[0-9]+$ || ! "${PER_SEED_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo 'Use an absolute OUTPUT_ROOT and integer seed/time limits' >&2
  exit 2
fi
if ((START_SEED < 3100 || END_SEED > 3149 || START_SEED > END_SEED)); then
  echo 'Seed range must be within 3100..3149' >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/logs"
shopt -s nullglob

for ((seed=START_SEED; seed<=END_SEED; seed++)); do
  episodes=("${OUTPUT_ROOT}/episodes/${seed}_expert_"*)
  if ((${#episodes[@]})); then
    if ((${#episodes[@]} != 1)) || ! python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
assert record["status"] == "complete"
assert record["metadata"]["seed"] == int(sys.argv[2])
assert record["metadata"]["sequence_source"] == "balanced_four_groups_8note_v1"
assert record["metadata"]["record_video"] is False
' "${episodes[0]}/result.json" "${seed}"; then
      echo "Existing seed ${seed} is incomplete or belongs to another suite; inspect ${episodes[*]}" >&2
      exit 1
    fi
    echo "seed ${seed}: existing complete episode preserved"
    continue
  fi
  log="${OUTPUT_ROOT}/logs/seed_${seed}.log"
  if [[ -e "${log}" ]]; then
    echo "Refusing to overwrite existing log: ${log}" >&2
    exit 1
  fi
  echo "seed ${seed}: collecting (log ${log})"
  set +e
  OUTPUT_ROOT="${OUTPUT_ROOT}" timeout --signal=TERM --kill-after=30s \
    "${PER_SEED_TIMEOUT_SECONDS}s" "${SCRIPT_DIR}/run_shadow_digit_pilot.sh" \
    --episode expert --expert-seed "${seed}" --balanced-eight --no-video \
    >"${log}" 2>&1
  code=$?
  set -e
  echo "seed ${seed}: exit_code=${code}"
  if ((code != 0)); then
    echo "Stopped after seed ${seed}; inspect ${log} before resuming" >&2
    exit "${code}"
  fi
done

echo "Balanced 50-seed collection range ${START_SEED}..${END_SEED} finished."
if ((END_SEED == 3149)); then
  eval_log="${OUTPUT_ROOT}/logs/depth_holdout.log"
  if [[ -e "${eval_log}" ]]; then
    echo "Refusing to overwrite existing evaluation log: ${eval_log}" >&2
    exit 1
  fi
  eval_python="${CONDA_ENV_NAME:-/mnt/data/ljz/opentacbench/conda-envs/UniVTAC}/bin/python"
  "${eval_python}" "${SCRIPT_DIR}/evaluate_shadow_digit_depth_50seed.py" \
    --episodes-root "${OUTPUT_ROOT}/episodes" \
    --output "${OUTPUT_ROOT}/depth_holdout_metrics.json" >"${eval_log}" 2>&1
  echo "Depth holdout metrics: ${OUTPUT_ROOT}/depth_holdout_metrics.json"
fi
