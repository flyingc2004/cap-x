#!/usr/bin/env bash
# One seed defaults to an expert/probe pilot pair; expand END_SEED after auditing.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ljz/opentacbench/shadow_digit_force_v3/matched_60hz_v1}"
START_SEED="${START_SEED:-3200}"
END_SEED="${END_SEED:-${START_SEED}}"
PROBE_PROFILE="${PROBE_PROFILE:-varied}"
PER_EPISODE_TIMEOUT_SECONDS="${PER_EPISODE_TIMEOUT_SECONDS:-1200}"
if [[ ! "${OUTPUT_ROOT}" = /* || ! "${START_SEED}" =~ ^[0-9]+$ ||
      ! "${END_SEED}" =~ ^[0-9]+$ || ! "${PER_EPISODE_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] ||
   ((START_SEED < 3200 || END_SEED > 3299 || START_SEED > END_SEED || PER_EPISODE_TIMEOUT_SECONDS < 60)); then
  echo 'Use an absolute output path, seeds 3200..3299 and a timeout >= 60 seconds' >&2
  exit 2
fi
case "${PROBE_PROFILE}" in runtime|higher|lower|slower|varied) ;; *) exit 2 ;; esac
if [[ ! "${GPU:-${CUDA_VISIBLE_DEVICES:-}}" =~ ^[0-9]+$ ]]; then
  echo 'Set one numeric GPU explicitly; check that it is free before collecting' >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/logs"
shopt -s nullglob
for ((seed=START_SEED; seed<=END_SEED; seed++)); do
  for kind in expert probe; do
    episodes=("${OUTPUT_ROOT}/episodes/${seed}_${kind}_"*)
    if ((${#episodes[@]})); then
      if ((${#episodes[@]} != 1)) || ! python3 - "${episodes[0]}" "${seed}" "${kind}" "${PROBE_PROFILE}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
record = json.loads((path / 'result.json').read_text(encoding='utf-8'))
meta = record['metadata']
assert record['status'] in ('complete', 'probe_finished')
assert meta['seed'] == int(sys.argv[2]) and meta['controller'] == sys.argv[3]
assert meta['action_repeat'] == 4 and meta['public_control_frequency_hz'] == 60
assert meta['sequence_source'] == 'balanced_four_groups_8note_v1'
assert meta['record_video'] is False
if sys.argv[3] == 'probe':
    assert meta['probe_plan']['profile'] == sys.argv[4]
audit = json.loads((path / 'audit.json').read_text(encoding='utf-8'))
assert audit['exact_step_alignment'] and audit['public_frames'] > 1
PY
      then
        echo "Existing seed ${seed}/${kind} is not a completed matching episode; inspect it" >&2
        exit 1
      fi
      echo "seed ${seed}/${kind}: existing data preserved"
      continue
    fi
    log="${OUTPUT_ROOT}/logs/seed_${seed}_${kind}.log"
    if [[ -e "${log}" ]]; then
      echo "Refusing to overwrite log ${log}" >&2
      exit 1
    fi
    args=(--episode "${kind}" --expert-seed "${seed}" --probe-seed "${seed}"
          --balanced-eight --action-repeat 4 --no-video)
    if [[ "${kind}" == probe ]]; then
      args+=(--matched-probe --probe-profile "${PROBE_PROFILE}")
    fi
    echo "seed ${seed}/${kind}: collecting serially; log=${log}"
    set +e
    OUTPUT_ROOT="${OUTPUT_ROOT}" timeout --signal=TERM --kill-after=30s \
      "${PER_EPISODE_TIMEOUT_SECONDS}s" bash "${SCRIPT_DIR}/run_shadow_digit_pilot.sh" \
      "${args[@]}" >"${log}" 2>&1
    code=$?
    set -e
    echo "seed ${seed}/${kind}: exit_code=${code}"
    if ((code != 0)); then exit "${code}"; fi
  done
done
echo "Finished ${START_SEED}..${END_SEED}; audit coverage before fitting a NEW model."
