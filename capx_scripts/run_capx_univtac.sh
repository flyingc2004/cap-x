#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-/mnt/sdc/ljz/UniVTAC}"
CONDA_SH="${CONDA_SH:-/mnt/sdc/ljz/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-UniVTAC}"

MODE="${1:-smoke}"
case "${MODE}" in
  smoke) TRIALS="${CAPX_TRIALS:-1}" ;;
  quick) TRIALS="${CAPX_TRIALS:-10}" ;;
  *) echo "Usage: bash ${BASH_SOURCE[0]} {smoke|quick}" >&2; exit 2 ;;
esac

CONFIG_PATH="${CAPX_CONFIG_PATH:-env_configs/univtac/tactile_memory_match_response_selection_easy_gt.yaml}"
if [[ "${CONFIG_PATH}" = /* ]]; then
  CONFIG_FILE="${CONFIG_PATH}"
else
  CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[capx-univtac] ERROR: config not found: ${CONFIG_FILE}" >&2
  exit 2
fi

export CAPX_ENV_AUTO_ACTIVATE=0
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -n "${TACEX_RUNTIME_ROOT:-}" ]]; then
  for package in tacex tacex_uipc tacex_assets tacex_tasks; do
    source_dir="${TACEX_RUNTIME_ROOT}/source/${package}"
    if [[ ! -d "${source_dir}" ]]; then
      echo "[capx-univtac] ERROR: missing TacEx source: ${source_dir}" >&2
      exit 2
    fi
    export PYTHONPATH="${source_dir}:${PYTHONPATH}"
  done
fi
export HEADLESS="${HEADLESS:-1}"
export LIVESTREAM="${LIVESTREAM:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

cd "${PROJECT_ROOT}"

python - <<PY
from pathlib import Path
cfg = Path("${CONFIG_FILE}")
print(f"[capx-univtac] config={cfg}")
print(f"[capx-univtac] univtac_root=${UNIVTAC_ROOT}")
print(f"[capx-univtac] mode=${MODE}, trials=${TRIALS}")
PY

if [[ -f "${CONDA_SH}" ]]; then
  # Conda's binutils activation hook probes optional tool variables (for
  # example ADDR2LINE) before defining them.  This launcher otherwise uses
  # `set -u`, which makes that hook abort before CaP-X starts.  Relax nounset
  # only while Conda runs its activation/deactivation hooks, then restore it
  # for the actual trial command below.
  set +u
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME}"
  set -u
fi

python capx/envs/launch_univtac.py \
  --config-path "${CONFIG_FILE}" \
  --total-trials "${TRIALS}" \
  --num-workers "${CAPX_WORKERS:-1}" \
  --record-video "${CAPX_RECORD_VIDEO:-True}" \
  --output-dir "${CAPX_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/univtac_tactile_response_selection}" \
  --model "${CAPX_MODEL:-gpt-4o}" \
  --server-url "${CAPX_SERVER_URL:-http://127.0.0.1:8110/chat/completions}" \
  --temperature "${CAPX_TEMPERATURE:-1.0}"
