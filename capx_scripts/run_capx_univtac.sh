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

CONFIG_PATH="${CAPX_CONFIG_PATH:-env_configs/univtac/grasp_classify_tactile.yaml}"
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
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME}"
fi

python capx/envs/launch_univtac.py \
  --config-path "${CONFIG_FILE}" \
  --total-trials "${TRIALS}" \
  --num-workers "${CAPX_WORKERS:-1}" \
  --record-video "${CAPX_RECORD_VIDEO:-True}" \
  --output-dir "${CAPX_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/univtac_grasp_classify_tactile}" \
  --model "${CAPX_MODEL:-gpt-4o}" \
  --server-url "${CAPX_SERVER_URL:-http://127.0.0.1:8110/chat/completions}" \
  --temperature "${CAPX_TEMPERATURE:-1.0}"
