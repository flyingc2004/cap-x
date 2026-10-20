#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAPX_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
UNIVTAC_ROOT="${UNIVTAC_ROOT:-$(cd -- "${CAPX_DIR}/../../OpenTacBench" && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ljz/opentacbench/shadow_digit_force_v1}"
CONDA_SH="${CONDA_SH:-/mnt/a/ljz/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/mnt/data/ljz/opentacbench/conda-envs/UniVTAC}"

# The collector runs from OpenTacBench, so resolve a caller-relative phrase
# before changing directories. Preserve every other argument verbatim.
PILOT_ARGS=("$@")
for ((i=0; i<${#PILOT_ARGS[@]}; i++)); do
  if [[ "${PILOT_ARGS[i]}" == "--sequence-config" ]]; then
    if ((i + 1 >= ${#PILOT_ARGS[@]})); then
      echo '--sequence-config requires a path' >&2
      exit 2
    fi
    if [[ "${PILOT_ARGS[i+1]}" != /* && "${PILOT_ARGS[i+1]}" != '~/'* ]]; then
      PILOT_ARGS[i+1]="${PWD}/${PILOT_ARGS[i+1]}"
    fi
    i=$((i + 1))
  elif [[ "${PILOT_ARGS[i]}" == --sequence-config=* ]]; then
    config_path="${PILOT_ARGS[i]#--sequence-config=}"
    if [[ "${config_path}" != /* && "${config_path}" != '~/'* ]]; then
      PILOT_ARGS[i]="--sequence-config=${PWD}/${config_path}"
    fi
  fi
done

if [[ -z "${GPU:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Set GPU or CUDA_VISIBLE_DEVICES explicitly before running the pilot' >&2
  exit 2
fi
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="${GPU}"; fi
export UNIVTAC_ROOT PYTHONUNBUFFERED=1 HEADLESS=1 LIVESTREAM=0 OMNI_KIT_ACCEPT_EULA=YES
export PYOPENGL_PLATFORM=egl
export HOME="${UNIVTAC_HOME:-/mnt/data/ljz/opentacbench/kit-home}"
export XDG_CACHE_HOME="${UNIVTAC_XDG_CACHE_HOME:-/mnt/data/ljz/opentacbench/kit-cache}"
export XDG_DATA_HOME="${UNIVTAC_XDG_DATA_HOME:-/mnt/data/ljz/opentacbench/kit-data}"
export XDG_CONFIG_HOME="${UNIVTAC_XDG_CONFIG_HOME:-/mnt/data/ljz/opentacbench/kit-config}"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib-cache"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_DATA_HOME}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
TACEX_RUNTIME_ROOT="${TACEX_RUNTIME_ROOT:-${UNIVTAC_ROOT}/third_party/TacEx}"
export PYTHONPATH="${CAPX_DIR}:${UNIVTAC_ROOT}:${TACEX_RUNTIME_ROOT}/source/tacex:${TACEX_RUNTIME_ROOT}/source/tacex_uipc:${TACEX_RUNTIME_ROOT}/source/tacex_assets:${TACEX_RUNTIME_ROOT}/source/tacex_tasks:${PYTHONPATH:-}"
set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
set -u
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
cd "${UNIVTAC_ROOT}"
exec python "${SCRIPT_DIR}/collect_shadow_digit_pilot.py" \
  --opentac-root "${UNIVTAC_ROOT}" --capx-root "${CAPX_DIR}" \
  --output-root "${OUTPUT_ROOT}" "${PILOT_ARGS[@]}"
