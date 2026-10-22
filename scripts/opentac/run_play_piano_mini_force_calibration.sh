#!/usr/bin/env bash
set -Eeuo pipefail

# Resume only the rich Franka piano batch. No OpenTacBench checkout is changed.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PATH="/mnt/data/ljz/opentacbench/tools/git-lfs/usr/bin:${PATH}"
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=/mnt/data/ljz/opentacbench/tools/modelscope_askpass.sh
export GIT_LFS_PROGRESS=/mnt/data/ljz/opentacbench/logs/play_piano_mini_lfs_progress.log

source_repo=/mnt/data/ljz/opentacbench/datasets/OpenTacBench-tactile-memory
revision=3d1285f8519832067091bf5e037925beabb40a4b
dataset=/mnt/data/ljz/opentacbench/datasets/play_piano_mini_depth_marker_3d1285f
output=/mnt/data/ljz/opentacbench/force-audit/play_piano_mini_3d1285f_20261020
python=/mnt/data/ljz/opentacbench/conda-envs/UniVTAC/bin/python
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output" /mnt/data/ljz/opentacbench/cache/mpl-piano-mini
status="$output/status.txt"
trap 'code=$?; printf "failed exit=%s stage=%s finished=%s\n" "$code" "${stage:-unknown}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"' ERR

stage=download
printf 'running stage=%s started=%s\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
git -C "$source_repo" -c lfs.concurrenttransfers=12 lfs fetch origin "$revision" \
  --include='play_piano/mini-depth-marker/episodes/**' \
  --exclude='play_piano/mini-depth-marker/episodes/*/video/*'

stage=materialize
printf 'running stage=%s started=%s\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
"$python" "$script_dir/materialize_play_piano_mini_depth_marker.py" \
  --source-repo "$source_repo" --revision "$revision" --output-dir "$dataset"

stage=train
printf 'running stage=%s started=%s\n' "$stage" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
MPLCONFIGDIR=/mnt/data/ljz/opentacbench/cache/mpl-piano-mini \
  "$python" "$script_dir/train_play_piano_public_key_force.py" \
  --episodes-root "$dataset/episodes" --output-dir "$output" \
  --train --far-plane-mm 34.0

printf 'complete commit=%s finished=%s\n' \
  "$revision" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"
