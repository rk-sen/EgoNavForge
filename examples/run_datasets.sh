#!/usr/bin/env bash
# Same workflow, two dataset bindings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/mnt/luozhaoyi/envs/pi3_moge/bin/python}"
cd "$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/luozhaoyi/cache/huggingface}"

"$PY" -m ego_nav_forge doctor
"$PY" -m ego_nav_forge run -c configs/datasets/umi_single_video_imu.yaml
"$PY" -m ego_nav_forge run -c configs/datasets/threeeyes_three_video_imu.yaml

# Scene twin (optional): needs zhesuan_API_KEY / MOONSHOT_API_KEY for full VLM path.
# Smoke rebuilds Blender/USD from seed_project without calling the API.
# "$PY" -m ego_nav_forge run -c configs/datasets/seeker_scene_twin_check.yaml --max-eps 1
# "$PY" -m ego_nav_forge run -c configs/datasets/seeker_scene_twin_smoke.yaml --max-eps 1
# "$PY" -m ego_nav_forge run -c configs/datasets/seeker_scene_twin.yaml --max-eps 1
echo "OK → $ROOT/work_dirs/"
