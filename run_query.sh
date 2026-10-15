#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/application/dejaview_env.sh"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" application/query_latest_frame.py   --scene_dir "$SCENE_DIR" --data_dir "$DATA_DIR" --clip_ckpt "$CLIP_CKPT"   --query "${1:-office chair}" --top_k "${TOP_K:-3}"
