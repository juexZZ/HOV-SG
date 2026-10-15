#!/bin/bash
set -euo pipefail
DATASET_NAME="${1:-${DATASET_NAME:-pantry2}}"
source "$(dirname "${BASH_SOURCE[0]}")/application/dejaview_env.sh"
if [[ "$DATA_DIR" != "$DATASET_ROOT/$DATASET_NAME" || "$SCENE_DIR" != "$SAVE_ROOT/scannet/$DATASET_NAME" ]]; then
  echo "For mapping, set DATASET_ROOT, DATASET_NAME and SAVE_ROOT; DATA_DIR/SCENE_DIR must match them" >&2
  exit 2
fi
if [[ -e "$SCENE_DIR" ]]; then
  echo "Scene output exists: $SCENE_DIR; choose a new RUN_ID or SAVE_ROOT" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" application/record_run.py   --manifest "$SCENE_DIR/run_manifest.json" --   "$PYTHON" application/create_graph.py main.dataset=scannet main.split=""   main.scene_id="$DATASET_NAME" main.dataset_path="$DATASET_ROOT"   main.save_path="$SAVE_ROOT" models.clip.checkpoint="$CLIP_CKPT"   models.sam.checkpoint="$SAM_CKPT" pipeline.skip_frames="${SKIP_FRAMES:-10}"   hydra.run.dir="$ROOT/outputs/$RUN_ID/$DATASET_NAME"
