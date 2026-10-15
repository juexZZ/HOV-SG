#!/bin/bash
set -euo pipefail
DATASET_NAME="${1:-${DATASET_NAME:-pantry2}}"
source "$(dirname "${BASH_SOURCE[0]}")/application/dejaview_env.sh"
: "${ANNOTATION_ROOT:?Set ANNOTATION_ROOT to the annotation scene parent}"
: "${OBJECT_LIST_ROOT:?Set OBJECT_LIST_ROOT to DejaView/eval/object_test_lists}"
CHAPTER_ID="${CHAPTER_ID:-1}"
DATASET_DIR="${DATASET_DIR:-$ANNOTATION_ROOT/$DATASET_NAME}"
case "$DATASET_NAME" in
  pantry*) OBJECT_LIST_TXT="${OBJECT_LIST_TXT:-$OBJECT_LIST_ROOT/pantry.txt}" ;;
  *) OBJECT_LIST_TXT="${OBJECT_LIST_TXT:-$OBJECT_LIST_ROOT/$DATASET_NAME.txt}" ;;
esac
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT/eval_results/runs/$RUN_ID/hovsg_${DATASET_NAME}_ch${CHAPTER_ID}.json}"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" application/eval_object_retrieval.py   --scene_dir "$SCENE_DIR" --data_dir "$DATA_DIR" --clip_ckpt "$CLIP_CKPT"   --dataset_name "$DATASET_NAME" --dataset_dir "$DATASET_DIR"   --object_list_txt "$OBJECT_LIST_TXT" --chapter_id "$CHAPTER_ID"   --output_path "$OUTPUT_PATH" --contributor_top_t "${CONTRIBUTOR_TOP_T:-1}"   --similarity_threshold -1.0 --ignore_similarity_threshold
