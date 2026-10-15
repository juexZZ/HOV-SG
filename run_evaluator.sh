#!/bin/bash

set -e

DATASET_NAME="pantry2"
MODEL_TYPE="hovsg" # hovsg
CHAPTER_ID=1
SCENE_DIR="/local_data/jz4725/HOV-SG/data/scene_graphs/scannet/${DATASET_NAME}"
DATA_DIR="/local_data/jz4725/Depth-Anything-3/metric_depth_results/${DATASET_NAME}"
DATASET_DIR="/local_data/jz4725/metacam/rebuttal/${DATASET_NAME}" # for annotation
OBJECT_LIST_TXT="/local_data/jz4725/DejaView/eval/object_test_lists/pantry.txt"
OUTPUT_PATH="/local_data/jz4725/HOV-SG/eval_results/${MODEL_TYPE}_${DATASET_NAME}_ch${CHAPTER_ID}.json"
SIMILARITY_THRESHOLD=-1.0
GPU_ID=7

CUDA_VISIBLE_DEVICES="$GPU_ID" python application/eval_object_retrieval.py \
  --scene_dir "$SCENE_DIR" \
  --data_dir "$DATA_DIR" \
  --dataset_name "$DATASET_NAME" \
  --dataset_dir "$DATASET_DIR" \
  --object_list_txt "$OBJECT_LIST_TXT" \
  --chapter_id "$CHAPTER_ID" \
  --output_path "$OUTPUT_PATH" \
  --similarity_threshold "$SIMILARITY_THRESHOLD" \
  --ignore_similarity_threshold

# #! rebuttal
# DATASET_NAME="pantry2"
# MODEL_TYPE="hovsg" # hovsg
# CHAPTER_ID=1
# SCENE_DIR="/local_data/jz4725/HOV-SG/data/scene_graphs/scannet/${DATASET_NAME}"
# DATA_DIR="/local_data/jz4725/Depth-Anything-3/metric_depth_results/${DATASET_NAME}"
# DATASET_DIR="/local_data/jz4725/metacam/rebuttal/${DATASET_NAME}" # for annotation
# OBJECT_LIST_TXT="/local_data/jz4725/DejaView/eval/object_test_lists/pantry.txt"
# OUTPUT_PATH="/local_data/jz4725/HOV-SG/eval_results/${MODEL_TYPE}_${DATASET_NAME}_ch${CHAPTER_ID}.json"
# SIMILARITY_THRESHOLD=-1.0
# GPU_ID=7

# CUDA_VISIBLE_DEVICES="$GPU_ID" python application/eval_object_retrieval.py \
#   --scene_dir "$SCENE_DIR" \
#   --data_dir "$DATA_DIR" \
#   --dataset_name "$DATASET_NAME" \
#   --dataset_dir "$DATASET_DIR" \
#   --object_list_txt "$OBJECT_LIST_TXT" \
#   --chapter_id "$CHAPTER_ID" \
#   --output_path "$OUTPUT_PATH" \
#   --similarity_threshold "$SIMILARITY_THRESHOLD" \
#   --ignore_similarity_threshold
