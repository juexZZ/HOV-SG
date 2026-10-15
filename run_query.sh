#!/bin/bash

set -e

DATASET_NAME="pantry2"
SCENE_DIR="/local_data/jz4725/HOV-SG/data/scene_graphs/scannet/${DATASET_NAME}"
DATA_DIR="/local_data/jz4725/Depth-Anything-3/metric_depth_results/${DATASET_NAME}"
QUERY="${1:-office chair}"
TOP_K=3
GPU_ID=7

CUDA_VISIBLE_DEVICES="$GPU_ID" python application/query_latest_frame.py \
  --scene_dir "$SCENE_DIR" \
  --data_dir "$DATA_DIR" \
  --query "$QUERY" \
  --top_k "$TOP_K" \
