# Shared paths; historical outputs remain outside the current/ subtree.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${DATASET_ROOT:?Set DATASET_ROOT to the ScanNet-style export parent}"
PYTHON="${PYTHON:-python}"
DATASET_NAME="${DATASET_NAME:-pantry2}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-manual}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
SAVE_ROOT="${SAVE_ROOT:-$ROOT/data/scene_graphs/current/$RUN_ID}"
SCENE_DIR="${SCENE_DIR:-$SAVE_ROOT/scannet/$DATASET_NAME}"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/$DATASET_NAME}"
CLIP_CKPT="${CLIP_CKPT:-$ROOT/checkpoints/laion2b_s32b_b79k.bin}"
SAM_CKPT="${SAM_CKPT:-$ROOT/checkpoints/sam_vit_h_4b8939.pth}"
