# DejaView HOV-SG baseline

Upstream base: `d6e65a53c8be6faec3f01f00d1644d967f89e605`.

This adaptation fixes root-path joining and adds an Agg fallback for headless
Matplotlib imports. The ScanNet entry point builds object features and point
clouds; upstream `application/create_graph.py` skips the hierarchical graph
for ScanNet. Evaluation uses `mask_feats.pt` and `objects/pcd_<id>.ply`, not
floor/room graph reasoning.

`application/split_chapter_data.py` materializes per-traversal or cumulative
prefix datasets without renumbering frames. `run.sh` builds maps,
`run_evaluator.sh` evaluates, and `run_query.sh` prints latest observations.
Install the upstream dependencies and checkpoints in a separate HOV-SG
environment. Text features use upstream multiple-template averaging.

After selecting one object, frames are searched in reverse numeric order
using 3D projection and depth consistency (defaults: voxel 0.05m, at least 20
visible points, depth tolerance 0.5m). These are inferred observing frames,
not recorded fusion contributors. Evaluation scans all supplied frames,
including frames omitted by mapping's skip_frames setting.

## Known legacy limitations

The upstream ScanNet loader sorts filenames lexicographically. With unpadded
integer filenames and skip_frames=10 this changes both the sampled frames and
sequential fusion order. On pantry2 the original loader selects frames 0 and
18 instead of 0 and 10. Historical results must not be attributed to a corrected
numeric-order implementation; rebuilding maps is required after that fix.
Runners contain machine paths. Missing-frame split mappings, no-depth query
counts, output paths without directories, and CPU device handling need fixes.

## Data and evaluation contract

Inputs are a DA3 ScanNet-style export: `color/<integer>.jpg`,
`depth/<integer>.png` (millimeters), camera-to-world `pose/<integer>.txt`,
`intrinsic/intrinsic_{color,depth}.txt`, and `filename_mapping.json` mapping
local integer frame IDs to original annotation filenames. The exporter is
`Depth-Anything-3/tools/run_metric_depth.py` at
`01accac4c0b73c8d6d35226d86a1c43fc32ea9aa`; this is a separate dependency.
Annotations use `stream_annotations.json`; query lists live in DejaView's
`eval/object_test_lists/`. Dataset names, chapter and prefix must agree.

The evaluator chooses one object, then its latest top-t frames. `hit@1` uses
only the primary frame; TP/recall use any of the top-t frames. In-chapter rates
use evaluated queries present in the chapter as denominator. These scripts
implement the original current-chapter protocol, not DejaView's newer
current+previous metrics. Archived runs use t=1 and threshold=-1 with hit
threshold bypass. Their ghost/precision fields are not calibrated object
presence detection measurements.

## Historical experiment snapshot

`dejaview-baseline-legacy-v1` preserves the imported experimental implementation
and JSON outputs. JSON bytes are unchanged. `eval_results/legacy_manifest.json`
records checksums and metadata. Historical configs are supporting evidence,
not proof of which invocation generated each artifact. Missing provenance is
explicitly unknown; archiving did not rerun inference. Absolute paths in these
historical artifacts document the original machine.

Model weights, maps, images, detections, compiled dependencies and full run
logs are excluded. Keep upstream LICENSE and citation information.

## Corrected workflow (after the legacy tag)

Use a fresh `RUN_ID` for each experiment. Runners resolve repository-relative
code and accept externally supplied absolute data paths. They refuse existing
map outputs and result files; the evaluator's explicit `--overwrite` option
is available for intentional JSON replacement. Current results go under
`eval_results/runs/` and are ignored. Existing legacy result JSON is unchanged.

Mapping records the exact command, source commit, tracked dirty state and
success/failure in a run manifest. Evaluation records its arguments, revision,
protocol-input SHA-256 hashes, and the map manifest if available. A missing
manifest means map provenance is unknown. These records do not certify the
contents of large geometry/checkpoint files: preserve their versions with
the experiment. Do not change a checkout while a run is active.

Set the paths below for your machine (absolute paths recommended):

```bash
export DATASET_ROOT=/path/to/Depth-Anything-3/metric_depth_results
export ANNOTATION_ROOT=/path/to/metacam/rebuttal
export OBJECT_LIST_ROOT=/path/to/DejaView/eval/object_test_lists
export RUN_ID=pantry1_numeric_v1
export GPU_ID=0
```

The loader and retriever pair color/depth/pose by exact numeric frame ID and
reject missing or duplicate IDs. Mapping now samples and fuses in chronological
numeric order. **Rebuild HOV-SG maps before evaluating this corrected workflow.**
The default output subtree is `data/scene_graphs/current/$RUN_ID`, separate
from historical scene maps.

```bash
./run.sh pantry1
CHAPTER_ID=1 ./run_evaluator.sh pantry1
DATASET_NAME=pantry1 ./run_query.sh remote
python application/split_chapter_data.py \
  --data_dir "$DATASET_ROOT/28_metfloor" \
  --annotation_file /path/to/28_metfloor/stream_annotations.json \
  --mode prefix --out_parent /path/to/new/splits --symlink
```

Set `SKIP_FRAMES` (default 10), `SAVE_ROOT`, `CLIP_CKPT`, `SAM_CKPT` and
`PYTHON` as needed. Query/evaluation additionally accept `SCENE_DIR`,
`DATA_DIR`, `OUTPUT_PATH`, `CONTRIBUTOR_TOP_T`; the query accepts `TOP_K`.
The evaluator CLI exposes visibility parameters. This remains the original
multiple-template text retrieval and geometric visibility method. Text tokens
now follow the model device instead of forcing CUDA.

Splits retain only materialized frame IDs in their mapping, preserve padded
source filenames, and reject an existing destination to prevent stale frames.
Single-traversal `--mode each` is supported. The query's `--no_depth_check`
counts only in-image points. Invalid evaluation chapters fail explicitly.

Validation: `python -m unittest discover -s tests -p test_dejaview_baseline.py -v`
checks the actual loader's sampling order, mismatched/duplicate frame IDs,
visibility, missing/padded split frames, single-traversal splitting, CPU text
features, output preservation and run provenance.

Both the shell syntax and new/modified Python syntax were checked. Tests run
in the existing baseline environments. No full GPU mapping, accuracy rerun,
or browser rendering validation was performed. The headless backend change
is a fallback at import time, not a guarantee for every GUI configuration.
