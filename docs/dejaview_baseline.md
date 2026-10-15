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
