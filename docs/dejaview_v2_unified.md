# v2 SAM2/SigLIP cumulative object-map variant

The experiment recipe and limitations are in
`/root/dejaview_experiments/baselines/hovsg/DESIGN.md`; the shared sealed protocol
is `/root/dejaview_experiments/coordination/unified-graphs-20261026-r01/PROTOCOL.md`.

The upstream ScanNet baseline in this branch evaluates the object map and skips
floor/room/navigation construction. This adapter preserves that scope and the
native sequential overlap merger and object-feature DBSCAN thresholds. It is a
user-requested frontend/temporal variant, **not an untouched upstream result**.

The shared cached SAM2 objects have overlapping masks and SigLIP1152 vectors.
Their original/padded boxes are both xywh. The former is the mask box, the latter
is the embedding crop context. The cache lacks predicted-IoU; do not substitute
stability score or refilter accepted objects. Neither cached vectors nor source
files are changed. Masks use nearest-neighbor resizing, and the native sum/L2/
half pixel-feature operation uses a compact lossless table of overlap patterns.
This replaces upstream OpenCLIP global/local feature mixing with cached SigLIP
object embeddings; the native crop mixing weight is consequently not applied.

`application/v2_scene.py` loads the complete geometry manifest once and iterates
only the sealed scene-global stereo pairs with pair_index%5==0. Actual image IDs
are retained. Each selected frame is ingested once; native sequential mask state
is checkpointed before evaluation. Boundary terminal merging runs on a deep
copy and is checked not to alter the stream. T2 and T3 resume the same stream.

Upstream masks snap to one fixed point cloud built from the entire input, which
would include future frames at T1/T2. The temporal adapter uses local per-frame
voxel/DBSCAN support for the initial projection and retains these compact local
point-feature observations. At a boundary, global support and final object
features are re-reduced from already-associated observations on the snapshot.
This unavoidable global assembly repeats; old image reading, mask projection,
perception, and sequential object merging do not. It is not numerically equivalent
to future-dependent batch support. Native indexed += has duplicate-ID assignment
semantics; the local adapter uses explicit last row-major pixel per support point,
eliminating thread-dependent indexed-write ambiguity.

The provenance extension to merge_3d_masks only optionally returns its existing
connected components. It makes no changes to adjacency, thresholds, pointcloud
merging or default return behavior. Retrieval ranks native object features using
the shared graph SigLIP text cache, then calls the original HovsgRetriever's
get_top_t_latest_frames visibility readout. It scans ALL original frames up to the
current run, including frames omitted from mapping, but cannot read a later run.
The exact native defaults remain voxel=.05 m, min_visible=20 points, depth scale
1000, depth tolerance=.5 m. The latest sampled actual fusion contributor is only
a diagnostic field and time-integrity check, not the primary prediction. This
corrects the original shared protocol's mistaken readout assumption, per the
parent's explicit 2026-10-26 instruction. GT is read only by the separate offline
v2_score script.

Standalone usage (source the independent runtime first):

```bash
source /root/dejaview_experiments/baselines/hovsg/runtime.sh
python -m unittest discover -s tests -p 'test_v2_unified.py'
python application/v2_scene.py --scene 2metro8floor --smoke
python application/v2_scene.py --scene 2metro8floor
python application/v2_score.py
```

Default native numeric parameters: voxel=.02m, initial overlap=.75, IoU=.05,
point-cloud DBSCAN eps=.01/min_points=100 for support and eps=.1/min_points=10
inside the native merger, feature DBSCAN cosine eps=.01/min_points=100,
object feature sampling voxel=.04m. The native merger does **not** downsample
its merged cloud despite a voxel_size argument; this behavior is preserved.
