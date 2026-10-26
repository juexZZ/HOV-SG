"""Read-only, GT-free v2 input contract and atomic experiment records."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SHARED = Path('/root/dejaview_experiments/runs/full-v2-20261023-r01')
COORD = Path('/root/dejaview_experiments/coordination/unified-graphs-20261026-r01')
GEOMETRY = Path('/root/dejaview_data/shared_geometry/da3-full-sequence-20261026-r01')
RUN = Path('/root/dejaview_experiments/runs/hovsg-unified-v2-20261026-r01')
BASE = Path('/root/dejaview_experiments/baselines/hovsg')
CONFIG = 'sam2_siglip_fullseq_stereo_stride5'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_inputs(scene):
    ready_path = GEOMETRY / scene / 'READY.json'
    ready = read(ready_path)
    assert ready.get('status') == 'COMPLETE', ready
    source = SHARED / 'inputs' / scene / 'frames.json'
    frames = read(source)['frames']
    geometry_frames = read(GEOMETRY / scene / 'frames.json')['frames']
    assert len(geometry_frames) == len(frames)
    for i, (f, g) in enumerate(zip(frames, geometry_frames)):
        for field in ('file', 'run', 'timestamp_ns', 'pair_index', 'sha256'):
            assert f[field] == g[field], (i, field)
        assert g.get('index', i) == i
    sample_ready = read(COORD / 'sampling/READY.json')
    entry = next(x for x in sample_ready['scenes'] if x['scene'] == scene)
    sampling_path = Path(entry['manifest'])
    assert sha(sampling_path) == entry['sha256']
    sample = read(sampling_path)
    selected = sample['selected_indices']
    assert selected == [i for i, f in enumerate(frames) if f['pair_index'] % 5 == 0]
    assert sample['selected_frame_names'] == [frames[i]['file'] for i in selected]
    assert selected == sorted(set(selected))
    by_pair = {}
    for i in selected:
        by_pair.setdefault(frames[i]['pair_index'], []).append(frames[i]['camera'])
    assert all(sorted(c) == ['left', 'right'] for c in by_pair.values())
    assert set(f['run'] for f in frames) == {1, 2, 3}
    assert [f['run'] for f in frames] == sorted(f['run'] for f in frames)
    mapping = read(GEOMETRY / scene / 'native/filename_mapping.json')
    assert {int(x['index']): x['original'] for x in mapping} == {
        i: f['file'] for i, f in enumerate(frames)}
    return frames, selected, {'source_manifest_sha256': sha(source),
        'sampling_manifest_sha256': entry['sha256'], 'geometry_ready_sha256': sha(ready_path),
        'shared_protocol_sha256': sha(COORD / 'PROTOCOL.md'),
        'readout_amendment_sha256': sha(COORD / 'HOVSG_READOUT_AMENDMENT.md')}
