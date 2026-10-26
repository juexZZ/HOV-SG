"""One scene, one forward mapping pass, three independent evaluation snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import pickle
import numpy as np
import torch
import faiss

from hovsg.graph.v2_stream import StreamMap, retrieve
from hovsg.utils.v2_io import RUN, load_inputs, write, read, utc


def state_digest(state):
    h = hashlib.sha256()
    for p in state.masks:
        h.update(np.asarray(p.points).tobytes())
        h.update(np.asarray(p.colors).tobytes())
    h.update(json.dumps(state.contributors).encode())
    return h.hexdigest()


def capture_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(states):
    py, np_state, cpu, gpu = states
    random.setstate(py)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu)
    if gpu:
        torch.cuda.set_rng_state_all(gpu)


def rng_digest(states):
    py, np_state, cpu, gpu = states
    h = hashlib.sha256(pickle.dumps((py, np_state), protocol=4))
    for tensor in [cpu] + gpu:
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU3 is required; no CPU fallback')
    assert torch.cuda.device_count() == 1
    frames, selected, provenance = load_inputs(args.scene)
    if args.smoke:
        selected = sum(([i for i in selected if frames[i]['run'] == r][:2] for r in (1,2,3)), [])
    root = args.root or RUN / ('smoke' if args.smoke else 'maps') / args.scene
    root.mkdir(parents=True, exist_ok=True)
    state = StreamMap(args.scene, frames, selected, root, provenance)
    completed = []
    for chapter in (1,2,3):
        snapshot = root / f't{chapter}'
        prediction = root / f'prediction_t{chapter}.json' if args.smoke else RUN/'predictions'/f'{args.scene}-t{chapter}.json'
        if (root / f'stream_t{chapter}.pkl.json').exists():
            # A complete checkpoint is valid even if evaluation had not started.
            state.load_checkpoint(chapter)
        else:
            for index in selected:
                if frames[index]['run'] != chapter:
                    continue
                state.ingest(index)
                write(root/'progress.json', {'phase':'MAPPING', 'status':'RUNNING', 'scene':args.scene,
                    'run':chapter, 'processed':len(state.processed), 'selected_total':len(selected),
                    'last_global_index':index, 'stream_objects':len(state.masks), 'updated':utc()}, replace=True)
            state.save_checkpoint(chapter)
        if not (snapshot/'complete.json').exists():
            write(root/'progress.json', {'phase':'SNAPSHOT', 'status':'RUNNING', 'scene':args.scene,
                'run':chapter, 'processed':len(state.processed), 'selected_total':len(selected),
                'stream_objects':len(state.masks), 'updated':utc()}, replace=True)
            before = state_digest(state)
            rng = capture_rng()
            rng_before = rng_digest(rng)
            try:
                snapshot = state.snapshot(chapter, smoke=args.smoke)
            finally:
                restore_rng(rng)
            assert before == state_digest(state), 'Evaluation changed streaming fusion state'
            assert rng_before == rng_digest(capture_rng()), 'Evaluation changed RNG state'
            write(root/f'snapshot_t{chapter}_isolation.json', {'status':'PASS','stream_digest':before,
                'rng_restored':True, 'rng_digest':rng_before,
                'open3d_rng_used':False, 'native_operations':'DBSCAN/FlatL2/voxel/KDTree are deterministic; no Open3D random sampler'})
        if not prediction.exists():
            retrieve(args.scene, chapter, frames, selected, snapshot, prediction, args.smoke)
        completed.append(chapter)
        write(root/'progress.json', {'phase':'BOUNDARY_COMPLETE', 'status':'RUNNING','scene':args.scene,
            'run':chapter, 'processed':len(state.processed), 'selected_total':len(selected),
            'completed_snapshots':completed, 'updated':utc()}, replace=True)
    assert state.processed == selected
    write(root/'complete.json', {'status':'SMOKE' if args.smoke else 'COMPLETE', 'scene':args.scene,
        'processed':len(selected), 'completed_snapshots':completed, 'provenance':provenance,
        'processed_indices':state.processed, 'finished':utc()})


if __name__ == '__main__':
    main()
