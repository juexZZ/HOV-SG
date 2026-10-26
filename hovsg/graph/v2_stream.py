"""Persistent HOV-SG object-layer fusion with snapshot-only global reduction.

This temporal adapter is described in docs/dejaview_v2_unified.md. It does not
invoke Graph.__init__ (which would load native SAM/OpenCLIP). Numerical mask
merging and final feature DBSCAN come directly from the original repository.
"""
import copy
import gzip
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch
from omegaconf import OmegaConf

from hovsg.dataloader.scannet import ScannetDataset
from hovsg.models.v2_cached_features import load_cached
from hovsg.utils.graph_utils import merge_3d_masks, pcd_denoise_dbscan, feats_denoise_dbscan
from hovsg.utils.v2_io import read, write, utc, sha, GEOMETRY, SHARED, CONFIG


def cloud(points, colors):
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    out.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return out


def serialize_cloud(pcd):
    return (np.asarray(pcd.points).copy(), np.asarray(pcd.colors).copy())


def fuse_masks(masks, contributors, cfg):
    merged, components = merge_3d_masks(masks,
        overlap_threshold=cfg.init_overlap_thresh, radius=cfg.voxel_size,
        iou_thresh=cfg.iou_thresh, return_components=True)
    tags = [sorted({tag for i in group for tag in contributors[i]}) for group in components]
    assert len(tags) == len(merged)
    return merged, tags


def last_pixel_per_support(indices, pixel_ids, n_points):
    """Reproduce the native per-frame indexed += duplicate-index semantics.

Upstream torch sum_features[idx] += F_2D and counter[idx] += 1 are indexed
assignment, not scatter_add: repeated pixel->point IDs count once per frame.
Use explicit last row-major pixel to remove CPU assignment-race ambiguity.
"""
    ids = np.asarray(indices)
    selected, reverse = np.unique(ids[::-1], return_index=True)
    positions = len(ids) - 1 - reverse
    out = np.zeros(n_points, dtype=np.int32)
    count = np.zeros(n_points, dtype=np.float32)
    out[selected] = pixel_ids[positions]
    count[selected] = 1
    return out, count


class StreamMap:
    def __init__(self, scene, frames, selected, root, provenance, device='cuda'):
        self.scene, self.frames, self.selected = scene, frames, selected
        self.root, self.provenance, self.device = Path(root), provenance, device
        self.root.mkdir(parents=True, exist_ok=True)
        cfg_path = Path(__file__).resolve().parents[2] / 'config/create_graph.yaml'
        self.cfg = OmegaConf.load(cfg_path).pipeline
        assert self.cfg.merge_type == 'sequential'
        self.dataset = ScannetDataset({'root_dir': str(GEOMETRY / scene / 'native'), 'transforms': None})
        assert len(self.dataset) == len(frames)
        self.masks, self.contributors, self.processed, self.observations = [], [], [], []
        self.source_cfg_sha = sha(cfg_path)

    def ingest(self, index):
        assert index in self.selected
        assert index not in self.processed
        assert not self.processed or index > self.processed[-1]
        started = time.monotonic()
        frame = self.frames[index]
        rgb, depth, pose, _, _ = self.dataset[index]
        depth_np = np.asarray(depth)
        assert depth_np.shape == (504, 504) and np.isfinite(pose).all()
        masks, pixel_ids, table, cache_record = load_cached(self.scene, frame, depth_np.shape, self.device)
        raw_pcd = self.dataset.create_pcd(rgb, depth, pose)
        local = raw_pcd.voxel_down_sample(self.cfg.voxel_size)
        local = pcd_denoise_dbscan(local, eps=0.01, min_points=100)
        assert not local.is_empty(), (index, 'empty geometry')
        tree = cKDTree(np.asarray(local.points))
        raw_points = np.asarray(raw_pcd.points)
        _, nearest = tree.query(raw_points, k=1, workers=4)
        valid_ids = pixel_ids[depth_np > 0]
        assert len(nearest) == len(valid_ids)
        point_ids, point_count = last_pixel_per_support(nearest, valid_ids, len(local.points))
        projected = self.dataset.create_3d_masks(masks, depth, local, tree, pose,
            down_size=self.cfg.voxel_size, filter_distance=self.cfg.max_mask_distance, query_workers=4)
        tags = [[(index, i)] for i in range(len(projected))]
        # Empty masks (possible after nearest-neighbor resize) have no geometry
        # and cannot be passed to upstream bounding-box/FAISS routines.
        valid = [i for i, p in enumerate(projected) if not p.is_empty()]
        projected, tags = [projected[i] for i in valid], [tags[i] for i in valid]
        obs_path = self.root / 'observations' / f'{index:06d}.npz'
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        with obs_path.open('xb') as stream:
            np.savez_compressed(stream, points=np.asarray(local.points), colors=np.asarray(local.colors),
                feature_ids=point_ids, feature_table=table, point_count=point_count,
                frame_index=np.int64(index), run=np.int64(frame['run']))
        if not self.processed:
            # seq_merge starts with frames_pcd[0] without an extra merge.
            self.masks, self.contributors = projected, tags
        elif projected or self.masks:
            self.masks, self.contributors = fuse_masks(self.masks + projected,
                self.contributors + tags, self.cfg)
        self.processed.append(index)
        self.observations.append(str(obs_path))
        record = {'status': 'COMPLETE', 'index': index, 'frame': frame['file'], 'run': frame['run'],
            'local_points': len(local.points), 'valid_projected_masks': len(projected),
            'stream_objects': len(self.masks), 'cache': cache_record,
            'observation_sha256': sha(obs_path), 'seconds': time.monotonic()-started}
        write(obs_path.with_suffix('.json'), record)
        print('INGEST', json.dumps({k:v for k,v in record.items() if k != 'cache'}), flush=True)
        return record

    def save_checkpoint(self, chapter):
        path = self.root / f'stream_t{chapter}.pkl.gz'
        temporary = path.with_suffix(path.suffix + '.tmp')
        with gzip.open(temporary, 'xb', compresslevel=1) as stream:
            pickle.dump({'format': 1, 'scene': self.scene, 'chapter': chapter,
                'processed': self.processed, 'observations': self.observations,
                'masks': [serialize_cloud(p) for p in self.masks],
                'contributors': self.contributors, 'provenance': self.provenance,
                'rng': (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                'source_cfg_sha256': self.source_cfg_sha}, stream, protocol=4)
        os.link(temporary, path)
        temporary.unlink()
        write(path.with_suffix('.json'), {'status': 'COMPLETE', 'chapter': chapter,
            'sha256': sha(path), 'processed': self.processed, 'objects': len(self.masks),
            'last_global_index': self.processed[-1], 'finished': utc(),
            'terminal_merge_applied': False, 'provenance': self.provenance})
        return path

    def load_checkpoint(self, chapter):
        path = self.root / f'stream_t{chapter}.pkl.gz'
        receipt = read(path.with_suffix('.json'))
        assert sha(path) == receipt['sha256']
        with gzip.open(path, 'rb') as stream:
            data = pickle.load(stream)
        assert data['scene'] == self.scene and data['provenance'] == self.provenance
        assert data['source_cfg_sha256'] == self.source_cfg_sha
        assert all(self.frames[i]['run'] <= chapter for i in data['processed'])
        expected = [i for i in self.selected if self.frames[i]['run'] <= chapter]
        assert data['processed'] == expected
        self.processed, self.observations = data['processed'], data['observations']
        self.masks = [cloud(*p) for p in data['masks']]
        self.contributors = data['contributors']
        py, np_state, cpu, gpu = data['rng']
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if gpu:
            torch.cuda.set_rng_state_all(gpu)

    def snapshot(self, chapter, smoke=False):
        """No mutation of stream masks or contributors and no frontend/image reads."""
        assert self.processed and all(self.frames[i]['run'] <= chapter for i in self.processed)
        out = self.root / f't{chapter}'
        if out.exists():
            raise FileExistsError(out)
        out.mkdir()
        started = time.monotonic()
        masks, contributors = fuse_masks(copy.deepcopy(self.masks), copy.deepcopy(self.contributors), self.cfg)
        # Preserve the existing native traversal/removal order, including its
        # list-pop behavior, while keeping provenance aligned with objects.
        for i, pcd in enumerate(masks):
            if pcd.is_empty() or len(pcd.points) < 100:
                masks.pop(i)
                contributors.pop(i)
        full = o3d.geometry.PointCloud()
        observed_indices = []
        for obs_path in self.observations:
            with np.load(obs_path, allow_pickle=False) as obs:
                index = int(obs['frame_index'])
                assert index in self.processed and int(obs['run']) <= chapter
                assert self.frames[index]['run'] == int(obs['run'])
                observed_indices.append(index)
                full += cloud(obs['points'], obs['colors'])
        assert observed_indices == self.processed and len(set(observed_indices)) == len(observed_indices)
        assert all(f in self.processed and self.frames[f]['run'] <= chapter
                   for tags in contributors for f, _ in tags)
        full = full.voxel_down_sample(self.cfg.voxel_size)
        full = pcd_denoise_dbscan(full, eps=0.01, min_points=100)
        assert len(full.points)
        tree = cKDTree(np.asarray(full.points))
        # Re-reduce compact *already associated* observations on snapshot support.
        # This is the unavoidable global HOV point-feature assembly, NOT redoing
        # old-frame inference, projection, mask creation or sequential fusion.
        with torch.inference_mode():
            sums = torch.zeros((len(full.points), 1152), device=self.device)
            counts = torch.zeros(len(full.points), device=self.device)
            for obs_path in self.observations:
                with np.load(obs_path, allow_pickle=False) as obs:
                    _, indices = tree.query(obs['points'], k=1, workers=4)
                    table = torch.from_numpy(obs['feature_table'].astype(np.float32)).to(self.device)
                    ids = obs['feature_ids']
                    weight = obs['point_count']
                    for start in range(0, len(indices), 16384):
                        stop = min(start+16384, len(indices))
                        dest = torch.from_numpy(indices[start:stop]).to(self.device)
                        fid = torch.from_numpy(ids[start:stop].astype(np.int64)).to(self.device)
                        cnt = torch.from_numpy(weight[start:stop]).to(self.device)
                        sums.index_add_(0, dest, table[fid] * cnt[:, None])
                        counts.index_add_(0, dest, cnt)
            sums /= counts.clamp_min(1e-5)[:, None]
            full_feats = sums.cpu().numpy()
            del sums, counts
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        object_feats = []
        (out / 'objects').mkdir()
        for i, mask in enumerate(masks):
            small = mask.voxel_down_sample(self.cfg.voxel_size*2)
            _, ids = tree.query(np.asarray(small.points), k=1, workers=4)
            feats = np.nan_to_num(full_feats[ids])
            feature = (feats_denoise_dbscan(feats, eps=0.01, min_points=100)
                       if len(feats) else np.zeros(1152, np.float32))
            feature = np.asarray(feature).reshape(-1, 1152).mean(0)
            object_feats.append(feature)
            o3d.io.write_point_cloud(str(out / 'objects' / f'pcd_{i}.ply'), mask)
            print('SNAPSHOT_OBJECT', chapter, i, len(masks), len(small.points), flush=True)
        feats = np.asarray(object_feats, dtype=np.float32).reshape(-1, 1152)
        torch.save(torch.from_numpy(feats), out / 'mask_feats.pt')
        o3d.io.write_point_cloud(str(out / 'full_pcd.ply'), full)
        torch.save(torch.from_numpy(full_feats), out / 'full_feats.pt')
        write(out / 'contributors.json', {'objects': [
            {'object_id': i, 'contributors': [{'frame_index': int(f), 'source_object_index': int(m),
                'frame_name': self.frames[f]['file'], 'run': self.frames[f]['run']} for f, m in tags]}
            for i, tags in enumerate(contributors)]})
        record = {'status': 'SMOKE' if smoke else 'COMPLETE', 'scene': self.scene, 'chapter': chapter,
            'objects': len(masks), 'full_points': len(full.points), 'processed_indices': self.processed,
            'mask_feats_sha256': sha(out/'mask_feats.pt'), 'contributors_sha256': sha(out/'contributors.json'),
            'snapshot_only_global_assembly': True, 'stream_modified': False,
            'future_contributions': 0, 'source_observation_indices': observed_indices,
            'non_bitwise_native_batch_variant': True, 'support_mode':'per-frame-local; cumulative snapshot only',
            'seconds': time.monotonic()-started, 'finished': utc(), 'provenance': self.provenance}
        write(out / 'complete.json', record)
        return out


def retrieve(scene, chapter, frames, selected, snapshot, output, smoke=False):
    from hovsg.graph.v2_retrieval import CachedNativeRetriever
    snapshot = Path(snapshot)
    native = CachedNativeRetriever(scene, chapter, frames, snapshot)
    features = native.mask_feats
    tags = read(snapshot/'contributors.json')['objects']
    metadata = read(SHARED/'inputs/query_embeddings.json')
    assert metadata['graph_prompt'] == 'a photo of a {query}'
    with np.load(SHARED/'inputs/query_embeddings.npz', allow_pickle=False) as data:
        vectors = data['graph']
    assert vectors.shape == (len(metadata['labels']), 1152)
    text = dict(zip(metadata['labels'], vectors))
    queries = [q for q in read(SHARED/'inputs/queries.json')['queries']
               if q['scene'] == scene and q['chapter'] == chapter]
    selected = set(selected)
    for obj in tags:
        for contributor in obj['contributors']:
            i = contributor['frame_index']
            assert i in selected and frames[i]['run'] <= chapter
            assert frames[i]['file'] == contributor['frame_name']
    predictions = []
    for query in queries:
        q = text[query['query']].reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-12)
        order, similarities = native.query_top_instance(q, top_k=1)
        top = int(order[0]) if len(order) else None
        candidates = tags[top]['contributors'] if top is not None else []
        latest = max(candidates, key=lambda c:c['frame_index']) if candidates else None
        visible = native.get_top_t_latest_frames(top, top_t=1) if top is not None else []
        name, index, n_visible = visible[0] if visible else (None, None, 0)
        assert index is None or frames[index]['run'] <= chapter
        current = bool(index is not None and frames[index]['run'] == chapter)
        predictions.append(dict(query, method='hovsg', config=CONFIG,
            frame_name=name, frame_index=index, visible_points=int(n_visible),
            contributor_frame_name=latest['frame_name'] if latest else None,
            contributor_frame_index=latest['frame_index'] if latest else None,
            returned_frame_was_mapping_input=bool(index in selected),
            in_current_traversal=current, candidate_frames=[name] if name and current else [],
            instance_id=top, similarity=float(similarities[0]) if top is not None else None))
    write(output, {'status': 'SMOKE' if smoke else 'COMPLETE', 'scene': scene, 'chapter': chapter,
        'predictions': predictions, 'gt_used': False, 'objects': len(features), 'finished': utc(),
        'snapshot': str(snapshot), 'template': metadata['graph_prompt'],
        'readout':'native HOV visibility over all frames in current cumulative prefix',
        'visibility':{'voxel':native.vox_size,'min_visible':native.min_visible,
            'depth_scale':native.depth_scale,'depth_tol':native.depth_tol,
            'prefix_frames':len(native.pose_files),'future_frames':0},
        'query_embeddings_sha256': sha(SHARED/'inputs/query_embeddings.npz')})
