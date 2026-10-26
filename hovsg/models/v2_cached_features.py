"""SAM2/SigLIP adapter; never loads a perception model or an exclusive ID cache.

Pixel features use the native HOV-SG sum-of-overlapping-object-features followed
by L2 normalization and half precision, with cached SigLIP replacing native F_p.
The compact overlap-pattern table is lossless relative to that operation.
"""
import pickle
from pathlib import Path
import cv2
import numpy as np
import torch
from hovsg.utils.v2_io import SHARED, read, sha


def compact_overlap_features(masks, embeddings, shape, device='cuda'):
    h, w = shape
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, 1152)
    if not len(masks):
        return np.zeros((h, w), np.int32), np.zeros((1, 1152), np.float16)
    membership = np.asarray(masks, dtype=np.bool_).reshape(len(masks), h*w).T
    packed = np.packbits(membership, axis=1)
    unique, inverse = np.unique(packed, axis=0, return_inverse=True)
    active = np.unpackbits(unique, axis=1)[:, :len(masks)].astype(bool)
    with torch.inference_mode():
        table = torch.zeros((len(unique), 1152), device=device, dtype=torch.float32)
        vectors = torch.from_numpy(embeddings).to(device)
        # Same accumulation order as upstream's per-mask += loop (not matmul).
        for i in range(len(masks)):
            rows = torch.from_numpy(active[:, i]).to(device)
            table[rows] += vectors[i]
        table = torch.nn.functional.normalize(table, p=2, dim=-1).half()
    return inverse.reshape(h, w).astype(np.int32), table.cpu().numpy()


def load_cached(scene, frame, shape, device='cuda'):
    folder = SHARED / 'features' / scene / Path(frame['file']).stem
    marker_path = folder / 'complete.json'
    marker = read(marker_path)
    assert marker['image_sha256'] == frame['sha256']
    assert marker['image'] == frame['file']
    assert sha(frame['path']) == frame['sha256']
    cache = folder / 'siglip_objects.pkl'
    with cache.open('rb') as stream:
        objects = pickle.load(stream)
    assert isinstance(objects, list) and marker['objects'] == len(objects)
    h, w = shape
    masks, embeddings, metadata = [], [], []
    for i, obj in enumerate(objects):
        mask = np.asarray(obj['mask'])
        feature = np.asarray(obj['embedding'], dtype=np.float32)
        assert mask.shape == (800, 800) and mask.dtype == np.bool_
        assert feature.shape == (1152,) and np.isfinite(feature).all()
        resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(resized)
        embeddings.append(feature)
        x, y, bw, bh = obj['original_bbox']
        cx, cy, cw, ch = obj['bbox']
        metadata.append({'source_object_index': i, 'source_area': int(obj['area']),
            'area': int(resized.sum()), 'bbox_xywh': [x*w/800, y*h/800, bw*w/800, bh*h/800],
            'context_crop_xywh': [cx*w/800, cy*h/800, cw*w/800, ch*h/800],
            'stability_score': float(obj['stability_score']), 'predicted_iou': None})
    pixel_ids, table = compact_overlap_features(masks, embeddings, shape, device)
    return [{'segmentation': m} for m in masks], pixel_ids, table, {
        'objects_sha256': sha(cache), 'marker_sha256': sha(marker_path),
        'source_image_sha256': frame['sha256'], 'source_objects': len(objects),
        'metadata': metadata, 'overlap_patterns': len(table), 'frontend': 'SAM2 + SigLIP1152'}
