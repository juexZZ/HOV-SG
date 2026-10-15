"""
Query an open-vocabulary 3D instance from HOV-SG outputs (mask_feats.pt +
objects/pcd_*.ply) and find the latest RGB frame (by numeric filename stem)
that observes the matched instance.

Example:
    python application/query_latest_frame.py \
        --scene_dir /local_data/jz4725/HOV-SG/data/scene_graphs/scannet/12thfloor \
        --data_dir /local_data/jz4725/Depth-Anything-3/metric_depth_results/12thfloor \
        --query "office chair" --top_k 3
"""
import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import open_clip
from PIL import Image
from tqdm import tqdm

from hovsg.utils.clip_utils import get_text_feats_multiple_templates
from hovsg.utils.constants import CLIP_DIM


def load_intrinsics(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append([float(x) for x in line.split()])
    return np.array(rows)[:3, :3]


def load_pose(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append([float(x) for x in line.split()])
    return np.array(rows)


def numeric_stem(name):
    return int(Path(name).stem)


def visibility_count(points_world, pose_c2w, K, H, W, depth=None, depth_scale=1000.0, tol=0.5):
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    pts_cam = points_world @ R.T + t
    z = pts_cam[:, 2]
    in_front = z > 1e-3
    if not in_front.any():
        return 0
    pts = pts_cam[in_front]
    z = pts[:, 2]
    u = K[0, 0] * pts[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inb.any():
        return 0
    u = u[inb].astype(np.int32)
    v = v[inb].astype(np.int32)
    z = z[inb]
    if depth is not None:
        d = depth[v, u].astype(np.float32) / depth_scale
        valid = (d > 0) & (np.abs(d - z) < tol)
        return int(valid.sum())
    return int(inb.shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True, help="HOV-SG output: mask_feats.pt and objects/")
    p.add_argument("--data_dir", required=True, help="ScanNet-style scene: color/ depth/ pose/ intrinsic/")
    p.add_argument("--query", required=True)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--clip_type", default="ViT-H-14")
    p.add_argument("--clip_ckpt", default="checkpoints/laion2b_s32b_b79k.bin")
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth_scale", type=float, default=1000.0)
    p.add_argument("--depth_tol", type=float, default=0.5, help="meters; loose for noisy depth")
    p.add_argument("--min_visible", type=int, default=20, help="visible voxel-points to count as observed")
    p.add_argument("--vox_size", type=float, default=0.05, help="downsample instance pcd before projection")
    p.add_argument("--no_depth_check", action="store_true", help="skip depth-consistency check (allows occluded)")
    args = p.parse_args()

    mask_feats = torch.load(os.path.join(args.scene_dir, "mask_feats.pt")).float()
    mask_feats = torch.nn.functional.normalize(mask_feats, p=2, dim=-1).cpu().numpy()
    print(f"loaded {mask_feats.shape[0]} mask features (dim={mask_feats.shape[-1]})")

    feat_dim = CLIP_DIM[args.clip_type]
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_type, pretrained=args.clip_ckpt, device=args.device,
    )
    model.eval()
    text_feats = get_text_feats_multiple_templates([args.query], model, feat_dim)
    sim = (mask_feats @ text_feats.T).reshape(-1)
    top = np.argsort(-sim)[: args.top_k]
    print(f'\ntop {args.top_k} instances for "{args.query}":')
    for r, i in enumerate(top):
        print(f"  rank {r+1}: instance {i}  score {sim[i]:.4f}")

    K = load_intrinsics(os.path.join(args.data_dir, "intrinsic/intrinsic_depth.txt"))
    color_files = sorted(glob.glob(os.path.join(args.data_dir, "color/*.jpg")), key=numeric_stem)
    pose_files = sorted(glob.glob(os.path.join(args.data_dir, "pose/*.txt")), key=numeric_stem)
    depth_files = sorted(glob.glob(os.path.join(args.data_dir, "depth/*.png")), key=numeric_stem)
    assert len(color_files) == len(pose_files) == len(depth_files), "frame count mismatch"
    H, W = np.array(Image.open(depth_files[0])).shape[:2]
    print(f"frames: {len(color_files)}  image: {W}x{H}")

    results = {}
    for inst_id in top:
        pcd = o3d.io.read_point_cloud(os.path.join(args.scene_dir, "objects", f"pcd_{inst_id}.ply"))
        pts = np.asarray(pcd.voxel_down_sample(args.vox_size).points)
        print(f"\n[inst {inst_id}] pcd pts: {len(pcd.points)} -> {len(pts)} after voxel {args.vox_size}m")
        latest, latest_count = None, 0
        for idx in tqdm(range(len(pose_files) - 1, -1, -1), desc=f"inst {inst_id}"):
            pose = load_pose(pose_files[idx])
            depth = None if args.no_depth_check else np.array(Image.open(depth_files[idx]))
            n = visibility_count(pts, pose, K, H, W, depth=depth,
                                 depth_scale=args.depth_scale, tol=args.depth_tol)
            if n >= args.min_visible:
                latest, latest_count = idx, n
                break
        if latest is None:
            print(f"  no frame >= {args.min_visible} visible pts")
            results[int(inst_id)] = None
        else:
            print(f"  latest observation -> {color_files[latest]}  visible={latest_count}")
            results[int(inst_id)] = (color_files[latest], latest_count)

    print("\n=== summary ===")
    for r, i in enumerate(top):
        info = results[int(i)]
        if info is None:
            print(f"rank {r+1} inst {i} (sim {sim[i]:.4f}): no observation found")
        else:
            print(f"rank {r+1} inst {i} (sim {sim[i]:.4f}): {info[0]} (visible={info[1]})")


if __name__ == "__main__":
    main()
