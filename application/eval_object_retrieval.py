"""
Object retrieval evaluation on a HOV-SG output directory.

Mirrors the metrics from DejaView's semantic_voxels/evaluator.py:
  hit_rate(TP) / miss_rate(FN) / ghost_rate(FP) / True_Negative
  hit@1 (primary) and any-of-top-t contributor hits
  In-chapter rates over only present queries
  recall / precision / f1

Per query: top-1 instance from mask_feats, then top-t latest observing
frames found by descending-order 3D->2D projection with depth-consistency.
The instance's frame name is mapped through filename_mapping.json so it
matches the original_filename in stream_annotations.json.

Example:
    python application/eval_object_retrieval.py \
        --scene_dir data/scene_graphs/scannet/12thfloor \
        --data_dir /local_data/jz4725/Depth-Anything-3/metric_depth_results/12thfloor \
        --dataset_name 12thfloor \
        --dataset_dir /path/to/annotations_dir \
        --object_list_txt /path/to/objects.txt \
        --chapter_id 1 \
        --output_path data/eval/12thfloor_chapter1.json
"""
import os
import json
import glob
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import open3d as o3d
import open_clip
from PIL import Image
from tqdm import tqdm

from hovsg.utils.clip_utils import get_text_feats_multiple_templates
from hovsg.utils.constants import CLIP_DIM


def numeric_stem(name):
    return int(Path(name).stem)


def _split_left_right(images: Iterable[dict]):
    left, right = [], []
    for entry in images:
        fn = entry.get("file")
        if not isinstance(fn, str):
            continue
        if fn.startswith("left_"):
            left.append(entry)
        elif fn.startswith("right_"):
            right.append(entry)
    return left, right


class ChapterAnnotation:
    """Parse stream_annotations.json into chapter -> object -> [files] (mirrors DejaView)."""

    def __init__(self, annotation_file: str, dataset_dir: str):
        self.annotation_file = annotation_file
        with open(annotation_file) as f:
            self.annotation = json.load(f)
        self.left_right = "realsense" not in dataset_dir
        self.present_objects = self.annotation["labels"]
        self.annotation_by_chapter = {}
        self._parse()

    def _parse(self):
        chapter_id = 1
        self.annotation_by_chapter[chapter_id] = {obj: [] for obj in self.present_objects}
        if self.left_right:
            left_images, right_images = _split_left_right(self.annotation["images"])
            for index, left in enumerate(left_images):
                if left["new_traversal"]:
                    chapter_id += 1
                    self.annotation_by_chapter[chapter_id] = {obj: [] for obj in self.present_objects}
                left_anns = set(a["category"] for a in left.get("annotations", []))
                right = right_images[index]
                right_anns = set(a["category"] for a in right.get("annotations", []))
                for obj in left_anns:
                    self.annotation_by_chapter[chapter_id][obj].append(left["file"])
                for obj in right_anns:
                    self.annotation_by_chapter[chapter_id][obj].append(right["file"])
        else:
            for image in self.annotation["images"]:
                if image.get("new_traversal"):
                    chapter_id += 1
                    self.annotation_by_chapter[chapter_id] = {obj: [] for obj in self.present_objects}
                anns = set(a["category"] for a in image.get("annotations", []))
                for obj in anns:
                    self.annotation_by_chapter[chapter_id][obj].append(image["file"])
        self.total_chapters = chapter_id


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


def load_filename_mapping(path):
    """Return dict {int_index -> original_filename}."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        out = {}
        for entry in raw:
            idx = int(entry["index"])
            orig = entry.get("original") or entry.get("original_filename")
            out[idx] = orig
        return out
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    raise ValueError("Unsupported filename_mapping format")


def visibility_count(points_world, pose_c2w, K, H, W, depth, depth_scale=1000.0, tol=0.5):
    pose_w2c = np.linalg.inv(pose_c2w)
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    pts_cam = points_world @ R.T + t
    z = pts_cam[:, 2]
    if not (z > 1e-3).any():
        return 0
    pts = pts_cam[z > 1e-3]
    z = pts[:, 2]
    u = K[0, 0] * pts[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inb.any():
        return 0
    u = u[inb].astype(np.int32)
    v = v[inb].astype(np.int32)
    z = z[inb]
    if depth is None:
        return int(inb.sum())
    d = depth[v, u].astype(np.float32) / depth_scale
    valid = (d > 0) & (np.abs(d - z) < tol)
    return int(valid.sum())


class HovsgRetriever:
    """Loads a HOV-SG output and exposes query -> top instance -> top-t latest observing frames."""

    def __init__(self, scene_dir, data_dir, mapping_path, clip_type, clip_ckpt, device,
                 vox_size=0.05, min_visible=20, depth_scale=1000.0, depth_tol=0.5):
        self.scene_dir = scene_dir
        self.data_dir = data_dir
        self.device = device
        self.clip_type = clip_type
        self.vox_size = vox_size
        self.min_visible = min_visible
        self.depth_scale = depth_scale
        self.depth_tol = depth_tol

        self.mask_feats = torch.load(os.path.join(scene_dir, "mask_feats.pt")).float()
        self.mask_feats = torch.nn.functional.normalize(self.mask_feats, p=2, dim=-1).cpu().numpy()

        self.feat_dim = CLIP_DIM[clip_type]
        self.model, _, _ = open_clip.create_model_and_transforms(
            clip_type, pretrained=clip_ckpt, device=device,
        )
        self.model.eval()

        self.color_files = sorted(glob.glob(os.path.join(data_dir, "color/*.jpg")), key=numeric_stem)
        self.pose_files = sorted(glob.glob(os.path.join(data_dir, "pose/*.txt")), key=numeric_stem)
        self.depth_files = sorted(glob.glob(os.path.join(data_dir, "depth/*.png")), key=numeric_stem)
        assert len(self.color_files) == len(self.pose_files) == len(self.depth_files), "frame count mismatch"

        self.K = load_intrinsics(os.path.join(data_dir, "intrinsic/intrinsic_depth.txt"))
        H, W = np.array(Image.open(self.depth_files[0])).shape[:2]
        self.H, self.W = H, W

        self.idx_to_orig = load_filename_mapping(mapping_path)

    def get_text_embedding(self, text):
        return get_text_feats_multiple_templates([text], self.model, self.feat_dim)

    def query_top_instance(self, text_feats, top_k=1):
        sims = (self.mask_feats @ text_feats.T).reshape(-1)
        order = np.argsort(-sims)[:top_k]
        return order, sims[order]

    def get_top_t_latest_frames(self, instance_id, top_t=1):
        pcd = o3d.io.read_point_cloud(
            os.path.join(self.scene_dir, "objects", f"pcd_{instance_id}.ply")
        )
        pts = np.asarray(pcd.voxel_down_sample(self.vox_size).points)
        if len(pts) == 0:
            return []
        out = []
        for idx in range(len(self.pose_files) - 1, -1, -1):
            pose = load_pose(self.pose_files[idx])
            depth = np.array(Image.open(self.depth_files[idx]))
            n = visibility_count(pts, pose, self.K, self.H, self.W, depth,
                                 depth_scale=self.depth_scale, tol=self.depth_tol)
            if n >= self.min_visible:
                frame_stem = numeric_stem(self.color_files[idx])
                orig = self.idx_to_orig.get(frame_stem)
                out.append((orig, frame_stem, n))
                if len(out) >= top_t:
                    break
        return out


class ObjectEvaluator:
    """Mirrors DejaView evaluator semantics on top of HovsgRetriever."""

    def __init__(self, dataset_name, dataset_dir, retriever,
                 similarity_threshold=0.2, contributor_top_t=1,
                 ignore_similarity_threshold=False):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.contributor_top_t = max(1, contributor_top_t)
        self.ignore_similarity_threshold = ignore_similarity_threshold
        annotation_file = os.path.join(dataset_dir, "stream_annotations.json")
        self.chapters = ChapterAnnotation(annotation_file, dataset_dir)

    def _evaluate_single(self, query, chapter_ann, query_present):
        text_feats = self.retriever.get_text_embedding(query)
        order, sims = self.retriever.query_top_instance(text_feats, top_k=1)
        if len(order) == 0:
            return {"hit_any_contributor": False, "hit_primary": False, "hit@1": False,
                    "detected": False, "results": []}

        instance_id = int(order[0])
        similarity = float(sims[0])
        contributors = self.retriever.get_top_t_latest_frames(
            instance_id, top_t=self.contributor_top_t
        )

        detected = similarity > self.similarity_threshold
        hit_primary = False
        hit_any_contributor = False
        results = []

        if not contributors:
            return {"hit_any_contributor": False, "hit_primary": False, "hit@1": False,
                    "detected": detected, "results": [{
                        "instance_id": instance_id,
                        "similarity": similarity,
                        "score": similarity,
                        "score_type": "cosine_similarity",
                        "frame_name": None,
                        "frame_index": None,
                        "contributor_frames": [],
                    }]}

        primary_fn, primary_idx, _ = contributors[0]
        contributor_frames = [
            {"frame_name": fn, "frame_index": int(fi), "visible_points": int(vp)}
            for fn, fi, vp in contributors
        ]
        results.append({
            "instance_id": instance_id,
            "similarity": similarity,
            "score": similarity,
            "score_type": "cosine_similarity",
            "frame_name": primary_fn,
            "frame_index": int(primary_idx),
            "contributor_frames": contributor_frames,
        })

        if query_present and (self.ignore_similarity_threshold or similarity > self.similarity_threshold):
            gt = set(chapter_ann.get(query, []))
            if primary_fn is not None and primary_fn in gt:
                hit_primary = True
            if any(fn is not None and fn in gt for fn, _, _ in contributors):
                hit_any_contributor = True

        return {
            "hit_any_contributor": hit_any_contributor,
            "hit_primary": hit_primary,
            "hit@1": hit_primary,
            "detected": detected,
            "results": results,
        }

    def evaluate(self, chapter_id, object_list):
        retrieval_results = {}
        hit_any_count = miss_any_count = 0
        hit_primary_count = miss_primary_count = 0
        ghost_count = 0
        chapter_ann = self.chapters.annotation_by_chapter.get(chapter_id, {})
        n_gt_in_chapter = sum(1 for files in chapter_ann.values() if files)

        for query in tqdm(object_list, desc=f"chapter {chapter_id}"):
            query_present = bool(chapter_ann.get(query, ()))
            qr = self._evaluate_single(query, chapter_ann, query_present)
            detected = qr["detected"]
            hit_any = qr["hit_any_contributor"]
            hit_primary = qr["hit_primary"]
            if query_present:
                if hit_any:
                    status = "hit"
                    hit_any_count += 1
                else:
                    status = "miss"
                    miss_any_count += 1
                if hit_primary:
                    hit_primary_count += 1
                else:
                    miss_primary_count += 1
            else:
                if detected:
                    status = "ghost"
                    ghost_count += 1
                else:
                    status = "True_Negative"
            qr["status"] = status
            qr["present_in_chapter"] = query_present
            retrieval_results[query] = qr

        total = max(len(object_list), 1)
        tp, fn, fp = hit_any_count, miss_any_count, ghost_count
        tn = len(object_list) - tp - fn - fp
        chapter_positive = hit_any_count + miss_any_count
        denom = max(chapter_positive, 1)
        recall = tp / (tp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "dataset_name": self.dataset_name,
            "chapter_id": chapter_id,
            "object_list": object_list,
            "retrieval_results": retrieval_results,
            "hit_rate_any_contributor": hit_any_count / total,
            "hit_rate(TP)": hit_any_count / total,
            "miss_rate(FN)": miss_any_count / total,
            "hit_rate_primary": hit_primary_count / total,
            "miss_rate_primary(FN)": miss_primary_count / total,
            "hit@1": hit_primary_count / total,
            "ghost_rate(FP)": ghost_count / total,
            "True_Negative(TN)": tn / total,
            "In chapter hit (any contributor)": hit_any_count / denom,
            "In chapter miss (any contributor)": miss_any_count / denom,
            "In chapter hit (primary only)": hit_primary_count / denom,
            "In chapter miss (primary only)": miss_primary_count / denom,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "meta": {
                "similarity_threshold": self.similarity_threshold,
                "ignore_similarity_threshold_for_hits": self.ignore_similarity_threshold,
                "contributor_top_t": self.contributor_top_t,
                "tp_fn_criterion": "any_contributor_in_gt",
                "primary_metric_alias_hit@1": "hit_rate_primary",
                "n_gt_objects_in_chapter_annotation": n_gt_in_chapter,
                "n_evaluated_positive_queries": chapter_positive,
                "in_chapter_rates_denominator": "object_list queries with GT in this chapter",
                "scene_dir": self.retriever.scene_dir,
                "data_dir": self.retriever.data_dir,
                "clip_type": self.retriever.clip_type,
                "score_type": "cosine_similarity",
                "vox_size": self.retriever.vox_size,
                "min_visible": self.retriever.min_visible,
                "depth_tol": self.retriever.depth_tol,
            },
        }


def _load_object_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True, help="HOV-SG output: mask_feats.pt + objects/")
    p.add_argument("--data_dir", required=True, help="ScanNet-style scene: color/ depth/ pose/ intrinsic/")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_dir", required=True, help="dir containing stream_annotations.json")
    p.add_argument("--object_list_txt", required=True)
    p.add_argument("--chapter_id", type=int, required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--mapping", default=None, help="filename_mapping.json (defaults to data_dir)")
    p.add_argument("--clip_type", default="ViT-H-14")
    p.add_argument("--clip_ckpt", default="checkpoints/laion2b_s32b_b79k.bin")
    p.add_argument("--similarity_threshold", type=float, default=0.2)
    p.add_argument("--ignore_similarity_threshold", action="store_true")
    p.add_argument("--contributor_top_t", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vox_size", type=float, default=0.05)
    p.add_argument("--min_visible", type=int, default=20)
    p.add_argument("--depth_tol", type=float, default=0.5)
    p.add_argument("--depth_scale", type=float, default=1000.0)
    args = p.parse_args()

    mapping_path = args.mapping or os.path.join(args.data_dir, "filename_mapping.json")
    retriever = HovsgRetriever(
        scene_dir=args.scene_dir, data_dir=args.data_dir, mapping_path=mapping_path,
        clip_type=args.clip_type, clip_ckpt=args.clip_ckpt, device=args.device,
        vox_size=args.vox_size, min_visible=args.min_visible,
        depth_scale=args.depth_scale, depth_tol=args.depth_tol,
    )
    object_list = _load_object_list(args.object_list_txt)
    evaluator = ObjectEvaluator(
        dataset_name=args.dataset_name, dataset_dir=args.dataset_dir, retriever=retriever,
        similarity_threshold=args.similarity_threshold,
        contributor_top_t=args.contributor_top_t,
        ignore_similarity_threshold=args.ignore_similarity_threshold,
    )
    results = evaluator.evaluate(args.chapter_id, object_list)

    print("***************** evaluation results *****************")
    for k, v in results.items():
        if k == "retrieval_results":
            for obj, r in v.items():
                print(f"Object: {obj}")
                for kk, vv in r.items():
                    print(f"  {kk}: {vv}")
                print("--------------------------------")
        print(f"{k}: {v}")
    print("*******************************************************")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
