"""
Split a HOV-SG-style data directory into per-traversal prefix subsets,
based on `new_traversal` flags in stream_annotations.json.

Input directory layout:
    <data_dir>/
        color/   0.jpg, 1.jpg, ...
        depth/   0.png, 1.png, ...
        pose/    0.txt, 1.txt, ...
        intrinsic/intrinsic_color.txt, intrinsic_depth.txt
        filename_mapping.json     [{"index": 0, "original": "left_xxx.png", ...}, ...]

For each prefix of traversals (trav 1, trav 1+2, ...), creates a sibling
directory `<base>_trav_<suffix>` containing only the frames whose original
left_*.png filename falls within that prefix. The filtered
filename_mapping.json is rewritten so the evaluator's lookup still works.

Mirrors the prefix-mode behavior of DejaView/split_chapter_image.py.

Example:
    python application/split_chapter_data.py \
        --data_dir /local_data/jz4725/Depth-Anything-3/metric_depth_results/28_metfloor \
        --annotation_file /local_data/jz4725/metacam/data_v3/28_metfloor/stream_annotations.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


def _load_annotation_images(annotation_file: Path) -> list[dict]:
    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError("annotation JSON missing 'images' list")
    return images


def _split_left_right(images: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    left, right = [], []
    for e in images:
        fn = e.get("file")
        if not isinstance(fn, str):
            continue
        if fn.startswith("left_"):
            left.append(e)
        elif fn.startswith("right_"):
            right.append(e)
    return left, right


def _traversal_ranges(left_images: list[dict]) -> list[tuple[int, int]]:
    if not left_images:
        return []
    starts = [0]
    for i, e in enumerate(left_images):
        if i == 0:
            continue
        if e.get("new_traversal") is True:
            starts.append(i)
    return [(s, starts[i + 1] if i + 1 < len(starts) else len(left_images))
            for i, s in enumerate(starts)]


def _load_orig_to_idx(mapping_path: Path) -> dict[str, int]:
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, int] = {}
    if isinstance(raw, list):
        for e in raw:
            idx = int(e["index"])
            orig = e.get("original") or e.get("original_filename")
            if orig:
                out[orig] = idx
    elif isinstance(raw, dict):
        for k, v in raw.items():
            out[v] = int(k)
    else:
        raise ValueError("unsupported filename_mapping format")
    return out


def _filtered_mapping_entries(mapping_path: Path, kept: set[int]) -> list[dict]:
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [e for e in raw if int(e["index"]) in kept]
    if isinstance(raw, dict):
        return [{"index": int(k), "original": v} for k, v in raw.items() if int(k) in kept]
    raise ValueError("unsupported filename_mapping format")


def _index_to_ext(folder: Path) -> dict[int, Path]:
    out = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        try:
            index = int(p.stem)
        except ValueError:
            continue
        if index in out:
            raise ValueError(f"duplicate frame ID {index} in {folder}")
        out[index] = p
    return out


def _materialize(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists():
        return
    if symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def _copy_split(
    src_dir: Path, dst_dir: Path, kept: set[int],
    mapping_entries: list[dict], symlink: bool,
) -> tuple[int, int]:
    if dst_dir.exists():
        raise FileExistsError(f"Refusing to mix a split with an existing directory: {dst_dir}")
    for sub in ("color", "depth", "pose", "intrinsic"):
        (dst_dir / sub).mkdir(parents=True, exist_ok=True)
    color_exts = _index_to_ext(src_dir / "color")
    depth_exts = _index_to_ext(src_dir / "depth")
    pose_exts = _index_to_ext(src_dir / "pose")
    copied, missing = 0, 0
    materialized = set()
    for idx in sorted(kept):
        triplet = [
            ("color", color_exts.get(idx)),
            ("depth", depth_exts.get(idx)),
            ("pose", pose_exts.get(idx)),
        ]
        if any(ext is None for _, ext in triplet):
            missing += 1
            continue
        for sub, ext in triplet:
            _materialize(ext, dst_dir / sub / ext.name, symlink)
        copied += 1
        materialized.add(idx)
    intrinsic_src = src_dir / "intrinsic"
    if intrinsic_src.is_dir():
        for p in intrinsic_src.iterdir():
            if p.is_file():
                _materialize(p, dst_dir / "intrinsic" / p.name, symlink)
    with (dst_dir / "filename_mapping.json").open("w", encoding="utf-8") as f:
        json.dump([e for e in mapping_entries if int(e["index"]) in materialized], f, indent=2)
    return copied, missing


def split_traversals(
    data_dir: Path, annotation_file: Path, mode: str,
    out_parent: Path | None = None, symlink: bool = False,
) -> None:
    images = _load_annotation_images(annotation_file)
    left_images, _ = _split_left_right(images)
    if not left_images:
        raise ValueError("no left_*.png entries in annotation")

    ranges = _traversal_ranges(left_images)
    print(f"detected {len(ranges)} traversal(s) in annotation")
    if len(ranges) <= 1 and mode == "prefix":
        print("only one traversal — no prefix splits to create")
        return

    mapping_path = data_dir / "filename_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"missing {mapping_path}")
    orig_to_idx = _load_orig_to_idx(mapping_path)

    parent = out_parent or data_dir.parent
    base = data_dir.name
    print(f"writing splits under: {parent}")

    if mode == "prefix":
        # trav_1, trav_12, ..., trav_1..(n-1); skip the full set (== input)
        for trav_idx in range(1, len(ranges)):
            _, end_left = ranges[trav_idx - 1]
            kept_originals = [left_images[i]["file"] for i in range(0, end_left)]
            kept = {orig_to_idx[fn] for fn in kept_originals if fn in orig_to_idx}
            unmapped = [fn for fn in kept_originals if fn not in orig_to_idx]
            suffix = "".join(str(i) for i in range(1, trav_idx + 1))
            dst = parent / f"{base}_trav_{suffix}"
            copied, missing = _copy_split(data_dir, dst, kept, _filtered_mapping_entries(mapping_path, kept), symlink)
            note = f"{copied} frames" + (f", {missing} missing on disk" if missing else "")
            note += f", {len(unmapped)} not in mapping" if unmapped else ""
            print(f"  -> {dst}: {note}")
    else:  # "each"
        for ti, (s, e) in enumerate(ranges, start=1):
            kept_originals = [left_images[i]["file"] for i in range(s, e)]
            kept = {orig_to_idx[fn] for fn in kept_originals if fn in orig_to_idx}
            dst = parent / f"{base}_trav_{ti}"
            copied, missing = _copy_split(data_dir, dst, kept, _filtered_mapping_entries(mapping_path, kept), symlink)
            print(f"  -> {dst}: {copied} frames" + (f", {missing} missing" if missing else ""))


def main():
    p = argparse.ArgumentParser(description="Split HOV-SG-style data into prefix-traversal subsets.")
    p.add_argument("--data_dir", type=Path, required=True,
                   help="scene dir containing color/ depth/ pose/ intrinsic/ filename_mapping.json")
    p.add_argument("--annotation_file", type=Path, required=True, help="stream_annotations.json")
    p.add_argument("--mode", choices=["each", "prefix"], default="prefix")
    p.add_argument("--out_parent", type=Path, default=None,
                   help="parent dir for splits (default: parent of --data_dir)")
    p.add_argument("--symlink", action="store_true",
                   help="symlink files instead of copying (saves disk space)")
    args = p.parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"data_dir not found: {args.data_dir}")
    if not args.annotation_file.exists():
        raise SystemExit(f"annotation_file not found: {args.annotation_file}")
    split_traversals(args.data_dir, args.annotation_file, args.mode, args.out_parent, args.symlink)


if __name__ == "__main__":
    main()
