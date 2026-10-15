"""Small input/output helpers for the DejaView baseline workflow."""
import hashlib
import json
from pathlib import Path
import subprocess


def frame_triplets(root):
    """Pair color/depth/pose by numeric ID, rejecting incomplete exports."""
    root = Path(root)
    tables = []
    for folder, extensions in (("color", {".jpg", ".jpeg", ".png"}),
                               ("depth", {".png"}), ("pose", {".txt"})):
        table = {}
        for path in (root / folder).iterdir():
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                index = int(path.stem)
            except ValueError as exc:
                raise ValueError(f"Non-numeric frame filename: {path}") from exc
            if index in table:
                raise ValueError(f"Duplicate frame ID {index} in {root / folder}")
            table[index] = str(path)
        tables.append(table)
    if not tables[0]:
        raise ValueError(f"No color frames in {root}")
    if not (tables[0].keys() == tables[1].keys() == tables[2].keys()):
        raise ValueError(f"Frame IDs differ between color/depth/pose in {root}")
    return [tuple(t[i] for t in tables) for i in sorted(tables[0])]


def save_result(path, result, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def provenance(repo, args, inputs):
    """Hash small protocol inputs; record the revision without claiming map origin."""
    repo = Path(repo)
    try:
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"protocol": "dejaview-current-chapter-v1", "code_commit": revision,
            "tracked_worktree_dirty": dirty, "arguments": vars(args),
            "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
            "map_generating_commit": None}
