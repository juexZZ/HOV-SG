import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "application"))
import numpy as np
import torch
from hovsg.dataloader.scannet import ScannetDataset
from hovsg.utils.baseline_io import frame_triplets, save_result
from hovsg.utils.clip_utils import get_text_feats
from split_chapter_data import _copy_split, split_traversals
from query_latest_frame import visibility_count


def scene(root, ids):
    for folder, ext in (("color", "jpg"), ("depth", "png"), ("pose", "txt")):
        (root / folder).mkdir(parents=True)
        for index in ids:
            (root / folder / f"{index}.{ext}").touch()
    (root / "intrinsic").mkdir()


class BaselineTests(unittest.TestCase):
    def test_real_loader_numeric_order_and_exact_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene(root, range(20))
            loader = ScannetDataset.__new__(ScannetDataset)
            loader.root_dir = str(root)
            rows = loader._get_data_list()
            self.assertEqual([Path(r[0]).stem for r in rows[::10]], ["0", "10"])
            (root / "pose/19.txt").rename(root / "pose/20.txt")
            with self.assertRaisesRegex(ValueError, "Frame IDs differ"):
                frame_triplets(root)

    def test_duplicate_numeric_frame_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene(root, [1])
            (root / "color/01.jpg").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                frame_triplets(root)

    def test_visibility_counts_only_points_in_image(self):
        points = np.array([[1., 1., 1.], [100., 1., 1.], [1., 1., -1.]])
        self.assertEqual(visibility_count(points, np.eye(4), np.eye(3), 10, 10), 1)
        self.assertEqual(visibility_count(points, np.eye(4), np.eye(3), 10, 10, depth=np.ones((10, 10)) * 2000), 0)

    def test_split_mapping_excludes_missing_frames_and_preserves_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dst = root / "src", root / "dst"
            scene(src, ["0001", "0002"])
            (src / "depth/0002.png").unlink()
            mapping = [{"index": 1, "original": "left_a.png"}, {"index": 2, "original": "left_b.png"}]
            self.assertEqual(_copy_split(src, dst, {1, 2}, mapping, False), (1, 1))
            self.assertTrue((dst / "color/0001.jpg").exists())
            self.assertEqual(json.loads((dst / "filename_mapping.json").read_text()), mapping[:1])
            with self.assertRaises(FileExistsError):
                _copy_split(src, dst, {1, 2}, mapping, False)

    def test_single_traversal_each_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            scene(src, [0])
            (src / "filename_mapping.json").write_text('[{"index": 0, "original": "left_0.png"}]')
            ann = root / "ann.json"
            ann.write_text('{"images": [{"file": "left_0.png", "new_traversal": true}]}')
            split_traversals(src, ann, "each")
            self.assertTrue((root / "src_trav_1/color/0.jpg").exists())

    def test_text_features_use_cpu_model_device(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
            def encode_text(self, tokens):
                assert tokens.device.type == "cpu"
                return torch.ones((len(tokens), 2))
        with patch("hovsg.utils.clip_utils.open_clip.tokenize", return_value=torch.zeros((1, 4), dtype=torch.long)):
            feats = get_text_feats(["chair"], Model(), 2)
        np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.)

    def test_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            save_result(path, {"v": 1})
            with self.assertRaises(FileExistsError):
                save_result(path, {"v": 2})
            self.assertEqual(json.loads(path.read_text()), {"v": 1})


    def test_run_manifest_records_failure_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "run.json"
            command = [sys.executable, str(ROOT / "application/record_run.py"), "--manifest", str(manifest), "--", sys.executable, "-c", "raise SystemExit(3)"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3)
            first = manifest.read_bytes()
            self.assertEqual(json.loads(first)["status"], "failed")
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(manifest.read_bytes(), first)


    def test_runners_route_same_scene_and_refuse_existing_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "python"
            log = root / "args.jsonl"
            fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ['ARG_LOG'], 'a') as f: print(json.dumps(sys.argv[1:]), file=f)\n")
            fake.chmod(0o755)
            env = {**os.environ, "PYTHON": str(fake), "ARG_LOG": str(log), "DATASET_ROOT": str(root / "input data"), "SAVE_ROOT": str(root / "maps"), "RUN_ID": "smoke", "ANNOTATION_ROOT": str(root / "annotations"), "OBJECT_LIST_ROOT": str(root / "queries")}
            for script in ("run.sh", "run_evaluator.sh"):
                subprocess.run(["bash", str(ROOT / script), "pantry1"], env=env, cwd=tmp, check=True)
            mapping, evaluation = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertIn("main.scene_id=pantry1", mapping)
            self.assertIn("main.dataset_path=" + env["DATASET_ROOT"], mapping)
            scene_dir = root / "maps/scannet/pantry1"
            self.assertEqual(evaluation[evaluation.index("--scene_dir") + 1], str(scene_dir))
            scene_dir.mkdir(parents=True)
            retry = subprocess.run(["bash", str(ROOT / "run.sh"), "pantry1"], env=env, cwd=tmp, capture_output=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(len(log.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
