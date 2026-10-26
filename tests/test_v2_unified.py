import copy
import unittest
import tempfile
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from hovsg.models.v2_cached_features import compact_overlap_features
from hovsg.graph.v2_stream import StreamMap, cloud, fuse_masks, last_pixel_per_support
from hovsg.utils.graph_utils import merge_3d_masks
from hovsg.utils.v2_io import load_inputs
from hovsg.graph.v2_retrieval import CachedNativeRetriever
from application.eval_object_retrieval import HovsgRetriever
from unittest.mock import patch
from pathlib import Path


class UnifiedTests(unittest.TestCase):
    def test_overlap_table_matches_native_pixel_sum_with_background(self):
        masks = np.zeros((3,4,5), dtype=bool)
        masks[0,:2,:3] = True
        masks[1,1:3,1:4] = True
        masks[2,2:,:2] = True
        embeddings = np.random.RandomState(7).randn(3,1152).astype(np.float32)
        ids, table = compact_overlap_features(masks, embeddings, (4,5), 'cpu')
        dense = torch.zeros((4,5,1152))
        for mask, vec in zip(masks, embeddings):
            dense[mask] += torch.from_numpy(vec)
        dense = torch.nn.functional.normalize(dense, p=2, dim=-1).half().numpy()
        np.testing.assert_array_equal(table[ids], dense)
        np.testing.assert_array_equal(table[ids][~masks.any(0)], 0)

    def test_no_objects(self):
        ids, table = compact_overlap_features([], [], (3,4), 'cpu')
        self.assertEqual(table.shape, (1,1152))
        self.assertEqual(ids.sum(), 0)
        self.assertEqual(table.sum(), 0)

    def test_duplicate_point_assignment_matches_serial_native(self):
        torch.set_num_threads(1)
        ix = np.array([0,1,0,2,1])
        fid = np.array([1,2,3,4,5])
        out, count = last_pixel_per_support(ix, fid, 4)
        ref = torch.zeros(4, dtype=torch.int64)
        cnt = torch.zeros(4)
        ref[ix] += torch.from_numpy(fid)
        cnt[ix] += 1
        np.testing.assert_array_equal(out, ref.numpy())
        np.testing.assert_array_equal(count, cnt.numpy())
        torch.set_num_threads(4)

    def test_provenance_path_preserves_native_merger(self):
        rng = np.random.RandomState(3)
        p = rng.rand(120,3)*0.02
        masks = [cloud(p, np.ones_like(p)), cloud(p+0.001, np.ones_like(p)),
                 cloud(p+2, np.ones_like(p))]
        cfg = OmegaConf.create({'init_overlap_thresh':.75,'voxel_size':.02,'iou_thresh':.05})
        original = merge_3d_masks(copy.deepcopy(masks), .75, .02, .05)
        updated, tags = fuse_masks(copy.deepcopy(masks), [[(0,0)],[(1,0)],[(10,0)]], cfg)
        self.assertEqual(len(original),len(updated))
        self.assertEqual(tags,[[(0,0),(1,0)],[(10,0)]])
        for a,b in zip(original,updated):
            np.testing.assert_array_equal(np.asarray(a.points),np.asarray(b.points))
            np.testing.assert_array_equal(np.asarray(a.colors),np.asarray(b.colors))

    def test_real_input_sampling_contract(self):
        frames, selected, provenance = load_inputs('2metro8floor')
        self.assertEqual(len(frames),928)
        self.assertEqual(len(selected),186)
        self.assertEqual(selected[:4],[0,1,10,11])
        self.assertEqual(len(provenance),5)

    def test_native_visibility_methods_inherited_and_prefix_only(self):
        self.assertIs(CachedNativeRetriever.get_top_t_latest_frames, HovsgRetriever.get_top_t_latest_frames)
        self.assertIs(CachedNativeRetriever.query_top_instance, HovsgRetriever.query_top_instance)
        frames, selected, _ = load_inputs('2metro8floor')
        root = Path('/root/dejaview_experiments/runs/hovsg-unified-v2-20261026-r01/smoke/2metro8floor/t1')
        if not (root/'complete.json').exists():
            self.skipTest('Real smoke snapshot not created yet')
        with patch('open_clip.create_model_and_transforms', side_effect=AssertionError('must not load OpenCLIP')):
            retriever = CachedNativeRetriever('2metro8floor',1,frames,root)
        self.assertEqual(len(retriever.pose_files),286)
        self.assertGreater(len(retriever.pose_files),sum(frames[i]['run']==1 for i in selected))
        self.assertEqual(int(Path(retriever.pose_files[-1]).stem),285)
        self.assertEqual((retriever.vox_size,retriever.min_visible,retriever.depth_tol,retriever.depth_scale),(.05,20,.5,1000.))

    def test_stream_checkpoint_restores_geometry_tags_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StreamMap.__new__(StreamMap)
            state.root, state.scene = Path(tmp), 'test'
            state.frames, state.selected = [{'run':1},{'run':2}], [0,1]
            state.processed, state.observations = [0], ['already_associated_local_observation.npz']
            p = np.array([[1.,2.,3.],[4.,5.,6.]])
            state.masks, state.contributors = [cloud(p,np.ones_like(p))], [[(0,0)]]
            state.provenance, state.source_cfg_sha = {'test':'same'}, 'config-test'
            state.save_checkpoint(1)
            expected = (random.random(),np.random.rand(),torch.rand(3))
            state.masks, state.contributors, state.processed = [], [], []
            state.load_checkpoint(1)
            self.assertEqual(state.processed,[0])
            self.assertEqual(state.contributors,[[(0,0)]])
            np.testing.assert_array_equal(np.asarray(state.masks[0].points),p)
            self.assertEqual(random.random(),expected[0])
            self.assertEqual(np.random.rand(),expected[1])
            torch.testing.assert_close(torch.rand(3),expected[2],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
