"""Frontend-only constructor adapter; ranking/visibility use native methods."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from application.eval_object_retrieval import HovsgRetriever, load_intrinsics
from hovsg.utils.baseline_io import frame_triplets
from hovsg.utils.v2_io import GEOMETRY


class CachedNativeRetriever(HovsgRetriever):
    def __init__(self, scene, chapter, frames, snapshot):
        # Do not call native __init__, which instantiates OpenCLIP. All numeric
        # defaults below are the native evaluator defaults, not tuned settings.
        self.scene_dir = str(snapshot)
        self.data_dir = str(GEOMETRY / scene / 'native')
        self.device = 'cpu'
        self.feat_dim = 1152
        self.vox_size, self.min_visible = 0.05, 20
        self.depth_scale, self.depth_tol = 1000.0, 0.5
        self.mask_feats = torch.load(Path(snapshot)/'mask_feats.pt', map_location='cpu', weights_only=True).float()
        self.mask_feats = torch.nn.functional.normalize(self.mask_feats, p=2, dim=-1).numpy()
        assert self.mask_feats.shape[1] == 1152
        all_triples = frame_triplets(self.data_dir)
        assert len(all_triples) == len(frames)
        triples = [triple for i, triple in enumerate(all_triples) if frames[i]['run'] <= chapter]
        assert [int(Path(t[0]).stem) for t in triples] == [
            i for i, f in enumerate(frames) if f['run'] <= chapter]
        self.color_files, self.depth_files, self.pose_files = map(list, zip(*triples))
        self.K = load_intrinsics(str(Path(self.data_dir)/'intrinsic/intrinsic_depth.txt'))
        self.H, self.W = np.asarray(Image.open(self.depth_files[0])).shape[:2]
        self.idx_to_orig = {i:f['file'] for i,f in enumerate(frames) if f['run'] <= chapter}
        self.chapter = chapter

    def get_text_embedding(self, text):
        raise RuntimeError('Use the sealed shared SigLIP graph query cache; no native OpenCLIP inference')
