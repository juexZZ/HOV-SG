"""Offline-only exact camera-frame scoring; never imported by the mapper."""
import argparse
from pathlib import Path
from hovsg.utils.v2_io import SHARED, RUN, CONFIG, COORD, read, write, sha, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    expected = read(SHARED/'inputs/queries.json')['queries']
    key = lambda x:(x['scene'],x['chapter'],x['query'])
    keys = {key(x) for x in expected}
    predictions, sources = [], []
    for path in sorted((args.run/'predictions').glob('*.json')):
        data = read(path)
        if data['status'] != 'COMPLETE':
            raise ValueError('Only sealed COMPLETE snapshots can be scored')
        predictions.extend(data['predictions'])
        sources.append({'path':str(path),'sha256':sha(path)})
    observed = {key(p) for p in predictions}
    assert len(observed) == len(predictions) and observed <= keys
    # GT first opened here, never in input loading, map building or retrieval.
    gt = {key(x):set(x['gt_frames']) for x in read(SHARED/'evaluation/scoring_gt.json')['queries']}
    for p in predictions:
        p['hit'] = bool(p['in_current_traversal'] and p['frame_name'] in gt[key(p)])
    by_traversal = {str(c):{'hits':sum(p['hit'] for p in predictions if p['chapter']==c),
        'total':sum(p['chapter']==c for p in predictions)} for c in (1,2,3)}
    complete = observed == keys and len(predictions)==198
    if complete:
        assert [by_traversal[str(c)]['total'] for c in (1,2,3)] == [66,67,65]
    write(args.run/'results/per_query_scored.json', {'status':'COMPLETE' if complete else 'PARTIAL',
        'predictions':predictions, 'sources':sources}, replace=True)
    write(args.run/'results/summary.json', {'status':'COMPLETE' if complete else 'PARTIAL',
        'method':'hovsg','config':CONFIG, 'by_traversal':by_traversal,
        'hits':sum(p['hit'] for p in predictions),'total':len(predictions),'expected_total':198,
        'finished':utc(),'protocol':str(COORD/'PROTOCOL.md'),
        'protocol_sha256':sha(COORD/'PROTOCOL.md'), 'frontend':'SAM2 + SigLIP1152',
        'readout_amendment':str(COORD/'HOVSG_READOUT_AMENDMENT.md'),
        'readout_amendment_sha256':sha(COORD/'HOVSG_READOUT_AMENDMENT.md'),
        'geometry':'offline full-sequence DA3; sampled cumulative object mapping',
        'mapping':'single-pass native sequential masks with local causal support; global snapshot feature reduction',
        'non_bitwise_native_batch_variant':True, 'support_mode':'per-frame-local; no future map contributions',
        'configuration':{'support_mode':'local','native_batch_bitwise_equivalent':False,
            'mask_merge':'native sequential','native_default_thresholds':True,
            'sample':'scene-global pair_index%5==0','readout':'native full-prefix visibility'},
        'hierarchy':'native ScanNet object-layer baseline; no floor/room/navigation',
        'readout':'native top1 object + latest depth-consistent visible frame over ALL prefix frames (including unmapped frames)',
        'visibility':{'voxel':.05,'min_visible':20,'depth_tol':.5,'depth_scale':1000.0},
        'actual_contributors':'time-proof and diagnostic only; not the primary frame readout'}, replace=True)


if __name__ == '__main__':
    main()
