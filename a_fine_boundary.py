"""Frozen R20 fine-width/two-edit-length follow-up on the existing TAMA panel."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import a_capacity_grid as grid
import a_policy_controls as policy
import data_io
from data_io import completed, rows

WIDTHS = (20, 21, 22, 24, 25, 26, 28, 32)
LENGTHS = (1, 5)
CELLS = tuple((direction, length, width) for direction in ("prepend", "append")
              for length in LENGTHS for width in WIDTHS)
METHODS = policy.POLICIES


def make_edit(doc, span, direction, length):
    """Transport original characters through copy and marker insertions."""
    geometry = grid.copy_geometry(doc, span, 20)
    old = doc['old']
    current, mapping = old, [(i, i+1) for i in range(len(old))]
    operations = []

    def insert(kind, at, value):
        nonlocal current, mapping
        operations.append(dict(kind=kind, at_cp=at,
                               at_u16=grid.ref.unit_boundaries(current)[at],
                               length_cp=len(value), length_u16=len(grid.ref.utf16(value))))
        current, next_map = grid.base.insert_text(current, at, value)
        mapping = grid.base.compose_maps(mapping, next_map)

    copied = old[geometry['copy_start_cp']:geometry['copy_end_cp']]
    insert('copy_then_space' if direction == 'prepend' else 'space_then_copy',
           0 if direction == 'prepend' else len(current),
           copied+' ' if direction == 'prepend' else ' '+copied)
    left, right = grid.base.interval_image(mapping, *span)
    insert('neighbor_hashes', right, '#'*length)
    insert('neighbor_hashes', left, '#'*length)
    shown, support = grid.base.normalization_trace(current, 'text/plain')
    truth = grid.base.project_display(support, *grid.base.interval_image(mapping, *span))
    assert truth and shown[slice(*truth)] == old[slice(*span)]
    return shown, truth, dict(**geometry, operations_json=json.dumps(operations),
                             storage_sha256=grid.text_digest(current), shown_sha256=grid.text_digest(shown))


def summarize(records, outcomes, stages, reps=grid.REPS):
    groups = sorted({r['exact_text_group'] for r in records})
    ids = np.array([groups.index(r['exact_text_group']) for r in records])
    counts = np.bincount(ids, minlength=len(groups))
    rng = np.random.default_rng(grid.SEED)
    weights = np.stack([np.bincount(rng.integers(len(groups), size=len(groups)),
                                   minlength=len(groups)) for _ in range(reps)])
    draw_n = weights @ counts

    def estimate(value, ci=False):
        total = np.bincount(ids, weights=value, minlength=len(groups))
        fractions = total/counts
        result = dict(annotation=float(total.sum()/len(records)), equal_document=float(fractions.mean()))
        if ci:
            for name, values in [('annotation', (weights @ total)/draw_n),
                                 ('equal_document', (weights @ fractions)/len(groups))]:
                result[name+'_lo'], result[name+'_hi'] = (float(x) for x in np.quantile(values, [.025,.975]))
        return result

    summary, pairs, transitions = [], [], []
    for j, (direction, length, width) in enumerate(CELLS):
        key = dict(cell_index=j, direction=direction, copy_radius_u16=20,
                   marker_length=length, saved_width_u16=width,
                   n=len(records), groups=len(groups))
        for k, method in enumerate(METHODS):
            row = dict(**key, method=method)
            for code, name in enumerate(grid.OUTCOMES):
                value = outcomes[:,j,k] == code
                row[name+'_n'] = int(value.sum())
                row.update({name+'_'+w:v for w,v in estimate(value).items()})
            row['native_bypass_n'] = int((stages[:,j] == 0).sum())
            assert sum(row[name+'_n'] for name in grid.OUTCOMES) == len(records)
            summary.append(row)
        for a,b in [(0,1),(1,2),(0,2)]:
            for x,name_x in enumerate(grid.OUTCOMES):
                for y,name_y in enumerate(grid.OUTCOMES):
                    value = (outcomes[:,j,a] == x) & (outcomes[:,j,b] == y)
                    transitions.append(dict(**key, source_method=METHODS[a], target_method=METHODS[b],
                        source_outcome=name_x, target_outcome=name_y, count=int(value.sum())))
    contrasts = []
    for direction in ('prepend','append'):
        for width in WIDTHS:
            contrasts.append(('K1_minus_K5', (direction,5,width), (direction,1,width)))
        for length in LENGTHS:
            contrasts.append(('S32_minus_S20', (direction,length,20), (direction,length,32)))
    for name,a,b in contrasts:
        first = outcomes[:,CELLS.index(a),0] == 0
        second = outcomes[:,CELLS.index(b),0] == 0
        pairs.append(dict(contrast=name, direction=a[0], old_marker_length=a[1], new_marker_length=b[1],
            old_width_u16=a[2], new_width_u16=b[2], n=len(records), groups=len(groups),
            bootstrap_replicates=reps, bootstrap_pool_groups=len(groups), bootstrap_seed=grid.SEED,
            repairs=int((~first & second).sum()), breakages=int((first & ~second).sum()),
            **estimate(second.astype(float)-first.astype(float), True)))
    return summary,pairs,transitions


def self_check():
    for text, span in [('a'*64+'Q'+'b'*64,(64,65)), ('😀 '+'a'*64+'Q'+'b'*64+' 😀',(66,67))]:
        doc = grid.document(text)
        for direction in ('prepend','append'):
            ours = make_edit(doc,span,direction,5)
            reference = grid.make_edit(doc,span,direction,20)
            assert ours[:2] == reference[:2]
            assert ours[2]['storage_sha256'] == reference[2]['storage_sha256']
            shown,truth,_ = make_edit(doc,span,direction,1)
            expected = span[0]+1+(len(doc['old'][grid.copy_geometry(doc,span,20)['copy_start_cp']:
                                                           grid.copy_geometry(doc,span,20)['copy_end_cp']])+1
                                       if direction=='prepend' else 0)
            assert truth == (expected,expected+1) and shown[slice(*truth)] == 'Q'
    doc = grid.document('a'*64+'Q'+'b'*64)
    shown,truth,_ = make_edit(doc,(64,65),'prepend',5)
    answers,stage,hits,components = policy.policy_answers(shown,'Q','a'*21,'b'*21,(64,65))
    assert stage == 1 and answers[0] != truth
    delta = 20*(components[hits.index(answers[0]),:2].sum()-components[hits.index(truth),:2].sum())
    assert abs(delta-160/21) < 1e-12 and delta > 2
    records = [dict(exact_text_group=g) for g in ('a','a','b')]
    out = np.zeros((3,len(CELLS),3),np.uint8)
    out[0,CELLS.index(('prepend',5,20)),0] = 2
    tables = summarize(records,out,np.ones(out.shape[:2],np.int8),reps=20)
    row = next(r for r in tables[1] if r['contrast']=='K1_minus_K5' and r['direction']=='prepend'
               and r['old_width_u16']==20)
    assert row['annotation']==1/3 and row['equal_document']==.25 and row['repairs']==1
    print('PASS: transported truth, K5 geometry parity, fine-width counterexample, grouped estimator')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_io.arguments(parser)
    parser.add_argument('--grid-run', type=Path)
    parser.add_argument('--policy-run', type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--self-check',action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    assert args.out and args.limit >= 0
    out = data_io.fresh_output(args.out)
    assert args.grid_run and args.policy_run, 'Both completed input banks are required'
    start = time.time()
    old_grid, old_policy = args.grid_run.resolve(), args.policy_run.resolve()
    manifests = [completed(old_grid, bool(args.limit)), completed(old_policy, bool(args.limit))]
    assert all(m['corpus'] == 'tama' for m in manifests)
    assert manifests[0]['n'] == manifests[1]['n']
    raw,texts,annotations = data_io.load('tama', args.input, args.annotations)
    all_records = rows(old_grid/'records.csv')
    assert len(all_records) == len(annotations)
    for record, annotation in zip(all_records, annotations):
        assert record['post_key'] == annotation['post_key']
        assert int(record['unit_index']) == int(annotation['unit_index'])
        assert record['exact_text_group'] == grid.text_digest(texts[record['post_key']])
        assert (int(record['raw_start_cp']), int(record['raw_end_cp'])) == (int(annotation['raw_start']), int(annotation['raw_end']))
        assert (int(record['old_start_cp']), int(record['old_end_cp'])) == (int(annotation['old_start']), int(annotation['old_end']))
    records = [r for r in all_records if int(r['common_panel'])]
    if not args.limit:
        assert len(records)==688 and len({r['exact_text_group'] for r in records})==457
    if args.limit:
        records = records[:args.limit]
    inputs = [Path(__file__), Path(grid.__file__), Path(policy.__file__), Path(data_io.__file__),
              old_grid/'manifest.json',old_policy/'manifest.json',
              old_grid/'outcomes.npz',old_policy/'outcomes.npz',old_grid/'records.csv',
              raw, args.annotations, Path(grid.ref.__file__)]
    input_hashes = {str(p.resolve()):grid.digest(p) for p in inputs}
    out.mkdir(parents=True)
    manifest = dict(status='running',pilot=bool(args.limit),corpus='tama',n=len(records),
        cells=[dict(cell_index=j,direction=d,marker_length=k,saved_width_u16=s,copy_radius_u16=20)
               for j,(d,k,s) in enumerate(CELLS)],methods=list(METHODS),
        input_sha256=input_hashes,bootstrap_replicates=grid.REPS,bootstrap_seed=grid.SEED,
        bootstrap_pool='contributing exact-original-text groups of the fixed TAMA common panel',
        outcome_codes={0:'correct',1:'abstain',2:'wrong'},training=False,new_annotations=False)
    grid.base.write_json(out/'manifest.json',manifest)
    n = len(records)
    codes = np.ones((n,len(CELLS),3),np.uint8)
    pred_cp = np.full((*codes.shape,2),-1,np.int32)
    pred_u16 = pred_cp.copy()
    truth_cp = np.empty((n,len(CELLS),2),np.int32)
    truth_u16 = np.empty_like(truth_cp)
    stage = np.empty((n,len(CELLS)),np.int8)
    selected = np.full((*codes.shape,3),np.nan)
    gold = np.empty((n,len(CELLS),3))
    recipes,cache = [],{}
    with np.load(old_grid/'outcomes.npz') as prior, np.load(old_policy/'outcomes.npz') as controls:
        for i,record in enumerate(records):
            unit = int(record['unit_index'])
            ann = annotations[unit]
            assert ann['post_key']==record['post_key']
            if ann['post_key'] not in cache:
                cache[ann['post_key']] = grid.document(texts[ann['post_key']])
            doc = cache[ann['post_key']]
            assert doc['group']==record['exact_text_group']
            span = (int(record['old_start_cp']),int(record['old_end_cp']))
            old = tuple(doc['bounds'][v] for v in span)
            quote = doc['units'][slice(*old)]
            for direction in ('prepend','append'):
                old_j = grid.EDITS.index((direction,20))
                for length in LENGTHS:
                    shown,truth,recipe = make_edit(doc,span,direction,length)
                    bounds = grid.ref.unit_boundaries(shown)
                    reverse = {u:cp for cp,u in enumerate(bounds)}
                    target = tuple(bounds[v] for v in truth)
                    recipes.append(dict(panel_index=i,unit_index=unit,direction=direction,
                        marker_length=length,truth_start_cp=truth[0],truth_end_cp=truth[1],**recipe))
                    for width in WIDTHS:
                        j = CELLS.index((direction,length,width))
                        prefix,suffix = doc['units'][old[0]-width:old[0]],doc['units'][old[1]:old[1]+width]
                        record[f'left_{width}_partial_surrogate'] = int(grid.partial_surrogate(prefix))
                        record[f'right_{width}_partial_surrogate'] = int(grid.partial_surrogate(suffix))
                        assert len(prefix)==len(suffix)==width
                        answers,stage[i,j],hits,components = policy.policy_answers(
                            grid.ref.utf16(shown),quote,prefix,suffix,old)
                        gold[i,j] = components[hits.index(target)]
                        truth_cp[i,j],truth_u16[i,j] = truth,target
                        for k,answer in enumerate(answers):
                            codes[i,j,k] = policy.outcome(answer,target)
                            if answer is not None:
                                pred_u16[i,j,k] = answer
                                pred_cp[i,j,k] = (reverse[answer[0]],reverse[answer[1]])
                                selected[i,j,k] = components[hits.index(answer)]
                        if length==5 and width in (20,32):
                            old_s = grid.GRID.index(width)
                            assert codes[i,j,0] == prior['outcome'][unit,old_j,old_s]
                            assert np.array_equal(pred_cp[i,j,0],prior['predicted_cp'][unit,old_j,old_s])
                            assert np.array_equal(pred_u16[i,j,0],prior['predicted_u16'][unit,old_j,old_s])
                            assert stage[i,j] == prior['stage'][unit,old_j,old_s]
                            assert np.array_equal(truth_cp[i,j],prior['truth_cp'][unit,old_j])
                        if length==5 and width==32:
                            assert np.array_equal(codes[i,j],controls['outcome'][unit,old_j,:3])
                            assert np.array_equal(pred_cp[i,j],controls['predicted_cp'][unit,old_j,:3])
            if (i+1)%100==0:
                print(f'{i+1}/{n}',flush=True)
    assert np.array_equal(codes==0,np.all(pred_cp==truth_cp[:,:,None,:],axis=-1))
    assert np.array_equal(codes==1,np.all(pred_cp==-1,axis=-1))
    np.savez_compressed(out/'outcomes.npz',outcome=codes,predicted_cp=pred_cp,predicted_u16=pred_u16,
                        truth_cp=truth_cp,truth_u16=truth_u16,native_stage=stage,
                        selected_components=selected,gold_components=gold)
    grid.base.write_csv(out/'records.csv',records)
    grid.base.write_csv(out/'edit_metadata.csv',recipes)
    for name,data in zip(('summary','paired_intervals','transitions'),summarize(records,codes,stage)):
        grid.base.write_csv(out/(name+'.csv'),data)
    assert all(grid.digest(p)==sha for p,sha in input_hashes.items())
    manifest.update(status='completed',groups=len({r['exact_text_group'] for r in records}),
        legacy_grid_policy_parity=True,elapsed_seconds=time.time()-start,
        output_sha256={p.name:grid.digest(p) for p in out.iterdir() if p.name!='manifest.json'})
    grid.base.write_json(out/'manifest.json',manifest)
    print(f'DONE: {n} intervals, {len(CELLS)} cells, {len(METHODS)} policies')


if __name__=='__main__':
    main()
