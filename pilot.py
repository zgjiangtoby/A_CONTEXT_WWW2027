"""Offline synthetic execution; all generated inputs and outputs are temporary."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import anchoring
import a_capacity_grid as grid
import a_policy_controls as policy
import a_natural_probe as natural
import data_io


def main():
    policy.self_check(include_reanchor=True)
    natural.self_check()
    root = Path(__file__).resolve().parent
    examples = [('p'*80+'bad'+'s'*80+' bad', 80, 83),
                ('p'*80+'bad'+'s'*80+' bad', 80, 83),
                ('😀'+'p'*79+'bad'+'s'*80, 80, 83),
                ('😀'+'p'*31+'q'+'s'*31+'😀', 32, 33)]
    with tempfile.TemporaryDirectory(prefix='reanchoring_smoke_') as temporary:
        out = Path(temporary)
        documents, annotations = [], []
        for i, (text, start, end) in enumerate(examples):
            identifier = f'synthetic-{i}'
            key = identifier if i else 'train_'+hashlib.sha256(identifier.encode()).hexdigest()[:12]
            documents.append(dict(id=identifier, text=text,
                                  hate_phrases=[dict(char_pos=f'{start}-{end-1}', content=text[start:end])]))
            shown, support = anchoring.normalization_trace(text, 'text/plain')
            a, b = anchoring.project_display(support, start, end)
            annotations.append(dict(unit_index=i, post_key=key, annotation_index=0,
                raw_start=start, raw_end=end, old_start=a, old_end=b, old_post_length=len(shown),
                natural_candidates=len(anchoring.occurrences(shown, shown[a:b]))))
        source, labels = out/'original.jsonl', out/'annotations.csv'
        source.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in documents), encoding='utf-8')
        anchoring.write_csv(labels, annotations)
        common = ['--input', str(source), '--annotations', str(labels), '--limit', '4']
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        for script, name, extra in [
            ('a_capacity_grid.py', 'grid', ['--corpus', 'tama']),
            ('a_policy_controls.py', 'policy', ['--corpus', 'tama', '--include-reanchor']),
            ('a_fine_boundary.py', 'fine', ['--grid-run', str(out/'grid'), '--policy-run', str(out/'policy')]),
        ]:
            subprocess.run([sys.executable, str(root/script), *common, *extra,
                            '--out', str(out/name)], check=True, env=env)
            manifest = data_io.completed(out/name, allow_pilot=True)
            assert manifest['pilot']
        with np.load(out/'grid/outcomes.npz') as a, np.load(out/'policy/outcomes.npz') as b:
            assert np.array_equal(a['outcome'][:, :, grid.GRID.index(32)], b['outcome'][:, :, 0])
            assert b['reanchor_eligible'].tolist() == [True, True, True, False]
        records = data_io.rows(out/'grid/records.csv')
        assert records[0]['exact_text_group'] == records[1]['exact_text_group']
        assert sum(int(r['common_panel']) for r in records) == 3
        annotations[0]['raw_end'] += 1
        bad_labels = out/'invalid_annotations.csv'
        anchoring.write_csv(bad_labels, annotations)
        try:
            data_io.load('tama', source, bad_labels)
        except AssertionError:
            pass
        else:
            raise AssertionError('Altered source annotation was accepted')
    print('PASS: offline synthetic grid/policy/fine/natural; exact-text groups; native eligibility; no retained data')


if __name__ == '__main__':
    main()
