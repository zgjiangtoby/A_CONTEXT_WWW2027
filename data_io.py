"""External corpus/annotation adapters; no bundled data or inferred partitions."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import anchoring


def arguments(parser):
    parser.add_argument('--input', type=Path, help='External original TAMA JSONL, Toxic test CSV, or PTC archive')
    parser.add_argument('--annotations', type=Path, help='External frozen annotation CSV; see README schema')


def hashes(*paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def fresh_output(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parent
    if path.is_relative_to(root) or path.exists():
        raise ValueError('Output must be a fresh directory outside the source repository')
    return path


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def completed(path, allow_pilot=False):
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and (allow_pilot or not manifest['pilot']), path
    for name, expected in manifest['output_sha256'].items():
        assert Path(name).name == name, 'Output hash names must be basenames'
        assert hashes(path / name)[str((path / name).resolve())] == expected, (path, name)
    return manifest


def contiguous(indices, text_length):
    assert isinstance(indices, list)
    assert all(type(i) is int and 0 <= i < text_length for i in indices)
    spans = []
    for index in sorted(set(indices)):
        if spans and spans[-1][1] == index:
            spans[-1] = (spans[-1][0], index + 1)
        else:
            spans.append((index, index + 1))
    return spans


def load(corpus, input_path, annotation_path):
    if input_path is None or annotation_path is None:
        raise ValueError('--input and --annotations must identify external source files')
    input_path, annotation_path = Path(input_path), Path(annotation_path)
    annotations = rows(annotation_path)
    required = {'unit_index', 'post_key', 'annotation_index', 'raw_start', 'raw_end',
                'old_start', 'old_end', 'old_post_length', 'natural_candidates'}
    assert annotations and required <= annotations[0].keys(), 'Missing annotation columns'
    texts, labels = {}, {}
    if corpus == 'tama':
        originals, suffixes = {}, {}
        with input_path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row['id'])
                value = (row['text'], row.get('hate_phrases', []))
                assert isinstance(value[0], str) and isinstance(value[1], list)
                assert key not in originals or originals[key] == value, 'Conflicting duplicate ID'
                originals.setdefault(key, value)
        for key in originals:
            suffix = hashlib.sha256(key.encode()).hexdigest()[:12]
            assert suffix not in suffixes, 'Ambiguous original-ID hash suffix'
            suffixes[suffix] = key
        for annotation in annotations:
            key = annotation['post_key']
            source = key if key in originals else suffixes.get(key.rsplit('_', 1)[-1])
            assert source in originals, ('Missing original ID for frozen key', key)
            texts[key], phrases = originals[source]
            spans = []
            for phrase in phrases:
                parts = str(phrase.get('char_pos', '')).split('-')
                span = tuple(map(int, parts)) if len(parts) == 2 and all(p.isdigit() for p in parts) else None
                spans.append(None if span is None else (span[0], span[1] + 1, phrase.get('content')))
            labels[key] = spans
    elif corpus == 'toxic':
        for i, row in enumerate(rows(input_path)):
            key, text = f'test:{i}', row['text']
            texts[key] = text
            labels[key] = [(a, b, text[a:b]) for a, b in contiguous(ast.literal_eval(row['spans']), len(text))]
    elif corpus == 'ptc':
        with tarfile.open(input_path) as archive:
            members = {m.name: m for m in archive.getmembers() if m.isfile()}
            for name in sorted(members):
                if not name.startswith('datasets/train-articles/') or not name.endswith('.txt'):
                    continue
                key = Path(name).stem
                text = archive.extractfile(members[name]).read().decode('utf-8')
                label = f'datasets/train-labels-task2-technique-classification/{key}.task2-TC.labels'
                spans = []
                for line in archive.extractfile(members[label]).read().decode('utf-8').splitlines():
                    if line.strip():
                        article_id, _, start, end = line.split('\t')
                        assert key == 'article' + article_id
                        spans.append((int(start), int(end)))
                texts[key] = text
                labels[key] = [(a, b, text[a:b]) for a, b in sorted(set(spans))]
    else:
        raise ValueError('Unknown corpus')
    for i, row in enumerate(annotations):
        assert int(row['unit_index']) == i, 'Frozen unit order must be contiguous'
        text = texts[row['post_key']]
        start, end = int(row['raw_start']), int(row['raw_end'])
        assert 0 <= start < end <= len(text)
        index = int(row['annotation_index'])
        source_labels = labels[row['post_key']]
        assert 0 <= index < len(source_labels)
        assert source_labels[index] == (start, end, text[start:end]), 'Not an existing source annotation'
        old, support = anchoring.normalization_trace(text, 'text/plain')
        span = (int(row['old_start']), int(row['old_end']))
        assert anchoring.project_display(support, start, end) == span
        assert len(old) == int(row['old_post_length'])
        assert len(anchoring.occurrences(old, old[slice(*span)])) == int(row['natural_candidates'])
    return input_path, texts, annotations
