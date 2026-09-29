# Reanchoring Web Evidence

CPU implementation of retained-neighborhood capacity, matched Hypothesis policy interventions, fine-boundary controls, and canonical natural-revision probes. No classifier is trained. The repository contains source only: obtain all corpus files and frozen annotation maps separately, and write results outside this directory.

## Install and check

Use Python 3.11; the experiments used NumPy 1.26.4.

```bash
python3 -m pip install -r requirements.txt
bash run.sh smoke
```

Set `PYTHON=/path/to/python` when needed. Smoke runs offline on CPU, generates synthetic inputs in a temporary directory, executes the actual grid/policy/fine CLI paths, and removes the temporary files. Synthetic checks are not paper results. Native reanchor is included as five pinned Python files; no installation or download is needed for it.

## External data interface

`grid`, `policy`, and `fine` require both the original corpus file (`--input`) and its frozen existing-annotation CSV (`--annotations`). The CSV is an input connection to the original experiment bank, not a new annotation exercise. Preserve its row order and repeated TAMA annotations. No train/dev/test assignments are inferred or reconstructed.

| Corpus | `--input` | Existing annotation definition |
|---|---|---|
| `tama` | Authorized original JSONL with `id`, `text`, `hate_phrases` | Each phrase retains its `annotation_index`; `char_pos` is inclusive `start-end` and `content` must equal that raw substring. Duplicate identical IDs are deduplicated. Invalid coordinate/content pairs are absent from the frozen map. |
| `toxic` | Released Toxic Spans `tsd_test.csv` with `text`, `spans` | Maximal contiguous runs of marked character indices; unmarked gaps remain gaps. `post_key` is `test:` followed by the zero-based CSV row index. |
| `ptc` | PTC released `datasets-v2.tgz` | Sorted `datasets/train-articles/*.txt`; distinct sorted coordinate pairs from corresponding training TC labels. `post_key` is the article filename stem. Training membership identifies the location corpus; no classifier is fitted. |

Required annotation CSV columns:

```text
unit_index,post_key,annotation_index,raw_start,raw_end,old_start,old_end,old_post_length,natural_candidates
```

All intervals are half-open Unicode code-point offsets. `unit_index` is the contiguous frozen row index; `annotation_index` is the source phrase/run index (PTC: index after sorting distinct coordinates). `old_start/end` project the raw interval through whitespace-run collapse to one ordinary space, with no trimming, case folding, or Unicode compatibility conversion. `old_post_length` is that normalized text's code-point length; `natural_candidates` counts overlapping exact occurrences of its designated quote. Additional CSV columns are ignored.

Use the experiment's existing annotation export. If reproducing that export in another system, derive only these coordinates and counts from the existing source annotations, and retain the original source/annotation order. TAMA's `post_key` may be its original ID or an existing frozen key ending in `SHA256(str(original_id))[:12]`; the loader resolves that suffix without reading partition or relationship metadata. It validates every supplied interval against the original label and normalization. The repository supplies no frozen ID maps, splits, source text, sample data, or results.

Data sources: [Toxic Spans pinned release](https://github.com/ipavlopoulos/toxic_spans/tree/beaf170538df8275c3447b2e26070f2f82e55272/SemEval2021), [PTC release](https://zenodo.org/records/3952415), and [PEER WikiAtomicSample release](https://zenodo.org/records/4478267). TAMA must be obtained through its authorized distribution. Existing source licenses/access terms remain applicable.

## Run controlled experiments

The following paths are placeholders for external files and fresh external output directories:

```bash
bash run.sh grid --corpus tama --input /external/tama/data.jsonl \
  --annotations /external/annotations/tama.csv --out /external/results/grid_tama
bash run.sh policy --corpus tama --input /external/tama/data.jsonl \
  --annotations /external/annotations/tama.csv --include-reanchor \
  --out /external/results/policy_tama
bash run.sh fine --input /external/tama/data.jsonl \
  --annotations /external/annotations/tama.csv \
  --grid-run /external/results/grid_tama --policy-run /external/results/policy_tama \
  --out /external/results/fine_tama
```

Run `grid` and `policy` with `--corpus toxic` or `--corpus ptc` and their corresponding inputs/maps for the other two banks. `--limit 4` makes a connector test/pilot; its manifest is marked `pilot=true`. Full controlled runs enforce the published interval/common-panel/group counts: TAMA 5547/688/457, Toxic 1850/226/192, PTC 6087/5807/353. Fine runs use the fixed 688-interval TAMA panel and verify their overlapping conditions against the supplied grid/policy outcomes. A limited fine run may use pilot banks. Existing output directories are never overwritten.

The grid uses R,S = 8,20,32,48,64 UTF-16 units, both copy directions, and the no-copy control. Policy comparisons fix S=32: the original Hypothesis reference, forced scoring, and forced scoring without the position-score term. `--include-reanchor` adds the distinct native matcher on losslessly decodable shared fields. Fine controls use R=20, K=1/5, and S=20,21,22,24,25,26,28,32. Shared paired bootstrap draws use 2000 replicates and seed 20260925. Grouping uses exact original text; fine resamples its fixed contributing groups.

Each run writes text-free coordinate arrays, outcome counts, paired intervals, and a manifest of actual input/source hashes. The grid produces Figure 2's measurements; fine produces Figure 3's measurements. Policy output includes repairs/breakages, score components, and native metadata. Records ineligible for the native Unicode interface are not counted as abstentions. The selector receives revised text and saved fields; evaluation provenance is separate.

## Natural revisions

Extract the released `WikiAtomicSample/insertions_deletions.jsonl` and `splits/test.txt` outside this repository. Both files are checked against their frozen release hashes; only the released test indices are evaluated.

```bash
bash run.sh natural --prepare --input /external/WikiAtomicSample/insertions_deletions.jsonl \
  --test-indices /external/WikiAtomicSample/splits/test.txt
bash run.sh natural --full --input /external/WikiAtomicSample/insertions_deletions.jsonl \
  --test-indices /external/WikiAtomicSample/splits/test.txt \
  --out /external/results/natural
```

`--prepare` checks unique aligned atomic edits and canonical three-token probe geometry without invoking selectors. `--pilot --pilot-revisions 32` evaluates a bounded prefix of probe-bearing released revisions. No raw page whitespace, revision chain, or new human judgment is inferred.

## Implementation scope

The existing numerical functions are retained: UTF-16 slicing (including partial-surrogate saved fields), whitespace normalization, original-occurrence transport, overlapping exact candidates, substring-edit context scores, stable leftmost ties, native conversion, and paired exact-text-group statistics. Portability changes replace historical directory imports, source-tar audits, and old run receipts with explicit external inputs, fresh external outputs, actual-input hashes, and pinned vendor-file checks. Fine-bank parity checks are retained. Dataset adapters validate existing labels and do not introduce new algorithms or partition rules.

The minimum package omits plot editors, manuscript compilation, historical experiment runners, upstream TypeScript differential checks, and the separate residual-report collector. Saved policy components and native metadata remain available for downstream diagnostics. No full corpus experiment is executed by smoke. See [THIRD_PARTY.md](THIRD_PARTY.md) for exact upstream revisions and notices; no project-wide license is supplied.
