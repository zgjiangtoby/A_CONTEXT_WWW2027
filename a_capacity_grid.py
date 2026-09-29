"""Paired copied-radius / saved-width experiment; external inputs only."""
import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

import anchoring as ref
import anchoring as base
import data_io

GRID = (8, 20, 32, 48, 64)
EDITS = (("none", 0),) + tuple((direction, radius)
                               for direction in ("prepend", "append") for radius in GRID)
CONTRASTS = ((8, 8, 20), (20, 20, 32), (32, 32, 48), (48, 48, 64))
REPS, SEED = 2000, 20260925
OUTCOMES = ("correct", "abstain", "wrong")
EXPECTED = {"tama": (5547, 688, 457), "toxic": (1850, 226, 192), "ptc": (6087, 5807, 353)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def partial_surrogate(units):
    return bool(units) and (0xDC00 <= ord(units[0]) <= 0xDFFF
                           or 0xD800 <= ord(units[-1]) <= 0xDBFF)


def document(text):
    old, support = base.normalization_trace(text, "text/plain")
    bounds = ref.unit_boundaries(old)
    return dict(old=old, support=support, bounds=bounds, boundary_set=set(bounds),
                units=ref.utf16(old), group=text_digest(text), raw_length=len(text))


def copy_geometry(doc, span, radius):
    """Whole-code-point copy within an at-most-radius UTF-16 budget per side."""
    bounds = doc["bounds"]
    start, end = (bounds[x] for x in span)
    requested_start = max(0, start - radius)
    requested_end = min(bounds[-1], end + radius)
    left = bisect_left(bounds, requested_start)
    right = bisect_right(bounds, requested_end) - 1
    assert left <= span[0] < span[1] <= right
    return dict(copy_start_cp=left, copy_end_cp=right,
                copy_start_u16=bounds[left], copy_end_u16=bounds[right],
                copy_left_u16=start - bounds[left], copy_right_u16=bounds[right] - end,
                copy_left_document_clipped=int(start < radius),
                copy_right_document_clipped=int(bounds[-1] - end < radius),
                copy_left_surrogate_aligned=int(bounds[left] != requested_start),
                copy_right_surrogate_aligned=int(bounds[right] != requested_end))


def make_edit(doc, span, direction, radius):
    """Edit only the original lineage; the added copy never becomes gold.

    The recipe plus the frozen raw text and normalization support reconstructs
    every character. Copy and marker insertion positions are before-operation
    code-point offsets, also recorded in UTF-16. No source strings are exported.
    """
    assert (direction == "none" and radius == 0) or (
        direction in ("prepend", "append") and radius >= 0)
    old = doc["old"]
    current, mapping = old, [(i, i + 1) for i in range(len(old))]
    geometry = copy_geometry(doc, span, radius)
    operations = []

    def insert(kind, position, value):
        nonlocal current, mapping
        operations.append(dict(kind=kind, at_cp=position,
                               at_u16=ref.unit_boundaries(current)[position],
                               length_cp=len(value), length_u16=len(ref.utf16(value))))
        current, next_map = base.insert_text(current, position, value)
        mapping = base.compose_maps(mapping, next_map)

    if direction != "none":
        copied = old[geometry["copy_start_cp"]:geometry["copy_end_cp"]]
        insert("copy_then_space" if direction == "prepend" else "space_then_copy",
               0 if direction == "prepend" else len(current),
               copied + " " if direction == "prepend" else " " + copied)
    left, right = base.interval_image(mapping, *span)
    insert("five_hashes", right, "#####")
    insert("five_hashes", left, "#####")
    shown, support = base.normalization_trace(current, "text/plain")
    truth = base.project_display(support, *base.interval_image(mapping, *span))
    assert truth is not None and shown[slice(*truth)] == old[slice(*span)]
    assert all(a < b for a, b in mapping)
    assert all(mapping[i][1] <= mapping[i + 1][0] for i in range(len(mapping) - 1))
    return shown, truth, dict(**geometry, operations_json=json.dumps(operations, separators=(",", ":")),
                             storage_sha256=text_digest(current), shown_sha256=text_digest(shown))


def selector_fields(doc, span):
    """Reusable parity interface: quote/position/fields in UTF-16 unit strings.

    A separate upstream caller can combine these with make_edit's shown text
    and saved predicted_u16/stage arrays; it need not rerun the local selector.
    Serialize with ensure_ascii=True to retain partial-surrogate field units.
    """
    bounds, units = doc["bounds"], doc["units"]
    old_span = tuple(bounds[x] for x in span)
    quote = ref.utf16(doc["old"][slice(*span)])
    fields = [(units[max(0, old_span[0] - width):old_span[0]],
               units[old_span[1]:old_span[1] + width]) for width in GRID]
    return quote, old_span, fields


def evaluate(doc, span, shown, truth):
    """One unchanged selector, five independent saved-field widths."""
    quote, old_span, fields = selector_fields(doc, span)
    shown_units = ref.utf16(shown)
    shown_bounds = ref.unit_boundaries(shown)
    reverse = {u: cp for cp, u in enumerate(shown_bounds)}
    codes = np.ones(len(GRID), dtype=np.uint8)
    predicted_cp = np.full((len(GRID), 2), -1, dtype=np.int32)
    predicted_u16 = predicted_cp.copy()
    stages = np.full(len(GRID), -1, dtype=np.int8)
    for k, (prefix, suffix) in enumerate(fields):
        answer, stages[k] = ref.hypothesis_text(shown_units, quote, prefix, suffix, old_span)
        if answer is not None:
            assert answer[0] in reverse and answer[1] in reverse, "Split-surrogate quote prediction"
            converted = (reverse[answer[0]], reverse[answer[1]])
            assert shown[slice(*converted)] == doc["old"][slice(*span)]
            predicted_cp[k], predicted_u16[k] = converted, answer
            codes[k] = 0 if converted == truth else 2
    assert np.all(stages == stages[0]), "Saved width must not change position-branch admission"
    if stages[0] == 0:
        assert np.all(predicted_u16 == predicted_u16[0]) and np.all(codes == codes[0])
    return codes, predicted_cp, predicted_u16, stages, tuple(shown_bounds[x] for x in truth)


def summarize(records, edits, outcome, stage, reps=REPS):
    """All grid points; paired CIs only for the eight predeclared width contrasts."""
    group_names = sorted({r["exact_text_group"] for r in records})
    lookup = {name: i for i, name in enumerate(group_names)}
    group = np.array([lookup[r["exact_text_group"]] for r in records])
    ng = len(group_names)
    rng = np.random.default_rng(SEED)
    weights = np.stack([np.bincount(rng.integers(ng, size=ng), minlength=ng)
                        for _ in range(reps)]).astype(float)
    panels = (("common_full64_aligned", np.array([r["common_panel"] for r in records], bool)),
              ("all_annotations", np.ones(len(records), bool)))
    summary, paired = [], []
    for panel, mask in panels:
        counts = np.bincount(group[mask], minlength=ng)
        active = counts > 0
        draw_n, draw_groups = weights @ counts, weights @ active.astype(float)
        valid = draw_n > 0
        for j, (direction, radius) in enumerate(EDITS):
            for k, width in enumerate(GRID):
                row = dict(panel=panel, edit_index=j, direction=direction, copy_radius_u16=radius,
                           saved_width_u16=width, n=int(mask.sum()), groups=int(active.sum()),
                           bootstrap_pool_groups=ng)
                for code, name in enumerate(OUTCOMES):
                    yes = (outcome[:, j, k] == code) & mask
                    total = np.bincount(group[yes], minlength=ng)
                    row[name] = int(yes.sum())
                    row[name + "_rate"] = float(yes.sum() / mask.sum()) if mask.any() else None
                    row[name + "_equal_document"] = float(np.mean(total[active] / counts[active])) if active.any() else None
                for branch, branch_code in (("position", 0), ("quote_match", 1), ("empty_quote", 2)):
                    chosen = (stage[:, j, k] == branch_code) & mask
                    row[branch + "_n"] = int(chosen.sum())
                    for code, name in enumerate(OUTCOMES):
                        row[branch + "_" + name] = int(((outcome[:, j, k] == code) & chosen).sum())
                for side in ("left", "right"):
                    lengths = np.array([min(width, r["available_" + side + "_u16"]) for r in records])[mask]
                    row["saved_" + side + "_mean_u16"] = float(lengths.mean()) if len(lengths) else None
                    row["saved_" + side + "_full_n"] = int((lengths == width).sum())
                    row["saved_" + side + "_partial_surrogate_n"] = sum(
                        r[f"saved_{side}_{width}_partial_surrogate"] for r, keep in zip(records, mask) if keep)
                    copy_lengths = [r["copy_" + side + "_u16"] for r, keep in zip(edits[j], mask) if keep]
                    row["copy_" + side + "_mean_u16"] = float(np.mean(copy_lengths)) if copy_lengths else None
                    row["copy_" + side + "_surrogate_aligned_n"] = sum(
                        r["copy_" + side + "_surrogate_aligned"] for r, keep in zip(edits[j], mask) if keep)
                assert sum(row[name] for name in OUTCOMES) == row["n"]
                summary.append(row)
        for direction in ("prepend", "append"):
            for radius, low, high in CONTRASTS:
                j, a, b = EDITS.index((direction, radius)), GRID.index(low), GRID.index(high)
                for code, metric in enumerate(OUTCOMES):
                    delta = (outcome[:, j, b] == code).astype(float) - (outcome[:, j, a] == code)
                    total = np.bincount(group[mask], weights=delta[mask], minlength=ng)
                    fractions = np.divide(total, counts, out=np.zeros(ng), where=active)
                    ann_draws = (weights @ total)[valid] / draw_n[valid]
                    doc_draws = (weights @ fractions)[valid] / draw_groups[valid]
                    row = dict(panel=panel, direction=direction, copy_radius_u16=radius,
                               saved_low_u16=low, saved_high_u16=high, metric=metric,
                               orientation="high_minus_low", n=int(mask.sum()), groups=int(active.sum()),
                               bootstrap_pool_groups=ng, bootstrap_replicates=reps, bootstrap_seed=SEED,
                               valid_replicates=int(valid.sum()), empty_replicates=int((~valid).sum()))
                    for name, point, samples in (
                        ("annotation", float(total.sum() / mask.sum()) if mask.any() else None, ann_draws),
                        ("equal_document", float(fractions[active].mean()) if active.any() else None, doc_draws)):
                        row[name + "_difference"] = point
                        row[name + "_lo"] = float(np.quantile(samples, .025)) if len(samples) else None
                        row[name + "_hi"] = float(np.quantile(samples, .975)) if len(samples) else None
                    paired.append(row)
    return summary, paired


def replay(old, span, recipe):
    """Independent recipe replay used only by bounded synthetic self-checks."""
    current, mapping = old, [(i, i + 1) for i in range(len(old))]
    copied = old[recipe["copy_start_cp"]:recipe["copy_end_cp"]]
    for op in json.loads(recipe["operations_json"]):
        value = "#####" if op["kind"] == "five_hashes" else (
            copied + " " if op["kind"] == "copy_then_space" else " " + copied)
        assert ref.unit_boundaries(current)[op["at_cp"]] == op["at_u16"]
        assert len(value) == op["length_cp"] and len(ref.utf16(value)) == op["length_u16"]
        current, step = base.insert_text(current, op["at_cp"], value)
        mapping = base.compose_maps(mapping, step)
    shown, support = base.normalization_trace(current, "text/plain")
    assert text_digest(current) == recipe["storage_sha256"] and text_digest(shown) == recipe["shown_sha256"]
    return shown, base.project_display(support, *base.interval_image(mapping, *span))


def self_check():
    ref.self_check()
    for raw, raw_span in (("p" * 80 + "bad" + "s" * 80 + " bad", (80, 83)),
                          ("😀" + "p" * 7 + "q" + "s" * 7 + "😀", (8, 9)),
                          ("A  \tbad \n B", (4, 7)), ("bad" + "s" * 100, (0, 3))):
        doc = document(raw)
        span = base.project_display(doc["support"], *raw_span)
        for direction, radius in EDITS:
            shown, truth, recipe = make_edit(doc, span, direction, radius)
            assert replay(doc["old"], span, recipe) == (shown, truth)
            codes, _, _, stages, _ = evaluate(doc, span, shown, truth)
            assert np.isin(codes, (0, 1, 2)).all()
            if raw.startswith("bad") and direction == "prepend":
                assert np.all(stages == 0) and np.all(codes == 2)
    doc = document("😀" + "p" * 7 + "q" + "s" * 7 + "😀")
    geo = copy_geometry(doc, (8, 9), 8)
    assert geo["copy_left_u16"] == geo["copy_right_u16"] == 7
    assert geo["copy_left_surrogate_aligned"] == geo["copy_right_surrogate_aligned"] == 1
    s, e = doc["bounds"][8:10]
    assert partial_surrogate(doc["units"][s-8:s]) and partial_surrogate(doc["units"][e:e+8])
    records = [dict(exact_text_group=g, common_panel=common, available_left_u16=64,
                    available_right_u16=64, **{f"saved_{side}_{w}_partial_surrogate": 0
                    for side in ("left", "right") for w in GRID})
               for g, common in (("a", True), ("a", False), ("b", True))]
    edits = [[dict(copy_left_u16=r, copy_right_u16=r, copy_left_surrogate_aligned=0,
                   copy_right_surrogate_aligned=0) for _ in records] for _, r in EDITS]
    out = np.zeros((3, len(EDITS), len(GRID)), np.uint8)
    out[0, EDITS.index(("prepend", 8)), GRID.index(8)] = 2
    summary, pairs = summarize(records, edits, out, np.ones_like(out), reps=100)
    row = next(r for r in pairs if r["panel"] == "all_annotations" and r["direction"] == "prepend"
               and r["copy_radius_u16"] == 8 and r["metric"] == "correct")
    assert row["annotation_difference"] == 1/3 and row["equal_document_difference"] == .25
    assert len(summary) == 110 and len(pairs) == 48
    assert all(r["n"] == sum(r[k] for k in OUTCOMES) for r in summary)
    print("PASS: frozen selector; repeated quotes; surrogate alignment; whitespace lineage; "
          "recipe replay; position-stage width invariance; paired group estimands", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_io.arguments(parser)
    parser.add_argument("--corpus", choices=("tama", "toxic", "ptc"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Pilot only; formal runs omit this option")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    assert args.corpus and args.out and args.limit >= 0
    out = data_io.fresh_output(args.out)
    started = time.time()
    raw, texts, annotations = data_io.load(args.corpus, args.input, args.annotations)
    input_hashes = data_io.hashes(raw, args.annotations, Path(__file__), Path(ref.__file__), Path(data_io.__file__))
    if not args.limit:
        assert len(annotations) == EXPECTED[args.corpus][0]
    if args.limit:
        annotations = annotations[:args.limit]
    assert annotations
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", corpus=args.corpus, pilot=bool(args.limit), limit=args.limit,
                    n=len(annotations), radii_u16=list(GRID), saved_widths_u16=list(GRID),
                    edits=[dict(edit_index=j, direction=d, radius_u16=r) for j, (d, r) in enumerate(EDITS)],
                    bootstrap_replicates=REPS, bootstrap_seed=SEED,
                    primary_contrasts=[dict(copy_radius_u16=r, saved_low_u16=a, saved_high_u16=b)
                                       for r, a, b in CONTRASTS],
                    source_sha256=digest(__file__), input_sha256=input_hashes,
                    python=platform.python_version(), numpy=np.__version__,
                    hypothesis_commit=ref.CLIENT_COMMIT, approximate_match_commit=ref.APPROX_COMMIT,
                    coordinate_frame="copy radius and saved width use old normalized-display UTF-16",
                    copy_policy="whole code points; boundary-clipped; surrogate cuts align inward",
                    saved_field_policy="exact UTF-16 slicing, including partial surrogate units",
                    common_panel="both sides >=64 units and all five saved/copy boundaries are code-point boundaries",
                    inference="fixed-corpus paired, pointwise exact-original-text-group percentile intervals; no multiplicity correction",
                    new_annotations=False, training=False, source_text_exported=False)
    base.write_json(out / "manifest.json", manifest)
    n = len(annotations)
    outcome = np.empty((n, len(EDITS), len(GRID)), np.uint8)
    stage = np.empty_like(outcome, dtype=np.int8)
    predicted_cp = np.full((*outcome.shape, 2), -1, np.int32)
    predicted_u16 = predicted_cp.copy()
    truth_cp = np.empty((n, len(EDITS), 2), np.int32)
    truth_u16 = np.empty_like(truth_cp)
    records, documents, cache = [], [], {}
    edit_rows = [[] for _ in EDITS]
    support_parts, support_offsets = [], [0]
    for i, annotation in enumerate(annotations):
        assert int(annotation["unit_index"]) == i
        key = annotation["post_key"]
        if key not in cache:
            doc = document(texts[key])
            doc["index"] = len(documents)
            cache[key] = doc
            support_parts.extend(doc["support"])
            support_offsets.append(len(support_parts))
            documents.append(dict(document_index=doc["index"], post_key=key, raw_length_cp=doc["raw_length"],
                                  old_length_cp=len(doc["old"]), old_length_u16=len(doc["units"]),
                                  exact_text_group=doc["group"], old_display_sha256=text_digest(doc["old"])))
        doc = cache[key]
        raw_span = (int(annotation["raw_start"]), int(annotation["raw_end"]))
        span = (int(annotation["old_start"]), int(annotation["old_end"]))
        assert base.project_display(doc["support"], *raw_span) == span
        assert len(doc["old"]) == int(annotation["old_post_length"])
        s, e = (doc["bounds"][x] for x in span)
        total = len(doc["units"])
        common = s >= 64 and total-e >= 64 and all(
            s-w in doc["boundary_set"] and e+w in doc["boundary_set"] for w in GRID)
        record = dict(unit_index=i, document_index=doc["index"], post_key=key,
                      annotation_index=annotation["annotation_index"], exact_text_group=doc["group"],
                      raw_start_cp=raw_span[0], raw_end_cp=raw_span[1],
                      old_start_cp=span[0], old_end_cp=span[1], old_start_u16=s, old_end_u16=e,
                      available_left_u16=s, available_right_u16=total-e, common_panel=int(common),
                      natural_candidates=int(annotation["natural_candidates"]))
        for width in GRID:
            for side, field in (("left", doc["units"][max(0, s-width):s]),
                                ("right", doc["units"][e:e+width])):
                record[f"saved_{side}_{width}_length_u16"] = len(field)
                record[f"saved_{side}_{width}_partial_surrogate"] = int(partial_surrogate(field))
        records.append(record)
        for j, (direction, radius) in enumerate(EDITS):
            shown, truth, recipe = make_edit(doc, span, direction, radius)
            result = evaluate(doc, span, shown, truth)
            outcome[i, j], predicted_cp[i, j], predicted_u16[i, j], stage[i, j], truth_u16[i, j] = result
            truth_cp[i, j] = truth
            edit_rows[j].append(dict(unit_index=i, edit_index=j, direction=direction, radius_u16=radius,
                                     new_candidates=len(base.occurrences(shown, doc["old"][slice(*span)])),
                                     shown_length_cp=len(shown), shown_length_u16=len(ref.utf16(shown)),
                                     truth_start_cp=truth[0], truth_end_cp=truth[1], **recipe))
        if (i+1) % 250 == 0:
            print(f"{args.corpus}: {i+1}/{n}", flush=True)
    assert np.isin(outcome, (0, 1, 2)).all()
    common_n = sum(r["common_panel"] for r in records)
    common_groups = len({r["exact_text_group"] for r in records if r["common_panel"]})
    if not args.limit:
        assert (n, common_n, common_groups) == EXPECTED[args.corpus], "Predeclared geometry changed"
    assert np.array_equal(outcome == 0, np.all(predicted_cp == truth_cp[:, :, None, :], axis=-1))
    assert np.array_equal(outcome == 1, np.all(predicted_cp == -1, axis=-1))
    np.savez_compressed(out / "outcomes.npz", outcome=outcome, stage=stage,
                        predicted_cp=predicted_cp, predicted_u16=predicted_u16,
                        truth_cp=truth_cp, truth_u16=truth_u16,
                        normalization_support=np.asarray(support_parts, dtype=np.int32),
                        normalization_support_offsets=np.asarray(support_offsets, dtype=np.int64))
    base.write_csv(out / "records.csv", records)
    base.write_csv(out / "documents.csv", documents)
    base.write_csv(out / "edit_metadata.csv", [r for rows in edit_rows for r in rows])
    summary, paired = summarize(records, edit_rows, outcome, stage)
    base.write_csv(out / "summary.csv", summary)
    base.write_csv(out / "paired_intervals.csv", paired)
    assert all(digest(p) == sha for p, sha in input_hashes.items()), "Input changed during run"
    manifest.update(status="completed", common_panel_n=common_n,
                    exact_text_groups=len({r["exact_text_group"] for r in records}),
                    common_panel_groups=common_groups,
                    npz_axes=["records.csv unit_index", "manifest edits", "saved_widths_u16"],
                    normalization_support_axes="documents.csv document_index; offsets slice flattened old-display-code-point/raw-code-point support pairs",
                    outcome_codes={0: "correct", 1: "abstain", 2: "wrong"},
                    stage_codes={0: "quote_checked_position", 1: "exact_quote_candidates_fuzzy_context", 2: "empty_quote"},
                    output_sha256={p.name: digest(p) for p in out.iterdir() if p.name != "manifest.json"},
                    elapsed_seconds=time.time()-started)
    base.write_json(out / "manifest.json", manifest)
    print(f"DONE {args.corpus}: {n} annotations, common panel {manifest['common_panel_n']}; {out}", flush=True)


if __name__ == "__main__":
    main()

