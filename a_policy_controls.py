"""Same-input H/F/N policies and the pinned native reanchor comparison."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

import a_capacity_grid as grid
import data_io

ref, base = grid.ref, grid.base
WIDTH = 32
POLICIES = ("hypothesis_pinned", "skip_position_fastpath", "skip_fastpath_no_position_score")
REANCHOR = "reanchor_native_no_old_position"
INCOMPARABLE = 3
COMMIT = "b91143e7f3f48e9356a8bec846da39a2527ab04d"
SOURCE = Path(__file__).resolve().parent / "vendor"
VENDOR_HASHES = {
    "reanchor/__init__.py": "e208187c9fa0ecc6b07ea01e5e199d5c1f1e92222eb4859078a7899b6f317bbb",
    "reanchor/describe.py": "308b5ddacf7f902393bf5ba5d20579d8b77d36655b3b15354343ff6da0bddbad",
    "reanchor/normalize.py": "96c8159115667c4e3e8b140dca6ca58cb4772e57a2e1c8112145d44a3b37e20b",
    "reanchor/resolve.py": "3928a7f65f250d8169d0bbf1e81f1685f1a5921332f2bf0d7c692e09851e00a5",
    "reanchor/search.py": "05e51ead19f4e6499dd0d12b539e78911b39542c941e7d4dc2e8dbbf1f5f4997",
    "reanchor/LICENSE": "77d48cae6c8e800e411964bd98c2e20bc5ba42d88ecea51fa863b2d0d545a78c",
}


def vendor_module():
    for name, sha in VENDOR_HASHES.items():
        assert grid.digest(SOURCE / name) == sha, ("Pinned vendor source changed", name)
    sys.path.insert(0, str(SOURCE))
    module = importlib.import_module("reanchor")
    assert Path(module.__file__).resolve() == SOURCE / "reanchor/__init__.py"
    return module


def candidate_components(text, candidates, prefix, suffix, old_span):
    """Original context_score and candidate order; no replacement distance."""
    return np.asarray([
        (ref.context_score(text[max(0, s-len(prefix)):s], prefix) if prefix else 1.0,
         ref.context_score(text[e:e+len(suffix)], suffix) if suffix else 1.0,
         1.0 - abs(s-old_span[0]) / len(text)) for s, e in candidates], dtype=float)


def score92(components, position=True):
    """Common /92 diagnostic scale; deleting position does not rescale the rest."""
    return (50.0 + 20.0*components[..., 0] + 20.0*components[..., 1]
            + (2.0*components[..., 2] if position else 0.0)) / 92.0


def policy_answers(text, quote, prefix, suffix, old_span):
    """Only matcher-visible fields enter here; truth is exclusively evaluator-side."""
    native, stage = ref.hypothesis_text(text, quote, prefix, suffix, old_span)
    if not quote:
        return [native, None, None], stage, [], np.empty((0, 3))
    candidates = base.occurrences(text, quote)
    if not candidates:
        raise ValueError("Reference scope requires at least one preserved exact quote")
    components = candidate_components(text, candidates, prefix, suffix, old_span)
    with_position, without_position = score92(components), score92(components, False)
    # np.argmax retains the first candidate on an exact score tie, as upstream max does.
    skip = candidates[int(np.argmax(with_position))]
    no_position = candidates[int(np.argmax(without_position))]
    if stage == 1:
        assert skip == native, "Removing a bypass that was not taken must change nothing"
    return [native, skip, no_position], stage, candidates, components


def unicode_fields(fields):
    """Strict lossless UTF-16-unit to code-point conversion, never repair/drop units."""
    try:
        converted = tuple(b"".join(ord(c).to_bytes(2, "little") for c in field)
                          .decode("utf-16-le", errors="strict") for field in fields)
    except UnicodeError:
        return None
    assert all(ref.utf16(decoded) == units for decoded, units in zip(converted, fields))
    return converted


def reanchor_answer(module, shown, fields):
    """Distinct native interface: no old-position parameter or old document."""
    converted = unicode_fields(fields)
    if converted is None:
        return False, None
    quote, prefix, suffix = converted
    assert len(prefix) <= WIDTH and len(suffix) <= WIDTH
    result = module.resolve_quote(shown, module.TextQuoteSelector(quote, prefix, suffix))
    if result is not None:
        assert 0 <= result.start <= result.end <= len(shown)
        assert shown[result.start:result.end] == result.text
    return True, result


def outcome(answer, truth):
    return 1 if answer is None else (0 if answer == truth else 2)


def group_summary(records, codes, stages, eligible, methods, reps=grid.REPS):
    """Fixed-frame rates, 3x3 transitions, paired group CIs and descriptive strata.

    Bootstrap shares draws across edits/policies, using the full corpus's exact
    raw-text groups as its pool. Frame membership never uses selector outcomes.
    Branch strata describe native decisions; they are not separate causal claims.
    """
    groups = sorted({r["exact_text_group"] for r in records})
    lookup = {g: i for i, g in enumerate(groups)}
    ids = np.array([lookup[r["exact_text_group"]] for r in records])
    ng, n = len(groups), len(records)
    rng = np.random.default_rng(grid.SEED)
    weights = np.stack([np.bincount(rng.integers(ng, size=ng), minlength=ng)
                        for _ in range(reps)]).astype(float)
    common = np.array([r["common_panel"] for r in records], dtype=bool)
    full32 = np.array([r["field_stratum"] == "both_full" for r in records], dtype=bool)
    panels = [("common_full64_aligned", common), ("all_annotations", np.ones(n, bool)),
              ("full32", full32), ("clipped32", ~full32)]
    if REANCHOR in methods:
        panels += [(name + "_reanchor_eligible", mask & eligible) for name, mask in panels.copy()]
    summaries, transitions, intervals, strata = [], [], [], []
    pairs = [(0, 1), (0, 2), (1, 2)]
    if REANCHOR in methods:
        pairs.append((0, 3))
    for panel, mask in panels:
        matched = panel.endswith("_reanchor_eligible")
        count = np.bincount(ids[mask], minlength=ng)
        active = count > 0
        denominators = weights @ count
        group_denominators = weights @ active.astype(float)
        valid = denominators > 0

        def stats(values, uncertainty=False):
            total = np.bincount(ids[mask], weights=values[mask], minlength=ng)
            fractions = np.divide(total, count, out=np.zeros(ng), where=active)
            result = dict(annotation=float(total.sum()/mask.sum()) if mask.any() else None,
                          equal_document=float(fractions[active].mean()) if active.any() else None)
            if uncertainty:
                for weighting, draws in (("annotation", (weights @ total)[valid]/denominators[valid]),
                                         ("equal_document", (weights @ fractions)[valid]/group_denominators[valid])):
                    result[weighting + "_lo"] = float(np.quantile(draws, .025)) if len(draws) else None
                    result[weighting + "_hi"] = float(np.quantile(draws, .975)) if len(draws) else None
            return result

        for j, (direction, radius) in enumerate(grid.EDITS):
            key = dict(panel=panel, edit_index=j, direction=direction, copy_radius_u16=radius,
                       saved_width_u16=WIDTH, n=int(mask.sum()), groups=int(active.sum()),
                       bootstrap_pool_groups=ng)
            for k, method in enumerate(methods):
                if k == 3 and not matched:
                    continue  # No missing-as-abstention or algorithm-specific denominator.
                row = dict(**key, method=method)
                for code, name in enumerate(grid.OUTCOMES):
                    yes = codes[:, j, k] == code
                    row[name + "_n"] = int((yes & mask).sum())
                    row.update({name + "_" + w: v for w, v in stats(yes).items()})
                assert sum(row[name + "_n"] for name in grid.OUTCOMES) == row["n"]
                summaries.append(row)
            for a, b in pairs:
                if b == 3 and not matched:
                    continue
                pair_key = dict(**key, baseline=methods[a], alternative=methods[b])
                before, after = codes[:, j, a], codes[:, j, b]
                for old, old_name in enumerate(grid.OUTCOMES):
                    for new, new_name in enumerate(grid.OUTCOMES):
                        yes = (before == old) & (after == new)
                        transitions.append(dict(**pair_key, before=old_name, after=new_name,
                                                count=int((yes & mask).sum()), **stats(yes)))
                values = {name + "_difference": (after == code).astype(float) - (before == code)
                          for code, name in enumerate(grid.OUTCOMES)}
                values.update(wrong_repaired=((before == 2) & (after == 0)).astype(float),
                              correct_broken=((before == 0) & (after == 2)).astype(float))
                for metric, value in values.items():
                    intervals.append(dict(**pair_key, metric=metric,
                                          orientation="alternative_minus_baseline_or_transition_fraction",
                                          bootstrap_replicates=reps, bootstrap_seed=grid.SEED,
                                          valid_replicates=int(valid.sum()), empty_replicates=int((~valid).sum()),
                                          **stats(value, True)))
                if matched:
                    continue
                levels = [("native_branch", name, stages[:, j] == s)
                          for s, name in ((0, "position"), (1, "quote_match"), (2, "empty_quote"))]
                levels += [("saved_fields", name, np.array([r["field_stratum"] == name for r in records]))
                           for name in ("both_full", "left_short", "right_short", "both_short")]
                levels += [("partial_surrogate", str(flag),
                            np.array([bool(r["partial_surrogate"]) == flag for r in records]))
                           for flag in (False, True)]
                for axis, level, member in levels:
                    subset = mask & member
                    for old, old_name in enumerate(grid.OUTCOMES):
                        for new, new_name in enumerate(grid.OUTCOMES):
                            strata.append(dict(**pair_key, stratum_axis=axis, stratum=level,
                                               stratum_n=int(subset.sum()), before=old_name, after=new_name,
                                               count=int(((before == old) & (after == new) & subset).sum())))
    return summaries, transitions, intervals, strata


def self_check(include_reanchor=False):
    ref.self_check()
    # Deliberate fast-path bypass; evaluation can distinguish repaired and broken records.
    # Saved fields are consistent with old text "xxxgood bad end", old span 8:11.
    args = ("zzzzzzzzbad xxx good bad end", "bad", "good ", " end", (8, 11))
    answers, stage, candidates, components = policy_answers(*args)
    assert stage == 0 and answers == [(8, 11), (21, 24), (21, 24)]
    assert [outcome(a, (21, 24)) for a in answers] == [2, 0, 0]
    assert [outcome(a, (8, 11)) for a in answers] == [0, 2, 2]
    answers, stage, _, components = policy_answers("bad bad", "bad", "", "", (4, 7))
    assert answers == [(4, 7), (4, 7), (0, 3)]  # position-only ordering versus stable left tie
    assert np.array_equal(score92(components, False), np.full(2, 90/92))
    assert policy_answers("bad", "", "", "", (0, 0))[0] == [None]*3
    answers, _, candidates, _ = policy_answers("aaaa", "aa", "", "", (8, 10))
    assert candidates == [(0, 2), (1, 3), (2, 4)] and answers[2] == (0, 2)
    assert unicode_fields((ref.utf16("😀q"), "", ref.utf16("é"))) == ("😀q", "", "é")
    assert unicode_fields(("q", "\ude00", "")) is None
    assert unicode_fields(("q", "", "\ud83d")) is None
    module = vendor_module() if include_reanchor else None
    for raw, raw_span in (("p"*80+"bad"+"s"*80+" bad", (80, 83)),
                          ("😀"+"p"*31+"q"+"s"*31+"😀", (32, 33)),
                          ("A  \tbad \n B", (4, 7)), ("bad"+"s"*100, (0, 3))):
        doc = grid.document(raw)
        span = base.project_display(doc["support"], *raw_span)
        quote, old, fields = grid.selector_fields(doc, span)
        prefix, suffix = fields[grid.GRID.index(WIDTH)]
        for direction, radius in grid.EDITS:
            shown, truth, recipe = grid.make_edit(doc, span, direction, radius)
            assert grid.replay(doc["old"], span, recipe) == (shown, truth)
            answers, stage, hits, components = policy_answers(ref.utf16(shown), quote, prefix, suffix, old)
            assert (answers[0], stage) == ref.hypothesis_text(ref.utf16(shown), quote, prefix, suffix, old)
            assert tuple(ref.unit_boundaries(shown)[x] for x in truth) in hits
            if module:
                applicable, result = reanchor_answer(module, shown, (quote, prefix, suffix))
                assert applicable == (not grid.partial_surrogate(prefix) and not grid.partial_surrogate(suffix))
                if not applicable:
                    assert result is None
    if module:
        ok, result = reanchor_answer(module, "😀 hello world", ("hello", ref.utf16("😀 "), " world"))
        assert ok and (result.start, result.end) == (2, 7)
        assert tuple(ref.unit_boundaries("😀 hello world")[x] for x in (result.start, result.end)) == (3, 8)
        ok, result = reanchor_answer(module, "😀 hello", ("hello", "", ""))
        assert ok and result is not None  # Empty context is eligible, not malformed Unicode.
        ok, result = reanchor_answer(module, "hello", ("", "", ""))
        assert ok and result is None  # Genuine native abstention, unlike ineligible partial fields.
    records = [dict(exact_text_group=g, common_panel=True, field_stratum="both_full", partial_surrogate=0)
               for g in ("a", "a", "b")]
    records[2].update(common_panel=False, field_stratum="left_short")
    codes = np.zeros((3, len(grid.EDITS), 3), dtype=np.uint8)
    codes[0, :, 0] = 2
    summary, transitions, intervals, _ = group_summary(
        records, codes, np.ones(codes.shape[:2], dtype=np.int8), np.ones(3, bool), POLICIES, reps=32)
    row = next(r for r in intervals if r["panel"] == "all_annotations" and r["edit_index"] == 0
               and r["alternative"] == POLICIES[1] and r["metric"] == "correct_difference")
    assert row["annotation"] == 1/3 and row["equal_document"] == .25
    assert all(sum(t["count"] for t in transitions if all(t[k] == row[k]
                   for k in ("panel", "edit_index", "baseline", "alternative"))) == row["n"]
               for row in intervals)
    with_vendor = np.concatenate((codes, np.zeros((*codes.shape[:2], 1), np.uint8)), axis=2)
    with_vendor[0, :, 3] = INCOMPARABLE
    summaries, transitions, _, _ = group_summary(records, with_vendor,
        np.ones(codes.shape[:2], dtype=np.int8), np.array([False, True, True]), (*POLICIES, REANCHOR), reps=16)
    matched = [r for r in summaries if r["panel"] == "all_annotations_reanchor_eligible"]
    assert len(matched) == len(grid.EDITS)*4 and all(r["n"] == 2 for r in matched)
    assert all(r["abstain_n"] == 0 for r in matched)
    expected_n = {"full32": 2, "clipped32": 1, "full32_reanchor_eligible": 1,
                  "clipped32_reanchor_eligible": 1}
    assert all(r["n"] == expected_n[r["panel"]] for r in summaries if r["panel"] in expected_n)
    assert not any(r["method"] == REANCHOR and not r["panel"].endswith("_reanchor_eligible") for r in summaries)
    print("PASS: exact candidates/context reuse; bypass repair/break; position removal/left ties; "
          "unchanged native parity; lineage; strict UTF-16 eligibility; paired group estimands"
          + ("; pinned native reanchor code-point mapping" if module else ""), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data_io.arguments(parser)
    parser.add_argument("--corpus", choices=tuple(grid.EXPECTED))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Nonzero is explicitly pilot-only")
    parser.add_argument("--include-reanchor", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check(args.include_reanchor)
    if args.self_check:
        return
    assert args.corpus and args.out and args.limit >= 0
    out = data_io.fresh_output(args.out)
    start_time = time.time()
    raw, texts, annotations = data_io.load(args.corpus, args.input, args.annotations)
    hashes = data_io.hashes(raw, args.annotations, Path(__file__), Path(grid.__file__), Path(ref.__file__), Path(data_io.__file__))
    if not args.limit:
        assert len(annotations) == grid.EXPECTED[args.corpus][0]
    module = vendor_module() if args.include_reanchor else None
    if module:
        for path in [SOURCE / p for p in VENDOR_HASHES]:
            hashes[str(path)] = grid.digest(path)
    annotations = annotations[:args.limit] if args.limit else annotations
    methods = list(POLICIES) + ([REANCHOR] if module else [])
    n, ne, nm = len(annotations), len(grid.EDITS), len(methods)
    codes = np.full((n, ne, nm), INCOMPARABLE, np.uint8)
    predicted_cp = np.full((n, ne, nm, 2), -1, np.int32)
    predicted_u16 = predicted_cp.copy()
    truth_cp = np.empty((n, ne, 2), np.int32)
    truth_u16 = truth_cp.copy()
    stages = np.empty((n, ne), np.int8)
    selected_components = np.full((n, ne, 3, 3), np.nan)
    gold_components = np.full((n, ne, 3), np.nan)
    selected_scores = np.full((n, ne, 3), np.nan)
    gold_scores = selected_scores.copy()
    eligible = np.empty(n, bool)
    candidate_counts = np.empty((n, ne), np.int32)
    records, edit_rows, reanchor_rows, cache = [], [], [], {}
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", corpus=args.corpus, pilot=bool(args.limit), limit=args.limit, n=n,
                    methods=methods, saved_width_u16=WIDTH,
                    edits=[dict(edit_index=j, direction=d, copy_radius_u16=r) for j, (d, r) in enumerate(grid.EDITS)],
                    input_sha256=hashes, python=platform.python_version(), numpy=np.__version__,
                    hypothesis_commit=ref.CLIENT_COMMIT, approximate_match_commit=ref.APPROX_COMMIT,
                    score_policy="50+20*left+20*right+2*position, common /92; last policy sets only position term to zero",
                    tie_policy="stable leftmost exact candidate; native fast path unchanged",
                    bootstrap_replicates=grid.REPS, bootstrap_seed=grid.SEED,
                    inference="pointwise paired exact-original-text-group percentile intervals, fixed corpus; no multiplicity correction",
                    construction="same a_capacity_grid make_edit; normalized-display copy radii; neighbor5 on both sides",
                    training=False, new_annotations=False, source_text_exported=False)
    if module:
        manifest["reanchor"] = dict(commit=COMMIT, pyproject_version="0.3.0", runtime_version=module.__version__,
            options=asdict(module.ResolveOptions()), interface="new document + quote/prefix/suffix; NO old position",
            conversion="lossless complete UTF-16 fields to native code points; partial fields incomparable, not abstentions",
            evaluation="actual returned input-document code-point interval versus lineage truth; no exact-quote assumption")
    base.write_json(out / "manifest.json", manifest)
    for i, annotation in enumerate(annotations):
        assert int(annotation["unit_index"]) == i
        key = annotation["post_key"]
        if key not in cache:
            cache[key] = grid.document(texts[key])
        doc = cache[key]
        span = (int(annotation["old_start"]), int(annotation["old_end"]))
        raw_span = (int(annotation["raw_start"]), int(annotation["raw_end"]))
        assert base.project_display(doc["support"], *raw_span) == span
        assert len(doc["old"]) == int(annotation["old_post_length"])
        quote, old_span, fields = grid.selector_fields(doc, span)
        prefix, suffix = fields[grid.GRID.index(WIDTH)]
        s, e = old_span
        common = s >= 64 and len(doc["units"])-e >= 64 and all(
            s-w in doc["boundary_set"] and e+w in doc["boundary_set"] for w in grid.GRID)
        eligible[i] = unicode_fields((quote, prefix, suffix)) is not None
        partial = grid.partial_surrogate(prefix) or grid.partial_surrogate(suffix)
        assert eligible[i] == (not partial)
        left_full, right_full = len(prefix) == WIDTH, len(suffix) == WIDTH
        field_stratum = ("both_full" if left_full and right_full else "right_short" if left_full
                         else "left_short" if right_full else "both_short")
        records.append(dict(unit_index=i, post_key=key, annotation_index=annotation["annotation_index"],
            exact_text_group=doc["group"], raw_start_cp=raw_span[0], raw_end_cp=raw_span[1],
            old_start_cp=span[0], old_end_cp=span[1], old_start_u16=s, old_end_u16=e,
            common_panel=int(common), saved_left_u16=len(prefix), saved_right_u16=len(suffix),
            field_stratum=field_stratum, partial_surrogate=int(partial), reanchor_eligible=int(eligible[i]),
            natural_candidates=int(annotation["natural_candidates"]), old_display_sha256=grid.text_digest(doc["old"])))
        for j, (direction, radius) in enumerate(grid.EDITS):
            shown, truth, recipe = grid.make_edit(doc, span, direction, radius)
            bounds = ref.unit_boundaries(shown)
            reverse = {u: cp for cp, u in enumerate(bounds)}
            truth_cp[i, j], truth_u16[i, j] = truth, tuple(bounds[x] for x in truth)
            answers, stages[i, j], hits, components = policy_answers(ref.utf16(shown), quote, prefix, suffix, old_span)
            candidate_counts[i, j] = len(hits)
            gold_components[i, j] = components[hits.index(tuple(truth_u16[i, j]))]
            for k, answer in enumerate(answers):
                cp = None
                if answer is not None:
                    assert answer[0] in reverse and answer[1] in reverse, "Split-surrogate quote prediction"
                    cp = (reverse[answer[0]], reverse[answer[1]])
                    assert shown[slice(*cp)] == doc["old"][slice(*span)]
                    predicted_cp[i, j, k], predicted_u16[i, j, k] = cp, answer
                    selected_components[i, j, k] = components[hits.index(answer)]
                    selected_scores[i, j, k] = score92(selected_components[i, j, k], k != 2)
                    gold_scores[i, j, k] = score92(gold_components[i, j], k != 2)
                codes[i, j, k] = outcome(cp, truth)
            if module:
                applicable, result = reanchor_answer(module, shown, (quote, prefix, suffix))
                assert applicable == eligible[i]
                if applicable:
                    cp = (result.start, result.end) if result is not None else None
                    codes[i, j, 3] = outcome(cp, truth)
                    if cp is not None:
                        predicted_cp[i, j, 3] = cp
                        predicted_u16[i, j, 3] = tuple(bounds[x] for x in cp)
                reanchor_rows.append(dict(unit_index=i, edit_index=j, eligible=int(applicable),
                    reason="" if applicable else "partial_surrogate_saved_field", method=result.method if result else "",
                    confidence=result.confidence if result else None, distance=result.distance if result else None,
                    rival_count=len(result.rivals) if result else 0,
                    predicted_exact_quote=int(result.text == doc["old"][slice(*span)]) if result else None))
            edit_rows.append(dict(unit_index=i, edit_index=j, direction=direction, copy_radius_u16=radius,
                native_stage=int(stages[i, j]), exact_candidates=len(hits), shown_length_cp=len(shown),
                shown_length_u16=bounds[-1], truth_start_cp=truth[0], truth_end_cp=truth[1], **recipe))
        if (i+1) % 250 == 0:
            print(f"{args.corpus}: {i+1}/{n}", flush=True)
    common_n = sum(r["common_panel"] for r in records)
    common_groups = len({r["exact_text_group"] for r in records if r["common_panel"]})
    if not args.limit:
        assert (n, common_n, common_groups) == grid.EXPECTED[args.corpus]
    assert np.isin(codes[:, :, :3], (0, 1, 2)).all()
    assert np.array_equal(codes == 0, np.all(predicted_cp == truth_cp[:, :, None, :], axis=-1))
    assert np.array_equal((codes == 1) | (codes == INCOMPARABLE), np.all(predicted_cp == -1, axis=-1))
    if module:
        assert np.array_equal(codes[:, :, 3] == INCOMPARABLE, np.broadcast_to(~eligible[:, None], (n, ne)))
    np.savez_compressed(out / "outcomes.npz", outcome=codes, native_stage=stages, predicted_cp=predicted_cp,
        predicted_u16=predicted_u16, truth_cp=truth_cp, truth_u16=truth_u16, reanchor_eligible=eligible,
        exact_candidate_count=candidate_counts, selected_components=selected_components, gold_components=gold_components,
        selected_score92=selected_scores, gold_score92=gold_scores, selected_minus_gold_score92=selected_scores-gold_scores)
    base.write_csv(out / "records.csv", records)
    base.write_csv(out / "edit_metadata.csv", edit_rows)
    base.write_csv(out / "reanchor_metadata.csv", reanchor_rows)
    for name, rows in zip(("summary", "transitions", "paired_intervals", "transition_strata"),
                          group_summary(records, codes, stages, eligible, methods)):
        base.write_csv(out / (name + ".csv"), rows)
    assert all(grid.digest(p) == sha for p, sha in hashes.items()), "Inputs changed during run"
    manifest.update(status="completed", common_panel_n=common_n, common_panel_groups=common_groups,
        exact_text_groups=len({r["exact_text_group"] for r in records}), reanchor_eligible_n=int(eligible.sum()),
        reanchor_incomparable_n=int((~eligible).sum()), outcome_codes={0: "correct", 1: "abstain", 2: "wrong", 3: "incomparable"},
        native_stage_codes={0: "quote_checked_position", 1: "exact_candidates_fuzzy_context", 2: "empty_quote"},
        npz_axes="records.csv unit_index, manifest edits, manifest methods; component arrays use first three policies only",
        component_axis=["left_context", "right_context", "position"], gold_components_axes="record, edit, component",
        score_note="native stage-0 diagnostic score is counterfactual; native shortcut did not consult it",
        transition_rates="fractions of the full stated panel, not conditional on baseline outcome",
        output_sha256={p.name: grid.digest(p) for p in out.iterdir() if p.name != "manifest.json"},
        elapsed_seconds=time.time()-start_time)
    base.write_json(out / "manifest.json", manifest)
    print(f"DONE {args.corpus}: {n} annotations; common={common_n}; {out}", flush=True)


if __name__ == "__main__":
    main()
