"""Canonical WikiAtomicSample probes using external released inputs only."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np

import a_policy_controls as policy
import data_io

grid, ref, base = policy.grid, policy.ref, policy.base
SOURCE_HASHES = {
    "insertions_deletions.jsonl": "8b0b8ec67e342737a7a71c9fdfff4ccd050feebdd97a7c91b3b99e222ec2809b",
    "splits/test.txt": "e089343b1bc973eeff98bcf4c9a799d3a218e9a0b3177c6536468a2b5e28fbe6",
}
ROLES = ("near-left", "near-right", "far-left", "far-right")
METHODS = (*policy.POLICIES, policy.REANCHOR)
CONTRASTS = ((0, 1), (1, 2), (0, 3))
PAD = "__EMPTY__"


def canonical(tokens):
    """The release has tokens, not recoverable original whitespace."""
    text = " ".join(tokens)
    starts, ends, cursor = [], [], 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(token)
        ends.append(cursor)
        cursor += 1
    return dict(text=text, starts=starts, ends=ends,
                bounds=ref.unit_boundaries(text), units=ref.utf16(text))


def unique_edit(row):
    """Validate aligned tags, then enumerate every possible atomic token edit.

    For short/long arrays with length difference k, a legal edit boundary b
    needs b <= common_prefix and len(short)-b <= common_suffix. The inclusive
    interval below is therefore the complete set, including repeated tokens.
    """
    keys = ("src", "tgt", "yin_before", "yin_after", "src_tag", "changed")
    if not isinstance(row, dict) or type(row.get("id")) is not int or any(
            not isinstance(row.get(k), list) or not all(isinstance(x, str) for x in row[k])
            for k in keys):
        return None, "invalid_schema"
    old, new = row["src"], row["tgt"]
    before, after, tags = (row[k] for k in ("yin_before", "yin_after", "src_tag"))
    if not (len(before) == len(after) == len(tags)):
        return None, "alignment_length_mismatch"
    if [x for x in before if x != PAD] != old or [x for x in after if x != PAD] != new:
        return None, "alignment_reconstruction_mismatch"
    for a, b, tag in zip(before, after, tags):
        if not ((tag == "equal" and a == b and a != PAD)
                or (tag == "insert" and a == PAD and b != PAD)
                or (tag == "delete" and b == PAD and a != PAD)):
            return None, "invalid_alignment_tag_semantics"
    changed = [i for i, tag in enumerate(tags) if tag != "equal"]
    if not changed:
        return None, "empty_edit"
    directions = {tags[i] for i in changed}
    if len(directions) != 1:
        return None, "mixed_edit_direction"
    if changed != list(range(changed[0], changed[-1] + 1)):
        return None, "noncontiguous_edit"
    direction = directions.pop()
    edit_tokens = [(before if direction == "delete" else after)[i] for i in changed]
    if edit_tokens != row["changed"]:
        return None, "changed_token_mismatch"
    short, long = (old, new) if direction == "insert" else (new, old)
    k = len(long) - len(short)
    if k != len(changed) or k <= 0:
        return None, "edit_length_mismatch"
    prefix = suffix = 0
    while prefix < len(short) and short[prefix] == long[prefix]:
        prefix += 1
    while suffix < len(short) and short[-suffix-1] == long[-suffix-1]:
        suffix += 1
    legal = range(max(0, len(short)-suffix), min(prefix, len(short))+1)
    if len(legal) != 1:
        return None, "ambiguous_edit_position" if len(legal) else "no_legal_atomic_edit"
    b = legal.start
    aligned_b = sum(x != PAD for x in before[:changed[0]])
    if aligned_b != b:
        return None, "unique_position_disagrees_with_alignment"
    assert short[:b] == long[:b] and short[b:] == long[b+k:]
    old_end = b if direction == "insert" else b+k
    mapping = [i if i < b else i+k if direction == "insert" else
               None if i < old_end else i-k for i in range(len(old))]
    assert all(new[j] == old[i] for i, j in enumerate(mapping) if j is not None)
    return dict(direction=direction, start_token=b, end_token=old_end,
                edit_tokens=k, mapping=mapping, legal_edit_positions=1), "eligible"


def probe_geometry(row, row_index):
    """Outcome-independent eligibility, token-map truth, and saved S32 fields."""
    edit, reason = unique_edit(row)
    revision = dict(row_index=row_index, released_id=row.get("id") if isinstance(row, dict) else None,
                    eligibility_reason=reason, probe_n=0)
    tags = row.get("src_tag", []) if isinstance(row, dict) else []
    directions = {tag for tag in tags if tag != "equal"} if isinstance(tags, list) and all(isinstance(t, str) for t in tags) else set()
    revision["released_direction"] = next(iter(directions)) if len(directions) == 1 else "mixed_or_unknown"
    if edit is None:
        return revision, [], None
    old, new = canonical(row["src"]), canonical(row["tgt"])
    b, e, n = edit["start_token"], edit["end_token"], len(row["src"])
    left_boundary = old["starts"][b] if b < n else len(old["text"])
    right_boundary = old["ends"][e-1] if e > b else left_boundary
    revision.update(direction=edit["direction"], edit_tokens=edit["edit_tokens"],
                    edit_start_token=b, edit_end_token=e, legal_edit_positions=1,
                    old_edit_left_cp=left_boundary, old_edit_right_cp=right_boundary,
                    old_edit_left_u16=old["bounds"][left_boundary],
                    old_edit_right_u16=old["bounds"][right_boundary],
                    old_tokens=n, new_tokens=len(row["tgt"]),
                    exact_text_group=grid.text_digest(old["text"]),
                    new_display_sha256=grid.text_digest(new["text"]))
    candidates = (("near-left", b-3, b), ("near-right", e, e+3),
                  ("far-left", 0, 3), ("far-right", n-3, n))
    probes, seen = [], set()
    for role, start, end in candidates:
        status = "available"
        if not (0 <= start < end <= n):
            status = "fewer_than_three_tokens"
        elif (role.endswith("left") and end > b) or (role.endswith("right") and start < e):
            status = "not_wholly_on_edit_side"
        else:
            mapped = edit["mapping"][start:end]
            if any(x is None for x in mapped) or mapped != list(range(mapped[0], mapped[0]+3)):
                status = "not_unchanged_contiguous_run"
            else:
                span = (old["starts"][start], old["ends"][end-1])
                distance = (old["bounds"][left_boundary]-old["bounds"][span[1]]
                            if role.endswith("left") else
                            old["bounds"][span[0]]-old["bounds"][right_boundary])
                if role.startswith("far") and distance < policy.WIDTH:
                    status = "distance_below_32_u16"
                elif (start, end) in seen:
                    status = "duplicate_probe_span"
        revision[role.replace("-", "_")+"_status"] = status
        if status != "available":
            continue
        seen.add((start, end))
        mapped = edit["mapping"][start:end]
        truth = (new["starts"][mapped[0]], new["ends"][mapped[-1]])
        quote = old["text"][slice(*span)]
        assert end-start == 3 and quote == new["text"][slice(*truth)]
        old_u16 = tuple(old["bounds"][x] for x in span)
        truth_u16 = tuple(new["bounds"][x] for x in truth)
        s, t = old_u16
        prefix, suffix = old["units"][max(0, s-policy.WIDTH):s], old["units"][t:t+policy.WIDTH]
        fields = (ref.utf16(quote), prefix, suffix)
        eligible = policy.unicode_fields(fields) is not None
        left_full, right_full = len(prefix) == policy.WIDTH, len(suffix) == policy.WIDTH
        probes.append(dict(row_index=row_index, released_id=row["id"],
            exact_text_group=revision["exact_text_group"], new_display_sha256=revision["new_display_sha256"],
            direction=edit["direction"], role=role, proximity=role.split("-")[0],
            old_start_token=start, old_end_token=end, new_start_token=mapped[0], new_end_token=mapped[-1]+1,
            old_start_cp=span[0], old_end_cp=span[1], old_start_u16=s, old_end_u16=t,
            truth_start_cp=truth[0], truth_end_cp=truth[1],
            truth_start_u16=truth_u16[0], truth_end_u16=truth_u16[1],
            distance_to_old_edit_u16=distance, saved_width_u16=policy.WIDTH,
            available_left_u16=s, available_right_u16=len(old["units"])-t,
            saved_left_u16=len(prefix), saved_right_u16=len(suffix),
            field_stratum=("both_full" if left_full and right_full else "right_short" if left_full
                           else "left_short" if right_full else "both_short"),
            partial_surrogate=int(grid.partial_surrogate(prefix) or grid.partial_surrogate(suffix)),
            reanchor_eligible=int(eligible),
            exact_quote_count_before=len(base.occurrences(old["units"], fields[0])),
            exact_quote_count_after=len(base.occurrences(new["units"], fields[0]))))
    revision["probe_n"] = len(probes)
    return revision, probes, (old, new)


def prepare(data_path, index_path):
    """All released test indices in their frozen file order; no selector calls."""
    sources = {"insertions_deletions.jsonl": data_path, "splits/test.txt": index_path}
    for name, expected in SOURCE_HASHES.items():
        assert grid.digest(sources[name]) == expected, ("Frozen PEER input changed", name)
    indices = list(map(int, index_path.read_text().split()))
    assert len(indices) == len(set(indices)) == 10400 and all(0 <= i < 104000 for i in indices)
    wanted, rows = set(indices), {}
    with data_path.open() as stream:
        for row_index, line in enumerate(stream):
            if row_index in wanted:
                rows[row_index] = json.loads(line)
    assert row_index+1 == 104000 and set(rows) == wanted
    revisions, records, texts = [], [], {}
    for row_index in indices:
        revision, probes, canonical_pair = probe_geometry(rows[row_index], row_index)
        revisions.append(revision)
        if probes:
            texts[row_index] = canonical_pair
        for probe in probes:
            records.append(dict(unit_index=len(records), **probe))
    return revisions, records, texts


def geometry_summary(revisions, records):
    eligible = [r for r in revisions if r["eligibility_reason"] == "eligible"]
    return dict(released_test_revisions=len(revisions), eligible_revisions=len(eligible),
        revisions_with_probes=sum(r["probe_n"] > 0 for r in revisions), probes=len(records),
        exact_old_sentence_groups=len({r["exact_text_group"] for r in records}),
        eligibility_reasons=dict(Counter(r["eligibility_reason"] for r in revisions)),
        released_direction_counts=dict(Counter(r["released_direction"] for r in revisions)),
        direction_eligibility_reasons=dict(Counter(r["released_direction"]+"/"+r["eligibility_reason"] for r in revisions)),
        eligible_direction_counts=dict(Counter(r["direction"] for r in eligible)),
        probes_per_revision=dict(sorted(Counter(r["probe_n"] for r in eligible).items())),
        role_status={role: dict(Counter(r[role.replace("-", "_")+"_status"] for r in eligible)) for role in ROLES},
        probe_direction_role=dict(Counter(r["direction"]+"/"+r["role"] for r in records)),
        reanchor_eligible_probes=sum(r["reanchor_eligible"] for r in records),
        partial_surrogate_probes=sum(r["partial_surrogate"] for r in records),
        field_availability=dict(Counter(r["field_stratum"] for r in records)))


def summarize(records, codes, reps=grid.REPS):
    """Natural probe axis only; shared exact-old-sentence bootstrap across methods."""
    groups = sorted({r["exact_text_group"] for r in records})
    lookup = {name: i for i, name in enumerate(groups)}
    ids = np.array([lookup[r["exact_text_group"]] for r in records])
    ng, n = len(groups), len(records)
    assert codes.shape == (n, len(METHODS)) and n and reps > 0
    rng = np.random.default_rng(grid.SEED)
    weights = np.stack([np.bincount(rng.integers(ng, size=ng), minlength=ng)
                        for _ in range(reps)]).astype(float)
    frames = (("all_probes", np.ones(n, bool)),
              ("reanchor_eligible", np.array([r["reanchor_eligible"] for r in records], bool)))
    summary, transitions, intervals, strata = [], [], [], []
    for frame, mask in frames:
        count = np.bincount(ids[mask], minlength=ng)
        active = count > 0
        den = weights @ count
        group_den = weights @ active.astype(np.int32)
        valid = den > 0

        def stats(values, uncertainty=False, subset=mask):
            counts = count if subset is mask else np.bincount(ids[subset], minlength=ng)
            present = counts > 0
            total = np.bincount(ids[subset], weights=values[subset], minlength=ng)
            fractions = np.divide(total, counts, out=np.zeros(ng), where=present)
            result = dict(annotation=float(total.sum()/subset.sum()) if subset.any() else None,
                          equal_old_sentence=float(fractions[present].mean()) if present.any() else None)
            if uncertainty:
                assert subset is mask
                for name, draws in (("annotation", (weights @ total)[valid]/den[valid]),
                                    ("equal_old_sentence", (weights @ fractions)[valid]/group_den[valid])):
                    result[name+"_lo"] = float(np.quantile(draws, .025)) if len(draws) else None
                    result[name+"_hi"] = float(np.quantile(draws, .975)) if len(draws) else None
            return result

        key = dict(frame=frame, n=int(mask.sum()), groups=int(active.sum()),
                   saved_width_u16=policy.WIDTH, bootstrap_pool_groups=ng)
        methods = range(4) if frame == "reanchor_eligible" else range(3)
        levels = [("direction", d, np.array([r["direction"] == d for r in records])) for d in ("insert", "delete")]
        levels += [("proximity", p, np.array([r["proximity"] == p for r in records])) for p in ("near", "far")]
        levels += [("role", p, np.array([r["role"] == p for r in records])) for p in ROLES]
        levels += [("direction_role", d+"/"+p, np.array([r["direction"] == d and r["role"] == p for r in records]))
                   for d in ("insert", "delete") for p in ROLES]
        for k in methods:
            row = dict(**key, method=METHODS[k])
            for code, name in enumerate(grid.OUTCOMES):
                yes = codes[:, k] == code
                row[name+"_n"] = int((yes & mask).sum())
                row.update({name+"_"+w: v for w, v in stats(yes).items()})
            assert sum(row[name+"_n"] for name in grid.OUTCOMES) == row["n"]
            summary.append(row)
            for axis, level, member in levels:
                subset = mask & member
                entry = dict(**key, method=METHODS[k], stratum_axis=axis, stratum=level,
                             stratum_n=int(subset.sum()), stratum_groups=len(set(ids[subset])))
                for code, name in enumerate(grid.OUTCOMES):
                    yes = codes[:, k] == code
                    entry[name+"_n"] = int((yes & subset).sum())
                    entry.update({name+"_"+w: v for w, v in stats(yes, subset=subset).items()})
                strata.append(entry)
        for a, b in CONTRASTS:
            if b == 3 and frame != "reanchor_eligible":
                continue
            pair = dict(**key, baseline=METHODS[a], alternative=METHODS[b])
            before, after = codes[:, a], codes[:, b]
            for old, old_name in enumerate(grid.OUTCOMES):
                for new, new_name in enumerate(grid.OUTCOMES):
                    yes = (before == old) & (after == new)
                    transitions.append(dict(**pair, before=old_name, after=new_name,
                                            count=int((yes & mask).sum()), **stats(yes)))
            values = {name+"_difference": (after == code).astype(float)-(before == code)
                      for code, name in enumerate(grid.OUTCOMES)}
            values.update(wrong_repaired=((before == 2) & (after == 0)).astype(float),
                          correct_broken=((before == 0) & (after == 2)).astype(float))
            for metric, value in values.items():
                intervals.append(dict(**pair, metric=metric,
                    orientation="alternative_minus_baseline_or_transition_fraction", bootstrap_replicates=reps,
                    bootstrap_seed=grid.SEED, valid_replicates=int(valid.sum()),
                    empty_replicates=int((~valid).sum()), **stats(value, True)))
    return summary, transitions, intervals, strata


def evaluate(records, texts, module):
    n = len(records)
    codes = np.full((n, 4), policy.INCOMPARABLE, np.uint8)
    prediction_cp = np.full((n, 4, 2), -1, np.int32)
    prediction_u16 = prediction_cp.copy()
    truth_cp = np.array([(r["truth_start_cp"], r["truth_end_cp"]) for r in records], np.int32)
    truth_u16 = np.array([(r["truth_start_u16"], r["truth_end_u16"]) for r in records], np.int32)
    stages, native_rows = np.empty(n, np.int8), []
    for i, r in enumerate(records):
        old, new = texts[r["row_index"]]
        span = (r["old_start_cp"], r["old_end_cp"])
        old_span = (r["old_start_u16"], r["old_end_u16"])
        quote = ref.utf16(old["text"][slice(*span)])
        prefix = old["units"][max(0, old_span[0]-policy.WIDTH):old_span[0]]
        suffix = old["units"][old_span[1]:old_span[1]+policy.WIDTH]
        # Truth already exists; matcher calls receive only their pinned interfaces.
        answers, stages[i], candidates, _ = policy.policy_answers(new["units"], quote, prefix, suffix, old_span)
        assert len(candidates) == r["exact_quote_count_after"] and tuple(truth_u16[i]) in candidates
        reverse = {u: cp for cp, u in enumerate(new["bounds"])}
        for k, answer in enumerate(answers):
            cp = None
            if answer is not None:
                assert answer[0] in reverse and answer[1] in reverse
                cp = (reverse[answer[0]], reverse[answer[1]])
                assert new["text"][slice(*cp)] == old["text"][slice(*span)]
                prediction_cp[i, k], prediction_u16[i, k] = cp, answer
            codes[i, k] = policy.outcome(cp, tuple(truth_cp[i]))
        eligible, result = policy.reanchor_answer(module, new["text"], (quote, prefix, suffix))
        assert eligible == bool(r["reanchor_eligible"])
        if eligible:
            cp = (result.start, result.end) if result is not None else None
            codes[i, 3] = policy.outcome(cp, tuple(truth_cp[i]))
            if cp is not None:
                prediction_cp[i, 3] = cp
                prediction_u16[i, 3] = tuple(new["bounds"][x] for x in cp)
        native_rows.append(dict(unit_index=i, native_stage=int(stages[i]),
            reanchor_eligible=int(eligible), incomparable_reason="" if eligible else "partial_surrogate_saved_field",
            reanchor_method=result.method if result else "", confidence=result.confidence if result else None,
            distance=result.distance if result else None, rival_count=len(result.rivals) if result else 0,
            predicted_exact_quote=int(result.text == old["text"][slice(*span)]) if result else None))
        if (i+1) % 1000 == 0:
            print(f"natural probes: {i+1}/{n}", flush=True)
    assert np.isin(codes[:, :3], (0, 1, 2)).all()
    assert np.array_equal(codes == 0, np.all(prediction_cp == truth_cp[:, None, :], axis=-1))
    assert np.array_equal((codes == 1) | (codes == policy.INCOMPARABLE), np.all(prediction_cp == -1, axis=-1))
    assert np.array_equal(codes[:, 3] == policy.INCOMPARABLE, ~np.array([r["reanchor_eligible"] for r in records], bool))
    return dict(outcome=codes, predicted_cp=prediction_cp, predicted_u16=prediction_u16,
                truth_cp=truth_cp, truth_u16=truth_u16, native_stage=stages), native_rows


def synthetic(old, edit, b, direction="insert"):
    """Tiny released-schema fixture; used only by the runnable self-check."""
    new = old[:b]+edit+old[b:] if direction == "insert" else old[:b]+old[b+len(edit):]
    before = old[:b]+[PAD]*len(edit)+old[b:] if direction == "insert" else old
    after = new if direction == "insert" else old[:b]+[PAD]*len(edit)+old[b+len(edit):]
    tags = ["equal"]*b+[direction]*len(edit)+["equal"]*(len(before)-b-len(edit))
    return dict(id=0, src=old, tgt=new, yin_before=before, yin_after=after, src_tag=tags, changed=edit)


def self_check():
    # Any accidental selector use during preparation fails before synthetic evaluation.
    original_answers, original_reanchor = policy.policy_answers, policy.reanchor_answer
    def forbidden(*args, **kwargs):
        raise AssertionError("Selector invoked while deriving geometry")
    policy.policy_answers = policy.reanchor_answer = forbidden
    try:
        assert unique_edit(synthetic(["a", "a"], ["a"], 1))[1] == "ambiguous_edit_position"
        assert unique_edit(synthetic(["a", "a"], ["a"], 0, "delete"))[1] == "ambiguous_edit_position"
        tokens = ["😀", "b", "c", "x"*40, "e", "f", "g", "h", "i", "j", "y"*40, "l", "m", "n"]
        fixtures = [synthetic(tokens, ["EDIT"], 7), synthetic(tokens[:7]+["EDIT"]+tokens[7:], ["EDIT"], 7, "delete")]
        examples = []
        for row in fixtures:
            rev, probes, texts = probe_geometry(row, 0)
            assert rev["eligibility_reason"] == "eligible" and len(probes) == 4
            assert [r["role"] for r in probes] == list(ROLES)
            assert all(r["old_end_token"]-r["old_start_token"] == 3 for r in probes)
            assert all(r["distance_to_old_edit_u16"] >= 32 for r in probes if r["proximity"] == "far")
            far = next(r for r in probes if r["role"] == "far-left")
            assert far["old_end_u16"] == far["old_end_cp"]+1
            old, new = texts
            for r in probes:
                assert old["text"][r["old_start_cp"]:r["old_end_cp"]] == new["text"][r["truth_start_cp"]:r["truth_end_cp"]]
            examples.append((probes, texts))
        _, probes, _ = probe_geometry(synthetic(["a", "b", "c", "d", "e", "f"], ["EDIT"], 3), 0)
        assert [r["role"] for r in probes] == ["near-left", "near-right"]
        rev, probes, _ = probe_geometry(synthetic(["a", "b"], ["EDIT"], 0), 0)
        assert not probes and rev["near_left_status"] == "fewer_than_three_tokens"
        # Equality at 32 is admitted; 31 is not. All coordinates use UTF-16.
        for padding, expected in ((29, False), (30, True)):
            rev, probes, _ = probe_geometry(synthetic(["a", "b", "c", "x"*padding, "d", "e", "f"], ["EDIT"], 4), 0)
            assert any(r["role"] == "far-left" for r in probes) == expected
        _, probes, _ = probe_geometry(synthetic(["😀"+"p"*30, "a", "b", "c", "d", "e", "f"], ["EDIT"], 4), 0)
        near = next(r for r in probes if r["role"] == "near-left")
        assert near["partial_surrogate"] == 1 and near["reanchor_eligible"] == 0
    finally:
        policy.policy_answers, policy.reanchor_answer = original_answers, original_reanchor
    module = policy.vendor_module()
    for probes, texts in examples:
        result, native = evaluate(probes, {0: texts}, module)
        assert result["outcome"].shape == (4, 4) and len(native) == 4
    records = [dict(exact_text_group=g, reanchor_eligible=e, direction="insert", role="near-left", proximity="near")
               for g, e in (("a", 0), ("a", 1), ("b", 1))]
    codes = np.zeros((3, 4), np.uint8)
    codes[0, 0], codes[0, 3] = 2, policy.INCOMPARABLE
    summaries, transitions, intervals, strata = summarize(records, codes, reps=16)
    row = next(r for r in intervals if r["frame"] == "all_probes" and r["alternative"] == METHODS[1] and r["metric"] == "correct_difference")
    assert row["annotation"] == 1/3 and row["equal_old_sentence"] == .25
    assert len(summaries) == 7 and len(transitions) == 45 and len(intervals) == 25
    assert all(r["n"] == 2 for r in summaries if r["frame"] == "reanchor_eligible")
    assert not any(r["method"] == METHODS[3] and r["frame"] == "all_probes" for r in summaries)
    assert all(sum(t["count"] for t in transitions if all(t[k] == row[k]
                   for k in ("frame", "baseline", "alternative"))) == row["n"] for row in intervals)
    print("PASS: selector-free token truth; ambiguous repeats excluded; insert/delete; near/far thresholds; "
          "three-token spans; astral UTF-16; native adapters; shared group bootstrap and matched frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="External released insertions_deletions.jsonl")
    parser.add_argument("--test-indices", type=Path, help="External released splits/test.txt")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "self-check", "pilot", "full"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--out", type=Path, help="Fresh external output directory; optional for geometry-only prepare")
    parser.add_argument("--pilot-revisions", type=int, default=32, help="First probe-bearing revisions in released test order")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.prepare and args.out is None:
        parser.error("--pilot/--full require --out")
    if args.pilot_revisions <= 0:
        parser.error("--pilot-revisions must be positive")
    started = time.time()
    if args.input is None or args.test_indices is None:
        parser.error("--input and --test-indices are required")
    inputs = [args.input, args.test_indices, Path(__file__), Path(policy.__file__),
              Path(grid.__file__), Path(ref.__file__), Path(data_io.__file__)]
    hashes = data_io.hashes(*inputs)
    revisions, records, texts = prepare(args.input, args.test_indices)
    geometry = geometry_summary(revisions, records)
    print(json.dumps(geometry, indent=2), flush=True)
    if args.prepare and args.out is None:
        assert all(grid.digest(p) == sha for p, sha in hashes.items()), "Input changed during geometry preparation"
        return
    out = data_io.fresh_output(args.out)
    if args.pilot:
        selected = set([r["row_index"] for r in revisions if r["probe_n"] > 0][:args.pilot_revisions])
        records = [r for r in records if r["row_index"] in selected]
        records = [dict(r, unit_index=i) for i, r in enumerate(records)]
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", mode="prepare" if args.prepare else "pilot" if args.pilot else "full",
        pilot=bool(args.pilot), pilot_revision_limit=args.pilot_revisions if args.pilot else None,
        full_geometry=geometry, evaluated_probes=0, input_sha256=hashes,
        python=platform.python_version(), numpy=np.__version__, saved_width_u16=policy.WIDTH,
        coordinate_frame="canonical tokens joined by U+0020; code-point and UTF-16 offsets; no raw whitespace recovery",
        edit_boundary="deletion token-content start/end; insertion next old-token start or document end",
        truth="unique token edit agreeing released alignment; preserved contiguous three-token map",
        group_definition="SHA-256 of exact old canonical sentence; not article or revision chain",
        training=False, new_annotations=False, source_text_exported=False)
    base.write_json(out / "manifest.json", manifest)
    base.write_csv(out / "revisions.csv", revisions)
    base.write_csv(out / "records.csv", records)
    if not args.prepare:
        module = policy.vendor_module()
        for p in [policy.SOURCE / p for p in policy.VENDOR_HASHES]:
            hashes[str(p)] = grid.digest(p)
        arrays, native = evaluate(records, texts, module)
        np.savez_compressed(out / "outcomes.npz", **arrays)
        base.write_csv(out / "native_metadata.csv", native)
        for name, rows in zip(("summary", "transitions", "paired_intervals", "descriptive_strata"), summarize(records, arrays["outcome"])):
            base.write_csv(out / (name+".csv"), rows)
        manifest.update(evaluated_probes=len(records), methods=METHODS,
            contrasts=[dict(baseline=METHODS[a], alternative=METHODS[b]) for a, b in CONTRASTS],
            outcome_codes={0: "correct", 1: "abstain", 2: "wrong", 3: "incomparable"},
            native_stage_codes={0: "quote_checked_position", 1: "exact_candidates_fuzzy_context", 2: "empty_quote"},
            npz_axes="records.csv unit_index, manifest methods; no synthetic edit axis",
            bootstrap_replicates=grid.REPS, bootstrap_seed=grid.SEED,
            inference="pointwise 95% paired exact-old-sentence group percentile intervals, shared draws; no multiplicity correction",
            transition_rates="fractions of full stated frame, not conditional on baseline outcome",
            hypothesis_commit=ref.CLIENT_COMMIT, approximate_match_commit=ref.APPROX_COMMIT,
            reanchor=dict(commit=policy.COMMIT, runtime_version=module.__version__, options=asdict(module.ResolveOptions()),
                          interface="new sentence + quote/prefix/suffix; no old position; lossless Unicode fields only"))
    assert all(grid.digest(p) == sha for p, sha in hashes.items()), "Inputs changed during run"
    manifest.update(status="completed", elapsed_seconds=time.time()-started,
        output_sha256={p.name: grid.digest(p) for p in out.iterdir() if p.name != "manifest.json"})
    base.write_json(out / "manifest.json", manifest)
    print(f"DONE {manifest['mode']}: {len(records)} probes; {out}", flush=True)


if __name__ == "__main__":
    main()

