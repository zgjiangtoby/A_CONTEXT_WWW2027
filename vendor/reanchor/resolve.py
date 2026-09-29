"""Quote resolution.

Given a W3C-style TextQuoteSelector -- the quoted text plus a little context
either side -- and a document that may have changed since the quote was taken,
find where the quote lives now.

The strategy is a ladder, cheapest and most trustworthy rung first:

1. Exact match of quote plus context, in the raw document.
2. Exact match of the quote alone; context disambiguates between hits.
3. Steps 1-2 again over normalized text, which absorbs re-typeset whitespace,
   curly quotes, and hyphenated line breaks.
4. Approximate match over normalized text, for genuine edits and OCR noise,
   scored by both quote distance and how well the context survived.

A resolver's most important property is knowing when it has failed. Every
result therefore carries the rung that produced it and a confidence, and
:func:`resolve_quote` returns ``None`` rather than the least-bad window when
nothing clears the threshold. A silently wrong citation is worse than an absent
one: it looks verified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Final, Literal

from .normalize import (
    DEFAULT_OPTIONS,
    NormalizedText,
    NormalizeOptions,
    normalize,
    to_source_span,
)
from .search import (
    GramIndex,
    SearchOptions,
    build_gram_index,
    find_approximate,
    find_exact,
)

__all__ = [
    "MatchMethod",
    "PreparedDocument",
    "ResolveOptions",
    "ResolvedQuote",
    "TextQuoteSelector",
    "prepare_document",
    "resolve_quote",
    "resolve_quotes",
]

#: The rung of the ladder that produced a match, from most to least
#: trustworthy. Callers that must not show a wrong citation can require
#: ``"exact-with-context"``; callers that prefer a probably-right location to
#: none can accept ``"approximate"``.
MatchMethod = Literal[
    "exact-with-context",
    "exact",
    "normalized-with-context",
    "normalized",
    "approximate",
]


@dataclass(frozen=True, slots=True)
class TextQuoteSelector:
    """A quote and the context it was taken with."""

    #: The quoted text.
    exact: str
    #: Text immediately before the quote, if it was captured.
    prefix: str = ""
    #: Text immediately after the quote, if it was captured.
    suffix: str = ""


@dataclass(frozen=True, slots=True)
class ResolveOptions:
    """Thresholds and limits for :func:`resolve_quote`."""

    #: Reject matches below this confidence. The default admits substantial
    #: editing while still refusing near-noise.
    min_confidence: float = 0.5
    #: Largest tolerated edit distance for the approximate rung, as a fraction
    #: of quote length. Above roughly 0.4 the matches stop being the same
    #: sentence.
    max_edit_ratio: float = 0.3
    #: How many characters of context to weigh either side. Longer context
    #: disambiguates repeated passages better but is likelier to have been
    #: edited itself.
    context_length: int = 32
    #: Stop at this rung; lower rungs are not attempted.
    max_method: MatchMethod = "approximate"
    #: Normalization, which must agree with any prepared document.
    normalize: NormalizeOptions = DEFAULT_OPTIONS
    #: How many rivals to report.
    max_rivals: int = 3


DEFAULT_RESOLVE_OPTIONS: Final = ResolveOptions()


@dataclass(frozen=True, slots=True)
class ResolvedQuote:
    """Where a quote is now, and how much to trust that answer."""

    #: Start of the quote in the document, as a string index.
    start: int
    #: End of the quote in the document, as a string index.
    end: int
    #: The document text actually spanned, useful for display and assertions.
    text: str
    #: Which rung produced this match.
    method: MatchMethod
    #: How much to trust this match, in ``(0, 1]``. Exact matches with agreeing
    #: context score 1. Approximate matches are penalized by their edit
    #: distance and by context that failed to survive.
    confidence: float
    #: Levenshtein distance to the quote, measured on normalized text.
    distance: int
    #: Other locations that matched comparably well. A non-empty value means
    #: the document repeats this passage and the context was not enough to tell
    #: the copies apart; treat the primary result with suspicion. Rivals
    #: themselves carry no rivals.
    rivals: tuple[ResolvedQuote, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    """A document with its normalization and seed index computed once.

    Resolving many quotes against one document -- the usual case for a citation
    checker -- should reuse this rather than re-normalizing per quote.
    """

    text: str
    normalized: NormalizedText
    index: GramIndex
    options: NormalizeOptions


_METHOD_ORDER: Final[tuple[MatchMethod, ...]] = (
    "exact-with-context",
    "exact",
    "normalized-with-context",
    "normalized",
    "approximate",
)

#: Confidence awarded by each rung before context and ambiguity adjustments.
#:
#: The exact rungs score 1: the quote was found character for character, and a
#: selector that recorded no context is not thereby less certain. The
#: normalized rungs give up a little because normalization can in principle
#: conflate texts that differ -- ``café`` and ``cafe`` reach the same needle.
#: The approximate rung starts lower still and is then scaled by similarity.
_METHOD_CONFIDENCE: Final[dict[MatchMethod, float]] = {
    "exact-with-context": 1.0,
    "exact": 1.0,
    "normalized-with-context": 0.98,
    "normalized": 0.98,
    "approximate": 0.95,
}

#: Applied to a candidate its own rung could not single out -- see
#: :func:`_penalize_ambiguous`.
_AMBIGUITY_PENALTY: Final = 0.8

#: Confidences within this distance count as tied, so neither wins on rounding.
_TIE_EPSILON: Final = 1e-9

#: Floor of the context factor: a match whose recorded context is entirely
#: absent from the document keeps this fraction of its confidence.
_CONTEXT_WEIGHT: Final = 0.7


def prepare_document(
    text: str, options: NormalizeOptions = DEFAULT_OPTIONS
) -> PreparedDocument:
    """Normalize and index ``text`` once, for reuse across many selectors."""
    normalized = normalize(text, options)
    return PreparedDocument(
        text=text,
        normalized=normalized,
        index=build_gram_index(normalized.text),
        options=options,
    )


def resolve_quote(
    document: str | PreparedDocument,
    selector: TextQuoteSelector,
    options: ResolveOptions = DEFAULT_RESOLVE_OPTIONS,
) -> ResolvedQuote | None:
    """Locate ``selector`` in ``document``.

    Returns ``None`` if the quote cannot be found with enough confidence.
    """
    if not selector.exact:
        return None

    limit = _METHOD_ORDER.index(options.max_method)

    prepared = (
        prepare_document(document, options.normalize)
        if isinstance(document, str)
        else document
    )
    prefix = _clip_end(selector.prefix, options.context_length)
    suffix = _clip_start(selector.suffix, options.context_length)

    candidates: list[_Candidate] = []
    for rung in range(limit + 1):
        candidates.extend(
            _penalize_ambiguous(
                _attempt(
                    _METHOD_ORDER[rung], prepared, selector, prefix, suffix, options
                )
            )
        )
        if candidates and not _can_be_beaten(candidates, rung, limit):
            break
    if not candidates:
        return None

    # One location can be reached by several rungs -- the exact rung and the
    # normalized rung find the same span whenever normalization changed nothing
    # relevant. Keep each location once, at its best score, so that a span cannot
    # appear as a rival to itself.
    by_location: dict[tuple[int, int], _Candidate] = {}
    for candidate in candidates:
        key = (candidate.start, candidate.end)
        seen = by_location.get(key)
        if seen is None or candidate.confidence > seen.confidence:
            by_location[key] = candidate
    distinct = list(by_location.values())

    distinct.sort(key=lambda candidate: (-candidate.confidence, candidate.start))
    best = distinct[0]
    if best.confidence < options.min_confidence:
        return None

    rivals = tuple(
        _materialize(candidate, prepared.text)
        for candidate in distinct[1:]
        if candidate.confidence >= options.min_confidence
        and candidate.confidence >= best.confidence - 0.05
    )[: options.max_rivals]

    return replace(_materialize(best, prepared.text), rivals=rivals)


def _can_be_beaten(candidates: Sequence[_Candidate], rung: int, limit: int) -> bool:
    """Whether a lower rung could still produce a better answer than these.

    The ladder is ordered by trustworthiness, so the obvious implementation stops
    at the first rung that yields anything. That is wrong when the quote survives
    verbatim somewhere it does not belong: a decoy found by the exact rung whose
    surroundings bear no resemblance to the recorded context scores *below* what
    the approximate rung can award the copy-edited original, whose surroundings
    still match. Stopping early means those two candidates are never compared.

    So descend while the best score so far is under the ceiling a lower rung
    could reach -- its base confidence, since similarity, context agreement and
    the ambiguity penalty can only reduce it. A match with fully agreeing context
    scores its rung's base confidence, which is at or above every lower rung's
    ceiling, so the unambiguous case still stops at the first rung and costs
    nothing.
    """
    ceiling = max(
        (
            _METHOD_CONFIDENCE[_METHOD_ORDER[next_]]
            for next_ in range(rung + 1, limit + 1)
        ),
        default=0.0,
    )
    return max(candidate.confidence for candidate in candidates) < ceiling


def _penalize_ambiguous(candidates: list[_Candidate]) -> list[_Candidate]:
    """Discount candidates that their own rung could not single out.

    Counting hits is the tempting test, and it is the wrong one: a rung that
    found three locations but whose recorded context matches exactly one of them
    *has* told them apart, and penalizing the survivor for the company it kept
    lets a worse rung's unambiguous-but-wrong answer outrank it. What ambiguity
    should cost is being indistinguishable, so the penalty falls on candidates
    tied at the top of their rung -- where the resolver is genuinely choosing by
    coin flip -- and a candidate that context put in front on its own keeps its
    score.

    A single hit is trivially in front and so passes through unchanged, which is
    why this is not a behaviour change for the common case.
    """
    if len(candidates) <= 1:
        return candidates
    best = max(candidate.confidence for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if candidate.confidence >= best - _TIE_EPSILON
    ]
    if len(tied) <= 1:
        return candidates
    return [
        replace(candidate, confidence=candidate.confidence * _AMBIGUITY_PENALTY)
        if candidate.confidence >= best - _TIE_EPSILON
        else candidate
        for candidate in candidates
    ]


def resolve_quotes(
    document: str,
    selectors: Sequence[TextQuoteSelector],
    options: ResolveOptions = DEFAULT_RESOLVE_OPTIONS,
) -> list[ResolvedQuote | None]:
    """Resolve many quotes against one document.

    Equivalent to calling :func:`resolve_quote` per selector, but normalizes
    and indexes the document once.
    """
    prepared = prepare_document(document, options.normalize)
    return [resolve_quote(prepared, selector, options) for selector in selectors]


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    method: MatchMethod
    confidence: float
    distance: int


def _attempt(
    method: MatchMethod,
    prepared: PreparedDocument,
    selector: TextQuoteSelector,
    prefix: str,
    suffix: str,
    options: ResolveOptions,
) -> list[_Candidate]:
    if method == "exact-with-context":
        if not prefix and not suffix:
            return []
        joined = prefix + selector.exact + suffix
        return [
            _Candidate(
                start=at + len(prefix),
                end=at + len(prefix) + len(selector.exact),
                method=method,
                confidence=_METHOD_CONFIDENCE[method],
                distance=0,
            )
            for at in find_exact(prepared.text, joined)
        ]

    if method == "exact":
        hits = find_exact(prepared.text, selector.exact)
        return [
            _Candidate(
                start=at,
                end=at + len(selector.exact),
                method=method,
                confidence=_score(
                    _METHOD_CONFIDENCE[method],
                    _context_agreement(
                        prepared.text,
                        at,
                        at + len(selector.exact),
                        prefix,
                        suffix,
                    ),
                ),
                distance=0,
            )
            for at in hits
        ]

    if method in ("normalized-with-context", "normalized"):
        with_context = method == "normalized-with-context"
        if with_context and not prefix and not suffix:
            return []
        normalized_quote = normalize(selector.exact, prepared.options).text
        if not normalized_quote:
            return []
        needle = (
            normalize(prefix + selector.exact + suffix, prepared.options).text
            if with_context
            else normalized_quote
        )
        if not needle:
            return []

        # Normalization can shift where the quote sits inside the joined needle,
        # so locate it rather than assuming the prefix length carried over.
        offset = needle.find(normalized_quote) if with_context else 0
        if offset < 0:
            return []

        hits = find_exact(prepared.normalized.text, needle)
        candidates: list[_Candidate] = []
        for at in hits:
            span = to_source_span(
                prepared.normalized,
                at + offset,
                at + offset + len(normalized_quote),
            )
            agreement = (
                1.0
                if with_context
                else _context_agreement(
                    prepared.text, span.start, span.end, prefix, suffix
                )
            )
            candidates.append(
                _Candidate(
                    start=span.start,
                    end=span.end,
                    method=method,
                    confidence=_score(_METHOD_CONFIDENCE[method], agreement),
                    distance=0,
                )
            )
        return candidates

    normalized_quote = normalize(selector.exact, prepared.options).text
    if not normalized_quote:
        return []
    matches = find_approximate(
        prepared.normalized.text,
        normalized_quote,
        SearchOptions(max_edit_ratio=options.max_edit_ratio, index=prepared.index),
    )
    candidates = []
    for match in matches:
        span = to_source_span(prepared.normalized, match.start, match.end)
        similarity = 1 - match.distance / max(len(normalized_quote), 1)
        agreement = _context_agreement(
            prepared.text, span.start, span.end, prefix, suffix
        )
        candidates.append(
            _Candidate(
                start=span.start,
                end=span.end,
                method="approximate",
                confidence=_score(
                    _METHOD_CONFIDENCE["approximate"] * similarity,
                    agreement,
                ),
                distance=match.distance,
            )
        )
    return candidates


def _score(base: float, agreement: float) -> float:
    """Combine a rung's base confidence with how well the context survived.

    Context agreement asks whether the surroundings still look like the ones
    recorded -- a selector that recorded no context scores 1 here, because absent
    evidence is not evidence against. Whether the rung could tell this location
    apart from the others it found is a separate question, answered afterwards by
    :func:`_penalize_ambiguous`, once every candidate on the rung has been scored
    and it is possible to see whether context put one of them in front.
    """
    return base * (_CONTEXT_WEIGHT + (1 - _CONTEXT_WEIGHT) * agreement)


def _context_agreement(
    document: str,
    start: int,
    end: int,
    prefix: str,
    suffix: str,
) -> float:
    """How well the text around a candidate matches the recorded context.

    In ``[0, 1]``. Scored on normalized text so that re-typesetting does not
    read as disagreement, and by longest common suffix/prefix so that context
    which was itself partly edited still contributes. With no recorded context
    the answer is 1: absent evidence is not evidence against.
    """
    total = 0
    agreement = 0

    if prefix:
        before = normalize(document[max(0, start - len(prefix) * 2) : start]).text
        want = normalize(prefix).text
        total += len(want)
        agreement += _common_suffix_length(before, want)
    if suffix:
        tail = document[end : min(len(document), end + len(suffix) * 2)]
        after = normalize(tail).text
        want = normalize(suffix).text
        total += len(want)
        agreement += _common_prefix_length(after, want)
    return 1.0 if total == 0 else agreement / total


def _common_prefix_length(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_length(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    return i


def _materialize(candidate: _Candidate, document: str) -> ResolvedQuote:
    return ResolvedQuote(
        start=candidate.start,
        end=candidate.end,
        text=document[candidate.start : candidate.end],
        method=candidate.method,
        confidence=min(1.0, round(candidate.confidence, 6)),
        distance=candidate.distance,
        rivals=(),
    )


def _clip_end(value: str, length: int) -> str:
    return value if len(value) <= length else value[len(value) - length :]


def _clip_start(value: str, length: int) -> str:
    return value if len(value) <= length else value[:length]
