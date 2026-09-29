"""Approximate substring search.

Locating a quote in a document that has since been edited is approximate
substring matching: find the window of the haystack whose edit distance to the
needle is smallest. Two well-known algorithms are combined here.

Sellers' variant of Levenshtein alignment does the actual measuring. It leaves
gaps at both ends of the haystack free, so it reports the best *infix*
alignment rather than forcing the needle to consume the whole window, and it
carries the alignment's origin alongside the cost so the matched window can be
recovered without a traceback matrix.

Running that over an entire document would cost O(needle x document), so a
k-gram diagonal vote narrows the field first: shared k-grams between needle and
haystack agree on the offset ``position - index`` when they belong to the same
occurrence, and disagree at random otherwise. Clustering those offsets yields a
handful of candidate windows, each of which is then measured exactly. This is
the standard seed-and-extend structure, and it keeps the cost proportional to
the number of plausible occurrences rather than to document length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, NamedTuple

__all__ = [
    "ApproximateMatch",
    "GramIndex",
    "SearchOptions",
    "build_gram_index",
    "find_approximate",
    "find_exact",
]

#: Characters per seed. Four is short enough to survive an edit every few words.
GRAM_SIZE: Final = 4

#: Seeds occurring more often than this carry no locality information -- in a
#: document about anchoring, ``" the "`` votes for everywhere -- so they are
#: skipped.
_MAX_POSTINGS: Final = 256

#: Never run the quadratic alignment over more than this many windows.
_DEFAULT_MAX_WINDOWS: Final = 8

#: Cap on unseeded whole-document alignment, which is quadratic.
_MAX_UNSEEDED_HAYSTACK: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class GramIndex:
    """Seed postings over one haystack, reusable across many needles."""

    size: int
    postings: dict[str, list[int]]


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """Tuning for :func:`find_approximate`."""

    #: Largest tolerated distance, as a fraction of needle length. Matches
    #: worse than this are not returned at all.
    max_edit_ratio: float = 0.3
    #: How many candidate windows to align.
    max_windows: int = _DEFAULT_MAX_WINDOWS
    #: Prebuilt index over the haystack, to amortize across many needles.
    index: GramIndex | None = field(default=None)


DEFAULT_SEARCH_OPTIONS: Final = SearchOptions()


class ApproximateMatch(NamedTuple):
    """A window of the haystack and how far it is from the needle."""

    #: Start of the matched window, in haystack coordinates.
    start: int
    #: End of the matched window, in haystack coordinates.
    end: int
    #: Levenshtein distance between the needle and that window.
    distance: int


def build_gram_index(text: str, size: int = GRAM_SIZE) -> GramIndex:
    """Index every ``size``-character seed of ``text`` by where it occurs."""
    postings: dict[str, list[int]] = {}
    for i in range(len(text) - size + 1):
        gram = text[i : i + size]
        existing = postings.get(gram)
        if existing is None:
            postings[gram] = [i]
        elif len(existing) < _MAX_POSTINGS:
            existing.append(i)
        elif len(existing) == _MAX_POSTINGS:
            # Mark as too common and stop growing it.
            existing.append(-1)
    return GramIndex(size=size, postings=postings)


def find_exact(haystack: str, needle: str) -> list[int]:
    """Every exact occurrence of ``needle`` in ``haystack``, left to right."""
    if not needle:
        return []
    found: list[int] = []
    start = 0
    while True:
        at = haystack.find(needle, start)
        if at < 0:
            return found
        found.append(at)
        start = at + 1


def find_approximate(
    haystack: str,
    needle: str,
    options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
) -> list[ApproximateMatch]:
    """Best approximate occurrences of ``needle``, by increasing distance.

    Overlapping windows are collapsed to their best representative, so the
    returned matches are genuinely distinct locations rather than neighbouring
    alignments of one occurrence.
    """
    if not needle or not haystack:
        return []

    max_distance = max(1, int(len(needle) * options.max_edit_ratio))

    windows = _seed_windows(
        haystack, needle, max_distance, options.max_windows, options.index
    )
    if not windows:
        if len(haystack) > _MAX_UNSEEDED_HAYSTACK:
            return []
        windows.append((0, len(haystack)))

    matches: list[ApproximateMatch] = []
    for start, end in windows:
        match = _align_in_window(haystack, needle, start, end, max_distance)
        if match is not None:
            matches.append(match)

    matches.sort(key=lambda match: (match.distance, match.start))
    return _drop_overlapping(matches)


def _seed_windows(
    haystack: str,
    needle: str,
    max_distance: int,
    max_windows: int,
    provided: GramIndex | None,
) -> list[tuple[int, int]]:
    index = provided if provided is not None else build_gram_index(haystack)
    if len(needle) < index.size:
        return []

    # Vote for candidate offsets. Seeds belonging to one occurrence agree on
    # `position - index` up to the edits between them.
    votes: dict[int, int] = {}
    for i in range(len(needle) - index.size + 1):
        posting = index.postings.get(needle[i : i + index.size])
        if posting is None or len(posting) > _MAX_POSTINGS:
            continue
        for position in posting:
            offset = position - i
            votes[offset] = votes.get(offset, 0) + 1
    if not votes:
        return []

    # Cluster nearby offsets: an edit inside the quote shifts the diagonal by
    # one, so the votes for a single occurrence spread over a small range.
    tolerance = max(index.size, max_distance)
    offsets = sorted(votes)
    clusters: list[tuple[int, int, int]] = []
    from_offset = to_offset = offsets[0]
    weight = votes[from_offset]
    for offset in offsets[1:]:
        if offset - to_offset <= tolerance:
            to_offset = offset
            weight += votes[offset]
        else:
            clusters.append((from_offset, to_offset, weight))
            from_offset = to_offset = offset
            weight = votes[offset]
    clusters.append((from_offset, to_offset, weight))

    clusters.sort(key=lambda cluster: -cluster[2])
    slack = max_distance + index.size
    return [
        (
            max(0, cluster_from - slack),
            min(len(haystack), cluster_to + len(needle) + slack),
        )
        for cluster_from, cluster_to, _ in clusters[:max_windows]
    ]


def _align_in_window(
    haystack: str,
    needle: str,
    from_index: int,
    to_index: int,
    max_distance: int,
) -> ApproximateMatch | None:
    """Sellers' algorithm over ``haystack[from_index:to_index]``.

    The cost row is seeded with zeros so an alignment may start anywhere, and
    the origin row records where each alignment began. The minimum of the final
    row is the best match.
    """
    width = to_index - from_index
    if width <= 0:
        return None

    ceiling = max_distance + 1
    window = haystack[from_index:to_index]
    cost = [0] * (width + 1)
    origin = list(range(width + 1))
    next_cost = [0] * (width + 1)
    next_origin = [0] * (width + 1)

    for i in range(1, len(needle) + 1):
        needle_char = needle[i - 1]
        next_cost[0] = min(i, ceiling)
        next_origin[0] = 0

        for j in range(1, width + 1):
            substitute = cost[j - 1] + (0 if window[j - 1] == needle_char else 1)
            delete_from_needle = cost[j] + 1
            insert_from_haystack = next_cost[j - 1] + 1

            best = substitute
            best_origin = origin[j - 1]
            if delete_from_needle < best:
                best = delete_from_needle
                best_origin = origin[j]
            if insert_from_haystack < best:
                best = insert_from_haystack
                best_origin = next_origin[j - 1]

            next_cost[j] = ceiling if best > ceiling else best
            next_origin[j] = best_origin

        cost, next_cost = next_cost, cost
        origin, next_origin = next_origin, origin

    # Among equal-cost alignments prefer the longest window, hence `<=`. Equal
    # cost means the extra characters were matched rather than paid for, and a
    # resolved quote is usually highlighted for a human: given the choice
    # between "brown fox jumps" and the equally-priced "brown fox jum", the one
    # that does not cut mid-word is the answer the reader expects.
    best_distance = ceiling
    best_end = -1
    for j in range(1, width + 1):
        if cost[j] <= best_distance:
            best_distance = cost[j]
            best_end = j
    if best_end < 0 or best_distance > max_distance:
        return None

    return ApproximateMatch(
        start=from_index + origin[best_end],
        end=from_index + best_end,
        distance=best_distance,
    )


def _drop_overlapping(matches: list[ApproximateMatch]) -> list[ApproximateMatch]:
    """Keep the best of each overlapping group; input must be sorted by distance."""
    kept: list[ApproximateMatch] = []
    for match in matches:
        if not any(
            match.start < other.end and other.start < match.end for other in kept
        ):
            kept.append(match)
    return kept
