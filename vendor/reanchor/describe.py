"""Quote description -- the inverse of resolution.

To cite a passage you must first record a selector for it. The quoted text
alone is often ambiguous: a document may contain "as discussed above" a dozen
times, and a selector that matches all of them tells a later resolver nothing.
So context is captured either side.

How much context? Fixed-length context is the common answer and it is wrong in
one direction: too little to disambiguate a repeated heading. So
:func:`describe_quote` grows the context until the selector identifies exactly
one location, then stops -- repeated passages get just enough.

It stops growing, but it does not start at nothing. Recording no context for a
quote that is unique today is a bet that it will still be unique when someone
comes to resolve it, and that bet loses in a way worth naming: a passage is
revised, a verbatim copy of the old wording survives elsewhere as a pull quote
or a syndicated excerpt, and the only surviving character-for-character match is
the copy. Reported against Hypothesis as
`client#7571 <https://github.com/hypothesis/client/issues/7571>`_. Context is
what tells the revised original from the stale copy, and a selector that
recorded none has thrown that evidence away before the ambiguity existed.

Context is not free -- every recorded character is more surface for a later edit
to damage -- but the resolver scores context by longest common affix rather than
requiring it to match, so partly-edited context degrades a candidate's confidence
instead of disqualifying it. A floor therefore costs little and buys the case
above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .resolve import TextQuoteSelector
from .search import find_exact

__all__ = ["DescribeOptions", "describe_quote"]


@dataclass(frozen=True, slots=True)
class DescribeOptions:
    """How far :func:`describe_quote` may grow the recorded context."""

    #: Context to record either side, always, and the length to start from when
    #: the quote repeats and the context has to grow.
    min_context_length: int = 16
    #: Stop growing here even if the quote is still ambiguous -- some documents
    #: genuinely repeat a passage verbatim, and unbounded growth would capture
    #: whole paragraphs.
    max_context_length: int = 128


DEFAULT_DESCRIBE_OPTIONS: Final = DescribeOptions()


def describe_quote(
    document: str,
    start: int,
    end: int,
    options: DescribeOptions = DEFAULT_DESCRIBE_OPTIONS,
) -> TextQuoteSelector:
    """Build a selector for ``document[start:end]``.

    Captures enough context to identify the span uniquely where possible.

    Callers that need a guarantee should re-resolve the result: if
    :func:`~reanchor.resolve.resolve_quote` reports rivals, this document
    repeats the passage beyond :attr:`DescribeOptions.max_context_length` and
    position must carry the disambiguation instead.
    """
    if start < 0 or end < start or end > len(document):
        raise ValueError(
            f"span {start}..{end} is outside the document (length {len(document)})"
        )

    exact = document[start:end]

    def context(length: int) -> TextQuoteSelector:
        return TextQuoteSelector(
            exact=exact,
            prefix=document[max(0, start - length) : start],
            suffix=document[end : end + length],
        )

    if not exact:
        return context(options.min_context_length)

    occurrences = find_exact(document, exact)
    if len(occurrences) <= 1:
        return context(options.min_context_length)

    length = options.min_context_length
    while True:
        selector = context(length)
        distinct = all(
            at == start
            or not _context_matches(
                document, at, at + len(exact), selector.prefix, selector.suffix
            )
            for at in occurrences
        )
        if distinct or length >= options.max_context_length:
            return selector
        length = min(length * 2, options.max_context_length)


def _context_matches(
    document: str,
    start: int,
    end: int,
    prefix: str,
    suffix: str,
) -> bool:
    # `max(0, ...)` rather than a negative index: a candidate too close to the
    # start of the document has no room for the full prefix, and comparing
    # against the document's opening characters is the honest answer -- a
    # negative index would silently compare against its tail instead.
    if prefix and not document.startswith(prefix, max(0, start - len(prefix))):
        return False
    return not suffix or document.startswith(suffix, end)
