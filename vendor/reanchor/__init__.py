"""reanchor -- find a quote again after the document changed.

Zero dependencies, no filesystem, no network. Give it a string and a
:class:`TextQuoteSelector`; it tells you where the quote is now, or admits it
cannot tell.

    >>> from reanchor import describe_quote, resolve_quote
    >>> document = "The highlight still renders, but over the wrong words."
    >>> selector = describe_quote(document, 4, 19)
    >>> edited = "A note: the highlight quietly renders, but elsewhere."
    >>> found = resolve_quote(edited, selector)
    >>> found.method
    'approximate'
"""

from .describe import DescribeOptions, describe_quote
from .normalize import (
    NormalizedText,
    NormalizeOptions,
    Span,
    normalize,
    to_source_span,
)
from .resolve import (
    MatchMethod,
    PreparedDocument,
    ResolvedQuote,
    ResolveOptions,
    TextQuoteSelector,
    prepare_document,
    resolve_quote,
    resolve_quotes,
)
from .search import (
    ApproximateMatch,
    GramIndex,
    SearchOptions,
    build_gram_index,
    find_approximate,
    find_exact,
)

__version__ = "0.1.0"

__all__ = [
    "ApproximateMatch",
    "DescribeOptions",
    "GramIndex",
    "MatchMethod",
    "NormalizeOptions",
    "NormalizedText",
    "PreparedDocument",
    "ResolveOptions",
    "ResolvedQuote",
    "SearchOptions",
    "Span",
    "TextQuoteSelector",
    "__version__",
    "build_gram_index",
    "describe_quote",
    "find_approximate",
    "find_exact",
    "normalize",
    "prepare_document",
    "resolve_quote",
    "resolve_quotes",
    "to_source_span",
]
