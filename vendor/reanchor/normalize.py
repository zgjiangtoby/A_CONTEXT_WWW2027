"""Offset-preserving text normalization.

Two copies of "the same" text rarely agree character for character. A PDF
re-extracted with a different tool hyphenates across line breaks; a CMS turns
straight quotes into curly ones; an editor rewraps paragraphs. Comparing such
texts requires normalizing both sides -- but a quote resolver must then report
offsets in the *original* string, not the normalized one.

:func:`normalize` therefore returns, alongside the normalized text, the source
range that produced each normalized character. :func:`to_source_span` maps a
span back. Every transformation here is deliberately offset-attributable: one
source character maps to zero or more normalized characters, and a run of
collapsed whitespace maps to a single space spanning the whole run.

Offsets are Python string indices, which is to say code points. The JavaScript
implementation of this library reports UTF-16 code unit indices, because that
is how JavaScript indexes strings. The two agree on text within the Basic
Multilingual Plane and diverge past it: an emoji occupies one Python index and
two JavaScript ones. Selectors are portable between the implementations because
they carry text rather than positions; raw offsets are not.
"""

from __future__ import annotations

import unicodedata
from array import array
from dataclasses import dataclass
from typing import Final, NamedTuple

__all__ = [
    "NormalizeOptions",
    "NormalizedText",
    "Span",
    "normalize",
    "to_source_span",
]


@dataclass(frozen=True, slots=True)
class NormalizeOptions:
    """How aggressively to fold text before comparing it.

    The defaults are tuned for the common case: the same passage passed through
    two different publishing pipelines. Turn individual steps off when a
    distinction matters to you -- ``case_fold=False`` if you are anchoring
    case-sensitive identifiers, ``strip_marks=False`` for languages where a
    combining mark is not decoration.
    """

    #: Apply per-character NFKD. Decomposing rather than composing makes
    #: precomposed and combining forms converge, and folds compatibility
    #: characters such as ligatures and full-width Latin.
    unicode: bool = True
    #: Drop combining marks left behind by decomposition, so ``café`` and
    #: ``cafe`` compare equal. Only meaningful with :attr:`unicode`.
    strip_marks: bool = True
    #: Lowercase the result.
    case_fold: bool = True
    #: Fold typographic punctuation onto ASCII: every dash to ``-``, every
    #: quotation mark to ``'`` or ``"``, the ellipsis to ``...``.
    fold_punctuation: bool = True
    #: Collapse each run of whitespace to a single space.
    collapse_whitespace: bool = True
    #: Join words broken across a line by a hyphen: ``exam-\\nple`` becomes
    #: ``example``. Applies only between two letters, and only when the
    #: intervening whitespace contains a line break.
    join_hyphenated_line_breaks: bool = True
    #: Drop leading and trailing whitespace.
    trim: bool = True


DEFAULT_OPTIONS: Final = NormalizeOptions()


class Span(NamedTuple):
    """A half-open range of string indices."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """Normalized text plus the map back to where it came from."""

    #: The normalized text.
    text: str
    #: ``src_start[i]`` is where the source range for ``text[i]`` begins.
    src_start: array[int]
    #: ``src_end[i]`` is just past the source range for ``text[i]``.
    src_end: array[int]
    #: Length of the string this was normalized from.
    source_length: int


#: Characters that carry no visible width and should never affect matching.
#: Spelled as codepoints rather than literals: a set of invisible characters
#: written out literally is a set nobody can review.
_ZERO_WIDTH: Final[frozenset[str]] = frozenset(
    map(
        chr,
        (
            0x00AD,  # SOFT HYPHEN
            0x200B,  # ZERO WIDTH SPACE
            0x200C,  # ZERO WIDTH NON-JOINER
            0x200D,  # ZERO WIDTH JOINER
            0x2060,  # WORD JOINER
            0xFEFF,  # ZERO WIDTH NO-BREAK SPACE
        ),
    )
)

_PUNCTUATION_FOLD: Final[dict[str, str]] = {
    # Dashes, hyphens, and minus signs.
    **{
        chr(c): "-"
        for c in (
            0x2010,  # HYPHEN
            0x2011,  # NON-BREAKING HYPHEN
            0x2012,  # FIGURE DASH
            0x2013,  # EN DASH
            0x2014,  # EM DASH
            0x2015,  # HORIZONTAL BAR
            0x2043,  # HYPHEN BULLET
            0x2212,  # MINUS SIGN
            0xFE58,  # SMALL EM DASH
            0xFE63,  # SMALL HYPHEN-MINUS
            0xFF0D,  # FULLWIDTH HYPHEN-MINUS
        )
    },
    # Single quotation marks, primes, and apostrophes.
    **{
        chr(c): "'"
        for c in (
            0x2018,  # LEFT SINGLE QUOTATION MARK
            0x2019,  # RIGHT SINGLE QUOTATION MARK
            0x201A,  # SINGLE LOW-9 QUOTATION MARK
            0x201B,  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
            0x2032,  # PRIME
            0x2035,  # REVERSED PRIME
            0x00B4,  # ACUTE ACCENT
            0x02BC,  # MODIFIER LETTER APOSTROPHE
            0xFF07,  # FULLWIDTH APOSTROPHE
        )
    },
    # Double quotation marks and guillemets.
    **{
        chr(c): '"'
        for c in (
            0x201C,  # LEFT DOUBLE QUOTATION MARK
            0x201D,  # RIGHT DOUBLE QUOTATION MARK
            0x201E,  # DOUBLE LOW-9 QUOTATION MARK
            0x201F,  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
            0x2033,  # DOUBLE PRIME
            0x2036,  # REVERSED DOUBLE PRIME
            0x00AB,  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
            0x00BB,  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
            0x301D,  # REVERSED DOUBLE PRIME QUOTATION MARK
            0x301E,  # DOUBLE PRIME QUOTATION MARK
            0xFF02,  # FULLWIDTH QUOTATION MARK
        )
    },
    # The ellipsis expands, so that `…` and `...` converge.
    chr(0x2026): "...",
}

#: Whitespace. Kept explicit rather than delegating to :meth:`str.isspace` so
#: the set cannot drift with the runtime's Unicode tables, and so it agrees
#: exactly with the JavaScript implementation. U+200B is deliberately absent:
#: it is zero width, and folding it to a space would insert a word boundary
#: that was never visible.
_WHITESPACE: Final[frozenset[str]] = frozenset(
    map(
        chr,
        (
            0x09,  # CHARACTER TABULATION
            0x0A,  # LINE FEED
            0x0B,  # LINE TABULATION
            0x0C,  # FORM FEED
            0x0D,  # CARRIAGE RETURN
            0x20,  # SPACE
            0x85,  # NEXT LINE
            0xA0,  # NO-BREAK SPACE
            0x1680,  # OGHAM SPACE MARK
            *range(0x2000, 0x200B),  # EN QUAD .. HAIR SPACE
            0x2028,  # LINE SEPARATOR
            0x2029,  # PARAGRAPH SEPARATOR
            0x202F,  # NARROW NO-BREAK SPACE
            0x205F,  # MEDIUM MATHEMATICAL SPACE
            0x3000,  # IDEOGRAPHIC SPACE
        ),
    )
)

_LINE_BREAK: Final[frozenset[str]] = frozenset(
    map(chr, (0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0x2028, 0x2029))
)

_DASHES: Final[frozenset[str]] = frozenset("-") | frozenset(
    char for char, folded in _PUNCTUATION_FOLD.items() if folded == "-"
)


def normalize(
    source: str, options: NormalizeOptions = DEFAULT_OPTIONS
) -> NormalizedText:
    """Normalize ``source``, recording where each output character came from."""
    chars: list[str] = []
    starts: array[int] = array("i")
    ends: array[int] = array("i")

    # A run of source whitespace is held back until we know whether anything
    # follows it, so trailing whitespace can be dropped without a second pass
    # and a collapsed space can span the entire run.
    pending_start = -1
    pending_end = -1

    def emit(value: str, start: int, end: int) -> None:
        for char in value:
            chars.append(char)
            starts.append(start)
            ends.append(end)

    def flush_whitespace(at_end: bool) -> None:
        nonlocal pending_start, pending_end
        if pending_start < 0:
            return
        leading = not chars
        if not (options.trim and (leading or at_end)):
            if options.collapse_whitespace:
                emit(" ", pending_start, pending_end)
            else:
                for k in range(pending_start, pending_end):
                    emit(" ", k, k + 1)
        pending_start = -1
        pending_end = -1

    length = len(source)
    i = 0
    while i < length:
        char = source[i]

        if char in _ZERO_WIDTH:
            i += 1
            continue

        if char in _WHITESPACE:
            if pending_start < 0:
                pending_start = i
            pending_end = i + 1
            i += 1
            continue

        if (
            options.join_hyphenated_line_breaks
            and char in _DASHES
            and pending_start < 0
            and chars
        ):
            joined = _try_join_hyphenated_break(source, i + 1, chars[-1])
            if joined >= 0:
                i = joined
                continue

        flush_whitespace(at_end=False)

        folded = _PUNCTUATION_FOLD.get(char) if options.fold_punctuation else None
        if folded is None:
            folded = unicodedata.normalize("NFKD", char) if options.unicode else char
            if options.strip_marks:
                folded = _strip_combining_marks(folded)
        if options.case_fold:
            folded = folded.lower()
        emit(folded, i, i + 1)
        i += 1

    flush_whitespace(at_end=True)

    return NormalizedText(
        text="".join(chars),
        src_start=starts,
        src_end=ends,
        source_length=length,
    )


def _try_join_hyphenated_break(source: str, start: int, previous: str) -> int:
    """Resume index past a hyphen that breaks a word across a line, else ``-1``.

    ``previous`` is the last character already emitted; a hyphen only joins
    when a letter precedes it, a line break follows, and a letter resumes
    after. Anything else is an ordinary hyphen and must be kept.
    """
    if not previous.isalpha():
        return -1

    k = start
    length = len(source)
    saw_line_break = False
    while k < length:
        char = source[k]
        if char in _ZERO_WIDTH:
            k += 1
            continue
        if char not in _WHITESPACE:
            break
        if char in _LINE_BREAK:
            saw_line_break = True
        k += 1

    if not saw_line_break or k >= length or not source[k].isalpha():
        return -1
    return k


def _is_combining(char: str) -> bool:
    # Unicode general category M*, matching the JavaScript implementation's
    # `\p{M}`. Not `unicodedata.combining`, which reports the canonical
    # combining class: that is zero for many marks, including every Devanagari
    # vowel sign, so it would leave exactly the marks that most need dropping.
    return unicodedata.category(char)[0] == "M"


def _strip_combining_marks(value: str) -> str:
    if len(value) == 1:
        return "" if _is_combining(value) else value
    return "".join(char for char in value if not _is_combining(char))


def to_source_span(normalized: NormalizedText, start: int, end: int) -> Span:
    """Map a span of normalized text back to the source string.

    An empty span collapses to the source position where the normalized
    character at ``start`` begins, or to the end of the source when ``start``
    is past the last normalized character.
    """
    length = len(normalized.text)
    if start < 0 or end < start or end > length:
        raise ValueError(
            f"span {start}..{end} is outside the normalized text (length {length})"
        )
    if start == length:
        return Span(normalized.source_length, normalized.source_length)
    source_start = normalized.src_start[start]
    if end == start:
        return Span(source_start, source_start)
    return Span(source_start, normalized.src_end[end - 1])
