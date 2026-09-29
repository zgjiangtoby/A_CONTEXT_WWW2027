"""Existing display-coordinate and pinned Hypothesis exact-quote functions."""
import csv
import html
from html.parser import HTMLParser
import json
import re

CLIENT_COMMIT = "b4d085a2f893aa6de3b61d8b8bc3ae4d0f24fc1a"
APPROX_COMMIT = "d45a0bfd65a4f1a0f8daebc5e75a4f6f0aea0cdd"

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def occurrences(text, exact):
    """Overlaps are candidates, never additional designated annotations."""
    result, start = [], 0
    while exact and (start := text.find(exact, start)) >= 0:
        result.append((start, start + len(exact)))
        start += 1
    return result


def interval_image(mapping, start, end):
    """Nonempty interval hull for an order-preserving non-erasing map."""
    assert 0 <= start < end <= len(mapping)
    return mapping[start][0], mapping[end - 1][1]


def compose_maps(first, second):
    return [interval_image(second, start, end) for start, end in first]


def replace_chars(text, replacement):
    chunks, mapping, end = [], [], 0
    for char in text:
        chunk = replacement(char)
        assert chunk, "This construction map must not erase source characters"
        chunks.append(chunk)
        mapping.append((end, end + len(chunk)))
        end += len(chunk)
    return "".join(chunks), mapping


def insert_text(text, position, value):
    assert 0 <= position <= len(text)
    mapping = [(i + (len(value) if i >= position else 0),
                i + 1 + (len(value) if i >= position else 0)) for i in range(len(text))]
    return text[:position] + value + text[position:], mapping


class TextReader(HTMLParser):
    """Only generated <pre> wrappers enter this parser; original posts are plain."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def parsed_text(storage, media):
    if media == "text/plain":
        return storage
    assert media == "text/html"
    reader = TextReader()
    reader.feed(storage)
    reader.close()
    return "".join(reader.parts)


def display_text(storage, media):
    # Same Python Unicode whitespace predicate as the trace; no strip/NFKC/casefold.
    return re.sub(r"\s+", " ", parsed_text(storage, media))


def normalization_trace(storage, media):
    """Display-character source supports. Whitespace collapse is many-to-one.

    This trace is independent of non-erasing map composition. HTML references
    are decoded exactly once, and only the generated wrapper is parsed as HTML.
    """
    if media == "text/plain":
        decoded = storage
        decoded_support = [(i, i + 1) for i in range(len(storage))]
    else:
        assert storage.startswith("<pre>") and storage.endswith("</pre>")
        body = storage[5:-6]
        chars, decoded_support = [], []
        # html.escape emits only these single-code-point entity forms.
        for match in re.finditer(r"&(?:amp|lt|gt|quot|#x27);|[\s\S]", body):
            char = html.unescape(match[0])
            assert len(char) == 1
            chars.append(char)
            decoded_support.append((match.start() + 5, match.end() + 5))
        decoded = "".join(chars)
        assert decoded == parsed_text(storage, media)
    output, support, in_whitespace = [], [], False
    for char, (start, end) in zip(decoded, decoded_support):
        if char.isspace():
            if in_whitespace:
                support[-1] = (support[-1][0], end)
            else:
                output.append(" ")
                support.append((start, end))
            in_whitespace = True
        else:
            output.append(char)
            support.append((start, end))
            in_whitespace = False
    text = "".join(output)
    assert text == display_text(storage, media)
    return text, support


def project_display(support, start, end):
    """First/last overlapping output characters, even within a whitespace run."""
    overlaps = [i for i, (left, right) in enumerate(support) if left < end and right > start]
    return (overlaps[0], overlaps[-1] + 1) if overlaps else None


def utf16(text):
    """Python characters representing JS UTF-16 units, including cut surrogates."""
    raw = text.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(raw[i] + 256 * raw[i + 1]) for i in range(0, len(raw), 2))


def unit_boundaries(text):
    result = [0]
    for char in text:
        result.append(result[-1] + (2 if ord(char) > 0xFFFF else 1))
    return result


def context_score(text, pattern):
    """Upstream textMatchScore: minimum substring Levenshtein error / length.

    Only the minimum error is consumed by Hypothesis; match endpoints are not.
    Inputs are UTF-16 units. Empty actual text scores zero, as in upstream.
    """
    if not text or not pattern:
        return 0.0
    if pattern in text:
        return 1.0
    # ponytail: quadratic DP for fields up to 64 UTF-16 units; use Myers only if profiling warrants it.
    previous = list(range(len(pattern) + 1))
    best = len(pattern)
    for char in text:
        current = [0]  # Starting a substring anywhere in the text is free.
        for i, expected in enumerate(pattern, 1):
            current.append(min(current[-1] + 1, previous[i] + 1,
                               previous[i - 1] + (char != expected)))
        best = min(best, current[-1])
        previous = current
    return 1.0 - best / len(pattern)


def hypothesis_text(text, quote, prefix, suffix, old_span):
    """Pinned text path in UTF-16 units; no truth/map or old full text accepted.

    Returns (span, stage). Stage 0 is quote-checked position, stage 1 quote
    matching, stage 2 empty-quote abstention. No-exact-quote inputs raise because
    fuzzy quote candidate generation is outside this reference's declared scope.
    """
    if not quote:
        return None, 2
    start, end = old_span
    if 0 <= start < end <= len(text) and text[start:end] == quote:
        return old_span, 0
    candidates = occurrences(text, quote)
    if not candidates:
        raise ValueError("Reference scope requires at least one preserved exact quote")

    def score(span):
        start, end = span
        left = context_score(text[max(0, start - len(prefix)):start], prefix) if prefix else 1.0
        right = context_score(text[end:end + len(suffix)], suffix) if suffix else 1.0
        position = 1.0 - abs(start - old_span[0]) / len(text)
        return (50.0 + 20.0 * left + 20.0 * right + 2.0 * position) / 92.0

    # max retains the first (leftmost) candidate on a score tie, like stable JS sort.
    return max(candidates, key=score), 1


def self_check():
    assert occurrences("aaaa", "aa") == [(0, 2), (1, 3), (2, 4)]
    assert context_score("xabc", "abcd") == .75
    assert context_score("", "x") == 0
    assert hypothesis_text("bad old bad new", "bad", "old ", " new", (0, 3)) == ((0, 3), 0)
    assert hypothesis_text("a---a", "a", "", "", (2, 3))[0] == (0, 1)
    assert hypothesis_text("", "", "", "", (0, 0)) == (None, 2)
    try:
        hypothesis_text("abc", "missing", "", "", (0, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing exact quote must fail closed")
    shown, support = normalization_trace("a \t\n b", "text/plain")
    assert shown == "a b" and project_display(support, 2, 5) == (1, 2)
    assert unit_boundaries("😀 bad") == [0, 2, 3, 4, 5, 6]
    assert hypothesis_text(utf16("++😀 bad"), "bad", utf16("😀 "), "", (3, 6))[0] == (5, 8)
