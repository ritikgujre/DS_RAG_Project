"""Sentence-aware chunking that preserves absolute character offsets.

Offsets are the whole point. A citation is only useful if it can be resolved
back to an exact span in the file the user uploaded, so every chunk carries its
absolute (start, end) in the source text and never mutates the text it covers:
`source_text[chunk.start:chunk.end] == chunk.text` always holds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from . import config

# Abbreviations that end in a period but do not end a sentence, and that can be
# followed by a capital letter or digit. Anything normally followed by lowercase
# ("5 mg. daily", "q.i.d. for a week") is already handled by the lowercase check
# in `split_sentences`, so listing units here would only swallow genuine
# sentence ends like "CRP 3.21 mg/dL. AFP was normal."
_ABBREVIATIONS = {
    # Titles and names
    "dr", "drs", "mr", "mrs", "ms", "prof", "st", "mt", "sr", "jr",
    # Structural / editorial
    "vs", "etc", "eg", "ie", "cf", "al", "fig", "figs", "ref", "refs",
    "dept", "approx", "est", "vol", "ch", "pp", "sec",
    # Months, which precede a year
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
    # Organisations
    "inc", "ltd", "co", "corp", "univ", "hosp", "assn",
}

# A sentence boundary: terminal punctuation, optional closing quote/bracket,
# then whitespace, then something that looks like a new sentence.
_BOUNDARY = re.compile(r'([.!?])(["\')\]]*)(\s+)')


@dataclass
class Chunk:
    """A retrievable, citable span of a source document."""

    index: int
    text: str
    start: int
    end: int
    doc_id: str
    # Section/speaker label when the source has structure worth keeping.
    label: str = ""

    def __post_init__(self) -> None:
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"chunk {self.index}: span {self.start}:{self.end} "
                f"({self.end - self.start} chars) does not match text "
                f"({len(self.text)} chars)"
            )

    @property
    def citation_title(self) -> str:
        """Short label shown to the model as the document title."""
        if self.label:
            return f"[{self.index}] {self.label}"
        return f"[{self.index}] chars {self.start}-{self.end}"


@dataclass
class Sentence:
    text: str
    start: int
    end: int


def _looks_like_abbreviation(text: str, period_pos: int) -> bool:
    """True if the period at `period_pos` closes an abbreviation, not a sentence."""
    # Walk back over the word preceding the period.
    i = period_pos - 1
    while i >= 0 and (text[i].isalnum() or text[i] == "."):
        i -= 1
    word = text[i + 1 : period_pos].lower().rstrip(".")

    if word in _ABBREVIATIONS:
        return True
    # Single initial: "J. Smith", or a lettered list item "a."
    if len(word) == 1 and word.isalpha():
        return True
    # Numbered list item: "1." / "12." -- but not a date or year ending a
    # sentence ("...on 07/29/2008. He returned..."), so only short runs.
    # Decimals like "0.5" never reach here: the boundary pattern requires
    # whitespace after the period.
    if word.isdigit() and len(word) <= 2:
        return True
    # Dotted acronym: "U.S.A." -- the internal periods keep it one token.
    if "." in text[i + 1 : period_pos]:
        return True
    return False


def split_sentences(text: str) -> list[Sentence]:
    """Split into sentences, keeping exact offsets into `text`."""
    sentences: list[Sentence] = []
    start = 0

    for match in _BOUNDARY.finditer(text):
        period_pos = match.start(1)
        if _looks_like_abbreviation(text, period_pos):
            continue

        # The next non-space character should plausibly start a sentence.
        after = match.end()
        if after < len(text) and text[after].islower():
            continue

        end = match.end(2)  # include terminal punctuation and closing quotes
        candidate = text[start:end].strip()
        if candidate:
            # Re-derive offsets after stripping so they stay exact.
            lead = len(text[start:end]) - len(text[start:end].lstrip())
            trail = len(text[start:end]) - len(text[start:end].rstrip())
            sentences.append(Sentence(candidate, start + lead, end - trail))
        start = match.end()

    tail = text[start:].strip()
    if tail:
        lead = len(text[start:]) - len(text[start:].lstrip())
        sentences.append(Sentence(tail, start + lead, start + lead + len(tail)))

    return sentences


def _split_oversized(sentence: Sentence, limit: int) -> list[Sentence]:
    """Break a sentence that is longer than `limit` into pieces on whitespace.

    Sentence boundaries normally bound chunk size, but nothing guarantees a
    document has any. multiclinsum_gs_en_410 is 21,511 characters with 11
    sentences, one of them 20,238 characters -- a repeated lab-value run joined
    by commas, whose 535 periods are all decimals the splitter correctly refuses
    to break on. Without a ceiling that yields a 20,000-character "citation",
    which is useless as attribution even though every offset in it is correct.

    Splits on whitespace so offsets stay exact and no word is cut in half.
    """
    if len(sentence.text) <= limit:
        return [sentence]

    pieces: list[Sentence] = []
    start = 0
    text = sentence.text
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            # Back off to the last space so the break lands between words.
            space = text.rfind(" ", start + limit // 2, end)
            if space > start:
                end = space
        piece = text[start:end]
        lead = len(piece) - len(piece.lstrip())
        body = piece.strip()
        if body:
            pieces.append(
                Sentence(body, sentence.start + start + lead,
                         sentence.start + start + lead + len(body))
            )
        start = end
    return pieces


def sentences_within(
    text: str, chunks: list[Chunk], min_chars: int = 0
) -> list[tuple[Sentence, int]]:
    """Sentences of `text` that lie inside one of `chunks`, with the owner index.

    Splitting the whole document once, rather than each chunk separately, keeps
    offsets absolute with no arithmetic and sidesteps the overlap between
    consecutive chunks -- a sentence shared by two chunks is still one sentence
    here. `chunk_text` never splits a sentence, so every sentence of the document
    is contained in at least one chunk.

    `min_chars` drops fragments: below roughly 40 characters a "sentence" is
    almost always a heading or a list bullet, which carries no standalone claim.

    Used by both the extractive backend (as the pool it selects from) and the
    local backend (as the pool it grounds generated sentences against).
    """
    windows = [(c.start, c.end, c.index) for c in chunks]
    out: list[tuple[Sentence, int]] = []
    for sentence in split_sentences(text):
        if len(sentence.text) < min_chars:
            continue
        owner = next(
            (i for lo, hi, i in windows if sentence.start >= lo and sentence.end <= hi),
            None,
        )
        if owner is not None:
            out.append((sentence, owner))
    return out


def chunk_text(
    text: str,
    doc_id: str = "doc",
    target_chars: int = config.CHUNK_TARGET_CHARS,
    overlap_sentences: int = config.CHUNK_OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Group sentences into chunks of roughly `target_chars`.

    Chunks never split a sentence, and consecutive chunks share
    `overlap_sentences` sentences so a fact spanning a boundary is still
    retrievable as a unit.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Enforce a hard ceiling before grouping. A chunk is the unit of citation,
    # so an unbounded one makes attribution technically correct and practically
    # worthless.
    ceiling = max(target_chars, config.CHUNK_MAX_CHARS)
    sentences = [p for s in sentences for p in _split_oversized(s, ceiling)]

    chunks: list[Chunk] = []
    current: list[Sentence] = []
    index = 0

    def flush(sents: list[Sentence]) -> None:
        nonlocal index
        if not sents:
            return
        start, end = sents[0].start, sents[-1].end
        chunks.append(
            Chunk(
                index=index,
                text=text[start:end],
                start=start,
                end=end,
                doc_id=doc_id,
            )
        )
        index += 1

    for sentence in sentences:
        # Enforce the ceiling before appending. Splitting oversized sentences is
        # not enough on its own: the one-sentence overlap carries a large
        # sentence into the next chunk, so two near-ceiling sentences would
        # combine into a chunk of twice the limit.
        if current and sentence.end - current[0].start > ceiling:
            flush(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            # Overlap alone can already exceed the ceiling; drop it if so.
            if current and sentence.end - current[0].start > ceiling:
                current = []

        current.append(sentence)
        span = current[-1].end - current[0].start
        if span >= target_chars:
            flush(current)
            current = current[-overlap_sentences:] if overlap_sentences else []

    # Avoid emitting a trailing chunk that is nothing but overlap.
    if current and (not chunks or current[-1].end > chunks[-1].end):
        flush(current)

    return chunks


def chunk_paragraphs(text: str, doc_id: str = "doc") -> list[Chunk]:
    """Alternative chunker: one chunk per paragraph, offsets preserved.

    Useful for dialogue transcripts and bulleted documents where blank-line
    structure carries more meaning than sentence count.
    """
    chunks: list[Chunk] = []
    index = 0
    for match in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        body = match.group()
        stripped = body.strip()
        if not stripped:
            continue
        lead = len(body) - len(body.lstrip())
        start = match.start() + lead
        chunks.append(
            Chunk(
                index=index,
                text=stripped,
                start=start,
                end=start + len(stripped),
                doc_id=doc_id,
            )
        )
        index += 1
    return chunks


def iter_windows(chunks: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield consecutive windows of chunks, for map-reduce over long documents."""
    for i in range(0, len(chunks), size):
        yield chunks[i : i + size]
