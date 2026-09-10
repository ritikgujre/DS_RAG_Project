"""Model-asserted citations, verified against the source before they are trusted.

This is the fourth attribution strategy, and it exists to answer a question the
project had asserted rather than measured.

The `api` backend is trustworthy because Claude's API extracts `cited_text`
itself: the model never writes the citation, so it cannot invent one. Without an
Anthropic key that guarantee is unavailable, and no other vendor reproduces it.
The `local` and `groq` backends sidestep the problem entirely by never asking the
model to attribute anything and recovering spans by similarity afterwards.

This backend takes the third road: **ask the model to cite, then check.** It
requests each claim together with a verbatim quote, then verifies every quote by
locating it in the text the model was actually shown. A quote that cannot be
found is not a citation -- it is a caught fabrication, and the claim is left
uncited.

That yields a guarantee strictly weaker than the API's and strictly stronger than
prompting alone:

    api      the model cannot fabricate a citation
    verified the model can fabricate one, and we will catch it
    groq     the model is never asked, so spans are inferred by similarity
    local    same, on your own hardware

The point of building it is the measurement it produces. `citation_precision`
(verified quotes / quotes asserted) is a direct number for how often a model's
self-reported citation lies, which is exactly the premise the whole design rests
on. Reported per run in `SummaryResult.usage`.

Two details that matter for the verification to be honest:

* A quote is only accepted if it appears in the **retrieved chunks**, not merely
  somewhere in the document. A model cannot legitimately quote text it was never
  sent, and allowing that would let a lucky guess pass as a citation.
* Matching normalises typography and whitespace but nothing else. Models rewrite
  hyphens and spaces -- gpt-oss-120b emits U+2011 and U+202F routinely -- and
  failing a citation over a hyphen would measure the tokenizer, not the model.
  Every substitution is one character for one character, so offsets survive and
  the span still slices exactly out of the original document.
"""

from __future__ import annotations

import json
import re

from . import config
from .chunking import Chunk, chunk_text
from .grounding import strip_artifacts
from .remote import RemoteError, _post
from .retrieval import ChunkIndex
from .summarizer import ASPECT_PRESETS, Fact, SourceSpan, SummaryResult

VERIFIED_SYSTEM_PROMPT = """\
You summarise documents and cite your sources.

You are given numbered excerpts from a single document. Produce a list of the \
claims the excerpts establish. For each claim, give a quote copied EXACTLY, \
character for character, from one of the excerpts, long enough to establish the \
claim on its own.

Rules:
- Every claim must come from the excerpts. Add no background knowledge and no \
inference of your own.
- The quote must be a literal substring of an excerpt. Do not paraphrase, \
shorten, join separate passages, or fix spelling inside a quote.
- Preserve every specific exactly: ages, dates, dosages, lab values and \
measurements.
- If you cannot support a claim with an exact quote, omit the claim.
- Order the claims as the document presents them.

Reply with JSON only, in this shape:
{"claims": [{"claim": "one factual sentence", "quote": "exact substring"}]}"""

# One-for-one character substitutions only, so a position in the normalised
# string is the same position in the original and offsets survive the round
# trip. Anything that changed length would silently corrupt every span.
_TYPOGRAPHIC = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
})


def _normalise(text: str) -> str:
    """Fold typographic variants to ASCII without changing any character's position."""
    return text.translate(_TYPOGRAPHIC)


def locate_quote(quote: str, chunks: list[Chunk]) -> tuple[int, int, int] | None:
    """Find `quote` inside the chunks the model was shown.

    Returns (chunk_index, absolute_start, absolute_end), or None when the quote
    does not occur -- which is the signal that the model invented it.

    Whitespace is matched flexibly because models re-wrap text freely; that is a
    formatting difference, not a fabricated citation.
    """
    normalised = _normalise(quote).strip()
    if len(normalised) < config.VERIFIED_MIN_QUOTE_CHARS:
        return None

    pattern = re.compile(r"\s+".join(re.escape(part) for part in normalised.split()))
    for chunk in chunks:
        match = pattern.search(_normalise(chunk.text))
        if match:
            return chunk.index, chunk.start + match.start(), chunk.start + match.end()
    return None


def _parse_claims(content: str) -> list[dict]:
    """Pull the claim list out of the model's reply, tolerating stray prose."""
    text = strip_artifacts(content or "")
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSON mode is requested, but a model may still wrap it in commentary.
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    claims = data.get("claims") if isinstance(data, dict) else data
    if not isinstance(claims, list):
        return []
    return [c for c in claims if isinstance(c, dict) and c.get("claim")]


def summarize_verified(
    text: str,
    *,
    doc_id: str = "doc",
    aspects: str | list[str] = "generic",
    instruction: str | None = None,
    top_k: int = config.TOP_K,
    chunk_budget: int | None = None,
    index: ChunkIndex | None = None,
    chunks: list[Chunk] | None = None,
    model: str = config.GROQ_MODEL,
    max_tokens: int = config.VERIFIED_MAX_TOKENS,
    timeout: float = config.GROQ_TIMEOUT,
) -> SummaryResult:
    """Summarise `text`, asking the model to cite and verifying every citation.

    Signature mirrors `summarizer.summarize` so callers can switch backends
    without restructuring.
    """
    if chunks is None:
        chunks = chunk_text(text, doc_id=doc_id)
    if not chunks:
        return SummaryResult(summary="", facts=[], chunks_used=[], source_text=text)

    queries = ASPECT_PRESETS[aspects] if isinstance(aspects, str) else aspects
    if index is None:
        index = ChunkIndex(chunks).build()

    selected = index.search_many(queries, top_k=top_k, budget=chunk_budget)
    if not selected:
        selected = chunks

    task = instruction or (
        "List every substantive claim these excerpts establish, each with an "
        "exact supporting quote."
    )
    excerpts = "\n\n".join(
        f"[excerpt {i}]\n{chunk.text}" for i, chunk in enumerate(selected)
    )

    try:
        data = _post(
            {
            "model": model,
            "messages": [
                {"role": "system", "content": VERIFIED_SYSTEM_PROMPT},
                {"role": "user", "content": f"{excerpts}\n\n---\n\n{task}"},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout,
        )
    except RemoteError as exc:
        # A truncated object fails JSON validation server-side with an empty
        # failed_generation, which looks like a prompt fault. It is a budget
        # fault: say so rather than sending the caller to rewrite the prompt.
        if "json_validate_failed" in str(exc):
            raise RemoteError(
                f"the model's JSON was cut off at max_tokens={max_tokens}. "
                f"Raise DOCSUM_VERIFIED_MAX_TOKENS (or pass max_tokens=) -- long "
                f"documents need more room because each claim carries a full quote."
            ) from exc
        raise

    choice = data["choices"][0]
    claims = _parse_claims(choice["message"].get("content"))

    facts: list[Fact] = []
    asserted = 0
    verified = 0
    rejected: list[str] = []

    for entry in claims:
        claim = str(entry.get("claim", "")).strip()
        quote = str(entry.get("quote", "") or "").strip()
        if not claim:
            continue

        if not quote:
            facts.append(Fact(text=claim, sources=[]))
            continue

        asserted += 1
        found = locate_quote(quote, selected)
        if found is None:
            # The model asserted a citation that does not exist. Do not attach
            # it: an uncited claim is visibly unsupported, a fabricated citation
            # looks verified.
            rejected.append(quote)
            facts.append(Fact(text=claim, sources=[]))
            continue

        verified += 1
        chunk_index, start, end = found
        facts.append(
            Fact(
                text=claim,
                sources=[
                    SourceSpan(
                        # The slice, not what the model wrote -- so the span is
                        # exact even where the quote differed in typography.
                        cited_text=text[start:end],
                        chunk_index=chunk_index,
                        start=start,
                        end=end,
                        support=1.0,  # verified, not estimated
                    )
                ],
            )
        )

    usage = data.get("usage", {})
    return SummaryResult(
        summary=" ".join(f.text for f in facts),
        facts=facts,
        chunks_used=selected,
        source_text=text,
        usage={
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model": data.get("model", model),
            # The measurement this backend exists to produce.
            "quotes_asserted": asserted,
            "quotes_verified": verified,
            "quotes_rejected": asserted - verified,
            "citation_precision": round(verified / asserted, 4) if asserted else 1.0,
            "rejected_quotes": rejected,
        },
        stop_reason=choice.get("finish_reason"),
    )
