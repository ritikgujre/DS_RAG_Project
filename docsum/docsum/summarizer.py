"""RAG summarisation with every fact traced back to its source span.

Pipeline
--------
1. Chunk the document, keeping absolute character offsets.
2. Retrieve chunks with a set of aspect queries (coverage, not just top-1).
3. Send each retrieved chunk as its own `document` block with citations enabled.
4. Parse the response into `Fact` objects, mapping each citation back to an
   absolute span in the original document.

Step 3 is the part that makes attribution trustworthy. Because each chunk is a
separate document block, a citation's `document_index` identifies the chunk and
its `start_char_index` is an offset *within* that chunk -- so the absolute
position in the uploaded file is just `chunk.start + start_char_index`. The
model cannot cite a span that does not exist: the API extracts `cited_text`
itself rather than letting the model write it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config
from .chunking import Chunk
from .retrieval import ChunkIndex

# Aspect queries drive retrieval for coverage. A summary built from a single
# query systematically drops whatever that query did not match, so we sweep
# several facets and merge.
GENERIC_ASPECTS = [
    "main topic, purpose and central claim",
    "key findings, results and figures",
    "background, context and prior work",
    "methods, process and how it was done",
    "conclusions, recommendations and next steps",
    "limitations, risks, caveats and exceptions",
]

CLINICAL_ASPECTS = [
    "patient age, sex, and presenting complaint",
    "history of present illness and symptom timeline",
    "past medical history, medications and allergies",
    "physical examination findings and vital signs",
    "laboratory results, imaging and diagnostic tests",
    "diagnosis and differential diagnosis",
    "treatment, procedures, medications given",
    "clinical course, outcome, and discharge disposition",
]

ASPECT_PRESETS = {
    "generic": GENERIC_ASPECTS,
    "clinical": CLINICAL_ASPECTS,
}

SYSTEM_PROMPT = """\
You write factual summaries that are fully grounded in the supplied source \
excerpts.

Rules:
- Every factual claim you make must be drawn from the excerpts provided. Do not \
add background knowledge, inferences, or context that is not in the sources.
- Cite the source for every claim. Citations are the point of this task.
- Do not speculate about information that is absent. If the excerpts do not \
establish something, leave it out rather than hedging about it.
- Preserve specifics exactly as written: ages, dates, dosages, lab values, \
measurements, and named entities. Never round, convert, or approximate a number.
- Write flowing prose, not a bulleted list of fragments. Aim for a summary that \
reads as a coherent whole.
- Prefer completeness over brevity. Omitting a materially relevant fact is a \
worse failure than a slightly longer summary."""


@dataclass
class SourceSpan:
    """One resolved citation: a span in the original uploaded document."""

    cited_text: str
    chunk_index: int
    # Absolute offsets into the original document text.
    start: int
    end: int
    # How strongly the claim was tied to this span, for backends that align
    # after the fact rather than being told. None means the attribution was
    # authoritative (the API reported it) rather than inferred -- see
    # docsum/local.py, where a float here is the alignment score.
    support: float | None = None

    def snippet(self, source_text: str, context: int = 0) -> str:
        lo = max(0, self.start - context)
        hi = min(len(source_text), self.end + context)
        return source_text[lo:hi]


@dataclass
class Fact:
    """A claim from the summary together with the sources that support it."""

    text: str
    sources: list[SourceSpan] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return bool(self.sources)

    @property
    def is_claim_like(self) -> bool:
        """Whether this block asserts something, as opposed to joining two claims.

        Citations attach to the clause the model was asserting, so the gaps
        between them are connectives ("The patient presented with", "and", ". "
        ). Those are not attribution failures, so they must not be counted as
        such. A block is treated as a claim if it carries a number or enough
        words to state something on its own.
        """
        stripped = self.text.strip()
        if any(ch.isdigit() for ch in stripped):
            return True
        return len(stripped.split()) >= 5


@dataclass
class SummaryResult:
    summary: str
    facts: list[Fact]
    chunks_used: list[Chunk]
    source_text: str
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    refusal: str | None = None

    @property
    def grounded_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.is_grounded]

    @property
    def unattributed_claims(self) -> list[Fact]:
        """Claim-like blocks the model wrote without attaching a citation.

        These are the ones worth reviewing: assertions with no source. Purely
        connective blocks are excluded, since they have nothing to attribute.
        """
        return [f for f in self.facts if f.is_claim_like and not f.is_grounded]

    @property
    def coverage(self) -> float:
        """Share of summary characters that sit inside a cited block.

        Character-weighted rather than block-counted: blocks vary from one word
        to a full clause, so counting them equally would let a long uncited
        passage hide behind several short cited ones.
        """
        total = sum(len(f.text.strip()) for f in self.facts)
        if not total:
            return 0.0
        cited = sum(len(f.text.strip()) for f in self.facts if f.is_grounded)
        return cited / total

    def all_spans(self) -> list[SourceSpan]:
        spans: list[SourceSpan] = []
        for fact in self.facts:
            spans.extend(fact.sources)
        return spans


def _build_document_blocks(chunks: list[Chunk]) -> list[dict]:
    """One `document` block per chunk, with citations enabled.

    Per the citations docs: putting each RAG chunk in its own plain-text
    document lets the model cite individual sentences within a chunk, and makes
    `document_index` a direct handle on which chunk was used.
    """
    blocks = []
    for chunk in chunks:
        blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": chunk.text,
                },
                "title": chunk.citation_title,
                # `context` is passed to the model but is not citable itself --
                # a good place for provenance metadata.
                "context": json.dumps(
                    {
                        "chunk_index": chunk.index,
                        "doc_id": chunk.doc_id,
                        "char_range": [chunk.start, chunk.end],
                    }
                ),
                "citations": {"enabled": True},
            }
        )
    return blocks


def _parse_response(response, chunks: list[Chunk]) -> tuple[str, list[Fact]]:
    """Turn the API response into summary text plus resolved facts."""
    facts: list[Fact] = []
    parts: list[str] = []

    for block in response.content:
        if block.type != "text":
            continue

        parts.append(block.text)
        raw_citations = getattr(block, "citations", None) or []
        sources: list[SourceSpan] = []

        for citation in raw_citations:
            # Plain-text documents yield char_location citations.
            if getattr(citation, "type", None) != "char_location":
                continue
            doc_i = citation.document_index
            if not (0 <= doc_i < len(chunks)):
                continue
            chunk = chunks[doc_i]
            sources.append(
                SourceSpan(
                    cited_text=citation.cited_text,
                    chunk_index=chunk.index,
                    # Citation offsets are relative to the chunk we sent, so
                    # shift by the chunk's own offset to get absolute position.
                    start=chunk.start + citation.start_char_index,
                    end=chunk.start + citation.end_char_index,
                )
            )

        facts.append(Fact(text=block.text, sources=sources))

    return "".join(parts), facts


def summarize(
    text: str,
    *,
    doc_id: str = "doc",
    aspects: str | list[str] = "generic",
    instruction: str | None = None,
    length: str | None = None,
    top_k: int = config.TOP_K,
    chunk_budget: int | None = None,
    index: ChunkIndex | None = None,
    chunks: list[Chunk] | None = None,
    client=None,
    model: str = config.GEN_MODEL,
) -> SummaryResult:
    """Summarise `text`, returning the summary and its source attributions.

    `aspects` is either a preset name ("generic", "clinical") or an explicit
    list of retrieval queries. `chunk_budget` caps how many chunks reach the
    model; leave it None to send everything the aspect sweep retrieved.
    """
    import anthropic

    from .chunking import chunk_text

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

    if client is None:
        client = anthropic.Anthropic()

    task = instruction or config.length_instruction(length)

    request = {
        "model": model,
        "max_tokens": config.MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "thinking": {"type": "adaptive"},
        # `effort` is fine alongside citations; `output_config.format` is not
        # (that combination returns a 400).
        "output_config": {"effort": config.EFFORT},
        "messages": [
            {
                "role": "user",
                "content": _build_document_blocks(selected)
                + [{"type": "text", "text": task}],
            }
        ],
    }

    response = client.messages.create(**request)

    # Clinical source material (overdose, self-harm, infectious disease) can trip
    # a safety classifier. A refusal arrives as HTTP 200, so check for it before
    # reading content.
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        return SummaryResult(
            summary="",
            facts=[],
            chunks_used=selected,
            source_text=text,
            stop_reason="refusal",
            refusal=getattr(details, "explanation", None) or "request was declined",
        )

    summary, facts = _parse_response(response, selected)

    return SummaryResult(
        summary=summary,
        facts=facts,
        chunks_used=selected,
        source_text=text,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        stop_reason=response.stop_reason,
    )
