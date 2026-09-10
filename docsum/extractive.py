"""Extractive summarisation: attribution exact by construction, no API key.

The generative backend asks a model to write prose and relies on the Claude
API's citations feature to prove where each claim came from. This backend
inverts that: it *selects* source sentences verbatim, so the summary is made of
the document. A citation is then not a claim to be verified but a fact of how
the text was assembled -- `source_text[span.start:span.end]` is literally the
sentence that was chosen.

Three properties follow for free, rather than being measured after the fact:

* `citation_integrity` is 1.0 -- the cited text is the slice.
* `numeric_fidelity` is 1.0 -- every number was copied, so none was invented.
* `attribution_coverage` is 1.0 -- every emitted sentence carries its source.

The cost is prose quality: the result reads as a highlight reel rather than a
flowing summary. That is the trade being made, and it is why this backend
returns the same `SummaryResult` as the generative one -- `report.py` and
`evaluate.py` treat them identically, so the two modes stay comparable.

Selection runs in two phases, in that order because the MTS-Dialog correlation
study puts omission (31-54%) an order of magnitude above hallucination (<4%):
coverage is the failure worth designing against.

1. Coverage: each aspect query claims its single best sentence, provided that
   sentence clears `MIN_ASPECT_SCORE`. The gate matters -- without it, an aspect
   the document never addresses ("past medical history" in a paper that has
   none) would still drag in its least-bad match.
2. MMR fill: remaining slots go to sentences that are relevant but unlike what
   is already selected, which is what stops five sentences all restating the
   diagnosis.
"""

from __future__ import annotations

import numpy as np

from . import config
from .chunking import Chunk, chunk_text, sentences_within
from .retrieval import ChunkIndex, _load_embedder
from .summarizer import ASPECT_PRESETS, Fact, SourceSpan, SummaryResult


def _sentence_budget(n_candidates: int) -> int:
    """How many sentences the summary may keep."""
    target = round(n_candidates * config.EXTRACTIVE_RATIO)
    return max(
        config.EXTRACTIVE_MIN_SENTENCES,
        min(config.EXTRACTIVE_MAX_SENTENCES, target, n_candidates),
    )


def _candidates(text: str, selected: list[Chunk]) -> list:
    """Sentences eligible for selection: those inside a retrieved chunk."""
    return sentences_within(text, selected, min_chars=config.EXTRACTIVE_MIN_SENTENCE_CHARS)


def _select(
    relevance: np.ndarray, similarity: np.ndarray, budget: int
) -> list[int]:
    """Phase 1 (per-aspect coverage) then phase 2 (MMR fill).

    `relevance` is (n_aspects, n_sentences); `similarity` is the sentence-by-
    sentence cosine matrix.
    """
    n_aspects, n_sentences = relevance.shape
    chosen: list[int] = []

    # Phase 1 -- guarantee each answerable aspect is represented.
    for aspect in range(n_aspects):
        if len(chosen) >= budget:
            break
        scores = relevance[aspect].copy()
        scores[chosen] = -np.inf
        best = int(np.argmax(scores))
        if scores[best] >= config.EXTRACTIVE_MIN_ASPECT_SCORE:
            chosen.append(best)

    # Phase 2 -- MMR over what is left.
    pooled = relevance.max(axis=0)
    lam = config.EXTRACTIVE_MMR_LAMBDA
    while len(chosen) < budget:
        remaining = [i for i in range(n_sentences) if i not in chosen]
        if not remaining:
            break
        if chosen:
            redundancy = similarity[np.ix_(remaining, chosen)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        mmr = lam * pooled[remaining] - (1.0 - lam) * redundancy
        chosen.append(remaining[int(np.argmax(mmr))])

    return chosen


def summarize_extractive(
    text: str,
    *,
    doc_id: str = "doc",
    aspects: str | list[str] = "generic",
    top_k: int = config.TOP_K,
    chunk_budget: int | None = None,
    max_sentences: int | None = None,
    index: ChunkIndex | None = None,
    chunks: list[Chunk] | None = None,
) -> SummaryResult:
    """Summarise `text` by selecting source sentences, with exact attribution.

    Signature deliberately mirrors `summarizer.summarize` minus the API-only
    arguments, so callers can switch backends without restructuring.
    """
    if chunks is None:
        chunks = chunk_text(text, doc_id=doc_id)
    if not chunks:
        return SummaryResult(summary="", facts=[], chunks_used=[], source_text=text)

    queries = ASPECT_PRESETS[aspects] if isinstance(aspects, str) else aspects

    if index is None:
        index = ChunkIndex(chunks).build()

    selected_chunks = index.search_many(queries, top_k=top_k, budget=chunk_budget)
    if not selected_chunks:
        selected_chunks = chunks

    candidates = _candidates(text, selected_chunks)
    if not candidates:
        return SummaryResult(
            summary="", facts=[], chunks_used=selected_chunks, source_text=text
        )

    embedder = _load_embedder(config.EMBED_MODEL)
    sent_vecs = np.asarray(
        embedder.encode(
            [s.text for s, _ in candidates],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
    )
    query_vecs = np.asarray(embedder.encode(queries, normalize_embeddings=True))

    # Both sides are L2-normalised, so these dot products are cosine similarity.
    relevance = query_vecs @ sent_vecs.T
    similarity = sent_vecs @ sent_vecs.T

    budget = max_sentences or _sentence_budget(len(candidates))
    chosen = _select(relevance, similarity, budget)

    # Document order, not relevance order: sentences pulled out of sequence read
    # as disconnected assertions, the same reason `search_many` restores order.
    chosen.sort()

    facts: list[Fact] = []
    for i in chosen:
        sentence, chunk_index = candidates[i]
        # cited_text is the slice itself, so integrity cannot drift.
        facts.append(
            Fact(
                text=sentence.text,
                sources=[
                    SourceSpan(
                        cited_text=text[sentence.start : sentence.end],
                        chunk_index=chunk_index,
                        start=sentence.start,
                        end=sentence.end,
                    )
                ],
            )
        )

    return SummaryResult(
        summary=" ".join(f.text for f in facts),
        facts=facts,
        chunks_used=selected_chunks,
        source_text=text,
        usage={"input_tokens": 0, "output_tokens": 0},
        stop_reason="end_turn",
    )
