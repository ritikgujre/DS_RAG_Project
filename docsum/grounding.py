"""Recovering attribution for text a model wrote freely.

The `api` backend gets spans from Claude, which extracts `cited_text` itself, so
a citation there cannot be fabricated. No other generator offers that: not a
local model, and not an OpenAI-compatible endpoint such as Groq. Asking those
models to emit their own offsets would be exactly the prompt-engineered
attribution this project exists to avoid -- it yields citations that look
authoritative and cannot be checked.

So the generative backends without a citations feature do not ask. They generate
unconstrained prose, and this module recovers the attribution afterwards by
aligning each generated sentence against the source independently. A sentence
that clears the threshold gets a real span and a support score; one that does not
is left **uncited**, which is the honest outcome and shows up in
`SummaryResult.unattributed_claims`.

Shared by `docsum/local.py` and `docsum/remote.py` so the two differ only in
where the text came from, never in how a claim earns its citation -- otherwise
comparing them would measure two alignment implementations rather than two
models.

The score blends embedding cosine with lexical containment. Cosine alone rates a
fluent paraphrase highly even when it swapped a dosage, because the sentence
embedding barely moves; containment is what notices that the tokens carrying the
specifics are absent. Clinical text needs both, for the same reason retrieval is
hybrid.

Validated, with limits: see `docsum/validation.py`. Against 400 human-scored
summaries this coverage moves the right way against hallucination and omission
but weakly, and it does not reproduce the humans' ranking of systems. It
measures **traceability**, not factuality.
"""

from __future__ import annotations

import re

import numpy as np

from . import config
from .chunking import Chunk, sentences_within, split_sentences
from .retrieval import _load_embedder, _tokenize
from .summarizer import Fact, SourceSpan

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_LEAD_IN = re.compile(r"^\s*(here is[^\n:]*:|here's[^\n:]*:|summary:)\s*", re.I)


def strip_artifacts(text: str) -> str:
    """Remove reasoning blocks and lead-ins that instruct models still emit.

    Some models return reasoning in a dedicated response field and some inline it
    in `<think>` tags; this handles the inline case so a reasoning trace never
    reaches the summary and gets scored as a claim.
    """
    return _LEAD_IN.sub("", _THINK_BLOCK.sub("", text)).strip()


def containment(generated: str, source: str) -> float:
    """Share of the generated sentence's tokens that occur in the source span.

    Deliberately asymmetric. The question is whether the claim is supported by
    the source, not whether the two say the same amount -- a short claim drawn
    from a long sentence is fully supported and should score as such.
    """
    gen = set(_tokenize(generated))
    if not gen:
        return 0.0
    return len(gen & set(_tokenize(source))) / len(gen)


def align(generated: list, candidates: list, embedder=None) -> list[list[SourceSpan]]:
    """Match each generated sentence to the source spans that support it.

    `generated` is a list of Sentence; `candidates` is a list of
    (Sentence, chunk_index) as produced by `chunking.sentences_within`.
    """
    if not candidates or not generated:
        return [[] for _ in generated]

    embedder = embedder or _load_embedder(config.EMBED_MODEL)
    gen_vecs = np.asarray(
        embedder.encode([s.text for s in generated], normalize_embeddings=True,
                        show_progress_bar=False, batch_size=32)
    )
    src_vecs = np.asarray(
        embedder.encode([s.text for s, _ in candidates], normalize_embeddings=True,
                        show_progress_bar=False, batch_size=32)
    )
    cosine = gen_vecs @ src_vecs.T

    w = config.LOCAL_ALIGN_DENSE_WEIGHT
    out: list[list[SourceSpan]] = []

    for gi, gen_sentence in enumerate(generated):
        scores = np.array([
            w * cosine[gi, ci]
            + (1.0 - w) * containment(gen_sentence.text, candidates[ci][0].text)
            for ci in range(len(candidates))
        ])
        best = float(scores.max())
        if best < config.LOCAL_ALIGN_THRESHOLD:
            out.append([])  # an unsupported claim -- surfaced, not hidden
            continue

        # A generated sentence often fuses several source facts, so keep every
        # span close to the best one rather than only the single argmax.
        keep = [
            int(i) for i in np.argsort(-scores)[: config.LOCAL_MAX_SUPPORT_SPANS]
            if scores[i] >= config.LOCAL_ALIGN_THRESHOLD
            and scores[i] >= best - config.LOCAL_SUPPORT_MARGIN
        ]
        out.append([
            SourceSpan(
                cited_text=candidates[i][0].text,
                chunk_index=candidates[i][1],
                start=candidates[i][0].start,
                end=candidates[i][0].end,
                support=round(float(scores[i]), 4),
            )
            for i in sorted(keep, key=lambda i: candidates[i][0].start)
        ])

    return out


def ground(summary: str, text: str, selected: list[Chunk]) -> list[Fact]:
    """Split a generated summary into claims and attach the spans supporting each.

    Candidates are the sentences of the chunks the model actually saw: a claim
    cannot honestly be traced to text that was never sent to it.
    """
    candidates = sentences_within(
        text, selected, min_chars=config.EXTRACTIVE_MIN_SENTENCE_CHARS
    )
    sentences = split_sentences(summary)
    aligned = align(sentences, candidates)
    return [
        Fact(text=sentence.text, sources=spans)
        for sentence, spans in zip(sentences, aligned)
    ]
