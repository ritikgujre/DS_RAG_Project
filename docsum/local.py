"""Local GPU generation, with attribution recovered by alignment.

This backend writes prose like the API backend, but on hardware you own and
with no key. The catch is the part that matters most here: **no local model has
the Claude API's citations feature**, so nothing forces a citation to be real.
Asking a model to emit its own span offsets is exactly the prompt-engineered
attribution this project was built to avoid -- it produces citations that look
authoritative and cannot be checked.

So this backend does not ask. It generates unconstrained prose, then aligns each
generated sentence back to the source independently:

    generate --> split into sentences --> align each against the source
                                                   |
                              score >= threshold --+-- below threshold
                                        |                    |
                                 attach the span      leave it uncited

The span attached is always a real slice of the document, so `citation_integrity`
stays 1.0 -- but that is now a trivial property, not evidence. The meaningful
signals for this backend are `attribution_coverage` (how much of the summary
could be tied to the source at all) and `numeric_fidelity` (whether generated
numbers actually occur in the document). Unlike the extractive backend, where
both are 1.0 by construction, here they can fail -- and when they do, that is a
real finding about the model rather than a bug.

Alignment blends embedding similarity with lexical containment. Cosine alone
rates a fluent paraphrase highly even when it swapped a dosage, because the
sentence embedding barely moves; containment is what notices that the tokens
carrying the specifics are absent. Clinical text needs both, for the same reason
retrieval is hybrid.

`SourceSpan.support` records the alignment score, so a reader can tell a
near-verbatim restatement from a loose thematic match.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from . import config
from .chunking import Chunk, chunk_text, sentences_within, split_sentences
from .retrieval import ChunkIndex, _load_embedder, _tokenize
from .summarizer import ASPECT_PRESETS, Fact, SourceSpan, SummaryResult

# No citation instructions: this model is not asked to attribute anything, so
# telling it to would only invite confident invention. Its whole job is to write
# an accurate summary of the excerpts; grounding happens afterwards.
LOCAL_SYSTEM_PROMPT = """\
You write factual summaries that stay strictly inside the source excerpts you \
are given.

Rules:
- Use only what the excerpts state. Add no background knowledge, no inference, \
and no context of your own.
- Preserve every specific exactly as written: ages, dates, dosages, lab values, \
measurements and named entities. Never round, convert or approximate a number.
- Do not speculate about what is absent. If the excerpts do not establish \
something, leave it out rather than hedging about it.
- Write flowing prose, not bullet points or fragments.
- Prefer completeness over brevity: omitting a materially relevant fact is a \
worse failure than a slightly longer summary.
- Output only the summary itself, with no preamble, heading or commentary."""


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load tokenizer + model onto the GPU once per process.

    Cached because loading a quantised 14B takes far longer than generating from
    it, so a batch run must not pay that cost per document.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "the local backend needs a CUDA GPU and torch reports none. A "
            "CPU-only torch build is the usual cause -- reinstall from the CUDA "
            "index, or use the extractive backend, which needs no GPU."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=config.LOCAL_DEVICE,
    )
    model.eval()
    return tokenizer, model


def _build_prompt(tokenizer, chunks: list[Chunk], task: str) -> str:
    """Render the excerpts and the task through the model's own chat template."""
    excerpts = "\n\n".join(
        f"[excerpt {i}]\n{chunk.text}" for i, chunk in enumerate(chunks)
    )
    messages = [
        {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"{excerpts}\n\n---\n\n{task}"},
    ]
    try:
        # Qwen3 and other hybrid-reasoning models default to a thinking pass.
        # Summarising supplied excerpts does not need one, and it would multiply
        # the wall clock, so opt out where the template understands the flag.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _strip_artifacts(text: str) -> str:
    """Remove reasoning blocks and lead-ins that instruct models still emit."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*(here is[^\n:]*:|here's[^\n:]*:|summary:)\s*", "", text, flags=re.I)
    return text.strip()


def _containment(generated: str, source: str) -> float:
    """Share of the generated sentence's tokens that occur in the source span.

    Deliberately asymmetric. The question is whether the claim is supported by
    the source, not whether the two say the same amount -- a short claim drawn
    from a long sentence is fully supported and should score as such.
    """
    gen = set(_tokenize(generated))
    if not gen:
        return 0.0
    return len(gen & set(_tokenize(source))) / len(gen)


def _align(generated: list, candidates: list, embedder) -> list[list[SourceSpan]]:
    """Match each generated sentence to the source spans that support it."""
    if not candidates or not generated:
        return [[] for _ in generated]

    gen_vecs = np.asarray(
        embedder.encode(
            [s.text for s in generated], normalize_embeddings=True,
            show_progress_bar=False, batch_size=32,
        )
    )
    src_vecs = np.asarray(
        embedder.encode(
            [s.text for s, _ in candidates], normalize_embeddings=True,
            show_progress_bar=False, batch_size=32,
        )
    )
    cosine = gen_vecs @ src_vecs.T

    w = config.LOCAL_ALIGN_DENSE_WEIGHT
    out: list[list[SourceSpan]] = []

    for gi, gen_sentence in enumerate(generated):
        scores = np.array([
            w * cosine[gi, ci]
            + (1.0 - w) * _containment(gen_sentence.text, candidates[ci][0].text)
            for ci in range(len(candidates))
        ])
        best = float(scores.max())
        if best < config.LOCAL_ALIGN_THRESHOLD:
            out.append([])  # an unsupported claim -- surfaced, not hidden
            continue

        # A generated sentence often fuses two source facts, so keep every span
        # close to the best one rather than only the single argmax.
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


def summarize_local(
    text: str,
    *,
    doc_id: str = "doc",
    aspects: str | list[str] = "generic",
    instruction: str | None = None,
    top_k: int = config.TOP_K,
    chunk_budget: int | None = None,
    index: ChunkIndex | None = None,
    chunks: list[Chunk] | None = None,
    model: str = config.LOCAL_MODEL,
    max_new_tokens: int = config.LOCAL_MAX_NEW_TOKENS,
) -> SummaryResult:
    """Summarise `text` with a local GPU model, grounding claims after the fact.

    Signature mirrors `summarizer.summarize` so callers can switch backends
    without restructuring.
    """
    import torch

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
        "Write a factual summary of the document these excerpts are drawn from. "
        "Cover every substantive point the excerpts establish."
    )

    tokenizer, lm = _load_model(model)
    prompt = _build_prompt(tokenizer, selected, task)
    inputs = tokenizer(prompt, return_tensors="pt").to(lm.device)

    with torch.inference_mode():
        generated = lm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Greedy: a summary should be reproducible, and an eval comparing
            # this backend against the others is meaningless if it resamples.
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    n_in = int(inputs["input_ids"].shape[1])
    completion = tokenizer.decode(generated[0][n_in:], skip_special_tokens=True)
    summary = _strip_artifacts(completion)
    if not summary:
        return SummaryResult(
            summary="", facts=[], chunks_used=selected, source_text=text,
            stop_reason="empty",
        )

    # Ground it. Candidates are the sentences of the chunks the model actually
    # saw: a claim cannot honestly be traced to text that was never sent to it.
    candidates = sentences_within(
        text, selected, min_chars=config.EXTRACTIVE_MIN_SENTENCE_CHARS
    )
    gen_sentences = split_sentences(summary)
    aligned = _align(gen_sentences, candidates, _load_embedder(config.EMBED_MODEL))

    facts = [
        Fact(text=sentence.text, sources=spans)
        for sentence, spans in zip(gen_sentences, aligned)
    ]

    return SummaryResult(
        summary=summary,
        facts=facts,
        chunks_used=selected,
        source_text=text,
        usage={
            "input_tokens": n_in,
            "output_tokens": int(generated.shape[1]) - n_in,
            "model": model,
        },
        stop_reason="end_turn",
    )
