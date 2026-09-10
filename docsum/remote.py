"""Generation through Groq's OpenAI-compatible API, grounded by alignment.

Same shape as the local backend and for the same reason: **Groq has no citations
feature.** It serves open models over an OpenAI-compatible endpoint, so nothing
in the protocol can guarantee that a citation the model writes is real. Asking
for span offsets would produce authoritative-looking fiction. So this backend
generates freely and recovers attribution afterwards with `docsum/grounding.py`
-- the identical alignment the local backend uses, so a comparison between them
measures the two models rather than two implementations.

What it buys over `local` is model size and speed: `openai/gpt-oss-120b` is an
order of magnitude larger than the 8B that fits on a 16GB card, and a summary
comes back in about a second instead of half a minute. What it costs is that the
document leaves the machine, which the local and extractive backends never do --
see the privacy note in the README before pointing this at anything sensitive.

Deliberately not using an SDK. The one call this needs is a JSON POST, `httpx`
is already a dependency via the Anthropic SDK, and adding `openai` or `groq`
would pull a second client library into the project to save about ten lines.

Reasoning models on this endpoint (the `gpt-oss` family) return their reasoning
in a separate `reasoning` field rather than inline, so `content` arrives clean.
Models that instead inline it in `<think>` tags are handled by
`grounding.strip_artifacts`, so both conventions are safe.
"""

from __future__ import annotations

import os

import httpx

from . import config
from .chunking import Chunk, chunk_text
from .grounding import ground, strip_artifacts
from .retrieval import ChunkIndex
from .summarizer import ASPECT_PRESETS, SummaryResult

# Reused verbatim from the local backend: the instruction not to attribute
# anything is the point, and having the two prompts drift would confound any
# comparison between the models.
from .local import LOCAL_SYSTEM_PROMPT as REMOTE_SYSTEM_PROMPT


class RemoteError(RuntimeError):
    """A Groq request failed in a way the caller should see verbatim."""


def _api_key() -> str:
    key = os.environ.get(config.GROQ_API_KEY_ENV)
    if not key:
        raise RemoteError(
            f"{config.GROQ_API_KEY_ENV} is not set. Export it, or use "
            f"--backend extractive (no credentials) or --backend local (no network)."
        )
    return key


def _post(payload: dict, timeout: float) -> dict:
    # Groq sits behind Cloudflare, which rejects requests carrying no
    # recognisable User-Agent with a 403 (error 1010) that reads like an auth
    # failure but is not one. Sending an explicit agent avoids the false alarm.
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": "docsum/0.1 (+https://github.com/)",
    }
    try:
        response = httpx.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            headers=headers, json=payload, timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise RemoteError(f"could not reach Groq: {exc}") from exc

    if response.status_code == 401:
        raise RemoteError("Groq rejected the API key (401).")
    if response.status_code == 429:
        raise RemoteError("Groq rate limit hit (429). Retry, or lower the batch size.")
    if response.status_code >= 400:
        raise RemoteError(f"Groq returned {response.status_code}: {response.text[:300]}")

    return response.json()


def available_models(timeout: float = 30.0) -> list[str]:
    """Model ids this key can use. Useful for checking access without spending."""
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "User-Agent": "docsum/0.1 (+https://github.com/)",
    }
    response = httpx.get(f"{config.GROQ_BASE_URL}/models", headers=headers, timeout=timeout)
    if response.status_code >= 400:
        raise RemoteError(f"Groq returned {response.status_code}: {response.text[:300]}")
    return sorted(m["id"] for m in response.json().get("data", []))


def summarize_remote(
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
    max_tokens: int = config.GROQ_MAX_TOKENS,
    timeout: float = config.GROQ_TIMEOUT,
) -> SummaryResult:
    """Summarise `text` with a Groq-hosted model, grounding claims after the fact.

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
        "Write a factual summary of the document these excerpts are drawn from. "
        "Cover every substantive point the excerpts establish."
    )
    excerpts = "\n\n".join(
        f"[excerpt {i}]\n{chunk.text}" for i, chunk in enumerate(selected)
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REMOTE_SYSTEM_PROMPT},
            {"role": "user", "content": f"{excerpts}\n\n---\n\n{task}"},
        ],
        # Deterministic: an eval comparing this backend against the others is
        # meaningless if it resamples between runs.
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    data = _post(payload, timeout)
    choice = data["choices"][0]
    summary = strip_artifacts(choice["message"].get("content") or "")

    if not summary:
        return SummaryResult(
            summary="", facts=[], chunks_used=selected, source_text=text,
            stop_reason=choice.get("finish_reason") or "empty",
        )

    facts = ground(summary, text, selected)
    usage = data.get("usage", {})

    return SummaryResult(
        summary=summary,
        facts=facts,
        chunks_used=selected,
        source_text=text,
        usage={
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model": data.get("model", model),
        },
        stop_reason=choice.get("finish_reason"),
    )
