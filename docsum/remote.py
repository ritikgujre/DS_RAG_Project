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
import random
import re
import time

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


# Groq expresses reset times as "630ms", "1m26.4s", "2.5s" rather than seconds.
_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def _parse_reset(value: str | None) -> float | None:
    """Seconds until a rate-limit window resets, from Groq's duration format."""
    if not value:
        return None
    value = value.strip()
    try:  # `retry-after` is plain seconds when present
        return float(value)
    except ValueError:
        pass
    m = _DURATION.match(value)
    if not m or not any(m.groups()):
        return None
    minutes, seconds, millis = (float(g) if g else 0.0 for g in m.groups())
    return minutes * 60 + seconds + millis / 1000


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """How long to wait before retrying a rate-limited request.

    Prefers what the server says over guessing. The token window is the binding
    one in practice: the free tier allows 1000 requests/minute but only 8000
    tokens/minute, and one case report costs roughly 2500 tokens round trip --
    so a batch run is paced by tokens, roughly three documents per minute.
    """
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        delay = _parse_reset(response.headers.get(header))
        if delay is not None:
            # Small jitter so concurrent callers do not resynchronise.
            return min(delay + random.uniform(0.1, 0.5), config.GROQ_MAX_BACKOFF)
    return min(2.0 ** attempt + random.uniform(0, 1), config.GROQ_MAX_BACKOFF)


def _post(payload: dict, timeout: float) -> dict:
    # Groq sits behind Cloudflare, which rejects requests carrying no
    # recognisable User-Agent with a 403 (error 1010) that reads like an auth
    # failure but is not one. Sending an explicit agent avoids the false alarm.
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": "docsum/0.1 (+https://github.com/)",
    }
    last_error = ""
    for attempt in range(config.GROQ_MAX_RETRIES + 1):
        try:
            response = httpx.post(
                f"{config.GROQ_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
        except httpx.HTTPError as exc:
            # Transient network trouble is worth one more attempt; a dead host
            # is not worth many.
            last_error = f"could not reach Groq: {exc}"
            if attempt >= config.GROQ_MAX_RETRIES:
                raise RemoteError(last_error) from exc
            time.sleep(min(2.0 ** attempt, config.GROQ_MAX_BACKOFF))
            continue

        if response.status_code == 401:
            raise RemoteError("Groq rejected the API key (401).")

        # Rate limits are the normal case for a batch run on the free tier, not
        # an error: wait out the window the server names rather than failing the
        # document. Without this a 20-document evaluation dies after two.
        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"Groq returned {response.status_code}"
            if attempt >= config.GROQ_MAX_RETRIES:
                raise RemoteError(
                    f"{last_error} after {attempt + 1} attempts: {response.text[:200]}"
                )
            time.sleep(_retry_delay(response, attempt))
            continue

        if response.status_code >= 400:
            raise RemoteError(f"Groq returned {response.status_code}: {response.text[:300]}")

        return response.json()

    raise RemoteError(last_error or "Groq request failed")


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
