"""Central configuration for the document summariser."""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent

MTS_DIR = DATA_ROOT / "MTS-Dialog-main"
MULTICLINSUM_DIR = DATA_ROOT / "15517617"

# Where extracted dataset files and cached embeddings live.
CACHE_DIR = PROJECT_ROOT / ".cache"

# --- Models ------------------------------------------------------------------

# Generation. Opus 5 has a 1M-token context window, so a retrieved chunk set
# never comes close to the limit.
GEN_MODEL = os.environ.get("DOCSUM_MODEL", "claude-opus-5")

# Retrieval embeddings run locally on CPU -- no API calls, no data leaving the
# machine for the retrieval half of the pipeline. MiniLM is 22M params and
# indexes a 100-page document in a few seconds on 8 cores.
EMBED_MODEL = os.environ.get("DOCSUM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking ----------------------------------------------------------------

# Target chunk size in characters. Chunks are the unit of retrieval AND the unit
# of citation, so this is the main quality knob: smaller chunks give tighter
# citations but more fragmented context.
CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_SENTENCES = 1

# --- Retrieval ---------------------------------------------------------------

# How many chunks each aspect query may contribute. On the full 592-document
# MultiClinSum English gold set, only 7 documents have more chunks than this, and
# the sweep returns every chunk for all but two of them -- so on a typical case
# report retrieval is effectively a pass-through and does not decide quality.
#
# The two exceptions are the ones to watch, and they are why this is not simply
# unbounded: gs_en_504 (32 chunks) retrieves 81%, and gs_en_296 (29 chunks)
# retrieves just 34%. On documents that large the sweep is discarding most of the
# source, and omission is already the dominant failure mode in this corpus
# (31-54% against under 4% hallucination in the MTS-Dialog correlation study).
# Raise TOP_K for long uploads where omission costs more than input tokens.
TOP_K = 8

# Weight of dense (embedding) score vs sparse (BM25) score in the hybrid rank.
DENSE_WEIGHT = 0.65

# --- Generation --------------------------------------------------------------

MAX_TOKENS = 8000
EFFORT = os.environ.get("DOCSUM_EFFORT", "high")

# --- Extractive backend ------------------------------------------------------

# Selects source sentences verbatim instead of generating prose, so attribution
# is exact by construction and no API key is needed. See docsum/extractive.py.

# Summary length as a share of candidate sentences, then clamped by the bounds
# below. A ratio alone would give a 3-sentence summary of a short case report
# and a 40-sentence one of a long paper.
EXTRACTIVE_RATIO = 0.25
EXTRACTIVE_MIN_SENTENCES = 3
EXTRACTIVE_MAX_SENTENCES = 15

# Relevance floor for the per-aspect coverage pass. An aspect the document never
# addresses should contribute nothing rather than its least-bad match. Cosine
# similarity against MiniLM, where unrelated clinical text typically sits near
# 0.1 and a genuine match clears 0.3.
EXTRACTIVE_MIN_ASPECT_SCORE = 0.25

# MMR trade-off: 1.0 is pure relevance (and duplicates), 0.0 is pure novelty.
EXTRACTIVE_MMR_LAMBDA = 0.7

# Below this, a "sentence" is nearly always a heading, bullet or fragment.
EXTRACTIVE_MIN_SENTENCE_CHARS = 40

# --- Local GPU backend -------------------------------------------------------

# Generates prose on a local GPU, then aligns each generated sentence back to a
# source span. See docsum/local.py for why alignment replaces citations here.

# Pre-quantised 4-bit: ~6GB of VRAM and a ~6GB download, rather than the ~16GB
# the bf16 weights would cost. Override for a different model; anything with a
# chat template and a transformers-loadable checkpoint works.
#
# Not 14B, despite it fitting in VRAM on paper: loading the 14B 4-bit checkpoint
# segfaults inside bitsandbytes on this machine. Every individual step of the
# load works in isolation (slicing all 947 tensors from a 5GB shard, moving them
# to CUDA), and disabling the loader's thread pool does not help -- but with
# memory-mapping off the same load raises MemoryError instead, which points at
# host memory during quantised-parameter construction. This box has 31.6GB of
# RAM against a 2GB pagefile, so the commit limit is only ~33GB. An 8B has half
# the peak and loads in 8 seconds. Raising the pagefile may make 14B viable.
LOCAL_MODEL = os.environ.get("DOCSUM_LOCAL_MODEL", "unsloth/Qwen3-8B-bnb-4bit")
LOCAL_DEVICE = os.environ.get("DOCSUM_LOCAL_DEVICE", "cuda:0")
LOCAL_MAX_NEW_TOKENS = int(os.environ.get("DOCSUM_LOCAL_MAX_NEW_TOKENS", "1024"))

# Alignment score below which a generated sentence is left uncited rather than
# tied to its least-bad match. A false citation is worse than a missing one:
# an uncited sentence is visibly unsupported, a wrong one looks verified.
LOCAL_ALIGN_THRESHOLD = 0.45

# Dense vs lexical split in the alignment score. Weighted towards containment
# more heavily than retrieval is, because the failure being caught here is a
# fluent paraphrase that changed a number -- which moves the embedding barely
# at all but drops containment sharply.
LOCAL_ALIGN_DENSE_WEIGHT = 0.5

# A generated sentence may fuse several source facts, so more than one span can
# support it; keep the near-ties, not just the argmax.
LOCAL_MAX_SUPPORT_SPANS = 3
LOCAL_SUPPORT_MARGIN = 0.08

# --- Groq (remote) backend ---------------------------------------------------

# Generates through Groq's OpenAI-compatible API, then grounds the result with
# the same alignment the local backend uses. Groq has no citations feature, so
# attribution here is recovered, not guaranteed -- see docsum/grounding.py.

GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY_ENV = "GROQ_API_KEY"

# gpt-oss-120b is roughly an order of magnitude larger than the 8B that fits in
# 16GB of VRAM, and returns in about a second. It also puts its reasoning in a
# separate response field rather than inline, so the summary arrives clean;
# qwen3.6-27b on this endpoint leaks <think> blocks into the content instead.
GROQ_MODEL = os.environ.get("DOCSUM_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_TOKENS = int(os.environ.get("DOCSUM_GROQ_MAX_TOKENS", "2048"))
GROQ_TIMEOUT = float(os.environ.get("DOCSUM_GROQ_TIMEOUT", "120"))

# Hard ceiling on chunk size. Sentence boundaries usually bound this, but nothing
# guarantees a document has any: one gold case report is 21k chars in 11
# sentences, the longest 20,238 chars. A chunk is the unit of citation, so an
# unbounded one makes attribution correct but useless.
CHUNK_MAX_CHARS = 2400

# Ceiling for the adaptive top_k in search_many. Bounded so a pathological
# document cannot push every chunk into the prompt.
TOP_K_MAX = 32

# Rate limiting. The free tier's binding constraint is tokens per minute (8000),
# not requests (1000), and one case report costs roughly 2500 tokens round trip
# -- so a batch run is paced at about three documents per minute. Retrying is
# the normal path for a batch run, not an error case.
GROQ_MAX_RETRIES = int(os.environ.get("DOCSUM_GROQ_MAX_RETRIES", "6"))
GROQ_MAX_BACKOFF = float(os.environ.get("DOCSUM_GROQ_MAX_BACKOFF", "90"))
