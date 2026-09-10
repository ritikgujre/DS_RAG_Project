"""Hybrid retrieval over document chunks: dense embeddings + BM25.

Dense retrieval catches paraphrase ("cardiac arrest" vs "the heart stopped");
BM25 catches the rare literal tokens that embeddings smooth away -- drug names,
dosages, lab values, identifiers. Clinical text needs both, so scores from each
are normalised and blended.

Everything here runs locally on CPU. No document text leaves the machine during
retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config
from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_score: float
    sparse_score: float


@lru_cache(maxsize=2)
def _load_embedder(model_name: str):
    """Load the sentence-transformer once per process (it is ~90MB on disk)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping digits and decimals intact for BM25."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A flat array maps to all-zeros rather than dividing by 0."""
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class ChunkIndex:
    """Searchable index over one document's chunks."""

    def __init__(self, chunks: list[Chunk], embed_model: str = config.EMBED_MODEL):
        self.chunks = chunks
        self.embed_model = embed_model
        self._embeddings: np.ndarray | None = None
        self._bm25 = None

    # -- building ------------------------------------------------------------

    def build(self) -> ChunkIndex:
        """Compute embeddings and the BM25 index. Safe to call more than once."""
        if not self.chunks:
            return self

        if self._embeddings is None:
            embedder = _load_embedder(self.embed_model)
            self._embeddings = embedder.encode(
                [c.text for c in self.chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )

        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            corpus = [_tokenize(c.text) for c in self.chunks]
            # BM25Okapi divides by the corpus average length; an all-empty
            # corpus would blow up, so guard it.
            if any(corpus):
                self._bm25 = BM25Okapi(corpus)

        return self

    # -- searching -----------------------------------------------------------

    def search(self, query: str, top_k: int = config.TOP_K) -> list[Hit]:
        """Return the top_k chunks for `query`, ranked by blended score."""
        if not self.chunks:
            return []
        self.build()

        embedder = _load_embedder(self.embed_model)
        q_vec = embedder.encode([query], normalize_embeddings=True)[0]
        # Embeddings are L2-normalised, so the dot product is cosine similarity.
        dense = np.asarray(self._embeddings) @ q_vec

        if self._bm25 is not None:
            sparse = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=float)
        else:
            sparse = np.zeros(len(self.chunks), dtype=float)

        w = config.DENSE_WEIGHT
        blended = w * _minmax(dense) + (1.0 - w) * _minmax(sparse)

        order = np.argsort(-blended)[:top_k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=float(blended[i]),
                dense_score=float(dense[i]),
                sparse_score=float(sparse[i]),
            )
            for i in order
        ]

    def search_many(
        self, queries: list[str], top_k: int = config.TOP_K, budget: int | None = None
    ) -> list[Chunk]:
        """Run several queries and merge the results into one ordered chunk set.

        Summarisation needs coverage, not just the single best-matching passage,
        so results are merged by best-score-per-chunk and then returned in
        *document order* -- a summary written from chunks shuffled into relevance
        order reads as disconnected facts.
        """
        best: dict[int, float] = {}
        for query in queries:
            for hit in self.search(query, top_k=top_k):
                key = hit.chunk.index
                if hit.score > best.get(key, float("-inf")):
                    best[key] = hit.score

        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        if budget is not None:
            ranked = ranked[:budget]

        by_index = {c.index: c for c in self.chunks}
        return [by_index[i] for i, _ in sorted(ranked, key=lambda kv: kv[0])]


def build_index(chunks: list[Chunk]) -> ChunkIndex:
    return ChunkIndex(chunks).build()
