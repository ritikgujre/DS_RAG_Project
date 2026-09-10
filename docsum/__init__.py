"""RAG document summariser with source attribution."""

from .chunking import Chunk, chunk_text, split_sentences
from .datasets import DocPair, load_mts_dialog, load_multiclinsum, read_document
from .extractive import summarize_extractive
from .local import summarize_local
from .remote import summarize_remote
from .verified import summarize_verified
from .retrieval import ChunkIndex, build_index
from .summarizer import Fact, SourceSpan, SummaryResult, summarize

__all__ = [
    "Chunk", "chunk_text", "split_sentences",
    "DocPair", "load_mts_dialog", "load_multiclinsum", "read_document",
    "ChunkIndex", "build_index",
    "Fact", "SourceSpan", "SummaryResult", "summarize",
    "summarize_extractive", "summarize_local", "summarize_remote", "summarize_verified",
]
