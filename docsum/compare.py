"""Run several backends over the same documents and score them side by side.

The three backends trade off against each other rather than ranking cleanly, so
the only useful comparison holds everything else fixed: same documents, same
chunking, same retrieved chunks, same metrics. This module builds the chunk
index once per document and hands it to every backend, so any difference in the
scores comes from generation and attribution rather than from retrieval drift.

Read the resulting table with the asymmetry in mind (see docsum/local.py):
`citation_integrity` is 1.0 by construction for `extractive` and trivially 1.0
for `local`, so it discriminates nothing between them -- it is there to catch
offset bugs. `attribution_coverage` and `numeric_fidelity` are the columns that
separate the backends, and ROUGE is the one where extractive is expected to lose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .chunking import chunk_text
from .datasets import DocPair
from .evaluate import Scores, aggregate, score
from .retrieval import ChunkIndex
from .summarizer import SummaryResult


@dataclass
class BackendRun:
    """Aggregate result of one backend over one document set."""

    backend: str
    rows: list[Scores] = field(default_factory=list)
    seconds: float = 0.0
    failed: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.rows)

    def summary_row(self) -> dict:
        row = aggregate(self.rows)
        n = row.pop("n_docs", 0)
        row["n_docs"] = n
        row["sec_per_doc"] = round(self.seconds / n, 2) if n else 0.0
        return row


def _run_one(backend: str, doc: DocPair, aspects: str, index, chunks) -> SummaryResult:
    """Dispatch to a backend, reusing the prebuilt index so retrieval is identical."""
    common = dict(doc_id=doc.doc_id, aspects=aspects, index=index, chunks=chunks)

    if backend == "extractive":
        from .extractive import summarize_extractive

        return summarize_extractive(doc.text, **common)
    if backend == "local":
        from .local import summarize_local

        return summarize_local(doc.text, **common)
    if backend == "api":
        from .summarizer import summarize

        return summarize(doc.text, **common)
    raise ValueError(f"unknown backend {backend!r}")


def compare(
    docs: list[DocPair],
    backends: list[str],
    aspects: str = "generic",
    verbose: bool = True,
) -> dict[str, BackendRun]:
    """Score every backend in `backends` over every document in `docs`."""
    # Chunk and index once per document, then reuse across backends.
    prepared = []
    for doc in docs:
        chunks = chunk_text(doc.text, doc_id=doc.doc_id)
        prepared.append((doc, chunks, ChunkIndex(chunks).build() if chunks else None))

    runs: dict[str, BackendRun] = {}

    for backend in backends:
        run = BackendRun(backend=backend)
        started = time.time()

        for doc, chunks, index in prepared:
            try:
                result = _run_one(backend, doc, aspects, index, chunks)
            except Exception as exc:
                run.failed += 1
                if not run.error:
                    run.error = f"{type(exc).__name__}: {exc}"
                # A backend that cannot run at all (no key, no GPU) should not
                # abort the comparison for the ones that can.
                if run.failed == 1 and len(run.rows) == 0:
                    break
                continue

            if result.refusal:
                run.failed += 1
                if not run.error:
                    run.error = f"refused: {result.refusal}"
                continue

            run.rows.append(score(result, reference=doc.reference))
            if verbose:
                print(f"  {backend:>10} {doc.doc_id}: "
                      f"{len(result.facts)} claims, "
                      f"{result.coverage:.0%} cited")

        run.seconds = time.time() - started
        runs[backend] = run

    return runs


# Columns worth showing, in the order that tells the story: what it produced,
# how much of it is attributable, then how close it lands to the gold summary.
_COLUMNS = [
    ("n_docs", "docs"),
    ("sec_per_doc", "s/doc"),
    ("attribution_coverage", "attrib"),
    ("numeric_fidelity", "numeric"),
    ("citation_integrity", "integrity"),
    ("rouge1_f1", "ROUGE-1"),
    ("rougeL_f1", "ROUGE-L"),
    ("n_claims", "claims"),
]


def format_table(runs: dict[str, BackendRun]) -> str:
    """Render the comparison as a fixed-width table."""
    lines = []
    header = f"{'backend':<12}" + "".join(f"{label:>11}" for _, label in _COLUMNS)
    lines.append(header)
    lines.append("-" * len(header))

    for backend, run in runs.items():
        if not run.ok:
            lines.append(f"{backend:<12}  did not run -- {run.error[:70]}")
            continue
        row = run.summary_row()
        cells = "".join(f"{row.get(key, 0):>11}" for key, _ in _COLUMNS)
        lines.append(f"{backend:<12}{cells}")
        if run.failed:
            lines.append(f"{'':<12}  ({run.failed} document(s) failed: {run.error[:60]})")

    lines.append("")
    lines.append("attrib/numeric are the discriminating columns; integrity is 1.0 by")
    lines.append("construction for extractive and trivially so for local (see docsum/local.py).")
    return "\n".join(lines)
