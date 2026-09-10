#!/usr/bin/env python
"""Command line entry point for the RAG document summariser.

    python cli.py summarize path/to/document.pdf
    python cli.py summarize --dataset multiclinsum --n 3 --aspects clinical
    python cli.py inspect path/to/document.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docsum import config
from docsum.chunking import chunk_text
from docsum.compare import compare, format_table
from docsum.datasets import load_mts_dialog, load_multiclinsum, read_document
from docsum.extractive import summarize_extractive
from docsum.local import summarize_local
from docsum.report import to_html, to_json, to_text
from docsum.retrieval import ChunkIndex
from docsum.summarizer import ASPECT_PRESETS, summarize


def _load_dataset_docs(name: str, n: int, lang: str, split: str):
    if name == "multiclinsum":
        return load_multiclinsum(split if split != "auto" else "gs", lang, limit=n)
    if name == "mts":
        return load_mts_dialog(split if split != "auto" else "validation", limit=n)
    raise SystemExit(f"unknown dataset {name!r} (use 'multiclinsum' or 'mts')")


def cmd_summarize(args: argparse.Namespace) -> int:
    if args.dataset:
        docs = _load_dataset_docs(args.dataset, args.n, args.lang, args.split)
        if not docs:
            print("no documents loaded", file=sys.stderr)
            return 1
        jobs = [(d.doc_id, d.text, d.reference) for d in docs]
    elif args.path:
        text = read_document(args.path)
        if not text.strip():
            print(f"no extractable text in {args.path}", file=sys.stderr)
            return 1
        jobs = [(Path(args.path).name, text, None)]
    else:
        print("give a file path or --dataset", file=sys.stderr)
        return 1

    for doc_id, text, reference in jobs:
        print(f"\n### {doc_id}  ({len(text):,} chars)")
        if args.backend == "extractive":
            result = summarize_extractive(
                text,
                doc_id=doc_id,
                aspects=args.aspects,
                top_k=args.top_k,
                chunk_budget=args.chunk_budget,
                max_sentences=args.max_sentences,
            )
        elif args.backend == "local":
            result = summarize_local(
                text,
                doc_id=doc_id,
                aspects=args.aspects,
                top_k=args.top_k,
                chunk_budget=args.chunk_budget,
                model=args.local_model,
            )
        else:
            result = summarize(
                text,
                doc_id=doc_id,
                aspects=args.aspects,
                top_k=args.top_k,
                chunk_budget=args.chunk_budget,
                model=args.model,
            )

        if args.format == "json":
            print(to_json(result))
        elif args.format == "html":
            out = Path(args.out or f"{doc_id}.html")
            out.write_text(to_html(result, title=doc_id), encoding="utf-8")
            print(f"wrote {out}")
        else:
            print(to_text(result))

        if reference and args.show_reference:
            print("\n--- reference summary (dataset gold) ---")
            print(reference)

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Chunk and retrieve without calling the API -- no key needed."""
    if args.dataset:
        docs = _load_dataset_docs(args.dataset, 1, args.lang, args.split)
        doc_id, text = docs[0].doc_id, docs[0].text
    else:
        doc_id, text = Path(args.path).name, read_document(args.path)

    chunks = chunk_text(text, doc_id=doc_id)
    print(f"{doc_id}: {len(text):,} chars -> {len(chunks)} chunks")
    for c in chunks:
        preview = " ".join(c.text.split())[:72]
        print(f"  [{c.index:>3}] {c.start:>6}-{c.end:<6} {len(c.text):>5}ch  {preview}")

    # Verify the invariant citations depend on.
    assert all(text[c.start : c.end] == c.text for c in chunks), "offset drift"
    print("\noffsets verified: every chunk maps exactly back to the source")

    if args.query:
        index = ChunkIndex(chunks).build()
        print(f"\ntop {args.top_k} chunks for {args.query!r}:")
        for hit in index.search(args.query, top_k=args.top_k):
            preview = " ".join(hit.chunk.text.split())[:64]
            print(
                f"  [{hit.chunk.index:>3}] score={hit.score:.3f} "
                f"(dense={hit.dense_score:.3f} bm25={hit.sparse_score:.2f})  {preview}"
            )
    else:
        queries = ASPECT_PRESETS[args.aspects]
        index = ChunkIndex(chunks).build()
        selected = index.search_many(queries, top_k=args.top_k, budget=args.chunk_budget)
        print(
            f"\naspect sweep ({args.aspects}, {len(queries)} queries) selected "
            f"{len(selected)}/{len(chunks)} chunks: {[c.index for c in selected]}"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Score several backends over the same documents."""
    if not args.dataset:
        print("compare needs --dataset", file=sys.stderr)
        return 1

    docs = _load_dataset_docs(args.dataset, args.n, args.lang, args.split)
    if not docs:
        print("no documents loaded", file=sys.stderr)
        return 1

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"comparing {backends} over {len(docs)} document(s), aspects={args.aspects}")
    print()

    runs = compare(docs, backends, aspects=args.aspects, verbose=not args.quiet)
    print()
    print(format_table(runs))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("path", nargs="?", help="document to summarise (.txt/.md/.pdf/.docx)")
        p.add_argument("--dataset", choices=["multiclinsum", "mts"], help="use a bundled corpus instead")
        p.add_argument("--split", default="auto", help="dataset split (gs/ls, or train/validation/test1/test2)")
        p.add_argument("--lang", default="en", help="MultiClinSum language: en/es/fr/pt")
        p.add_argument("--aspects", default="generic", choices=sorted(ASPECT_PRESETS), help="retrieval aspect preset")
        p.add_argument("--top-k", type=int, default=config.TOP_K, help="chunks retrieved per aspect query")
        p.add_argument("--chunk-budget", type=int, default=None, help="cap total chunks sent to the model")

    p_sum = sub.add_parser("summarize", help="summarise a document with source attribution")
    add_common(p_sum)
    p_sum.add_argument("-n", type=int, default=1, help="how many dataset docs to run")
    p_sum.add_argument("--model", default=config.GEN_MODEL)
    p_sum.add_argument("--backend", default="api", choices=["api", "extractive", "local"],
                       help="api: Claude with native citations (needs a key). "
                            "extractive: select source sentences verbatim, no key. "
                            "local: generate on a local GPU, ground spans by alignment.")
    p_sum.add_argument("--local-model", default=config.LOCAL_MODEL,
                       help="model for --backend local")
    p_sum.add_argument("--max-sentences", type=int, default=None,
                       help="extractive backend: cap summary length in sentences")
    p_sum.add_argument("--format", default="text", choices=["text", "json", "html"])
    p_sum.add_argument("--out", help="output path for --format html")
    p_sum.add_argument("--show-reference", action="store_true", help="print the dataset's gold summary too")
    p_sum.set_defaults(func=cmd_summarize)

    p_cmp = sub.add_parser("compare", help="score several backends over the same documents")
    add_common(p_cmp)
    p_cmp.add_argument("-n", type=int, default=5, help="how many dataset docs to run")
    p_cmp.add_argument("--backends", default="extractive,local",
                       help="comma-separated: extractive, local, api")
    p_cmp.add_argument("--quiet", action="store_true", help="table only, no per-document lines")
    p_cmp.set_defaults(func=cmd_compare)

    p_ins = sub.add_parser("inspect", help="chunk and retrieve only, no API call")
    add_common(p_ins)
    p_ins.add_argument("--query", help="run a single retrieval query instead of the aspect sweep")
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
