# docsum — RAG document summariser with source attribution

Upload a document, get a summary where **every fact points back to the exact
span it came from**.

## How it works

```
document ──► chunk (offsets kept) ──► embed + BM25 index
                                            │
                        aspect queries ──────┤
                                            ▼
                                   retrieved chunks
                                            │
                     one `document` block per chunk,
                        citations enabled
                                            ▼
                                    Claude (Opus 5)
                                            │
                                            ▼
                     summary + char-level citations
                                            │
                          chunk.start + citation offset
                                            ▼
                        absolute span in your document
```

The attribution is not prompt-engineered. Each retrieved chunk is sent as its
own `document` content block with `citations: {enabled: true}`, so the API
returns `document_index` + `start_char_index`/`end_char_index` for every claim,
and extracts `cited_text` itself. The model cannot cite a span that does not
exist. Mapping back to the uploaded file is then just:

```python
absolute_start = chunk.start + citation.start_char_index
```

Two design choices worth knowing:

**Chunks are the unit of both retrieval and citation.** `CHUNK_TARGET_CHARS` in
`docsum/config.py` is the main quality knob — smaller chunks give tighter
citations, larger ones give the model more context per chunk.

**Retrieval sweeps aspects rather than taking top-k once.** A summary must cover
the whole document, so `search_many()` runs several queries (see
`ASPECT_PRESETS`) and merges the hits, then returns them in *document order* so
the summary reads as prose rather than a pile of ranked fragments.

## Three backends

All three produce the same `SummaryResult`, so `report.py` and `evaluate.py`
treat them identically and the three stay directly comparable.

| | `api` (default) | `extractive` | `local` |
|---|---|---|---|
| Output | generated prose | source sentences, verbatim | generated prose |
| Attribution | Claude's native citations | exact by construction | alignment, after the fact |
| Can a claim be uncited? | no | no | **yes — and that is the point** |
| `citation_integrity` | measured | 1.0 always | 1.0, but trivially |
| `numeric_fidelity` | measured | 1.0 always | **measured** |
| Needs | an API key | nothing | a CUDA GPU |
| Speed | one API round trip | ~0.5s/doc | seconds/doc after model load |

**`extractive`** selects sentences instead of writing them, so the summary is
made *of* the document: `source_text[span.start:span.end]` is literally the
sentence chosen. Nothing can be fabricated and no number can drift, because
nothing is rewritten. Selection runs a per-aspect coverage pass (each aspect
query claims its best sentence, if it clears a relevance floor) followed by an
MMR fill that adds relevant-but-different sentences. That order is deliberate:
the MTS-Dialog correlation study puts omission at 31–54% against hallucination
below 4%, so coverage is the failure worth designing against.

The trade is prose quality — it reads as a highlight reel. Measured on 30
MultiClinSum English gold documents: ROUGE-1 0.342, ROUGE-L 0.219, with
attribution coverage, citation integrity and numeric fidelity all 1.0 on every
document.

```bash
python cli.py summarize case.txt --backend extractive --aspects clinical
```

**`local`** generates prose on your own GPU. No local model has the API's
citations feature, so this backend does not ask the model to attribute anything
— that would be exactly the prompt-engineered attribution this project avoids.
It generates freely, then aligns each generated sentence back to the source
using embedding similarity blended with lexical containment. A sentence that
clears the threshold gets a real span and a `support` score; one that does not
is **left uncited** and shows up in `result.unattributed_claims`.

Read the metrics accordingly. `citation_integrity` is 1.0 here trivially — the
spans are real slices, so of course they resolve. The signals that carry
information are `attribution_coverage` (how much of the summary could be tied to
the source at all) and `numeric_fidelity` (whether generated numbers actually
occur in the document). Both can fail on this backend, and a failure is a
finding about the model rather than a bug.

Defaults to `unsloth/Qwen3-8B-bnb-4bit` — ~6 GB of VRAM, loads in about 8s.
Override with `DOCSUM_LOCAL_MODEL`, and see the `LOCAL_*` knobs in
`docsum/config.py` for the alignment threshold.

```bash
python cli.py summarize case.txt --backend local --aspects clinical
```

Measured against extractive on the same 12 MultiClinSum English gold documents
(`cli.py compare`), which is the honest way to read this backend:

| | extractive | local (Qwen3-8B) |
|---|---|---|
| attribution_coverage | 1.0 | 0.988 |
| numeric_fidelity | 1.0 | 0.992 |
| ROUGE-1 / ROUGE-L | 0.348 / 0.212 | 0.331 / 0.209 |
| claims per summary | 5.0 | 14.9 |
| seconds per document | 0.25 | 34.8 |

The local backend writes markedly more detailed, readable prose — roughly three
times the claims — and it is the only backend here whose attribution can fail,
which is the point of measuring it. Across those 12 documents exactly one number
is flagged: a fabricated platelet count (`230,000/mm3`, absent from the source).
The alignment layer independently flagged that same sentence as uncited, so the
two signals agree on the one document that is actually wrong. ROUGE is a wash,
and extractive is ~100x faster.

**On `numeric_fidelity`:** it compares numeric tokens, breaking composites on
`/` at both ends, so a date reformatted from `31/05/2023` into "May 31, 2023" —
or a diastolic `80` quoted from a `110/80` blood pressure — counts as supported.
Decimals and thousands separators are deliberately *not* decomposed: `3.21` is
not evidence for a claim of `21`, and `10,200` is not evidence for `200`.
Splitting those would let real inventions through.

It remains a screen rather than a verdict. It matches numerals only, so a source
that spells a number out ("Six months") will not support a summary that writes
`6`. Check what it flags before treating it as proof. It cannot fire for
`extractive`, which copies digits verbatim.

### Why not a 14B

A 14B 4-bit checkpoint fits in 16 GB of VRAM on paper, but loading one segfaults
inside bitsandbytes on the machine this was built on. Every step of the load
works in isolation — slicing all 947 tensors out of a 5 GB shard, moving each to
CUDA — and disabling the loader's thread pool changes nothing, but loading with
`disable_mmap=True` raises `MemoryError` instead, which points at host memory
during quantised-parameter construction rather than at VRAM. That box has 31.6 GB
of RAM against a **2 GB pagefile**, so its commit limit is only ~33 GB. An 8B has
half the peak and loads in 8 seconds. A larger pagefile may make 14B viable.

## Setup

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Embeddings run locally on CPU, so retrieval needs no API key and no document
text leaves the machine for that half of the pipeline. The `extractive`
backend needs nothing further; `local` additionally needs a CUDA GPU and a
CUDA build of torch. Only the `api` backend needs a key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

## Use

Summarise a file:

```bash
python cli.py summarize report.pdf
```

Summarise with the clinical aspect preset and write an interactive HTML report
(hover a claim, its source highlights):

```bash
python cli.py summarize case.txt --aspects clinical --format html --out case.html
```

Run against the bundled corpora:

```bash
python cli.py summarize --dataset multiclinsum -n 3 --aspects clinical --show-reference
```

Score backends against each other on the same documents, same chunking, same
retrieved chunks — so any difference comes from generation, not retrieval drift:

```bash
python cli.py compare --dataset multiclinsum -n 12 --aspects clinical --backends extractive,local
```

A backend that cannot run at all (no key, no GPU) is reported as such and does
not abort the comparison for the others.

Inspect chunking and retrieval **without spending a token** — useful for tuning
chunk size and checking what retrieval actually selects:

```bash
python cli.py inspect --dataset multiclinsum --query "treatment and outcome"
```

## Library

```python
from docsum import read_document, summarize, summarize_extractive, summarize_local

text = read_document("case.pdf")
result = summarize(text, aspects="clinical")            # needs a key
result = summarize_extractive(text, aspects="clinical")  # needs nothing
result = summarize_local(text, aspects="clinical")       # needs a GPU

print(result.summary)
print(f"{result.coverage:.0%} of claims cited")

for fact in result.grounded_facts:
    for span in fact.sources:
        print(f"{fact.text[:50]!r} <- chars {span.start}-{span.end}: {span.cited_text!r}")
```

## Evaluating

`docsum/evaluate.py` scores a result two ways:

| Metric | Needs gold summary? | What it catches |
|---|---|---|
| `rouge1/2/L_f1` | yes | drift from a reference summary |
| `attribution_coverage` | no | claims written without a citation |
| `citation_integrity` | no | broken offset arithmetic — re-slices the source and compares against what the API reported |
| `numeric_fidelity` | no | invented ages, dosages, lab values |

`citation_integrity` is the one to watch. If it drops below 1.0, the offset
mapping is wrong and the attribution feature is silently lying.

`numeric_fidelity` exists because ROUGE cannot tell a paraphrase from a
hallucinated dosage, and on clinical text that distinction is the whole point.

## Datasets

Both corpora in the parent directory load through `docsum/datasets.py`:

- **MultiClinSum** (`15517617/`) — full case report → abstract. 592 gold pairs
  per language (en/es/fr/pt), plus 25,902 large-scale pairs each. Read straight
  out of the zips; nothing is extracted to disk.
- **MTS-Dialog** (`MTS-Dialog-main/`) — doctor-patient dialogue → clinical note
  section. 1,201 train / 100 validation / 200+200 test.

Note: MultiClinSum fr/pt read as machine translations of English source
material, and at least one gold pair (`multiclinsum_gs_en_48`) has a sex
mismatch between fulltext and summary — worth knowing before treating gold as
ground truth for factuality specifically.

## Layout

| File | Role |
|---|---|
| `docsum/chunking.py` | sentence splitting + chunking, offsets preserved exactly |
| `docsum/retrieval.py` | hybrid dense + BM25 search over chunks |
| `docsum/summarizer.py` | the RAG call, citation parsing, offset mapping |
| `docsum/extractive.py` | key-free backend: selects source sentences verbatim |
| `docsum/local.py` | local-GPU backend: generates prose, aligns spans after |
| `docsum/compare.py` | runs several backends over the same docs and scores them |
| `docsum/report.py` | text / JSON / interactive HTML output |
| `docsum/evaluate.py` | ROUGE + attribution + numeric fidelity |
| `docsum/datasets.py` | corpus loaders and document text extraction |
| `cli.py` | `summarize`, `compare` and `inspect` commands |
