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

## Five backends

All five produce the same `SummaryResult`, so `report.py` and `evaluate.py`
treat them identically and the five stay directly comparable.

They differ in one thing that matters more than any of the others: **how a
citation earns the right to be believed.**

| strategy | guarantee | backend |
|---|---|---|
| the API extracts the quote itself | the model **cannot** fabricate a citation | `api` |
| the summary *is* the source | there is nothing to fabricate | `extractive` |
| the model cites, we check every quote | it **can** fabricate — and we catch it | `verified` |
| the model is never asked; spans inferred | no assertion to check; similarity only | `local`, `groq` |

| | `api` | `extractive` | `verified` | `local` / `groq` |
|---|---|---|---|---|
| Output | prose | source sentences | claims + quotes | prose |
| Attribution | native citations | by construction | asserted, then verified | alignment |
| Can a claim be uncited? | no | no | **yes** | **yes** |
| `citation_integrity` | measured | 1.0 always | 1.0 always | 1.0, but trivially |
| `numeric_fidelity` | measured | 1.0 always | **measured** | **measured** |
| Needs | an Anthropic key | nothing | a CUDA GPU | a Groq key |
| Document leaves the machine | yes | **no** | **no** | yes |
| Speed | one API round trip | ~0.2s/doc | ~27s/doc after model load | ~1s/doc, rate limited |

**`extractive`** selects sentences instead of writing them, so the summary is
made *of* the document: `source_text[span.start:span.end]` is literally the
sentence chosen. Nothing can be fabricated and no number can drift, because
nothing is rewritten. Selection runs a per-aspect coverage pass (each aspect
query claims its best sentence, if it clears a relevance floor) followed by an
MMR fill that adds relevant-but-different sentences. That order is deliberate:
the MTS-Dialog correlation study puts omission at 31–54% against hallucination
below 4%, so coverage is the failure worth designing against.

The trade is prose quality — it reads as a highlight reel. Measured on the
**entire MultiClinSum gold set, all four languages, 2,368 documents**:

| lang | docs | s/doc | ROUGE-1 | ROUGE-L | claims | attrib / integrity / numeric |
|---|---|---|---|---|---|---|
| en | 592 | 0.21 | 0.328 | 0.207 | 6.4 | 1.0 / 1.0 / 1.0 |
| es | 592 | 0.25 | 0.382 | 0.229 | 6.4 | 1.0 / 1.0 / 1.0 |
| fr | 592 | 0.26 | 0.390 | 0.221 | 6.4 | 1.0 / 1.0 / 1.0 |
| pt | 592 | 0.27 | 0.363 | 0.221 | 6.4 | 1.0 / 1.0 / 1.0 |

The result that matters is the last column: **attribution coverage, citation
integrity and numeric fidelity were exactly 1.0 on all 2,368 documents, with zero
invalid citations.** These are by-construction properties, so anything less would
mean the offset arithmetic had broken somewhere in the corpus — this is the run
that shows it does not, in four languages.

English ROUGE is the *lowest* of the four, which is worth reading carefully
rather than as a cross-lingual win: MultiClinSum's fr and pt splits read as
machine translations of English source material, and translated prose is more
formulaic, so verbatim sentence selection lines up with the reference more
easily. Treat the es/fr/pt figures as a property of the corpus, not evidence
that the method transfers better to those languages.

The English spread is wide — sd 0.091, median 0.318, range 0.050–0.755 — so
quote the distribution, not the mean alone.

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
information are `attribution_coverage` and `numeric_fidelity`; both can fail on
this backend, and a failure is a finding about the model rather than a bug.

They measure **traceability, not factuality** — see "Are the metrics
trustworthy?" below, where this alignment mechanism is checked against 400 human
judgements. It moves the right way against hallucination but weakly, and it does
not reproduce the humans' ranking of systems.

Defaults to `unsloth/Qwen3-8B-bnb-4bit` — ~6 GB of VRAM, loads in about 8s.
Override with `DOCSUM_LOCAL_MODEL`, and see the `LOCAL_*` knobs in
`docsum/config.py` for the alignment threshold.

```bash
python cli.py summarize case.txt --backend local --aspects clinical
```

Measured head-to-head on the same 20 MultiClinSum English gold documents
(`cli.py compare`), which is the honest way to read the generative backends:

| | extractive | local (Qwen3-8B) | groq (gpt-oss-120b) |
|---|---|---|---|
| attribution_coverage | **1.0** | 0.993 | 0.987 |
| numeric_fidelity | **1.0** | 0.988 | 0.982 |
| citation_integrity | 1.0 | 1.0 | 1.0 |
| ROUGE-1 / ROUGE-L | **0.341** / **0.221** | 0.311 / 0.209 | 0.276 / 0.185 |
| claims per summary | 5.6 | 16.0 | 16.6 |
| seconds per document | **0.15** | 26.9 | 16.4 |

The 120B model scoring *below* the 8B is the result most likely to be
misread, so it is worth stating what it does and does not mean. Its lower ROUGE
is real: it writes at greater length and in its own register, which diverges
from the reference. Its lower `numeric_fidelity` is mostly **not** real, and
chasing that down produced three separate confounds worth knowing about — see
the caveats under "Are the metrics trustworthy?". Across all 20 documents
exactly **one** genuine fabrication survives scrutiny: an invented platelet
count of 230,000 on `gs_en_8`, produced independently by *both* generative
models, and flagged as uncited by the alignment layer in both cases.

The local backend writes markedly more detailed, readable prose — roughly three
times the claims — and it is the only backend here whose attribution can fail,
which is the point of measuring it. Across those 12 documents exactly one number
is flagged: a fabricated platelet count (`230,000/mm3`, absent from the source).
The alignment layer independently flagged that same sentence as uncited, so the
two signals agree on the one document that is actually wrong. ROUGE is a wash,
and extractive is ~100x faster.

**`groq`** is architecturally the same backend with a different generator:
generate freely, then recover spans with the identical alignment. **Groq has no
citations feature**, so it does *not* close the `api` gap — only Claude's
citations can, because only there does the API extract `cited_text` itself.

What it buys is size and speed. `openai/gpt-oss-120b` is roughly an order of
magnitude larger than the 8B that fits in 16 GB of VRAM, and returns in about a
second. What it costs is that the document leaves the machine — `extractive` and
`local` never send anything anywhere.

Rate limits are the practical constraint, and they are tighter than they look:
the free tier allows 1000 requests/minute but only **8000 tokens/minute**, and
one case report costs ~2500 tokens round trip. That paces a batch run at roughly
three documents per minute, so 429 is the normal path and the backend waits out
the window the server names rather than failing the document.

```bash
export GROQ_API_KEY=gsk_...
python cli.py summarize case.txt --backend groq --aspects clinical
```

### Summary length

The default task tells the model to cover every substantive point, and on a
short document retrieval is a pass-through — so the model receives the whole
source and is asked to keep all of it. The result is a reworded restatement
rather than a summary, which surprises people using the web UI.

`--length` (and `length=` in the library) controls this. Measured on
`multiclinsum_gs_en_1`, 4,954 chars, whose gold reference is 695 chars (14%):

| preset | extractive | local (8B) | attribution coverage |
|---|---|---|---|
| `brief` | 482 (10%) | 710 (14%) | 0.70 |
| `standard` | 1,107 (22%) | 1,533 (31%) | 0.96 |
| `full` *(default)* | 1,234 (25%) | 2,015 (41%) | 1.00 |

Compression costs attribution coverage, which is the honest trade and the reason
`full` stays the default: a compressed sentence fuses several source facts and
aligns less cleanly, so fewer claims earn a span. **Every measured table in this
README describes `full`.** The web form opens on `standard` instead, because a
reader summarising one document wants a summary.

`extractive` honours the same presets through a sentence cap rather than a
prompt. `verified` ignores them — it emits a claim list, and its task string is
the thing its measurement is about.

```bash
python cli.py summarize case.txt --backend local --aspects clinical --length brief
```

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

**`verified`** exists because the project *asserted* something it had never
measured: that self-reported citations cannot be trusted. Without an Anthropic
key the `api` guarantee is unavailable and no other vendor reproduces it — so
rather than approximate it, this backend tests the premise.

It asks the model for each claim together with a verbatim quote, then locates
every quote in the text the model was actually shown. A quote that cannot be
found is not a citation, it is a **caught fabrication**: the claim is emitted
uncited and the rejection is counted. The resulting `citation_precision`
(verified ÷ asserted, reported in `SummaryResult.usage`) is a direct measurement
of how often a model's self-reported citation lies.

Two details keep the verification honest. A quote is only accepted if it appears
in the **retrieved chunks** — a model cannot legitimately quote text it was never
sent, and allowing document-wide matching would let a lucky guess pass. And
matching folds typography and whitespace but nothing else, because models rewrite
hyphens and spaces constantly (gpt-oss-120b emits U+2011 and U+202F routinely);
every substitution is one character for one character, so offsets survive and the
span still slices exactly out of the original.

```bash
export GROQ_API_KEY=gsk_...
python cli.py summarize case.txt --backend verified --aspects clinical
```

**What it measured.** Across 6 MultiClinSum gold documents, `gpt-oss-120b`
asserted **96 citations and all 96 were real** — `citation_precision` 1.000, zero
fabrications caught.

That is a small sample and one model, and the honest reading is narrower than it
first looks. It does **not** vindicate prompt-engineered attribution in general.
Asking for a *verbatim quote* is a far easier and more self-checkable task than
asking a model to emit character offsets, which is the pattern this project
avoids — a quote either appears in the source or it does not, and the model knows
what it just copied. Nor does the result mean the verification is unnecessary:
without it there would be no way to know the rate was 100% rather than 80%, and
an unverified citation is worth nothing regardless of how often it happens to be
right.

Sample size is limited by the free tier, not by patience. Each verified request
reserves its full `max_tokens` against an 8000/minute budget, and Groq **queues**
rather than refusing when you exceed it — requests hang instead of returning 429,
which looks like a network fault and is not one. Sustained batches need ~25s
between documents.

It needs a larger token budget than the prose backends — each claim carries a
full quote, and reasoning tokens come from the same allowance. At 2048 the JSON
is truncated and Groq rejects the whole response with an empty
`failed_generation`, which reads like a prompt fault and is not one;
`VERIFIED_MAX_TOKENS` defaults to 4096.

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

Python 3.10+ (developed on 3.14).

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

> **On Windows, check `python` is real first.** A bare `python` is often a
> Microsoft Store stub that prints an install message and exits, which makes the
> line above fail confusingly. Verify with `python --version`; if it is a stub,
> use the full interpreter path, e.g.
> `C:\Users\<you>\AppData\Local\Programs\Python\Python314\python.exe -m venv .venv`.

Verify the install — this needs no API key and no GPU:

```bash
PYTHONPATH=. ./.venv/Scripts/python.exe tests/test_pipeline.py
```

Embeddings run locally on CPU, so retrieval needs no API key and no document
text leaves the machine for that half of the pipeline. The `extractive` backend
needs nothing further.

**For `--backend local`** you also need a CUDA GPU and a CUDA build of torch.
The plain PyPI wheel is CPU-only and will silently leave the GPU idle, so install
it from the index matching your CUDA version:

```bash
./.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130
```

Confirm with `torch.cuda.is_available()`. The model downloads on first use
(~6 GB). If that fails with a spurious `not enough space on the disk` error, set
`HF_HUB_DISABLE_XET=1` to fall back to plain HTTP transfer.

**For `--backend groq`** you need a Groq key (free tier works; it is rate
limited to 8000 tokens/minute, about three documents per minute):

```bash
export GROQ_API_KEY=gsk_...
```

**For `--backend api`** you need an Anthropic key:

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

## Are the metrics trustworthy?

Every other number here is an automatic metric grading itself, which is circular.
MTS-Dialog ships the one escape: 400 machine summaries with their source
dialogues, aligned to 400 human fact-based scores. `cli.py validate` scores those
400 with our metrics and correlates the result.

```bash
python cli.py validate --out validation.csv
```

Spearman rho against the human labels, n=400 (`*` = p < 0.01):

| | FactualF1 | Hallucination | Omission |
|---|---|---|---|
| `grounding_coverage` | 0.117 | **−0.168*** | −0.137* |
| `numeric_fidelity` | 0.283* | −0.085 | −0.206* |
| ROUGE-1 | 0.361* | −0.020 | **−0.467*** |
| ROUGE-L | 0.367* | −0.030 | −0.459* |

Read this before quoting any metric in this repository:

**The corpus cannot strongly validate hallucination detection.** `HallucinationRate`
is exactly zero for **92%** of the 400 summaries (mean 0.023, sd 0.090). With that
floor effect, *any* metric would correlate weakly, so a weak number here is a fact
about the data, not a verdict on the metric. Restricted to the 33 summaries that
do contain hallucination, `grounding_coverage` reaches rho = −0.367 (p = 0.036).

**`grounding_coverage` moves the right way, weakly.** This is the local backend's
alignment step used as a measurement, so it is the closest thing to a test of that
backend's attribution: the sign is correct against both hallucination and omission
and both are significant, but the magnitudes are small.

**`numeric_fidelity` does not track human-judged hallucination** (−0.085, not
significant; and it inverts on the hallucinating subset). This does not make it
useless — it caught a genuinely fabricated platelet count in the MultiClinSum
run — but it is a *narrow, precise* instrument for invented numerals, not a
general factuality score, and must not be presented as one.

**ROUGE is the strongest correlate of human factual judgement on this data**, and
the only signal here that recovers the humans' ranking of the four systems:

| block | FactualF1 (human) | grounding_coverage | numeric_fidelity | ROUGE-1 |
|---|---|---|---|---|
| 0 | 0.614 | 0.583 | 0.730 | 0.295 |
| 1 | 0.714 | 0.705 | 0.590 | 0.384 |
| 2 | 0.709 | 0.622 | 0.630 | 0.401 |
| 3 | **0.728** | 0.515 | 0.670 | **0.409** |

ROUGE orders the systems 3 > 2 > 1 > 0 against the humans' 3 > 1 > 2 > 0 (blocks 1
and 2 differ by 0.005, i.e. noise). Both attribution metrics get it wrong:
`grounding_coverage` ranks block 3 *last* where humans rank it first, and
`numeric_fidelity` ranks block 0 *first* where humans rank it last.

**What this means for the project's claims.** Attribution and factuality are not
the same property, and this run is the evidence. A summary can be perfectly
traceable and still omit most of the source, or be fluent and faithful while
citing nothing. The attribution metrics answer "can a reader check this claim
against the source?" — which is what this project is for — and should be reported
as traceability, not as a factuality score. Where factuality is the claim, ROUGE
against a reference remains the better-supported signal on this data.

### Three ways `numeric_fidelity` lies to you

Found while investigating why a 120B model appeared to hallucinate more than an
8B. All three are formatting or corpus artifacts, not fabrication:

1. **Typographic separators — fixed.** `gpt-oss-120b` writes thousands with
   U+202F (narrow no-break space), so a faithful `25 000` tokenised as `25` plus
   `000`: two inventions reported for a copied value. Now normalised, along with
   no-break and thin spaces. This alone accounted for most of the gap.
2. **List enumerators — not fixed.** A summary written as `(1) … (2) … (3) …`
   is flagged for indices that are structure, not claims. Models that format
   with numbered lists score worse than models writing plain prose regardless of
   faithfulness. Detecting these reliably means distinguishing `(2)` as an index
   from `(0.21)` as a value, which is not safely automatable.
3. **Source corruption — not fixable here.** `gs_en_20` lost its subscripts in
   text extraction and reads `pCO23.4 kPa, pO211.7 kPa, HCO319.5 mmol/L` and
   `SO294%`. A model that correctly recovers pCO₂ 3.4, pO₂ 11.7, HCO₃ 19.5 and
   SO₂ 94% is flagged for four inventions — **penalised for getting it right.**

Together with the spelled-out-numeral case, that is four known false-positive
modes. The metric is a screen that tells you where to look, never a verdict.
Always read what it flagged before believing it.

One data defect worth knowing: `OmissionRate` includes negative values (min
−1.0), so it is not a clean rate. Treat it as an ordinal signal.

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
| `docsum/remote.py` | Groq backend: same shape, hosted generation, rate-limit aware |
| `docsum/verified.py` | model-asserted citations, verified against the source |
| `docsum/grounding.py` | the alignment shared by the local and Groq backends |
| `docsum/compare.py` | runs several backends over the same docs and scores them |
| `docsum/validation.py` | correlates our metrics against 400 human judgements |
| `docsum/report.py` | text / JSON / interactive HTML output |
| `docsum/evaluate.py` | ROUGE + attribution + numeric fidelity |
| `docsum/datasets.py` | corpus loaders and document text extraction |
| `cli.py` | `summarize`, `compare`, `validate` and `inspect` commands |
