"""Invariants the attribution feature depends on.

Run with:
    PYTHONPATH=. ./.venv/Scripts/python.exe tests/test_pipeline.py

No API key required -- the generation step is exercised through a mock client
that returns the documented citations response shape, so these tests check our
offset arithmetic rather than the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsum.chunking import chunk_text, split_sentences
from docsum.datasets import load_mts_dialog, load_multiclinsum
from docsum.evaluate import citation_integrity, numeric_fidelity, rouge_l, rouge_n
from docsum.extractive import summarize_extractive
from docsum.summarizer import summarize

PASS: list[str] = []
FAIL: list[str] = []
SKIP: list[str] = []


def _corpora_present() -> bool:
    """Whether the datasets are sitting beside the project.

    They are large, separately licensed, and deliberately not in the repository,
    so a fresh clone has the code but not the corpora. Those tests skip rather
    than fail, which keeps `python tests/test_pipeline.py` meaningful in CI and
    for anyone who just wants to check the build.
    """
    from docsum import config

    return config.MULTICLINSUM_DIR.exists() and config.MTS_DIR.exists()


CORPORA = _corpora_present()


def need_corpora(test_name: str) -> bool:
    """Record a skip and tell the caller to bail out."""
    if CORPORA:
        return False
    SKIP.append(test_name)
    print(f"  [SKIP] {test_name} -- corpora not present beside the project")
    return True


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))


# --- 1. Sentence splitting ---------------------------------------------------

def test_sentence_splitting() -> None:
    print("\nsentence splitting")
    cases = [
        # (text, expected sentence count)
        ("Pt. is a 55 y.o. male seen by Dr. Smith on 07/29/2008. He improved.", 2),
        ("Labs: CRP 3.21 mg/dL. AFP was <1.3 ng/mL. Normal.", 3),
        ("He took 5 mg. daily without issue. Then it stopped.", 2),
        ("Doctor: Any fever? \nPatient: No. \nDoctor: Good.", 3),
        ("In Jan. 2009 she relapsed. Treatment resumed.", 2),
        ("She was born in the U.S.A. and moved later. Dr. Lee agreed.", 2),
        ("1. First item. 2. Second one. Then prose.", 3),
    ]
    for text, expected in cases:
        sents = split_sentences(text)
        check(f"{text[:44]!r} -> {expected} sentences",
              len(sents) == expected, f"got {len(sents)}: {[s.text for s in sents]}")
        # Offsets must be exact for every sentence.
        exact = all(text[s.start : s.end] == s.text for s in sents)
        check(f"  offsets exact for {text[:30]!r}", exact)


# --- 2. Chunk offsets, the load-bearing invariant ----------------------------

def test_chunk_offsets() -> None:
    if need_corpora("test_chunk_offsets"):
        return
    print("\nchunk offset invariant (source[c.start:c.end] == c.text)")
    total = bad = 0
    for lang in ("en", "es", "fr", "pt"):
        for pair in load_multiclinsum("gs", lang, limit=50):
            for c in chunk_text(pair.text, doc_id=pair.doc_id):
                total += 1
                if pair.text[c.start : c.end] != c.text:
                    bad += 1
    for split in ("train", "validation", "test1", "test2"):
        for pair in load_mts_dialog(split, limit=100):
            for c in chunk_text(pair.text, doc_id=pair.doc_id):
                total += 1
                if pair.text[c.start : c.end] != c.text:
                    bad += 1
    check(f"{total} chunks across 4 languages + 4 MTS splits", bad == 0, f"{bad} mismatched")


def test_chunk_coverage() -> None:
    if need_corpora("test_chunk_coverage"):
        return
    print("\nchunk coverage")
    pair = load_multiclinsum("gs", "en", limit=1)[0]
    chunks = chunk_text(pair.text, doc_id=pair.doc_id)
    check("first chunk starts at or near 0", chunks[0].start < 50)
    check("last chunk reaches the end", chunks[-1].end == len(pair.text.rstrip()),
          f"{chunks[-1].end} vs {len(pair.text)}")
    gaps = [
        (chunks[i].end, chunks[i + 1].start)
        for i in range(len(chunks) - 1)
        if chunks[i + 1].start > chunks[i].end
    ]
    check("no uncovered gaps between consecutive chunks", not gaps, str(gaps[:3]))


# --- 3. Citation offset mapping ----------------------------------------------

def _mock_client(cite_local: tuple[int, int] = (5, 45)):
    """A client that cites a known local span out of every document block."""
    captured: dict = {}

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                docs = [
                    b for b in kw["messages"][0]["content"] if b.get("type") == "document"
                ]
                captured["docs"] = docs
                captured["request"] = kw
                blocks = []
                for di, d in enumerate(docs):
                    data = d["source"]["data"]
                    lo = min(cite_local[0], max(0, len(data) - 1))
                    hi = min(cite_local[1], len(data))
                    blocks.append(
                        NS(
                            type="text",
                            text=f"claim about section {di} of the document ",
                            citations=[
                                NS(
                                    type="char_location",
                                    cited_text=data[lo:hi],
                                    document_index=di,
                                    document_title=d["title"],
                                    start_char_index=lo,
                                    end_char_index=hi,
                                )
                            ],
                        )
                    )
                return NS(
                    content=blocks,
                    stop_reason="end_turn",
                    stop_details=None,
                    usage=NS(input_tokens=1, output_tokens=1),
                )

    return Client(), captured


def test_request_shape() -> None:
    if need_corpora("test_request_shape"):
        return
    print("\nrequest shape")
    pair = load_multiclinsum("gs", "en", limit=1)[0]
    client, captured = _mock_client()
    summarize(pair.text, doc_id=pair.doc_id, aspects="clinical", client=client)

    docs = captured["docs"]
    req = captured["request"]
    check("every chunk is its own document block", len(docs) >= 1)
    check("citations enabled on every document block",
          all(d["citations"] == {"enabled": True} for d in docs))
    check("plain-text source type (yields char_location citations)",
          all(d["source"]["type"] == "text" for d in docs))
    check("adaptive thinking requested", req["thinking"] == {"type": "adaptive"})
    check("no output_config.format (incompatible with citations)",
          "format" not in req.get("output_config", {}))


def test_citation_mapping_full() -> None:
    if need_corpora("test_citation_mapping_full"):
        return
    print("\ncitation -> absolute offset mapping (all chunks sent)")
    pair = load_multiclinsum("gs", "en", limit=1)[0]
    client, _ = _mock_client()
    result = summarize(pair.text, doc_id=pair.doc_id, aspects="clinical", client=client)

    spans = result.all_spans()
    check("citations were parsed", len(spans) > 0)
    exact = all(pair.text[s.start : s.end] == s.cited_text for s in spans)
    check(f"all {len(spans)} spans slice back to their cited_text", exact)
    share, invalid = citation_integrity(result)
    check("citation_integrity == 1.0", share == 1.0, f"{invalid} invalid")


def test_citation_mapping_subset() -> None:
    if need_corpora("test_citation_mapping_subset"):
        return
    print("\ncitation mapping when retrieval sends a non-contiguous subset")
    pairs = load_multiclinsum("gs", "en", limit=120)
    pair = max(pairs, key=lambda p: len(p.text))
    all_chunks = chunk_text(pair.text, doc_id=pair.doc_id)

    client, captured = _mock_client()
    result = summarize(
        pair.text, doc_id=pair.doc_id, aspects="clinical",
        chunk_budget=4, client=client,
    )
    selected = [c.index for c in result.chunks_used]
    check("budget honoured", len(selected) == 4, str(selected))
    check("subset is non-contiguous (test is meaningful)",
          selected != list(range(len(selected))) or len(all_chunks) <= 4, str(selected))

    exact = all(pair.text[s.start : s.end] == s.cited_text for s in result.all_spans())
    check(f"document_index resolves to the Nth SELECTED chunk (subset {selected})", exact)


def test_refusal_handling() -> None:
    if need_corpora("test_refusal_handling"):
        return
    print("\nrefusal handling")

    class RefusingClient:
        class messages:
            @staticmethod
            def create(**kw):
                return NS(
                    content=[],
                    stop_reason="refusal",
                    stop_details=NS(type="refusal", category="test",
                                    explanation="declined for testing"),
                    usage=NS(input_tokens=1, output_tokens=0),
                )

    pair = load_multiclinsum("gs", "en", limit=1)[0]
    result = summarize(pair.text, doc_id=pair.doc_id, client=RefusingClient())
    check("refusal surfaced instead of crashing", result.refusal is not None)
    check("refusal explanation captured", "declined" in (result.refusal or ""))
    check("no phantom summary on refusal", result.summary == "")


# --- 4. Metrics --------------------------------------------------------------

def test_metrics() -> None:
    print("\nmetrics")
    check("rouge1 identical text -> 1.0", rouge_n("a b c", "a b c", 1)["f1"] == 1.0)
    check("rouge1 disjoint text -> 0.0", rouge_n("a b c", "x y z", 1)["f1"] == 0.0)
    check("rougeL respects order",
          rouge_l("a b c", "c b a")["f1"] < rouge_l("a b c", "a b c")["f1"])

    score, missing = numeric_fidelity("Patient is 29 and got 500 mg.",
                                      "A 29-year-old given 500 mg daily.")
    check("numeric_fidelity accepts faithful numbers", score == 1.0, str(missing))
    score, missing = numeric_fidelity("Patient is 42 and got 750 mg.",
                                      "A 29-year-old given 500 mg daily.")
    check("numeric_fidelity flags invented numbers",
          score == 0.0 and set(missing) == {"42", "750"}, str(missing))
    score, _ = numeric_fidelity("WBC 10200", "WBC was 10,200 today")
    check("numeric_fidelity normalises thousands separators", score == 1.0)

    # -- composite numeric tokens (dates, ratios) -----------------------------
    # A date reformatted between source and summary is not an invention. Before
    # this was handled, "May 31, 2023" against a source of "31/05/2023" reported
    # three fabricated numbers.
    s, missing = numeric_fidelity("Seen on May 31, 2023.", "Visit dated 31/05/2023.")
    check("numeric: reformatted date is not an invention", s == 1.0, f"missing {missing}")

    s, _ = numeric_fidelity("Recorded 2023.", "dated 31/05/2023")
    check("numeric: year from a composite date is supported", s == 1.0)

    s, _ = numeric_fidelity("The day was 5.", "dated 31/05/2023")
    check("numeric: zero-padded component matches unpadded", s == 1.0)

    s, _ = numeric_fidelity("Diastolic was 80.", "BP was 110/80 mmHg")
    check("numeric: component of a ratio is supported", s == 1.0)

    s, _ = numeric_fidelity("Seen on 31/05/2023.", "Visit dated May 31, 2023.")
    check("numeric: decomposition works in both directions", s == 1.0)

    # -- what must STILL be caught -------------------------------------------
    # Decimals and thousands separators must not be split, or real inventions
    # slip through: "3.21" is not evidence for a claim of "21".
    s, missing = numeric_fidelity("AFP was 21.", "CRP 3.21 mg/dL")
    check("numeric: decimals are NOT decomposed", s == 0.0, f"got {s}, missing {missing}")

    s, missing = numeric_fidelity("Platelets 200.", "WBC was 10,200")
    check("numeric: thousands separators are NOT decomposed", s == 0.0,
          f"got {s}, missing {missing}")

    s, missing = numeric_fidelity("Platelets of 230,000/mm3.", "Platelets were normal.")
    check("numeric: a genuinely invented value is still flagged", s == 0.0,
          f"got {s}, missing {missing}")

    # -- typographic separators ----------------------------------------------
    # gpt-oss-120b writes thousands with U+202F (narrow no-break space), so
    # "25 000" tokenised as "25" and "000" -- two fabrications reported for a
    # value the model had copied faithfully. It cost the Groq backend most of
    # its apparent numeric_fidelity gap against the local model.
    src = "Platelet count was 25,000 per uL and the dose was 50000 units."
    s, missing = numeric_fidelity("Platelets were 25 000 per uL with 50 000 units.", src)
    check("numeric: narrow no-break space thousands separator is handled", s == 1.0,
          f"got {s}, missing {missing}")
    s, _ = numeric_fidelity("Platelets were 25 000 per uL.", src)
    check("numeric: non-breaking space separator is handled", s == 1.0, f"got {s}")
    s, _ = numeric_fidelity("Platelets were 25 000 per uL.", src)
    check("numeric: thin space separator is handled", s == 1.0, f"got {s}")
    s, missing = numeric_fidelity("Platelets were 99 999 per uL.", src)
    check("numeric: normalisation does not mask a real invention", s == 0.0,
          f"got {s}, missing {missing}")
    # A space that is not between digits must not be swallowed.
    s, _ = numeric_fidelity("Dose 50000 units.", "Dose 50000 units given.")
    check("numeric: ordinary text unaffected by normalisation", s == 1.0)

    # The real hallucination found on multiclinsum_gs_en_8 must stay caught.
    if not CORPORA:
        SKIP.append("test_metrics::gs_en_8 regression")
        print("  [SKIP] gs_en_8 regression -- corpora not present")
        return
    doc8 = load_multiclinsum("gs", "en", limit=8)[7]
    s, missing = numeric_fidelity(
        "Laboratory results included platelets of 230,000/mm3.", doc8.text)
    check("numeric: the gs_en_8 fabricated platelet count is still flagged",
          "230,000" in missing, f"missing {missing}")


# --- 5. Dataset loaders ------------------------------------------------------

def test_loaders() -> None:
    if need_corpora("test_loaders"):
        return
    print("\ndataset loaders")
    gs = load_multiclinsum("gs", "en")
    # 592, not 593: the archive's `fulltext/` directory entry is not a document.
    check("MultiClinSum gs/en has 592 pairs", len(gs) == 592, f"got {len(gs)}")
    check("every gs pair has text and reference",
          all(p.text and p.reference for p in gs))
    check("summaries are shorter than fulltexts on average",
          sum(len(p.reference) for p in gs) < sum(len(p.text) for p in gs))

    for lang in ("es", "fr", "pt"):
        pairs = load_multiclinsum("gs", lang, limit=10)
        check(f"gs/{lang} loads (nested .zip in fr ignored)", len(pairs) == 10)

    ls = load_multiclinsum("ls", "en", limit=5)
    check("large-scale split loads from zip", len(ls) == 5)

    mts = load_mts_dialog("train")
    check("MTS-Dialog train has 1201 pairs", len(mts) == 1201, f"got {len(mts)}")
    check("MTS section headers preserved",
          all(p.meta.get("section_header") for p in mts[:50]))


# --- 10. Extractive backend: attribution exact by construction ---------------

def test_extractive_backend() -> None:
    """The extractive backend's guarantees are structural, so assert them as such.

    Selecting sentences verbatim should make integrity, numeric fidelity and
    coverage exactly 1.0 -- not approximately. Anything less means a span is
    being derived rather than copied, which is the bug this backend exists to
    make impossible.
    """
    if need_corpora("test_extractive_backend"):
        return
    print()
    print("extractive backend")

    docs = load_multiclinsum("gs", "en", limit=3)
    check("gold docs available for extractive test", len(docs) == 3)

    for doc in docs:
        result = summarize_extractive(doc.text, doc_id=doc.doc_id, aspects="clinical")
        tag = doc.doc_id

        check(f"{tag}: produced a summary", bool(result.summary.strip()))
        check(f"{tag}: every block is grounded",
              len(result.grounded_facts) == len(result.facts))
        check(f"{tag}: no unattributed claims", not result.unattributed_claims)

        spans = result.all_spans()
        exact = all(doc.text[s.start : s.end] == s.cited_text for s in spans)
        check(f"{tag}: every span re-slices to its cited text", exact)

        integrity, invalid = citation_integrity(result)
        check(f"{tag}: citation_integrity == 1.0", integrity == 1.0, f"got {integrity}")
        check(f"{tag}: zero invalid citations", invalid == 0, f"got {invalid}")

        fidelity, missing = numeric_fidelity(result.summary, doc.text)
        check(f"{tag}: numeric_fidelity == 1.0 (nothing can be invented)",
              fidelity == 1.0, f"missing {missing}")

        check(f"{tag}: attribution coverage == 1.0", result.coverage == 1.0,
              f"got {result.coverage}")

        # Document order, and no sentence selected twice.
        starts = [s.start for s in spans]
        check(f"{tag}: spans emitted in document order", starts == sorted(starts))
        check(f"{tag}: no duplicate spans", len(set(starts)) == len(starts))

        # Selected sentences must come from chunks retrieval actually chose.
        windows = [(c.start, c.end) for c in result.chunks_used]
        inside = all(any(lo <= s.start and s.end <= hi for lo, hi in windows)
                     for s in spans)
        check(f"{tag}: every span lies inside a retrieved chunk", inside)

    # Explicit budget is honoured.
    doc = docs[0]
    capped = summarize_extractive(doc.text, doc_id=doc.doc_id,
                                  aspects="clinical", max_sentences=4)
    check("max_sentences caps the summary", len(capped.facts) == 4,
          f"got {len(capped.facts)}")

    # Degenerate input must not raise.
    empty = summarize_extractive("", doc_id="empty")
    check("empty document returns an empty result",
          empty.summary == "" and not empty.facts)


# --- 11. Local backend: alignment replaces citations -------------------------

def test_local_alignment() -> None:
    """The local backend has no native citations, so grounding is inferred.

    These check the inference itself -- no GPU and no language model involved.
    The properties that matter: a real claim finds its span, a fabricated one
    finds nothing, and a fluent paraphrase that changed a number is not waved
    through just because it still reads like the source.
    """
    if need_corpora("test_local_alignment"):
        return
    print()
    print("local backend alignment")

    from docsum import config
    from docsum.chunking import Sentence, chunk_text, sentences_within
    from docsum.grounding import align, containment, strip_artifacts
    from docsum.retrieval import _load_embedder

    # Containment is asymmetric and specifics-sensitive.
    check("containment: identical text -> 1.0",
          containment("the patient was given 100 mg", "the patient was given 100 mg") == 1.0)
    check("containment: short claim inside a long source -> 1.0",
          containment("given 100 mg", "the patient was given 100 mg orally daily") == 1.0)
    swapped = containment("the patient was given 500 mg", "the patient was given 100 mg")
    check("containment: a swapped dosage drops below 1.0", swapped < 1.0, f"got {swapped}")
    check("containment: empty claim -> 0.0", containment("", "anything") == 0.0)

    # Generation artifacts must not survive into the summary.
    check("strip: <think> block removed",
          strip_artifacts("<think>reasoning here</think>The patient improved.")
          == "The patient improved.")
    check("strip: lead-in removed",
          strip_artifacts("Here is the summary: The patient improved.")
          == "The patient improved.")
    check("strip: ordinary prose untouched",
          strip_artifacts("The patient improved.") == "The patient improved.")

    # Alignment against a real document.
    doc = load_multiclinsum("gs", "en", limit=1)[0]
    chunks = chunk_text(doc.text, doc_id=doc.doc_id)
    candidates = sentences_within(doc.text, chunks,
                                  min_chars=config.EXTRACTIVE_MIN_SENTENCE_CHARS)
    check("candidate pool is non-empty", len(candidates) > 3)

    verbatim = candidates[2][0].text
    fabricated = ("The spacecraft completed its orbital insertion burn and "
                  "transmitted telemetry back to the ground station.")
    generated = [
        Sentence(verbatim, 0, len(verbatim)),
        Sentence(fabricated, 0, len(fabricated)),
    ]

    aligned = align(generated, candidates, _load_embedder(config.EMBED_MODEL))
    check("alignment returns one entry per generated sentence", len(aligned) == 2)

    # A sentence lifted straight from the source must ground to itself.
    check("verbatim claim is grounded", len(aligned[0]) >= 1)
    if aligned[0]:
        span = aligned[0][0]
        check("verbatim claim maps to the correct span",
              doc.text[span.start:span.end] == verbatim)
        check("verbatim claim scores near 1.0", span.support >= 0.9,
              f"got {span.support}")
        check("support score is recorded", span.support is not None)

    # An off-topic claim must attach to nothing rather than its least-bad match.
    check("fabricated claim is left uncited", aligned[1] == [],
          f"got {[(s.start, s.support) for s in aligned[1]]}")

    # Every emitted span must still re-slice exactly, as for every backend.
    all_spans = [s for group in aligned for s in group]
    check("every aligned span re-slices to its cited text",
          all(doc.text[s.start:s.end] == s.cited_text for s in all_spans))
    check("spans within a claim are in document order",
          all(g == sorted(g, key=lambda s: s.start) for g in aligned))
    check("no claim exceeds the support-span cap",
          all(len(g) <= config.LOCAL_MAX_SUPPORT_SPANS for g in aligned))

    # An unsupported claim must show up in the report surface, not vanish.
    from docsum.summarizer import Fact, SummaryResult
    result = SummaryResult(
        summary=" ".join(s.text for s in generated),
        facts=[Fact(text=s.text, sources=sp) for s, sp in zip(generated, aligned)],
        chunks_used=chunks,
        source_text=doc.text,
    )
    check("the fabricated claim is reported as unattributed",
          len(result.unattributed_claims) == 1)
    check("coverage is below 1.0 when a claim is unsupported",
          result.coverage < 1.0, f"got {result.coverage}")
    integrity, invalid = citation_integrity(result)
    check("citation_integrity still 1.0 (spans are real slices)", integrity == 1.0)
    check("no invalid citations", invalid == 0)


# --- 12. Metric validation against human judgement ---------------------------

def test_metric_validation() -> None:
    """The correlation study is the only non-circular check we have.

    Every other number in this project is a metric grading itself. These tests
    guard the plumbing that connects our metrics to the 400 human scores --
    especially the positional alignment, which has no join key and would fail
    silently if either file changed length.
    """
    if need_corpora("test_metric_validation"):
        return
    print()
    print("metric validation")

    from docsum import validation as V

    check("correlation study files are present",
          V.SUMMARIES_CSV.exists() and V.SCORES_CSV.exists())

    pairs = V._read_pairs()
    check("400 summaries pair with 400 human scores", len(pairs) == 400,
          f"got {len(pairs)}")
    check("summary rows carry dialogue, reference and automatic summary",
          all(k in pairs[0][0] for k in ("Dialogue", "Reference Summary", "Automatic Summary")))
    check("score rows carry the human labels",
          all(k in pairs[0][1] for k in ("FactualF1", "HallucinationRate", "OmissionRate")))

    # The BOM on both files would corrupt the first column name if it were read
    # as plain utf-8, making every lookup miss.
    check("BOM handled (first column name is clean)",
          "ID" in pairs[0][0] and "FactualPrecision" in pairs[0][1])

    # grounding_coverage must behave sensibly at both extremes, or the
    # correlations computed from it mean nothing.
    source = ("Doctor: When did the pain begin? Patient: About eight years ago. "
              "Doctor: Any surgery? Patient: I had a discectomy in 2011.")
    copied = "The patient has had pain for about eight years. She had a discectomy in 2011."
    unrelated = ("The spacecraft completed its orbital insertion burn. "
                 "Telemetry was transmitted to the ground station.")

    cov_copied, sup_copied = V.grounding_coverage(copied, source)
    cov_unrel, _ = V.grounding_coverage(unrelated, source)
    check("grounding_coverage is high for a faithful summary", cov_copied >= 0.5,
          f"got {cov_copied}")
    check("grounding_coverage is low for an unrelated summary", cov_unrel == 0.0,
          f"got {cov_unrel}")
    check("faithful summary carries real support", sup_copied > 0.4, f"got {sup_copied}")
    check("grounding_coverage separates the two cases", cov_copied > cov_unrel)

    # Empty inputs must not raise -- validation runs unattended over 400 rows.
    empty_cov, empty_sup = V.grounding_coverage("", source)
    check("empty summary yields zero coverage, no exception",
          empty_cov == 0.0 and empty_sup == 0.0)

    rows = V.run(limit=6)
    check("run(limit) returns that many rows", len(rows) == 6, f"got {len(rows)}")
    check("every metric field is populated",
          all(isinstance(getattr(rows[0], k), float) for k in V.AUTOMATIC + V.HUMAN))

    disc = V.label_discrimination(rows)
    check("label_discrimination reports every human label",
          set(disc) == set(V.HUMAN))
    check("discrimination entries carry a zero_fraction",
          "zero_fraction" in disc["HallucinationRate"])

    blocks = V.by_system(rows[:3])
    check("by_system groups into blocks", blocks[0]["n"] == 3)

    report = V.format_report(rows)
    check("format_report mentions the correlation and the labels",
          "Spearman" in report and "HallucinationRate" in report)


# --- 13. Robustness: chunk ceiling and adaptive retrieval --------------------

def test_chunk_ceiling() -> None:
    """A chunk is the unit of citation, so an unbounded one is useless.

    Sentence boundaries usually bound chunk size, but nothing guarantees a
    document has any: gs_en_410 is 21k characters across 11 sentences, the
    longest 20,238 chars. Before the ceiling that produced a single 20,625-char
    chunk -- every offset in it correct, and worthless as attribution.
    """
    if need_corpora("test_chunk_ceiling"):
        return
    print()
    print("chunk ceiling")

    from docsum import config
    from docsum.chunking import chunk_text

    # A document with no sentence boundaries at all.
    runaway = ", ".join(f"value {i} was 3.2{i % 10} units" for i in range(600))
    chunks = chunk_text(runaway, doc_id="runaway")
    biggest = max(len(c.text) for c in chunks)
    check("a document with no sentence breaks still chunks", len(chunks) > 1)
    check("no chunk exceeds the ceiling", biggest <= config.CHUNK_MAX_CHARS,
          f"biggest {biggest} > {config.CHUNK_MAX_CHARS}")
    check("offsets stay exact after splitting an oversized sentence",
          all(runaway[c.start:c.end] == c.text for c in chunks))
    check("splitting does not cut mid-word",
          all(not c.text.startswith(" ") and not c.text.endswith(" ") for c in chunks))

    # The real document that motivated this.
    doc = next(d for d in load_multiclinsum("gs", "en", limit=420)
               if d.doc_id == "multiclinsum_gs_en_410")
    ch = chunk_text(doc.text, doc_id=doc.doc_id)
    big = max(len(c.text) for c in ch)
    check("gs_en_410 no longer yields a 20k-char chunk", big <= config.CHUNK_MAX_CHARS,
          f"biggest {big}")
    check("gs_en_410 offsets still exact",
          all(doc.text[c.start:c.end] == c.text for c in ch))

    # Ordinary text must be unaffected.
    normal = "The patient improved. She was discharged on day four. Follow-up was arranged."
    check("short ordinary text is untouched by the ceiling",
          len(chunk_text(normal, doc_id="n")) == 1)


def test_adaptive_top_k() -> None:
    """Long documents must not silently lose most of their source.

    Omission already dominates hallucination in this corpus, so dropping two
    thirds of a document at the retrieval stage is the worst failure available.
    """
    if need_corpora("test_adaptive_top_k"):
        return
    print()
    print("adaptive top_k")

    from docsum import config
    from docsum.chunking import chunk_text
    from docsum.retrieval import ChunkIndex
    from docsum.summarizer import ASPECT_PRESETS

    small = ChunkIndex(chunk_text("One. Two. Three.", doc_id="s"))
    check("small documents keep the configured top_k",
          small.effective_top_k(config.TOP_K, 8) == config.TOP_K)

    docs = {d.doc_id: d for d in load_multiclinsum("gs", "en", limit=520)}
    for name, floor in (("multiclinsum_gs_en_296", 0.9), ("multiclinsum_gs_en_504", 0.9)):
        doc = docs[name]
        chunks = chunk_text(doc.text, doc_id=name)
        selected = ChunkIndex(chunks).build().search_many(
            ASPECT_PRESETS["clinical"], top_k=config.TOP_K)
        ratio = len(selected) / len(chunks)
        check(f"{name} retrieves most of its source", ratio >= floor,
              f"only {ratio:.0%} of {len(chunks)} chunks")

    # The cap must still bind, or a pathological document floods the prompt.
    huge = ChunkIndex([object()] * 5000)
    check("effective_top_k is capped", huge.effective_top_k(config.TOP_K, 8) == config.TOP_K_MAX)


# --- 14. Groq backend: rate limits and failure handling ---------------------

def test_remote_backend() -> None:
    """The Groq backend, exercised without a network or a key.

    Rate limiting is the interesting part. The free tier allows 1000
    requests/minute but only 8000 tokens/minute, and one case report costs
    roughly 2500 tokens round trip -- so a batch run is paced by tokens at about
    three documents per minute, and 429 is the normal path rather than an error.
    A 20-document comparison died after two before backoff was added.
    """
    print()
    print("groq backend")

    import os
    import httpx
    from docsum import config
    from docsum.remote import (
        RemoteError, _parse_reset, _retry_delay, _post, summarize_remote,
    )

    # Groq reports reset windows in its own duration format, not seconds.
    for raw, expected in [("630ms", 0.63), ("2.5s", 2.5), ("1m", 60.0), ("1m26.4s", 86.4)]:
        got = _parse_reset(raw)
        check(f"parse reset {raw!r} -> {expected}", got is not None and abs(got - expected) < 1e-6,
              f"got {got}")
    check("plain seconds parse", _parse_reset("12") == 12.0)
    check("unparseable reset returns None", _parse_reset("nonsense") is None)
    check("missing reset returns None", _parse_reset(None) is None)

    # The server's stated wait is preferred over guessing, and always capped.
    resp = httpx.Response(429, headers={"x-ratelimit-reset-tokens": "2s"})
    delay = _retry_delay(resp, attempt=0)
    check("retry delay follows the server's reset header", 2.0 <= delay <= 2.6,
          f"got {delay}")
    capped = _retry_delay(httpx.Response(429, headers={"retry-after": "99999"}), 0)
    check("retry delay is capped", capped <= config.GROQ_MAX_BACKOFF, f"got {capped}")
    blind = _retry_delay(httpx.Response(500), attempt=3)
    check("falls back to exponential backoff with no headers", 8.0 <= blind <= 9.1,
          f"got {blind}")

    # A missing key must explain itself and name the backends that need nothing.
    saved = os.environ.pop(config.GROQ_API_KEY_ENV, None)
    try:
        try:
            summarize_remote("Some text. More text.", doc_id="t")
            check("missing key raises RemoteError", False, "no exception")
        except RemoteError as exc:
            msg = str(exc)
            check("missing key raises RemoteError", True)
            check("the error names the env var", config.GROQ_API_KEY_ENV in msg)
            check("the error points at a backend needing no credentials",
                  "extractive" in msg)
    finally:
        if saved is not None:
            os.environ[config.GROQ_API_KEY_ENV] = saved

    # An empty document short-circuits before any network call, key or not.
    saved = os.environ.get(config.GROQ_API_KEY_ENV)
    os.environ[config.GROQ_API_KEY_ENV] = "not-a-real-key"
    try:
        empty = summarize_remote("", doc_id="empty")
        check("empty document returns an empty result without calling out",
              empty.summary == "" and not empty.facts)
    finally:
        if saved is None:
            os.environ.pop(config.GROQ_API_KEY_ENV, None)
        else:
            os.environ[config.GROQ_API_KEY_ENV] = saved

    # Retry must eventually give up rather than hang forever.
    calls = {"n": 0}

    def always_429(*a, **kw):
        calls["n"] += 1
        return httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1ms"},
                              text="rate limited", request=httpx.Request("POST", "http://x"))

    real_post, real_key = httpx.post, os.environ.get(config.GROQ_API_KEY_ENV)
    os.environ[config.GROQ_API_KEY_ENV] = "not-a-real-key"
    httpx.post = always_429
    try:
        try:
            _post({"model": "x", "messages": []}, timeout=5)
            check("persistent 429 eventually raises", False, "no exception")
        except RemoteError as exc:
            check("persistent 429 eventually raises", True)
            check("gives up after GROQ_MAX_RETRIES + 1 attempts",
                  calls["n"] == config.GROQ_MAX_RETRIES + 1,
                  f"made {calls['n']} attempts")
            check("the error says it retried", "attempts" in str(exc))
    finally:
        httpx.post = real_post
        if real_key is None:
            os.environ.pop(config.GROQ_API_KEY_ENV, None)
        else:
            os.environ[config.GROQ_API_KEY_ENV] = real_key

    # A bad key must fail fast, not burn the whole retry budget.
    calls["n"] = 0

    def always_401(*a, **kw):
        calls["n"] += 1
        return httpx.Response(401, text="unauthorized", request=httpx.Request("POST", "http://x"))

    os.environ[config.GROQ_API_KEY_ENV] = "not-a-real-key"
    httpx.post = always_401
    try:
        try:
            _post({"model": "x", "messages": []}, timeout=5)
            check("401 raises", False, "no exception")
        except RemoteError:
            check("401 raises", True)
            check("401 is not retried", calls["n"] == 1, f"made {calls['n']} attempts")
    finally:
        httpx.post = real_post
        if real_key is None:
            os.environ.pop(config.GROQ_API_KEY_ENV, None)
        else:
            os.environ[config.GROQ_API_KEY_ENV] = real_key


def main() -> int:
    print("=" * 74)
    print("docsum pipeline invariants")
    print("=" * 74)
    for fn in (
        test_sentence_splitting,
        test_chunk_offsets,
        test_chunk_coverage,
        test_request_shape,
        test_citation_mapping_full,
        test_citation_mapping_subset,
        test_refusal_handling,
        test_metrics,
        test_loaders,
        test_extractive_backend,
        test_local_alignment,
        test_metric_validation,
        test_chunk_ceiling,
        test_adaptive_top_k,
        test_remote_backend,
    ):
        fn()

    print("\n" + "=" * 74)
    tail = f", {len(SKIP)} skipped (corpora absent)" if SKIP else ""
    print(f"{len(PASS)} passed, {len(FAIL)} failed" + tail)
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
