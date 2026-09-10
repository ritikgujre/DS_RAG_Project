"""Do our automatic metrics agree with human judgement?

Every other number this project reports is an automatic metric scoring itself.
That is circular: a metric can be perfectly self-consistent and still measure
nothing a reader cares about. The MTS-Dialog correlation study is the only
escape from that circle in this repository -- 400 machine summaries with their
source dialogues, positionally aligned to 400 human fact-based scores
(factual precision/recall/F1, hallucination rate, omission rate).

This module scores those 400 summaries with our metrics and correlates the
result against the human labels, so claims about what the metrics mean can be
checked rather than asserted.

What the run actually shows (see README, "Are the metrics trustworthy?"):

* `grounding_coverage` -- the local backend's alignment used as a metric -- moves
  the right way against hallucination, and on the subset where hallucination
  actually occurs the relationship is moderate. It is weak across the full set
  because the full set has almost nothing to detect.
* `numeric_fidelity` does **not** track human-judged hallucination. It is a
  narrow, precise instrument (invented numerals) and must not be presented as a
  general factuality score.
* ROUGE against the reference remains the strongest correlate of human factual
  judgement on this data, and is the only signal here that recovers the humans'
  ranking of the four systems.

The single most important caveat is a property of the corpus, not of the
metrics: **hallucination rate is exactly zero for 92% of the 400 summaries**.
With that floor effect, a weak full-set correlation is what any metric would
produce, so this data cannot be used to validate hallucination detection
strongly in either direction.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .chunking import chunk_text, sentences_within, split_sentences
from .evaluate import numeric_fidelity, rouge_l, rouge_n
from .retrieval import _load_embedder

CORRELATION_DIR = config.MTS_DIR / "Correlation-Study"
SUMMARIES_CSV = CORRELATION_DIR / "MTS-Dialog-Automatic-Summaries-ValidationSet.csv"
SCORES_CSV = CORRELATION_DIR / "MTS-Dialog-Manual-Scores4CorrelationStudy.csv"

# The two files carry no join key -- the study aligns them by row position, and
# the summaries file restarts its IDs at 0 every 100 rows because it stacks four
# systems over the same 100 validation dialogues.
BLOCK_SIZE = 100

AUTOMATIC = ["grounding_coverage", "mean_support", "numeric_fidelity", "rouge1", "rougeL"]
HUMAN = ["FactualF1", "FactualPrecision", "FactualRecall", "HallucinationRate", "OmissionRate"]


@dataclass
class ValidationRow:
    """One machine summary: our metrics beside the humans' scores."""

    grounding_coverage: float
    mean_support: float
    numeric_fidelity: float
    rouge1: float
    rougeL: float
    FactualF1: float
    FactualPrecision: float
    FactualRecall: float
    HallucinationRate: float
    OmissionRate: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in AUTOMATIC + HUMAN}


def _read_pairs() -> list[tuple[dict, dict]]:
    """Load the two CSVs and pair them by row position."""
    for path in (SUMMARIES_CSV, SCORES_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. The correlation study ships with MTS-Dialog; "
                f"expected it under {CORRELATION_DIR}."
            )
    # utf-8-sig: both files carry a BOM, which would otherwise corrupt the first
    # column name and make every lookup fail.
    with open(SUMMARIES_CSV, encoding="utf-8-sig") as fh:
        summaries = list(csv.DictReader(fh))
    with open(SCORES_CSV, encoding="utf-8-sig") as fh:
        scores = list(csv.DictReader(fh))

    if len(summaries) != len(scores):
        raise ValueError(
            f"positional alignment is the only link between these files, and the "
            f"row counts disagree: {len(summaries)} summaries vs {len(scores)} scores"
        )
    return list(zip(summaries, scores))


def grounding_coverage(summary: str, source: str, embedder=None) -> tuple[float, float]:
    """Share of summary sentences that align to a source span, and mean support.

    This is exactly the local backend's attribution step (`grounding.align`) applied
    as a measurement, which is what makes it testable against human scores: if
    the mechanism the backend relies on does not track human factuality, the
    attribution it produces is decoration.
    """
    from .grounding import align

    chunks = chunk_text(source, doc_id="src")
    candidates = sentences_within(source, chunks, min_chars=0)
    generated = split_sentences(summary)
    if not generated or not candidates:
        return 0.0, 0.0

    aligned = align(generated, candidates, embedder or _load_embedder(config.EMBED_MODEL))
    covered = sum(1 for spans in aligned if spans) / len(generated)
    supports = [s.support for spans in aligned for s in spans]
    return covered, float(np.mean(supports)) if supports else 0.0


def run(limit: int | None = None, progress=None) -> list[ValidationRow]:
    """Score every summary in the correlation study with our metrics."""
    pairs = _read_pairs()
    if limit:
        pairs = pairs[:limit]

    embedder = _load_embedder(config.EMBED_MODEL)
    rows: list[ValidationRow] = []

    for i, (summary_row, score_row) in enumerate(pairs, 1):
        dialogue = summary_row["Dialogue"]
        summary = summary_row["Automatic Summary"]
        reference = summary_row["Reference Summary"]

        covered, support = grounding_coverage(summary, dialogue, embedder)
        numeric, _ = numeric_fidelity(summary, dialogue)

        rows.append(
            ValidationRow(
                grounding_coverage=covered,
                mean_support=support,
                numeric_fidelity=numeric,
                rouge1=rouge_n(summary, reference, 1)["f1"],
                rougeL=rouge_l(summary, reference)["f1"],
                FactualF1=float(score_row["FactualF1"]),
                FactualPrecision=float(score_row["FactualPrecision"]),
                FactualRecall=float(score_row["FactualRecall"]),
                HallucinationRate=float(score_row["HallucinationRate"]),
                OmissionRate=float(score_row["OmissionRate"]),
            )
        )
        if progress and i % 100 == 0:
            progress(i, len(pairs))

    return rows


def correlations(rows: list[ValidationRow]) -> dict[tuple[str, str], tuple[float, float]]:
    """Spearman rho and p-value for every automatic/human metric pair.

    Spearman rather than Pearson: several of these are bounded, heavily skewed
    scores (hallucination is zero for most summaries), so a rank correlation is
    the honest choice.
    """
    from scipy import stats

    data = {k: np.array([getattr(r, k) for r in rows]) for k in AUTOMATIC + HUMAN}
    out = {}
    for a in AUTOMATIC:
        for h in HUMAN:
            rho, p = stats.spearmanr(data[a], data[h])
            out[(a, h)] = (float(rho), float(p))
    return out


def label_discrimination(rows: list[ValidationRow]) -> dict[str, dict]:
    """How much signal each human label actually carries.

    Reported alongside the correlations because it decides how to read them: a
    label that is constant cannot be predicted by anything, so a weak
    correlation against it says nothing about the metric.
    """
    out = {}
    for h in HUMAN:
        v = np.array([getattr(r, h) for r in rows])
        out[h] = {
            "mean": float(v.mean()),
            "sd": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "zero_fraction": float((v == 0).mean()),
        }
    return out


def by_system(rows: list[ValidationRow]) -> list[dict]:
    """Per-block means, to see whether a metric ranks the systems as humans did."""
    blocks = []
    for start in range(0, len(rows), BLOCK_SIZE):
        seg = rows[start : start + BLOCK_SIZE]
        if not seg:
            continue
        entry = {"block": start // BLOCK_SIZE, "n": len(seg)}
        for k in AUTOMATIC + HUMAN:
            entry[k] = float(np.mean([getattr(r, k) for r in seg]))
        blocks.append(entry)
    return blocks


def format_report(rows: list[ValidationRow]) -> str:
    """Render the whole validation as text."""
    lines: list[str] = []
    corr = correlations(rows)

    lines.append(f"Spearman correlation, {len(rows)} human-scored summaries")
    lines.append("(* = p < 0.01)")
    lines.append("")
    lines.append(f"{'':<20}" + "".join(f"{h[:13]:>15}" for h in HUMAN))
    for a in AUTOMATIC:
        cells = ""
        for h in HUMAN:
            rho, p = corr[(a, h)]
            cells += f"{rho:>13.3f}{'*' if p < 0.01 else ' ':<2}"
        lines.append(f"{a:<20}{cells}")

    lines.append("")
    lines.append("How much signal each human label carries:")
    for h, d in label_discrimination(rows).items():
        lines.append(
            f"  {h:<20} mean={d['mean']:.3f} sd={d['sd']:.3f} "
            f"zero for {d['zero_fraction']:.0%} of summaries"
        )

    lines.append("")
    lines.append("Per-system means (does a metric rank the systems as humans did?):")
    header = f"  {'block':<7}" + "".join(
        f"{k[:14]:>16}" for k in ["FactualF1", "HallucinationRate", "grounding_coverage",
                                  "numeric_fidelity", "rouge1"])
    lines.append(header)
    for b in by_system(rows):
        lines.append(
            f"  {b['block']:<7}{b['FactualF1']:>16.3f}{b['HallucinationRate']:>16.3f}"
            f"{b['grounding_coverage']:>16.3f}{b['numeric_fidelity']:>16.3f}{b['rouge1']:>16.3f}"
        )

    return "\n".join(lines)


def write_csv(rows: list[ValidationRow], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUTOMATIC + HUMAN)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.as_dict())
