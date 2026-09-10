"""Scoring for generated summaries.

Two families of metric, because they answer different questions:

* Reference-based (ROUGE) -- how close is the summary to the dataset's gold
  summary? Useful for tracking regressions, but it cannot tell a hallucinated
  number from a paraphrase.
* Attribution-based -- are the citations real, and do the summary's specifics
  actually occur in the source? These need no gold summary at all, so they work
  on documents a user uploads.

`citation_integrity` is the one that matters most: it re-derives every cited
span from the source text and checks it against what the API reported. A
mismatch means the offset mapping is broken, which would make the whole
attribution feature quietly untrustworthy.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .summarizer import SummaryResult

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Numbers, dosages, ages, dates -- the specifics a clinical summary must not
# invent. Captures "55", "2.5", "10,200", "07/29/2008", "110/80".
_NUMERIC = re.compile(r"\d+(?:[.,/]\d+)*")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, pred_n: int, ref_n: int) -> tuple[float, float, float]:
    precision = overlap / pred_n if pred_n else 0.0
    recall = overlap / ref_n if ref_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def rouge_n(prediction: str, reference: str, n: int = 1) -> dict[str, float]:
    pred, ref = _ngrams(_tokens(prediction), n), _ngrams(_tokens(reference), n)
    overlap = sum((pred & ref).values())
    p, r, f = _f1(overlap, sum(pred.values()), sum(ref.values()))
    return {"precision": p, "recall": r, "f1": f}


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, O(len(a) * len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        curr = [0]
        for j, other in enumerate(b):
            if token == other:
                curr.append(prev[j] + 1)
            else:
                curr.append(max(curr[j], prev[j + 1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    pred, ref = _tokens(prediction), _tokens(reference)
    p, r, f = _f1(_lcs_length(pred, ref), len(pred), len(ref))
    return {"precision": p, "recall": r, "f1": f}


@dataclass
class Scores:
    rouge1_f1: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_f1: float = 0.0
    # Share of summary characters sitting inside a cited block.
    attribution_coverage: float = 0.0
    # Share of cited spans whose offsets resolve to the text the API reported.
    citation_integrity: float = 1.0
    # Share of numbers in the summary that also appear in the source document.
    numeric_fidelity: float = 1.0
    unsupported_numbers: list[str] = field(default_factory=list)
    invalid_citations: int = 0
    n_claims: int = 0
    n_spans: int = 0

    def as_row(self) -> dict:
        return {
            "rouge1_f1": round(self.rouge1_f1, 4),
            "rouge2_f1": round(self.rouge2_f1, 4),
            "rougeL_f1": round(self.rougeL_f1, 4),
            "attribution_coverage": round(self.attribution_coverage, 4),
            "citation_integrity": round(self.citation_integrity, 4),
            "numeric_fidelity": round(self.numeric_fidelity, 4),
            "invalid_citations": self.invalid_citations,
            "n_claims": self.n_claims,
            "n_spans": self.n_spans,
        }


def _normalise_ws(text: str) -> str:
    return " ".join(text.split())


def citation_integrity(result: SummaryResult) -> tuple[float, int]:
    """Check every citation's offsets against the source text.

    The API reports `cited_text` alongside the offsets; if our chunk-offset
    arithmetic is right, slicing the original document at those offsets
    reproduces it. Returns (share valid, count invalid).
    """
    spans = result.all_spans()
    if not spans:
        return 1.0, 0

    invalid = 0
    for span in spans:
        if not (0 <= span.start <= span.end <= len(result.source_text)):
            invalid += 1
            continue
        actual = _normalise_ws(result.source_text[span.start : span.end])
        expected = _normalise_ws(span.cited_text)
        if actual != expected:
            invalid += 1

    return (len(spans) - invalid) / len(spans), invalid


def _numeric_forms(token: str) -> set[str]:
    """Every form a numeric token can legitimately be written as.

    Composite tokens joined by "/" are dates ("31/05/2023") or ratios
    ("110/80"), whose components are real values in their own right: a summary
    writing "31" or "2023" from a source "31/05/2023", or "80" from a blood
    pressure of "110/80", is faithful. Reformatting a date into "May 31, 2023"
    is the common case, and without this the metric reports three inventions
    for a summary that invented nothing.

    Decimals and thousands separators are deliberately NOT decomposed: "3.21"
    does not support a claim of "21", and "10,200" does not support "200".
    Splitting those would let genuinely invented values through, which is the
    failure this metric exists to catch.
    """
    forms = {token, token.replace(",", "")}
    if "/" in token:
        for part in token.split("/"):
            if part.isdigit():
                # Zero-padded date components: "05" and "5" are the same day.
                forms.add(part)
                forms.add(part.lstrip("0") or "0")
            elif part:
                forms.add(part)
    return forms


# Typographic separators models use inside numbers: narrow no-break space,
# no-break space, thin space, figure space. gpt-oss-120b writes "25 000" with
# U+202F, which without normalisation tokenises as "25" and "000" -- two
# fabricated numbers where the model in fact copied the value faithfully.
_DIGIT_SEPARATORS = re.compile(r"(?<=\d)[    ](?=\d)")


def _normalise_numerals(text: str) -> str:
    """Collapse typographic thousands separators so numerals tokenise correctly."""
    return _DIGIT_SEPARATORS.sub("", text)


def numeric_fidelity(summary: str, source_text: str) -> tuple[float, list[str]]:
    """Share of numbers in the summary that also occur in the source.

    Catches invented dosages, ages and lab values -- the failure mode that
    matters most on clinical text and that ROUGE is blind to.

    Comparison is per numeric token, with composite tokens broken on "/" at both
    ends so a date or ratio written differently in the summary than in the
    source is not miscounted as invention. See `_numeric_forms` for exactly what
    is and is not decomposed.

    Still a screen rather than a verdict, with two known false positives:

    * It matches numerals only, so a source spelling a number out ("Six months")
      does not support a summary that writes it as a digit.
    * List enumerators count as numbers. A summary that writes "(1) ... (2) ..."
      is flagged for indices that are structure, not claims. Models that format
      with numbered lists therefore score worse than models writing plain prose,
      independently of how faithful either is.
    * Source corruption inverts the test. multiclinsum_gs_en_20 lost its
      subscripts during text extraction, so the document reads "pCO23.4 kPa,
      pO211.7 kPa, HCO319.5 mmol/L" and "SO294%". A model that correctly reads
      those as pCO2 3.4, pO2 11.7, HCO3 19.5 and SO2 94% is then flagged for
      four inventions -- penalised precisely for getting it right.

    Typographic thousands separators are handled (see `_normalise_numerals`).
    """
    summary_numbers = _NUMERIC.findall(_normalise_numerals(summary))
    if not summary_numbers:
        return 1.0, []

    supported: set[str] = set()
    for token in _NUMERIC.findall(_normalise_numerals(source_text)):
        supported |= _numeric_forms(token)

    missing = [n for n in summary_numbers if not (_numeric_forms(n) & supported)]
    return (len(summary_numbers) - len(missing)) / len(summary_numbers), missing


def score(result: SummaryResult, reference: str | None = None) -> Scores:
    """Score one summarisation result. `reference` is optional."""
    scores = Scores()

    claims = [f for f in result.facts if f.is_claim_like]
    scores.n_claims = len(claims)
    scores.n_spans = len(result.all_spans())
    scores.attribution_coverage = result.coverage

    scores.citation_integrity, scores.invalid_citations = citation_integrity(result)
    scores.numeric_fidelity, scores.unsupported_numbers = numeric_fidelity(
        result.summary, result.source_text
    )

    if reference:
        scores.rouge1_f1 = rouge_n(result.summary, reference, 1)["f1"]
        scores.rouge2_f1 = rouge_n(result.summary, reference, 2)["f1"]
        scores.rougeL_f1 = rouge_l(result.summary, reference)["f1"]

    return scores


def aggregate(rows: list[Scores]) -> dict:
    """Mean of each numeric field across runs."""
    if not rows:
        return {}
    keys = rows[0].as_row().keys()
    out = {}
    for key in keys:
        values = [r.as_row()[key] for r in rows]
        out[key] = round(sum(values) / len(values), 4)
    out["n_docs"] = len(rows)
    return out
