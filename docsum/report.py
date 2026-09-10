"""Rendering a SummaryResult for humans -- terminal, HTML, or JSON."""

from __future__ import annotations

import html
import json

from .summarizer import SummaryResult


def _line_of(text: str, pos: int) -> int:
    """1-indexed line number of character offset `pos`."""
    return text.count("\n", 0, pos) + 1


def to_text(result: SummaryResult, show_spans: bool = True, width: int = 88) -> str:
    """Plain-text report: the summary, then every fact with its source."""
    out: list[str] = []

    if result.refusal:
        return f"Request declined by the model: {result.refusal}"

    out.append("=" * width)
    out.append("SUMMARY")
    out.append("=" * width)
    out.append(result.summary.strip())
    out.append("")

    out.append("=" * width)
    out.append("FACTS AND SOURCES")
    out.append("=" * width)

    n = 0
    for fact in result.facts:
        body = fact.text.strip()
        # Skip pure connective fragments -- they carry no claim to attribute.
        if not fact.is_claim_like and not fact.sources:
            continue
        n += 1
        marker = " " if fact.sources else "!"
        out.append(f"\n{marker}[{n}] {body}")
        if not fact.sources:
            out.append("      (no citation attached)")
            continue
        for span in fact.sources:
            line = _line_of(result.source_text, span.start)
            out.append(
                f"      -> chunk {span.chunk_index}, chars {span.start}-{span.end}, line {line}"
            )
            if show_spans:
                quoted = " ".join(span.cited_text.split())
                if len(quoted) > width - 12:
                    quoted = quoted[: width - 15] + "..."
                out.append(f'         "{quoted}"')

    out.append("")
    out.append("=" * width)
    claims = [f for f in result.facts if f.is_claim_like]
    cited_claims = [f for f in claims if f.is_grounded]
    out.append(
        f"attribution: {len(cited_claims)}/{len(claims)} claims cited, "
        f"{result.coverage:.0%} of summary text inside a cited span"
    )
    out.append(
        f"retrieval:   {len(result.chunks_used)} chunks used, "
        f"{len(result.all_spans())} source spans cited"
    )
    if result.usage:
        out.append(
            f"tokens:      {result.usage.get('input_tokens', 0):,} in / "
            f"{result.usage.get('output_tokens', 0):,} out"
        )
    return "\n".join(out)


def to_json(result: SummaryResult) -> str:
    return json.dumps(
        {
            "summary": result.summary,
            "coverage": result.coverage,
            "stop_reason": result.stop_reason,
            "refusal": result.refusal,
            "usage": result.usage,
            "chunks_used": [
                {"index": c.index, "start": c.start, "end": c.end}
                for c in result.chunks_used
            ],
            "facts": [
                {
                    "text": f.text,
                    "grounded": f.is_grounded,
                    "sources": [
                        {
                            "cited_text": s.cited_text,
                            "chunk_index": s.chunk_index,
                            "start": s.start,
                            "end": s.end,
                            "line": _line_of(result.source_text, s.start),
                        }
                        for s in f.sources
                    ],
                }
                for f in result.facts
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


def to_html(result: SummaryResult, title: str = "Document summary") -> str:
    """Self-contained HTML: hover a claim to highlight its source span."""
    # Assign every cited span an id so the summary can point at it.
    spans = sorted(result.all_spans(), key=lambda s: (s.start, s.end))
    merged: list[tuple[int, int, list[int]]] = []
    for i, span in enumerate(spans):
        if merged and span.start <= merged[-1][1]:
            lo, hi, ids = merged[-1]
            merged[-1] = (lo, max(hi, span.end), ids + [i])
        else:
            merged.append((span.start, span.end, [i]))

    span_id = {}
    for slot, (_, _, ids) in enumerate(merged):
        for i in ids:
            span_id[(spans[i].start, spans[i].end)] = slot

    # Rebuild the source document with <mark> around every cited region.
    src = result.source_text
    pieces: list[str] = []
    cursor = 0
    for slot, (lo, hi, _) in enumerate(merged):
        if lo > cursor:
            pieces.append(html.escape(src[cursor:lo]))
        pieces.append(
            f'<mark id="s{slot}" class="src">{html.escape(src[lo:hi])}</mark>'
        )
        cursor = hi
    pieces.append(html.escape(src[cursor:]))
    source_html = "".join(pieces).replace("\n", "<br>")

    # The summary, with each cited claim linked to its span.
    sum_parts: list[str] = []
    for fact in result.facts:
        body = html.escape(fact.text)
        if not fact.sources:
            sum_parts.append(body)
            continue
        targets = sorted({span_id.get((s.start, s.end), 0) for s in fact.sources})
        refs = " ".join(
            f'<a href="#s{t}" class="cite" data-target="s{t}">{t + 1}</a>' for t in targets
        )
        sum_parts.append(f'<span class="claim" data-refs="{",".join(f"s{t}" for t in targets)}">{body}</span><sup>{refs}</sup>')
    summary_html = "".join(sum_parts)

    claims = [f for f in result.facts if f.is_claim_like]
    cited_claims = [f for f in claims if f.is_grounded]

    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3;
           --hl:#fff3bf; --hl-active:#ffd43b; --accent:#1971c2; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#16181c; --fg:#e6e6e6; --muted:#9aa0a6; --line:#2c2f34;
             --hl:#4d3d00; --hl-active:#8a6d00; --accent:#74a9e8; }}
  }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
          font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ padding:20px 28px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0 0 4px; font-size:17px; }}
  .meta {{ color:var(--muted); font-size:13px; }}
  .wrap {{ display:grid; grid-template-columns:1fr 1fr; gap:0; align-items:start; }}
  @media (max-width:900px) {{ .wrap {{ grid-template-columns:1fr; }} }}
  section {{ padding:24px 28px; }}
  section + section {{ border-left:1px solid var(--line); }}
  @media (max-width:900px) {{ section + section {{ border-left:0; border-top:1px solid var(--line); }} }}
  h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em;
        color:var(--muted); margin:0 0 14px; font-weight:600; }}
  .source {{ white-space:normal; font-size:14px; color:var(--fg);
             max-height:78vh; overflow-y:auto; }}
  mark.src {{ background:var(--hl); color:inherit; padding:1px 0; border-radius:2px; }}
  mark.src.active {{ background:var(--hl-active); }}
  .claim {{ border-radius:2px; padding:1px 0; transition:background .12s; }}
  .claim.active {{ background:var(--hl); }}
  sup a.cite {{ color:var(--accent); text-decoration:none; font-size:11px;
                padding:0 1px; font-weight:600; }}
  sup a.cite:hover {{ text-decoration:underline; }}
</style>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">{len(cited_claims)}/{len(claims)} claims cited
   &middot; {len(result.chunks_used)} chunks retrieved
   &middot; {len(spans)} source spans</div>
</header>
<div class="wrap">
  <section>
    <h2>Summary &mdash; hover a claim to locate its source</h2>
    <div class="summary">{summary_html}</div>
  </section>
  <section>
    <h2>Source document &mdash; cited spans highlighted</h2>
    <div class="source">{source_html}</div>
  </section>
</div>
<script>
  for (const claim of document.querySelectorAll('.claim')) {{
    const ids = (claim.dataset.refs || '').split(',').filter(Boolean);
    const marks = ids.map(id => document.getElementById(id)).filter(Boolean);
    const on = () => {{
      claim.classList.add('active');
      marks.forEach(m => m.classList.add('active'));
      if (marks[0]) marks[0].scrollIntoView({{block:'center', behavior:'smooth'}});
    }};
    const off = () => {{
      claim.classList.remove('active');
      marks.forEach(m => m.classList.remove('active'));
    }};
    claim.addEventListener('mouseenter', on);
    claim.addEventListener('mouseleave', off);
  }}
</script>
"""
