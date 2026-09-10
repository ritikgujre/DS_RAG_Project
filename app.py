#!/usr/bin/env python
"""Web front end: upload a document, get a summary with sources highlighted.

    ./.venv/Scripts/python.exe -m uvicorn app:app --reload --port 8000

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from docsum import config
from docsum.datasets import read_document
from docsum.extractive import summarize_extractive
from docsum.local import summarize_local
from docsum.remote import summarize_remote
from docsum.verified import summarize_verified
from docsum.report import to_html
from docsum.summarizer import ASPECT_PRESETS, summarize

app = FastAPI(title="docsum")

# Uploads are held in memory only long enough to extract text. Nothing is
# persisted to disk beyond the tempfile that PDF/DOCX parsing requires.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".txt", ".md", ".csv", ".pdf", ".docx"}

UPLOAD_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>docsum</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#dcdcdc; --accent:#1971c2; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#16181c; --fg:#e6e6e6; --muted:#9aa0a6; --line:#2c2f34; --accent:#74a9e8; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
    background:var(--bg); color:var(--fg);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .card {{ width:min(92vw,540px); padding:36px; }}
  h1 {{ margin:0 0 6px; font-size:22px; }}
  p.sub {{ margin:0 0 28px; color:var(--muted); font-size:14px; }}
  label {{ display:block; font-size:12px; text-transform:uppercase;
    letter-spacing:.07em; color:var(--muted); font-weight:600; margin:0 0 7px; }}
  .field {{ margin-bottom:20px; }}
  input[type=file], select {{ width:100%; padding:11px 12px; border-radius:7px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg); font-size:14px; }}
  .drop {{ border:1.5px dashed var(--line); border-radius:9px; padding:26px 18px;
    text-align:center; color:var(--muted); font-size:14px; }}
  button {{ width:100%; padding:13px; border:0; border-radius:7px; cursor:pointer;
    background:var(--accent); color:#fff; font-size:15px; font-weight:600; }}
  button:disabled {{ opacity:.55; cursor:progress; }}
  .note {{ margin-top:18px; color:var(--muted); font-size:12.5px; }}
  .err {{ background:#fdecec; color:#a32020; border-radius:7px; padding:12px 14px;
    font-size:14px; margin-bottom:22px; }}
  @media (prefers-color-scheme: dark) {{ .err {{ background:#3a1d1d; color:#f3a9a9; }} }}
</style>
<div class="card">
  <h1>Document summariser</h1>
  <p class="sub">Every fact in the summary links back to the exact span it came from.</p>
  {error}
  <form method="post" action="/summarize" enctype="multipart/form-data" id="f">
    <div class="field">
      <label>Document</label>
      <div class="drop">
        <input type="file" name="file" required
               accept=".txt,.md,.csv,.pdf,.docx">
        <div style="margin-top:8px">.txt &middot; .md &middot; .csv &middot; .pdf &middot; .docx &nbsp;&mdash;&nbsp; up to 25 MB</div>
      </div>
    </div>
    <div class="field">
      <label>Retrieval aspects</label>
      <select name="aspects">{aspect_options}</select>
    </div>
    <div class="field">
      <label>Backend</label>
      <select name="backend">{backend_options}</select>
    </div>
    <div class="field">
      <label>Summary length</label>
      <select name="length">{length_options}</select>
    </div>
    <button type="submit" id="b">Summarise</button>
  </form>
  <p class="note">Retrieval runs locally on CPU. <b>Extractive</b> and <b>Local GPU</b>
     both stay entirely on this machine — the first selects source sentences
     verbatim, the second writes prose and aligns each sentence back to a span.
     <b>Generated</b> sends the retrieved excerpts to {model}, and needs an API key.</p>
</div>
<script>
  document.getElementById('f').addEventListener('submit', () => {{
    const b = document.getElementById('b');
    b.disabled = true; b.textContent = 'Summarising…';
  }});
</script>
"""


# Extractive is the default: it needs no credentials, so a fresh checkout is
# usable immediately rather than erroring on a missing key.
BACKENDS = {
    "extractive": "Extractive — verbatim source sentences, no API key",
    "local": "Local GPU — prose from a local model, spans aligned after",
    "groq": "Groq — prose from a hosted model, spans aligned after",
    "verified": "Verified — the model cites, every quote is checked",
    "api": "Generated — Claude writes prose, needs an API key",
}

LENGTHS = {
    "brief": "Brief — abstract, roughly 120 words",
    "standard": "Standard — the significant findings, roughly 250 words",
    "full": "Full — every substantive point, barely condensed",
}

# The CLI and library default to `full`, because that is what every measured
# table in the README describes and changing it would invalidate them. A reader
# uploading a single document wants a summary, though, and `full` over a short
# document reads as a reworded copy of it -- retrieval is a pass-through at that
# size, so "cover everything" is applied to the whole source. So the web form
# opens on `standard` instead. Both remain one dropdown apart.
WEB_DEFAULT_LENGTH = "standard"


def _render_upload(
    error: str = "", backend: str = "extractive", length: str = WEB_DEFAULT_LENGTH
) -> str:
    options = "".join(
        f'<option value="{name}"{" selected" if name == "generic" else ""}>{name}</option>'
        for name in sorted(ASPECT_PRESETS)
    )
    backends = "".join(
        f'<option value="{key}"{" selected" if key == backend else ""}>{label}</option>'
        for key, label in BACKENDS.items()
    )
    lengths = "".join(
        f'<option value="{key}"{" selected" if key == length else ""}>{label}</option>'
        for key, label in LENGTHS.items()
    )
    return UPLOAD_PAGE.format(
        error=f'<div class="err">{error}</div>' if error else "",
        aspect_options=options,
        backend_options=backends,
        length_options=lengths,
        model=config.GEN_MODEL,
    )


def _is_missing_credentials(exc: BaseException) -> bool:
    """Whether `exc` is the SDK's unresolved-authentication failure."""
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return True
    return isinstance(exc, TypeError) and "authentication method" in str(exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render_upload()


@app.post("/summarize", response_class=HTMLResponse)
async def do_summarize(
    file: UploadFile = File(...),
    aspects: str = Form("generic"),
    backend: str = Form("extractive"),
    length: str = Form(WEB_DEFAULT_LENGTH),
) -> HTMLResponse:
    if backend not in BACKENDS:
        return HTMLResponse(_render_upload(f"Unknown backend {backend!r}."), status_code=400)
    if length not in LENGTHS:
        return HTMLResponse(
            _render_upload(f"Unknown length {length!r}.", backend=backend), status_code=400
        )

    name = Path(file.filename or "upload.txt").name
    suffix = Path(name).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        return HTMLResponse(
            _render_upload(f"Unsupported file type {suffix or '(none)'}.", backend=backend, length=length), status_code=400
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return HTMLResponse(
            _render_upload(f"File is {len(raw) / 1e6:.1f} MB; the limit is 25 MB.", backend=backend, length=length),
            status_code=413,
        )
    if not raw:
        return HTMLResponse(_render_upload("That file is empty.", backend=backend, length=length), status_code=400)

    # PDF and DOCX parsing needs a real path; delete it as soon as text is out.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(raw)
        try:
            text = read_document(path)
        except Exception as exc:
            return HTMLResponse(
                _render_upload(f"Could not read that file: {exc}", backend=backend, length=length), status_code=400
            )

    if not text.strip():
        return HTMLResponse(
            _render_upload(
                "No extractable text. If this is a scanned PDF it needs OCR first."
            ),
            status_code=400,
        )

    try:
        if backend == "extractive":
            result = summarize_extractive(text, doc_id=name, aspects=aspects, length=length)
        elif backend == "local":
            result = summarize_local(text, doc_id=name, aspects=aspects, length=length)
        elif backend == "groq":
            result = summarize_remote(text, doc_id=name, aspects=aspects, length=length)
        elif backend == "verified":
            # No length here: this backend emits a claim list rather than prose,
            # and its task string is the thing its measurement is about.
            result = summarize_verified(text, doc_id=name, aspects=aspects)
        else:
            result = summarize(text, doc_id=name, aspects=aspects, length=length)
    except Exception as exc:
        traceback.print_exc()
        # The SDK raises a bare TypeError when it cannot resolve credentials,
        # which is the one failure a first-time user is most likely to hit.
        # Point them at the backend that needs none rather than at a log file.
        if _is_missing_credentials(exc):
            return HTMLResponse(
                _render_upload(
                    "No API key found, so the generated backend cannot run. "
                    "Set ANTHROPIC_API_KEY and restart, or choose the "
                    "extractive backend — it needs no key.",
                    backend=backend,
                    length=length,
                ),
                status_code=503,
            )
        return HTMLResponse(
            _render_upload("Summarisation failed — see the server log.", backend=backend, length=length),
            status_code=500,
        )

    if result.refusal:
        return HTMLResponse(
            _render_upload(f"The model declined this document: {result.refusal}"),
            status_code=422,
        )

    return HTMLResponse(to_html(result, title=name))
