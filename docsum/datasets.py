"""Loaders for the two corpora in this repo.

MultiClinSum is read straight out of its zips -- the large-scale splits hold
~26k documents per language and there is no reason to expand 90MB of text onto
disk to sample a few hundred of them.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import config

# The CSV `dialogue` and `section_text` fields run long.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

LANGUAGES = ("en", "es", "fr", "pt")


@dataclass
class DocPair:
    """A document and its reference summary."""

    doc_id: str
    text: str
    reference: str
    lang: str = "en"
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}


# --- MultiClinSum ------------------------------------------------------------


def _multiclinsum_zip(split: str, lang: str) -> Path:
    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {LANGUAGES}, got {lang!r}")
    if split == "gs":
        name = f"multiclinsum_gs_train_{lang}.zip"
    elif split in ("ls", "large-scale"):
        name = f"multiclinsum_large-scale_train_{lang}.zip"
    else:
        raise ValueError(f"split must be 'gs' or 'ls', got {split!r}")

    path = config.MULTICLINSUM_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing MultiClinSum archive: {path}")
    return path


def load_multiclinsum(
    split: str = "gs", lang: str = "en", limit: int | None = None
) -> list[DocPair]:
    """Load fulltext/summary pairs from a MultiClinSum archive.

    `split` is "gs" (593 human-curated pairs per language) or "ls" (~25.9k).
    """
    archive = _multiclinsum_zip(split, lang)
    pairs: list[DocPair] = []

    with zipfile.ZipFile(archive) as zf:
        # Map stem -> (fulltext name, summary name). The archives also contain a
        # stray nested `.zip` duplicate in the French gold standard; filtering on
        # the .txt suffix skips it.
        fulltexts: dict[str, str] = {}
        summaries: dict[str, str] = {}

        for info in zf.infolist():
            if info.is_dir() or not info.filename.endswith(".txt"):
                continue
            stem = Path(info.filename).stem
            if "/fulltext/" in info.filename:
                fulltexts[stem] = info.filename
            elif "/summaries/" in info.filename:
                summaries[re.sub(r"_sum$", "", stem)] = info.filename

        # Sort numerically by trailing id so `limit` gives a stable sample.
        def sort_key(stem: str) -> tuple[int, str]:
            match = re.search(r"_(\d+)$", stem)
            return (int(match.group(1)) if match else 0, stem)

        for stem in sorted(fulltexts.keys() & summaries.keys(), key=sort_key):
            text = zf.read(fulltexts[stem]).decode("utf-8", errors="replace").strip()
            ref = zf.read(summaries[stem]).decode("utf-8", errors="replace").strip()
            if not text or not ref:
                continue
            pairs.append(
                DocPair(doc_id=stem, text=text, reference=ref, lang=lang,
                        meta={"split": split, "source": archive.name})
            )
            if limit is not None and len(pairs) >= limit:
                break

    return pairs


# --- MTS-Dialog --------------------------------------------------------------

MTS_FILES = {
    "train": config.MTS_DIR / "Main-Dataset" / "MTS-Dialog-TrainingSet.csv",
    "validation": config.MTS_DIR / "Main-Dataset" / "MTS-Dialog-ValidationSet.csv",
    "test1": config.MTS_DIR / "Main-Dataset" / "MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
    "test2": config.MTS_DIR / "Main-Dataset" / "MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv",
}


def load_mts_dialog(split: str = "validation", limit: int | None = None) -> list[DocPair]:
    """Load MTS-Dialog conversation/summary pairs.

    The dialogue is the document; `section_text` is the reference summary and
    `section_header` is kept in meta.
    """
    if split not in MTS_FILES:
        raise ValueError(f"split must be one of {sorted(MTS_FILES)}, got {split!r}")
    path = MTS_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"missing MTS-Dialog file: {path}")

    pairs: list[DocPair] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            dialogue = (row.get("dialogue") or "").strip()
            summary = (row.get("section_text") or "").strip()
            if not dialogue or not summary:
                continue
            pairs.append(
                DocPair(
                    doc_id=f"mts-{split}-{row.get('ID')}",
                    text=dialogue,
                    reference=summary,
                    lang="en",
                    meta={"section_header": row.get("section_header", ""), "split": split},
                )
            )
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


# --- Arbitrary user uploads --------------------------------------------------


def read_document(path: str | Path) -> str:
    """Extract plain text from a file the user uploaded.

    Text is returned as-is apart from newline normalisation, because citation
    offsets must index the same string the summariser was given.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            text = "\n\n".join(page.get_text() for page in doc)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        # .txt, .md, .csv and anything else that is already text.
        text = path.read_text(encoding="utf-8", errors="replace")

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _read_docx(path: Path) -> str:
    """Pull paragraph text out of a .docx without a hard python-docx dependency."""
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    # Paragraph breaks first, then strip the remaining tags.
    xml = re.sub(r"</w:p>", "\n\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return re.sub(r"\n{3,}", "\n\n", xml)
