# DS_RAG

Working repository for a RAG document-summarisation project: the summariser
itself, plus the clinical corpora used to evaluate it.

| Path | What it is |
| --- | --- |
| [`docsum/`](docsum/) | **The project.** A RAG document summariser that attributes every fact in a summary back to the exact span it came from. See [`docsum/README.md`](docsum/README.md) for the architecture, the evaluation tables and setup instructions. |
| `15517617/` | MultiClinSum training data (Zenodo record 15517617) — gold-standard and large-scale clinical case reports in English, Spanish, French and Portuguese, as distributed. |
| `MTS-Dialog-main/` | The MTS-Dialog corpus of doctor–patient dialogues and clinical notes, used for the metric-correlation study. |

## Datasets

Both corpora are redistributed here as released upstream, for reproducibility of
the evaluation numbers quoted in `docsum/README.md`.

- **MTS-Dialog** is licensed CC BY 4.0; its `LICENSE.txt` and `README.md` are
  included unchanged. Cite the original authors, not this repository.
- **MultiClinSum** comes from the BioASQ/Zenodo release under record
  `15517617`. Consult the upstream record for its licence and citation terms
  before reusing it.

The virtual environment (`docsum/.venv/`) is deliberately not tracked — install
from `docsum/requirements.txt` or the pinned `docsum/requirements.lock`.
