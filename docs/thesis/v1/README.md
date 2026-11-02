# M100 thesis methodology/results — documentation v1

Start with THESIS_METHODS_RESULTS.pdf (19 A4 pages), or the editable Word format
THESIS_METHODS_RESULTS.docx. Markdown and self-contained local-asset HTML contain
the same evidence-grounded material. Keep this hashed version intact; use Save As
to make an editing copy for institution-specific formatting. No author, university,
external bibliography, experimental outcome or citation detail was invented.

The document contains research objectives/scope, exact 14-channel feature order,
causal preprocessing and chronology, residual GRU/embedding architecture,
FedAvg and schedule-matched central SGD, persistence, selected validation/test
results for all three seeds, mean/sample SD, pooled/equal-node and per-node
analysis, limitations, provenance and proposed future work. All source references
are local repository evidence identifiers. The test period is explicitly already
observed; seeds are not independent datasets, forecasts do not prove savings,
and the simulation does not establish a fully privacy-preserving deployment.
The future 1/6/24-hour-history, environmental-ablation and optimization stages
are labeled PROPOSED, not completed. No new experiment was launched.

SOURCES.md/SOURCES.json identify82 direct source files with SHA-256, including
both completed test evaluations, final and validation reports, manifests,
configurations, selected exports/receipts, frozen metadata, source code and
historical docs. EVIDENCE.json contains exact saved numerical table inputs.
Existing completion inventories were checked without regenerating any report.
The final-report verifier also checks2652 recorded upstream inputs. The figures
are byte-identical copies, not newly evaluated or regenerated plots.

Documentation verification passed44 checks: source hashes, citations, exact
per-seed values, feature order/units/mask, architecture configurations, stated
limitations, Word XML/CRC/text/tables/media relationships, PDF results and page
bounds. Representative release PDF pages4 and15 were visually inspected; result
plots/tables were also inspected during preparation. Word was checked structurally
and for content; no Word/LibreOffice pagination renderer was available. PDF is
fully rendered. All3713 pre-existing non-venv files match before/after by bytes,
size, modification time and permissions. Access-time changes are ignored.
THESIS_COMPLETED.json is a NEW documentation receipt; no historical expected
hashes, sources, configurations, arrays, models, scores or receipts were modified.

No model imports, checkpoint deserialization, training, inference or evaluation
rerun occurred. The documentation compiler reads JSON/text and hashes artifacts;
its only arithmetic is transcription checking of stored score summaries. HTML
was rendered to PDF with a fresh isolated headless Chrome profile in /tmp, with
GPU and background networking disabled. Temporary work is outside the project.

Read-only package/source preservation verification, from project root:

```bash
cd /media/csedept/cse2014/Project/m100_training_template
.venv/bin/python -B docs/thesis/v1/code/verify_thesis.py
```

Documentation checks only (requires installed pdftotext for PDF checks):

```bash
.venv/bin/python -B docs/thesis/v1/code/check_documentation.py --directory docs/thesis/v1 --project-root .
```

The included compiler accepts project-root/output-dir arguments and refuses an
existing output. It produces Markdown, HTML and Word from DOCUMENT.json content
built from the protected saved evidence; PDF rendering is separate. Rebuilding
requires a new unused destination. Do not overwrite this snapshot or alter old
hashes to resolve a mismatch. Preserve and investigate any discrepancy.
