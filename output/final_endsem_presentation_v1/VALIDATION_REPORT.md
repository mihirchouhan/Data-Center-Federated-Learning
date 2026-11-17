# Presentation validation

**Status:** complete, rendered and inspected. There are 24 slides, with a planned delivery time of 19 minutes 55 seconds including pauses. Presentation: 19 December 2026. Evidence cutoff: 17 November 2026.

## Evidence and arithmetic checks

- All 109 updated-report inventory entries match byte-for-byte.
- The 647 prior content comparisons freshly reproduce 638 expected matches and nine known failures. The nine newer history-audit/preflight failures remain FAILED; no hash or baseline was repaired.
- All 279 audited source/configuration/test bindings match at the beginning and final scoped verification.
- All 472 saved-record arithmetic/identity checks pass. These cover v5 common-window bindings and paired comparisons, B1 seed/node/pooled/macro reconciliation, T5/T6 means and sample SD, and T7 uniqueness and all 33 groups. No model predictions were recomputed.
- All 1,367 package/preservation checks pass, including 24 slide parts, 24 note parts, image identity, internal relationships, content types, theme style lists, 16:9 geometry, PDF page titles and source-ID resolution. This count does not convert the nine historical failed bindings into passes.
- Supporting files contain 43 viva answers, 24 complete script sections and 210 stored metric records.

## Rendering and visual inspection

Every PDF page was independently rasterized with `pdftoppm` and viewed. Text-bound checks show zero canvas overflow. Visual review checked card margins, chart labels, captions, alignment, units, metrics and terminology. Crowded text, card heights and caption placement were corrected. Split labels use half-open intervals; bar axes start at zero. Detailed slide records are in `EVIDENCE/visual_inspection.json`.

The PPTX embeds the exact 24 lossless slide images used in the design, plus actual speaker notes. The PDF uses the same source artwork with vector graphics and text. Native PowerPoint/LibreOffice rendering was unavailable and was not tested. Editable SVG, Mermaid and slide-content sources are included; native editable PowerPoint text boxes are not provided.

## Scientific qualifications retained

- One physical facility and 979 known compute-node clients, simulated on one server.
- Separate historical v1/v2/v3, v5, B1, old history, T4, T5, T6 and T7 contracts; no global cross-cohort leaderboard.
- B1 has mixed persistence outcomes. Its centralized March metrics are not verified present.
- T4 pooled MAE 24.8133 W pairs with pooled persistence 33.0661 W; macro MAE 25.2242 W pairs with macro persistence 33.3504 W. Its central control is not exact B1 schedule replay.
- T7 clean RMSE is 57.5502 W from the CSV, not the erroneous 44.97 W narrative. Training-mean input fallback, condition-specific persistence and fixed masks are disclosed.
- The newer residual-history benchmark was externally progressing at the fixed 20:24:32+05:30 snapshot. Provisional scores are excluded. All 979 clients fail its stated 1,024-window eligibility predicate; that conflict and the nine manifest failures remain unresolved.
- Historical strict preflight is FAILED; semantic reproduction is UNKNOWN. No universal FL, long-history or weather benefit, energy saving, PUE/SLA improvement, formal privacy or untouched-confirmatory March claim is made.
- The official v3 paper PDF was read and byte-verified. The specifically named uploaded copy and mid-semester deck were unavailable locally.

## Preservation

Metadata for 174,893 pre-existing regular files was compared using bytes, modification time and mode; access time was excluded. Two changed log paths belong to the observed externally progressing history-run output. Their exact writer/invocation is UNKNOWN. No process was launched, resumed, signalled or controlled by this task, and no metadata change was observed outside that output prefix.

All listed core result, source and report byte bindings remain unchanged. This is a bounded byte check plus a wider metadata check; whole-project binary integrity or full-tree zero-change is not claimed. See `EVIDENCE/preservation_verification.json` for exact paths and fields.

## Execution boundary

Only reporting, plotting, XML/PDF validation and read-only access to saved evidence occurred. No training, inference, prediction scoring, model initialization, scaler fitting, raw preparation, download, extraction, GPU operation, baseline repair or protocol approval occurred. Publication creates one new directory and preserves existing artifacts.
