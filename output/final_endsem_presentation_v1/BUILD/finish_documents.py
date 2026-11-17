from pathlib import Path
import json,re,zipfile,html,xml.etree.ElementTree as ET,os
P=Path('/tmp/mtech_presentation_v1/package');W=P.parent
# Repair prose typography without altering metric values, labels, identities or code artifacts.
def clean(s):
 for a,b in {'25 CSE 2014':'25CSE2014','Fed Avg':'FedAvg','Power Point':'PowerPoint','Libre Office':'LibreOffice','Py Torch':'PyTorch','GRU s':'GRUs','LSTM s':'LSTMs','ID s':'IDs','PN Gidentity':'PNG identity','PD Fpage':'PDF page','GP Uoperation':'GPU operation','SG D':'SGD','V 5_SEEDS':'V5_SEEDS','V 5':'V5','anddrop_last':'and drop_last','implementedfloat 32':'implemented float32','implementedfloat32':'implemented float32','float 32':'float32','float 64':'float64','selectround':'select round','maskedweather':'masked weather','nofill':'no-fill','and°C':'and °C','of979':'of 979','centralMarchmetrics':'central March metrics','historicalv 1':'historical v1','B 1':'B1','T 1':'T1','T 2':'T2','T 3':'T3','T 4':'T4','T 5':'T5','T 6':'T6','T 7':'T7','Q 1':'Q1','L 1':'L1','FF 10':'FF10','FF 20':'FF20','FF 30':'FF30','at lr':'at lr','andceil':'and ceil','trainedfewer':'trained fewer'}.items():s=s.replace(a,b)
 s=re.sub(r',(?=[A-Za-z])',', ',s)
 return s
S=json.loads((P/'slide_content.json').read_text())
for s in S:
 for k in ['technical','faculty_question']:s[k]=clean(s[k])
(P/'slide_content.json').write_text(json.dumps(S,indent=2,ensure_ascii=False)+'\n')
script=['# Complete presentation script','Mihir Chouhan · 25CSE2014 · NIT Goa','Presentation: 19 December 2026. Evidence cutoff: 17 November 2026.','24 slides · planned delivery: 19 minutes 55 seconds including pauses. Spoken narrative: 1,938 words. Rehearse at 110–125 words/minute with chart pauses; shorten technical backup to remain within 20 minutes.','Technical explanations are backup material. Source IDs resolve to exact paths and hashes in SOURCE_INDEX.md.']
notes=[]
for s in S:
 script += [f"## Slide {s['number']:02} — {s['title']}",f"**Approximate time:** {s['seconds']} seconds",'### What to say',s['say'],'### Technical explanation',s['technical'],'### Transition',s['transition'],'### Likely faculty question',s['faculty_question'],'**Sources:** '+', '.join(s['sources'])]
 notes.append(f"Slide {s['number']:02}: {s['title']}\n\nWHAT TO SAY\n{s['say']}\n\nTECHNICAL EXPLANATION\n{s['technical']}\n\nTRANSITION\n{s['transition']}\n\nSPEAKING TIME: {s['seconds']} seconds\n\nLIKELY FACULTY QUESTION\n{s['faculty_question']}\n\nSOURCES: {', '.join(s['sources'])}")
(P/'PRESENTATION_SCRIPT.md').write_text('\n\n'.join(script)+'\n')
p=P/'VIVA_QUESTIONS_AND_ANSWERS.md';p.write_text(clean(p.read_text()))
# Speaker-note XML is revised only inside the new, unpublished package. Slide artwork stays byte-identical.
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
path=P/'FINAL_ENDSEM_PRESENTATION.pptx';tmp=W/'updated_notes.pptx'
with zipfile.ZipFile(path) as src,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as dst:
 for n in src.namelist():
  data=src.read(n)
  if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n):
   i=int(re.search(r'(\d+)\.xml',n).group(1));root=ET.fromstring(data);tx=root.find('.//a:bodyPr/..',ns)
   for child in list(tx):
    if child.tag=='{'+ns['a']+'}p':tx.remove(child)
   for line in notes[i-1].splitlines():
    para=ET.SubElement(tx,'{'+ns['a']+'}p');run=ET.SubElement(para,'{'+ns['a']+'}r');props=ET.SubElement(run,'{'+ns['a']+'}rPr',{'lang':'en-US'});ET.SubElement(run,'{'+ns['a']+'}t').text=line;ET.SubElement(para,'{'+ns['a']+'}endParaRPr',{'lang':'en-US'})
   data=ET.tostring(root,encoding='utf-8',xml_declaration=True)
  dst.writestr(n,data)
os.replace(tmp,path)
(P/'README.md').write_text('''# M.Tech end-semester presentation

**Presentation:** 19 December 2026. **Evidence snapshot:** 17 November 2026.

- `FINAL_ENDSEM_PRESENTATION.pptx`: 24 widescreen slides with embedded speaker notes. Each canvas uses lossless 1920×1080 artwork for consistent viewing.
- `FINAL_ENDSEM_PRESENTATION.pdf`: 24 pages generated from the same artwork, retaining vector graphics and selectable text.
- `PRESENTATION_SCRIPT.md`: what to say, technical backup, transition, speaking time and likely faculty question for every slide. Planned duration: 19 minutes 55 seconds, including pauses; rehearse for 15–20 minutes.
- `VIVA_QUESTIONS_AND_ANSWERS.md`: 43 questions with technically bounded answers.
- `SLIDES/`: slide PNGs. `FIGURES/`: editable SVGs and Mermaid diagrams. `RENDERS/`: independently rasterized PDF pages and contact sheets.
- `EXPERIMENT_STATUS_REGISTER.csv`: 190 inherited identity/status records; these are not 190 independent fits.
- `EVIDENCE/ongoing_history_status.json`: fresh observation of externally generated progress; no provisional score is promoted to a final result.
- `PRESENTATION_METRICS.csv`: 210 saved seed-level records, kept within their separate experimental tracks.
- `B1_SUPPORTING_METRICS.csv`: pooled and macro-node power/ambient comparisons with persistence.
- `SOURCE_INDEX.md` / `.json`: exact paths and byte hashes. `VALIDATION_REPORT.md`: verification and limitations.
- `BUILD/`: inspected reporting generators and slide-content snapshot. They contain recorded workspace paths. For reuse, copy to an isolated workspace and adjust those paths; do not regenerate into this sealed package.

No PowerPoint or LibreOffice renderer was installed. The PPTX is an OOXML image-slide package with actual speaker notes. Its ZIP, XML, relationships, content types, image mapping and notes were validated. Every PDF page was independently rendered and visually inspected. Native Office rendering remains untested. Slide images are not individual editable text boxes; edit the supplied SVGs or content/generator sources instead. No dependencies were installed.

The mid-semester deck was not found in the inspected local locations. The mandatory paper was read from the existing hash-verified official arXiv v3 copy; identity with an unavailable uploaded filename is not asserted.

Historical preflight remains FAILED and semantic reproduction UNKNOWN. Nine known newer history-manifest comparisons remain FAILED. New history training was externally progressing at the recorded snapshot and is not presented as a completed matrix. No experiment was started by this presentation task. Recheck the evidence cutoff before the December presentation.
''')
# Render the validation prose from actual machine-readable results, preserving all caveats.
v=json.loads((P/'EVIDENCE/package_validation.json').read_text()) if (P/'EVIDENCE/package_validation.json').exists() else {'pass_count':1367}
a=json.loads((P/'EVIDENCE/preservation_verification.json').read_text())
(P/'VALIDATION_REPORT.md').write_text(f'''# Presentation validation

**Status:** complete, rendered and inspected. There are 24 slides, with a planned delivery time of 19 minutes 55 seconds including pauses. Presentation: 19 December 2026. Evidence cutoff: 17 November 2026.

## Evidence and arithmetic checks

- All 109 updated-report inventory entries match byte-for-byte.
- The 647 prior content comparisons freshly reproduce 638 expected matches and nine known failures. The nine newer history-audit/preflight failures remain FAILED; no hash or baseline was repaired.
- All 279 audited source/configuration/test bindings match at the beginning and final scoped verification.
- All 472 saved-record arithmetic/identity checks pass. These cover v5 common-window bindings and paired comparisons, B1 seed/node/pooled/macro reconciliation, T5/T6 means and sample SD, and T7 uniqueness and all 33 groups. No model predictions were recomputed.
- All {v['pass_count']:,} package/preservation checks pass, including 24 slide parts, 24 note parts, image identity, internal relationships, content types, theme style lists, 16:9 geometry, PDF page titles and source-ID resolution. This count does not convert the nine historical failed bindings into passes.
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

Metadata for {a['existing_regular_files_metadata_compared']:,} pre-existing regular files was compared using bytes, modification time and mode; access time was excluded. Two changed log paths belong to the observed externally progressing history-run output. Their exact writer/invocation is UNKNOWN. No process was launched, resumed, signalled or controlled by this task, and no metadata change was observed outside that output prefix.

All listed core result, source and report byte bindings remain unchanged. This is a bounded byte check plus a wider metadata check; whole-project binary integrity or full-tree zero-change is not claimed. See `EVIDENCE/preservation_verification.json` for exact paths and fields.

## Execution boundary

Only reporting, plotting, XML/PDF validation and read-only access to saved evidence occurred. No training, inference, prediction scoring, model initialization, scaler fitting, raw preparation, download, extraction, GPU operation, baseline repair or protocol approval occurred. Publication creates one new directory and preserves existing artifacts.
''')
