from pathlib import Path
import os,json,csv,hashlib,zipfile,xml.etree.ElementTree as ET,posixpath,subprocess,datetime,shutil
R=Path('/media/csedept/cse2014/Project/m100_training_template');W=Path('/tmp/mtech_presentation_v1');P=W/'package'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
S=json.loads((P/'slide_content.json').read_text());checks=[]
def check(name,ok,scope):checks.append({'name':name,'pass':bool(ok),'scope':scope});assert ok,(name,scope)
check('24-slide specification',len(S)==24,'slide_content.json')
check('planned <=20min',sum(s['seconds'] for s in S)<=1200,'1195 seconds incl pauses; narrative rehearsing needed')
with zipfile.ZipFile(P/'FINAL_ENDSEM_PRESENTATION.pptx') as z:
 check('ZIP CRC',z.testzip() is None,'Whole PPTX container')
 names=set(z.namelist())
 check('slide count',len([x for x in names if x.startswith('ppt/slides/slide') and x.endswith('.xml')])==24,'24 OOXML slides')
 check('notes count',len([x for x in names if x.startswith('ppt/notesSlides/notesSlide') and x.endswith('.xml')])==24,'24 embedded note parts')
 for n in names:
  if n.endswith(('.xml','.rels')):ET.fromstring(z.read(n))
 for n in names:
  if not n.endswith('.rels'):continue
  base='' if n=='_rels/.rels' else n.replace('/_rels/','/').removesuffix('.rels').rsplit('/',1)[0]
  for e in ET.fromstring(z.read(n)):
   target=posixpath.normpath(posixpath.join(base,e.attrib['Target']));check('PPTX relationship',target in names,n+' -> '+target)
 for i,s in enumerate(S,1):
  check('PPTX image exact slide '+str(i),hashlib.sha256(z.read(f'ppt/media/image{i}.png')).hexdigest()==sha(P/f'SLIDES/slide_{i:02}.png'),'PPTX artwork agrees exactly with rendered design source')
  notes=' '.join(ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml')).itertext())
  check('PPTX notes slide '+str(i),s['say'] in notes and s['technical'] in notes and s['faculty_question'] in notes,'Spoken narrative and technical/faculty backup')
  rel=ET.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'));it=next(e.attrib['Target'] for e in rel if e.attrib['Type'].endswith('/image'));check('PPTX correct image relationship '+str(i),it==f'../media/image{i}.png','Identity mapping, not merely image count')
 ct=ET.fromstring(z.read('[Content_Types].xml'));main=next(e for e in ct if e.attrib.get('PartName')=='/ppt/presentation.xml');check('PPTX main content type',main.attrib['ContentType']=='application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml','Presentation compatible container type')
 pres=ET.fromstring(z.read('ppt/presentation.xml'));ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main'};size=pres.find('p:sldSz',ns);check('16:9 PPTX geometry',int(size.attrib['cx'])*9==int(size.attrib['cy'])*16,'12192000x6858000EMU')
 # Theme format lists conform to required3styles; XML well-formedness alone is insufficient.
 theme=ET.fromstring(z.read('ppt/theme/theme1.xml'));an={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
 for key in ['fillStyleLst','lnStyleLst','effectStyleLst','bgFillStyleLst']:check('PPTX theme '+key,len(theme.find('.//a:'+key,an))>=3,'DrawingML minimum style entries')
text=subprocess.check_output(['pdftotext','-layout',str(P/'FINAL_ENDSEM_PRESENTATION.pdf'),'-'],text=True);pages=text.split('\f');pages=[p for p in pages if p.strip()]
check('PDF page count',len(pages)==24,'pdftotext independent extraction')
for i,(s,page) in enumerate(zip(S,pages),1):
 check('PDF title '+str(i),' '.join(s['title'].split()) in ' '.join(page.split()),'Each correct slide title appears in its own page')
 check('PDF rendered page '+str(i),(P/f'RENDERS/pdf_slide-{i:02}.png').is_file(),'pdftoppm -r90 PNG for every page')
 layout=json.loads((P/'EVIDENCE/layout_checks.json').read_text())[i-1];check('Canvas text bounds '+str(i),not layout['canvas_text_overflow'],'Renderer-based figure bounds; visual review also performed')
refs=json.loads((P/'SOURCE_INDEX.json').read_text());refids={r['id'] for r in refs}
for s in S:check('Citation resolution '+str(s['number']),set(s['sources'])<=refids,'SOURCE_INDEX resolves every slide source ID')
# Exact recheck of every original byte binding used by presentation audit.
content=json.loads((P/'EVIDENCE/fresh_content_checks.json').read_text());cache={}
for r in content:
 p=R/r['path']
 if r['path'] not in cache:cache[r['path']]=sha(p)
 check('Preserved result byte binding',cache[r['path']]==r['fresh_actual'],r['path'])
source=json.loads((P/'EVIDENCE/source_hash_verification.json').read_text())
for r in source:check('Preserved source binding',sha(R/r['path'])==r['actual'],r['path'])
report=json.loads((P/'EVIDENCE/report_package_verification.json').read_text())
for r in report:check('Preserved report binding',sha(R/r['path'])==r['actual'],r['path'])
# Metadata is kept distinct from byte verification. Active experiment writes are observations only.
before=json.loads((W/'before_metadata.json').read_text());changes=[]
for rel,a in before.items():
 p=R/rel
 if not p.is_file():changes.append({'path':rel,'before':a,'after':None});continue
 t=p.stat();b=[t.st_size,t.st_mtime_ns,t.st_mode&0o7777]
 if a!=b:changes.append({'path':rel,'before':a,'after':b})
liveprefix='output/experiments/t1_2_controlled_history_benchmark_v1/'
unexpected=[r for r in changes if not r['path'].startswith(liveprefix)]
check('No unexpected pre-existing metadata changes',not unexpected,'Externally progressing history run is separately disclosed, not controlled')
preservation={'existing_regular_files_metadata_compared':len(before),'fields':['bytes','mtime_ns','mode'],'atime_checked':False,'metadata_changes':changes,'unexpected_changes':unexpected,'all_changes_within_observed_active_run':all(c['path'].startswith(liveprefix) for c in changes),'task_modified_existing_files':False,'exact_writer_of_concurrent_changes':'UNKNOWN; paths belong to observed externally progressing run; no process was launched/controlled by this task','fresh_content_comparisons':len(content),'fresh_expected_matches':sum(r['fresh_matches_expected'] for r in content),'known_failed_comparisons_preserved':sum(not r['fresh_matches_expected'] for r in content),'report_artifacts_preserved':len(report),'source_bindings_preserved':len(source),'historical_preflight':'FAILED unchanged','semantic_reproduction':'UNKNOWN; not executed','scope_note':'Metadata checks do not establish binary contents. Actual byte hashes checked only for listed result/source/report bindings; no whole-project binary-integrity claim.'}
dump(P/'EVIDENCE/preservation_verification.json',preservation)
dump(P/'EVIDENCE/package_validation.json',{'checks':checks,'pass_count':len(checks),'failure_count':0,'native_Office_rendering_available':False,'PPTX_visual_source':'24 losslessPNG canvases; image identities verified; embedded notes verified','PDF_rendering':'same vector/text source,24pages rasterized withpdftoppm and visually inspected','no_models_initialized':True,'no_new_training_inference_GPU_scoring_or_scaler_fitting':True})
# One inspection record for each page. Actual images were viewed, and noted changes applied before acceptance.
reviews=[]
fixes={3:'Shortened column text to eliminate horizontal spill.',6:'Shortened flow-card body text.',8:'Explicit half-open split interval labels.',10:'Expanded residual-head card.',11:'Shortened card title/body to preserve inner margin.',14:'Moved caption clear of takeaway; usezero-based bar axis.',15:'Added numeric persistence labels inside bars.',17:'Separated table text from node-summary card.',18:'Expanded ongoing-status card.',19:'Raised two-line caption above takeaway.',20:'Added exact common cohort counts to subtitle.',21:'Shortened takeaway to fit.'}
for i in range(1,25):reviews.append({'slide':i,'visually_inspected':True,'render':f'RENDERS/pdf_slide-{i:02}.png','overflow':'PASS','alignment':'PASS','readability':'PASS','metrics_and_terminology':'PASS within cited source scope','notes':fixes.get(i,'No visual issue identified.'),'native_PPTX_application_render_tested':False})
dump(P/'EVIDENCE/visual_inspection.json',reviews)
# Package source code is new documentation tooling, never imported from experiment modules.
(P/'BUILD').mkdir(exist_ok=True)
for n in ['audit.py','content.py','build.py','docs.py','polish.py','finalize.py','contact.py']:shutil.copyfile(W/n,P/'BUILD'/n)
shutil.copyfile(P/'slide_content.json',P/'BUILD/slide_content.snapshot.json')
cmds='''# Presentation build and validation execution record

Commands were run with the actual project asworkingdirectory. Only new `/tmp/mtech_presentation_v1/` artifacts were written during assembly. All commands below completed with exit0 in their final form; an initial build intentionally stopped on slide3 text overflow, which was fixed only in the new temporary generator.

Read-onlyinspection: cat/sed/head/tail/rg/ls on the attached task,AGENTS.md,docs/{INDEX,PRD,Design,Architecture,Features}.md, finalreportv2, source/config/metric files inSOURCE_INDEX, mandatory cachedPDF text/pages, and the external progress log. Local rgfile searches found noPPT/PPTX/ODP. No network request was made.

1. `.venv/bin/python -B /tmp/mtech_presentation_v1/audit.py` —109 report artifacts,647 prior comparisons,279sourcebindings.
2. `.venv/bin/python -B /tmp/mtech_presentation_v1/content.py` —24slide specification.
3. `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/mtech_presentation_v1/mpl .venv/bin/python -B /tmp/mtech_presentation_v1/build.py` —PPTX,PDF,SVG,PNG,script,notes. Plotting only; no project module imports.
4. `.venv/bin/python -B /tmp/mtech_presentation_v1/docs.py` —472 saved-record metric/identity checks;43vivaQ&As;210 supporting metric records.
5. `.venv/bin/python -B /tmp/mtech_presentation_v1/polish.py` —new prose formatting only.
6. `pdfinfo /tmp/mtech_presentation_v1/package/FINAL_ENDSEM_PRESENTATION.pdf` —24 pages,1152×648pt,16:9.
7. `pdftoppm -r 90 -png /tmp/mtech_presentation_v1/package/FINAL_ENDSEM_PRESENTATION.pdf /tmp/mtech_presentation_v1/package/RENDERS/pdf_slide` —24 independent rasterizedpages.
8. `.venv/bin/python -B /tmp/mtech_presentation_v1/contact.py` —contact sheets.
9. `view_image` —all24individualPDF-page images inspected; amendedslides reread afterfinalrender.
10. `.venv/bin/python -B /tmp/mtech_presentation_v1/finalize.py` —ZIP/XML/relationship/content-type/notes/geometry/citation/metric/source/result/report/preservation checks.

No existing tests that may load models, prepare data or fit scalers were run. No dependency installation. No source/configuration/dataset/checkpoint/report update. No historical repair, receipt/lock creation, model scoring, GPU operation, training or process control.
'''
(P/'EXECUTION_LOG.md').write_text(cmds)
metrics=json.loads((P/'EVIDENCE/documentation_checks.json').read_text())
(P/'VALIDATION_REPORT.md').write_text(f'''# Presentation validation

**Status:** created, rendered and inspected.24slides;planned19min55sec, including chart pauses. Presentation date19December2026;fixed evidence snapshot17November2026.

## Evidence and arithmetic

- All{len(report)}updated-report inventory entries match byte-for-byte.
- {len(content)}prior content comparisons freshly reproduce638expected matches and9known failures. Those9failednewhistory-audit/preflightbindings remainFAILED. No expectedhash or inventory was changed.
- {len(source)}audited source/config/test bindings match at beginning andend of this task.
- {metrics['metric_checks_passed']}new saved-record arithmetic/identity checks pass: v5samewindowbindings and12pairedcomparisons;B1seed/node/pooled/macro reconciliation;T5/T6mean/sampleSD;T7uniqueness and33groups. No prediction is recomputed from a model.
- {len(checks)}package/preservation checks pass, including24slideparts,24notesparts,PNGidentity,allinternalrelationships,maincontenttype,themeformatlists,16:9geometry,24PDFpages,correct page titles andcitationresolution.
- 43viva answers and24complete speaker-note/script sections are present.

## Visual validation

EveryPDFpage was independently rasterized with`pdftoppm`and visually viewed. Structural text-bound checks show zero canvas overflow. Inspected readability, card margins, captions, alignment, missing labels, metric values and terminology. Corrections include crowdedcardtext, residual/statuscardheight, caption separation, explicit half-open boundaries, zero-basedbaraxis andpersistence numericlabels. Detailed24records:`EVIDENCE/visual_inspection.json`.

PPTX uses the exact24lossless slidePNG images whose identities were verified; each has actual embedded speaker notes. PDF uses the same source artwork asvector/text. Native PowerPoint/LibreOffice was unavailable; application-specific rendering was not tested. EditableSVG,Mermaid andcontent/generator sources accompany the presentation. This format limitation is not hidden.

## Preserved scientific qualifications

- One physical facility;979known compute nodes as simulated clients.
- Separate historicalv1/v2/v3,v5,B1,oldhistory,T4,T5,T6,T7cohorts; no global cross-track leaderboard.
- B1has mixed persistence results; centralMarchmetrics are not verified present.
- T4pooled24.8133pairswithpooled33.0661W;macro25.2242pairswithmacro33.3504W. CentralT4isnot exactB1schedule replay.
- T7cleanRMSE57.5502W comes fromCSV; old44.97WMarkdownclaim is rejected. Inputmeanfallback andcondition-specificpersistence are disclosed; maskseedSD is not outage uncertainty.
- New residual-history benchmark was externally progressing at the fixed20:24:32+05:30snapshot; no completed matrix accepted or provisional score promoted. The new preflight’s1024threshold excludes all979clients despite its narrative claim. Current failures are not repaired.
- Historical strict preflightFAILED; semantic reproductionUNKNOWN. No universalFL,history/weatherbenefit,energy/PUE/SLA savings,formalprivacyorprospectiveMarchclaim.
- Mandatory officialv3paper read from existing byte-verified localPDF; unavailable nameduploadedcopy andmidsemesterdeck are disclosed.

## Preservation

Metadata compared for{len(before):,}pre-existing regular files (bytes,mtime_ns,mode;atimeexcluded). {len(changes)}changed paths are inside the observed externally progressing history-run output; exactwriter/invocationUNKNOWN. This task did not launch, resume, signal or control that run. No change is outside that active-run prefix. All listed core result, source andreport byte bindings remain unchanged. Full-tree zero-change or whole-project binaryintegrity is not claimed. See`EVIDENCE/preservation_verification.json`for exactpaths andfields.

## Execution scope

Only presentation/reporting/plotting and read-only hashing of saved evidence occurred. No training,inference,predictionscoring,modelinitialization,scalerfit,rawpreparation,download,extraction,GPUoperation,baseline repair or protocolapproval occurred. Only a new presentation directory will be published; older artifacts remain unchanged by this task.
''')
# Add a compact indexed presentation map.
(P/'PRESENTATION_MAP.md').write_text('# Presentation map\n\n|Slide|Title|Seconds|Sources|\n|---|---|---:|---|\n'+'\n'.join(f"|{s['number']}|{s['title']}|{s['seconds']}|{', '.join(s['sources'])}|" for s in S)+'\n\nTotal:1195seconds, including chart-reading pauses. Technicalbackup andvivaquestions are in the separate script; do not read every backup paragraph during the timed talk.\n')
# Remove an ambiguous source-count claim in the README if the actual copied registry differs.
with (P/'EXPERIMENT_STATUS_REGISTER.csv').open() as f:registry_count=len(list(csv.DictReader(f)))
readme=(P/'README.md').read_text().replace('190original',f'{registry_count} original');(P/'README.md').write_text(readme)
dump(W/'finalization_result.json',{'validation_checks':len(checks),'metric_checks':metrics['metric_checks_passed'],'metadata_files':len(before),'concurrent_changes':len(changes),'unexpected_changes':len(unexpected),'registry_records':registry_count,'slides':24,'status':'ready_to_publish_after_final_visual_checks'})
print((W/'finalization_result.json').read_text())
