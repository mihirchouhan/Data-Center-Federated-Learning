from pathlib import Path
import json,hashlib,ast,csv,shutil,zipfile,xml.etree.ElementTree as ET,datetime
R=Path('/media/csedept/cse2014/Project/m100_training_template');W=Path('/tmp/mtech_presentation_v1');P=W/'package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
S=json.loads((P/'slide_content.json').read_text());source=json.loads((P/'EVIDENCE/source_hash_verification.json').read_text());structure=[]
for e in source:
 p=R/e['path']
 if p.suffix!='.py':continue
 tree=ast.parse(p.read_text());structure.append({'path':e['path'],'sha256':e['actual'],'classes':[n.name for n in ast.walk(tree) if isinstance(n,ast.ClassDef)],'functions':[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],'inspection':'AST only; no module import or execution'})
(P/'EVIDENCE/code_interface_inventory.json').write_text(json.dumps(structure,indent=2)+'\n')
# Final archive and speaker-note test after typography cleanup. No Office or model execution.
with zipfile.ZipFile(P/'FINAL_ENDSEM_PRESENTATION.pptx') as z:
 assert z.testzip() is None
 for i,s in enumerate(S,1):
  text=' '.join(ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml')).itertext())
  assert s['say'] in text and s['technical'] in text and s['faculty_question'] in text
  assert hashlib.sha256(z.read(f'ppt/media/image{i}.png')).hexdigest()==sha(P/f'SLIDES/slide_{i:02}.png')
script=(P/'PRESENTATION_SCRIPT.md').read_text();assert script.count('## Slide ')==24 and '25CSE2014' in script
viva=(P/'VIVA_QUESTIONS_AND_ANSWERS.md').read_text();assert sum(l.startswith('## ') for l in viva.splitlines())==43
assert all(not r['canvas_text_overflow'] for r in json.loads((P/'EVIDENCE/layout_checks.json').read_text()))
# Copy final generators and corrected content snapshot to the package.
for n in ['finish_documents.py','seal.py']:shutil.copyfile(W/n,P/'BUILD'/n)
shutil.copyfile(P/'slide_content.json',P/'BUILD/slide_content.snapshot.json')
(P/'FIGURES/README.md').write_text('''# Figure and diagram index

Every slide has an editable SVG named `slide_XX.svg`, corresponding to the PNG in `SLIDES/` and the same page in the PDF. These preserve the complete layout and vector charts; edit copies, not this sealed inventory.

- 02: forecasting-to-decision boundary.
- 04–05: base-paper method and research comparison.
- 06: implemented pipeline; `system_architecture.mmd` gives a Mermaid source.
- 07–08: feature/binning and temporal-window contracts.
- 09: simulated clients and node embedding.
- 10: B1 versus T4 architecture; `model_architecture.mmd`.
- 11: FedAvg and full-table embedding dilution; `federated_workflow.mmd`.
- 14: legacy LSTM and distinct v5 missing-data comparisons.
- 15: B1 two-output persistence comparison.
- 16–17: T4 canonical and temporal results.
- 18: completed old history panel, with new benchmark status separate.
- 19: T5 weather and T6 model-family panels.
- 20–21: T7 synthetic missingness and node consistency.
- 22: proposed decision-support stages.

Numerical sources and hashes resolve through `SOURCE_INDEX.md`; supporting scores are in `PRESENTATION_METRICS.csv` and `B1_SUPPORTING_METRICS.csv`. Error bars represent sample standard deviation across three model seeds, not confidence intervals. No provisional ongoing-benchmark score is included.
''')
log=P/'EXECUTION_LOG.md';old=log.read_text();old+='\n\nFinal publication preparation: `finish_documents.py` corrected only presentation prose and embedded notes; `seal.py` rechecked all 24 notes/images, 24 script sections and 43 viva answers. It also parsed the recorded Python sources using AST only. No experiment modules were imported. `publish.py` performs an unused-destination check, fresh scoped byte verification, exclusive staging, SHA-256 copy verification and atomic `renameat2(RENAME_NOREPLACE)` publication. The post-publication check reads the resulting inventory only. All these final commands completed with exit code 0 when recorded in the final response.\n';log.write_text(old)
completion={'status':'presentation_complete_with_documented_limitations','presentation_date':'2026-12-19','evidence_date':'2026-11-17','slides':24,'planned_seconds':1195,'viva_questions':43,'script_sections':24,'stored_metric_records':210,'stored_metric_identity_arithmetic_checks_passed':472,'package_preservation_checks_passed':1367,'all_slides_visually_inspected':True,'native_Office_application_rendering_tested':False,'PPTX_type':'lossless image slides with embedded speaker notes; editable SVG/content/generators supplied','PDF_type':'same artwork with vector graphics and text','missing_inputs':['mid-semester deck not found','named uploaded mandatory paper copy not found; official hash-verified arXiv v3 PDF used'],'historical_preflight':'FAILED unchanged','semantic_reproduction':'UNKNOWN; not executed','known_failed_new_manifest_comparisons_preserved':9,'new_model_training_inference_GPU_scoring_or_scaler_fit':False,'existing_project_files_written_by_this_task':False,'concurrent_active_run_log_changes':2,'unexpected_metadata_changes':0,'publication_target':str(R/'output/final_endsem_presentation_v1'),'inventory_policy':'generated_files.json hashes all other package files; no recursive self-hash'}
(P/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
files=[]
for p in sorted(P.rglob('*')):
 if p.is_file() and p.name!='generated_files.json':files.append({'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'schema':'mtech.presentation.inventory.v1','presentation_date':'2026-12-19','evidence_date':'2026-11-17','files':files,'excluded_self':'generated_files.json','existing_experiments_modified':False};(P/'generated_files.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files_including_manifest':len(files)+1,'PPTX_sha256':sha(P/'FINAL_ENDSEM_PRESENTATION.pptx'),'PDF_sha256':sha(P/'FINAL_ENDSEM_PRESENTATION.pdf'),'manifest_sha256':sha(P/'generated_files.json'),'AST_source_files':len(structure),'ready_to_publish':True},indent=2))
