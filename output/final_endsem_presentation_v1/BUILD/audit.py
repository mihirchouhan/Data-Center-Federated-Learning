from pathlib import Path
import hashlib,json,csv,os,datetime
R=Path('/media/csedept/cse2014/Project/m100_training_template'); W=Path('/tmp/mtech_presentation_v1'); P=W/'package'; P.mkdir(exist_ok=True)
for d in ['EVIDENCE','FIGURES','SLIDES','RENDERS']: (P/d).mkdir(exist_ok=True)
D=R/'output/final_endsem_report_v2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def inventory():
 out={}
 for root,ds,fs in os.walk(R):
  ds[:]=[x for x in ds if x not in ['.git','.venv','__pycache__','.pytest_cache']]
  for n in fs:
   p=Path(root)/n
   if p.is_symlink() or not p.is_file():continue
   s=p.stat();out[str(p.relative_to(R))]=[s.st_size,s.st_mtime_ns,s.st_mode&0o7777]
 return out
before=inventory();dump(W/'before_metadata.json',before)
report_manifest=json.loads((D/'generated_files.json').read_text());rv=[]
for e in report_manifest['files']:
 p=D/e['path'];actual=sha(p);rv.append({'path':str(p.relative_to(R)),'expected':e['sha256'],'actual':actual,'match':actual==e['sha256']})
assert all(x['match'] for x in rv),'REPORT PACKAGE MISMATCH'
old=json.loads((D/'EVIDENCE/UPDATE_V2/content_checks.json').read_text()); cache={};checks=[]
for e in old:
 p=R/e['path']
 if str(p) not in cache:cache[str(p)]=sha(p)
 checks.append({**e,'fresh_actual':cache[str(p)],'matches_prior_actual':cache[str(p)]==e['actual'],'fresh_matches_expected':cache[str(p)]==e['expected']})
assert all(e['matches_prior_actual'] for e in checks),'NEW PROTECTED CONTENT DISCREPANCY'
sourcebase=json.loads((D/'EVIDENCE/UPDATE_V2/audited_source_hashes.json').read_text());sources=[]
for rel,e in sourcebase.items():
 p=R/rel;h=sha(p);sources.append({'path':rel,'expected':e['sha256'],'actual':h,'match':h==e['sha256']})
assert all(e['match'] for e in sources),'SOURCE CHANGED SINCE REVIEW'
dump(P/'EVIDENCE/report_package_verification.json',rv);dump(P/'EVIDENCE/fresh_content_checks.json',checks);dump(P/'EVIDENCE/source_hash_verification.json',sources)
for n in ['EXPERIMENT_STATUS_REGISTER.csv','LATEST_RESEARCH_STATUS.csv','UNVERIFIED_CLAIMS.csv']:(P/n).write_bytes((D/n).read_bytes())
log=R/'output/experiments/t1_2_controlled_history_benchmark_v1/progress.log';snap=log.read_text();(P/'EVIDENCE/ongoing_history_log_snapshot.txt').write_text(snap)
status={'snapshot_time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),'complete_matrix_claimed':False,'progress_log_sha256_at_read':hashlib.sha256(snap.encode()).hexdigest(),'last_lines':snap.splitlines()[-7:],'note':'Externally running experiment. Snapshot only; no provisional scores used. No process launched or controlled.'};dump(P/'EVIDENCE/ongoing_history_status.json',status)
paper=Path('/tmp/endsem_report_work_v1/base_paper_arxiv_v3.pdf');assert sha(paper)=='c081a18f0a7ac2ea920698904734f667981fe2f707fbb5d1d869efe8ec4d8aef'
dump(P/'EVIDENCE/audit_summary.json',{'report_artifacts_checked':len(rv),'prior_content_comparisons':len(checks),'fresh_expected_pass':sum(x['fresh_matches_expected'] for x in checks),'known_unrepaired_failures':sum(not x['fresh_matches_expected'] for x in checks),'all_match_prior_observation':True,'source_bindings':len(sources),'paper_sha256':sha(paper),'midsemester_deck':'NOT FOUND in repository, Project roots, Downloads, Documents, attachments and /tmp search','no_model_execution':True,'whole_tree_metadata_before_count':len(before)})
print(json.dumps({'report_artifacts':len(rv),'content_comparisons':len(checks),'known_failed':sum(not x['fresh_matches_expected'] for x in checks),'source_checks':len(sources),'snapshot':status},indent=2))
