"""Read-only thesis-package and source-preservation check. No models or arrays."""
from pathlib import Path
import hashlib,json,stat

def verify(directory):
 directory=Path(directory).resolve();baseline=json.loads((directory/'PRESERVATION_BASELINE.json').read_text())
 receipt=json.loads((directory/'THESIS_COMPLETED.json').read_text());root=Path(baseline['project_root']).resolve()
 sources=json.loads((directory/'SOURCES.json').read_text())['sources'];bad=[];cache={}
 def sha(path):
  if path not in cache:
   h=hashlib.sha256()
   with path.open('rb') as f:
    for b in iter(lambda:f.read(4194304),b''):h.update(b)
   cache[path]=h.hexdigest()
  return cache[path]
 def checked(base,name,expected):
  p=base/name
  if Path(name).is_absolute() or '..' in Path(name).parts or not p.resolve().is_relative_to(base):bad.append('unsafe path: '+name);return None
  if not p.is_file():bad.append('missing: '+str(p));return None
  if sha(p)!=expected:bad.append('hash: '+str(p))
  return p
 actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name!='THESIS_COMPLETED.json'}
 if actual!=set(receipt['artifacts']):bad.append('Thesis artifact inventory differs')
 if receipt['status']!='complete':bad.append('Thesis package incomplete')
 for name,value in receipt['artifacts'].items():checked(directory,name,value)
 for name,value in baseline['files'].items():
  p=checked(root,name,value['sha256'])
  if p:
   s=p.stat()
   if (s.st_size,s.st_mtime_ns,stat.S_IMODE(s.st_mode))!=(value['size'],value['mtime_ns'],value['mode']):bad.append('metadata: '+name)
 for s in sources:checked(root,s['path'],s['sha256'])
 return {'status':'verified_complete' if not bad else 'failed','pre_existing_files_checked':len(baseline['files']),
  'thesis_artifacts_checked':len(receipt['artifacts']),'direct_sources_checked':len(sources),'mismatches':bad,
  'training_executed':False,'model_inference_executed':False,'evaluation_rerun':False}

if __name__=='__main__':
 result=verify(Path(__file__).resolve().parents[1]);print(json.dumps(result,indent=2));raise SystemExit(bool(result['mismatches']))
