"""Documentation/content/format checks only. No model or evaluation imports."""
import argparse,ast,hashlib,json,re,subprocess,tempfile,zipfile
from pathlib import Path
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

FINAL='output/experiments/v3_centralized_sgd/reports/final_test_comparison_v1'
VALID='output/experiments/v3_centralized_sgd/reports/validation_comparison_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def run(directory,root):
 directory=Path(directory);root=Path(root);passed=[]
 def check(value,name):
  if not value:raise ValueError('Documentation check failed: '+name)
  passed.append(name)
 def read(p):return json.loads(p.read_text())
 doc=read(directory/'DOCUMENT.json');evidence=read(directory/'EVIDENCE.json');sources=read(directory/'SOURCES.json')['sources']
 md=(directory/'THESIS_METHODS_RESULTS.md').read_text();ht=(directory/'THESIS_METHODS_RESULTS.html').read_text()
 check(all(sha(root/s['path'])==s['sha256'] for s in sources),'82 direct source hashes match current repository')
 check(len(sources)==82 and len({s['id'] for s in sources})==82,'Source inventory complete and uniquely identified')
 ids={s['id'] for s in sources};citations=set(re.findall(r'\[(S\d+)\]',md))
 check(citations<=ids and len(citations)>30,'All repository citations resolve')
 check(not any(x in md for x in ['TODO','TBD','[INSERT','<insert']),'No unresolved placeholders')
 check(len(doc['blocks'])==118 and len([b for b in doc['blocks'] if b['kind']=='table'])==17,'Document sections and 17 tables present')
 meta=read(root/'output/experiments/v1/frozen/preprocessing.json');contract=read(root/'output/experiments/v1/frozen/contract.json')
 check(evidence['sample_counts']==meta['sample_counts']=={'train':4506133,'val':199670,'test':75566},'Cohort counts copied exactly')
 rows=evidence['numeric_tables']['features']['rows']
 check(len(rows)==14 and [r[1] for r in rows]==meta['feature_names']==contract['feature_names'],'Exact zero-based 14-feature order')
 check([int(r[0]) for r in rows]==list(range(14)) and [r[3] for r in rows]==meta['feature_units'],'Feature indices and units match')
 check([int(r[0]) for r in rows if r[-1]=='Yes']==[7,8] and rows[6][-1]=='No','Outdoor-only masking and indoor retention explicit')
 val=read(root/(VALID+'/comparison.json'));a=read(root/(FINAL+'/analysis.json'))
 check(evidence['validation']==val and evidence['test']==a,'All tabulated evidence equals saved validation/test JSON')
 for kind,values in [('validation',val),('test',a)]:
  expected=[]
  if kind=='validation':
   for i,seed in enumerate([42,43,44]):
    c,f=values['centralized_sgd'][i],values['fedavg'][i]
    expected.append([str(seed),str(c['selected_block']),f"{c['rmse_w']:.6f}",f"{c['mae_w']:.6f}",str(f['selected_round']),f"{f['rmse_w']:.6f}",f"{f['mae_w']:.6f}"])
  else:
   for i,seed in enumerate([42,43,44]):
    c,f=(values['per_seed'][m][i] for m in ['centralized_sgd','fedavg'])
    expected.append([str(seed),str([190,190,180][i]),f"{c['rmse_w']:.6f}",f"{c['mae_w']:.6f}",'200',f"{f['rmse_w']:.6f}",f"{f['mae_w']:.6f}"])
  check(evidence['numeric_tables'][kind+'_seeds']['rows']==expected,'Per-seed '+kind+' values and locked selections exact')
 check('ddof' not in md or 'sample' in md,'Sample standard deviation convention present')
 check(all(s in md.casefold() for s in ['sample standard deviations','not independent datasets','not a confidence interval','no best test seed is selected']),'Seed dependence and no selection clearly stated')
 check('not untouched confirmatory' in md and 'test period has already been observed' in md,'Historical test exposure disclosed')
 check(all(s.casefold() in md.casefold() for s in ['prediction does not prove energy savings','no measured kWh reduction','No measured carbon savings','fully privacy-preserving deployment were not established']),'Energy/carbon/privacy limits explicit')
 check(all(s.casefold() in md.casefold() for s in ['Proposed future work — not executed','6, 36 and 144','intersection of eligible node/origin','Controlled environmental-feature ablation','Separate energy-management optimization study']),'Three required future-work stages labeled proposed')
 check(all(s.casefold() in md.casefold() for s in ['Pooled MAE','Equal-node mean RMSE','94 distinct forecast origins','six calendar dates']),'Metric aggregation and coverage caveats present')
 check(evidence['architecture']=={'recurrent_input_width':21,'hidden_size':64,'num_layers':2,'node_embedding_size':8,'node_count':979},'Recorded architecture dimensions correct')
 for entry in sources:
  if entry['path'].endswith('/manifest.json') and '/main/' in entry['path']:
   config=read(root/entry['path'])['config']
   check(config['model']=='gru' and config['weather']=='masked' and config['hidden_size']==64 and config['num_layers']==2 and config['node_embedding_size']==8,'Architecture/configuration matches '+entry['id'])
 class Assets(HTMLParser):
  def __init__(self):super().__init__();self.images=[]
  def handle_starttag(self,tag,attrs):
   if tag=='img':self.images.append(dict(attrs)['src'])
 assets=Assets();assets.feed(ht)
 check(assets.images==['figures/validation_curves.png','figures/per_node_comparison.png'],'HTML has two local figure links')
 check(sha(directory/assets.images[0])==sha(root/(VALID+'/validation_curves.png')) and sha(directory/assets.images[1])==sha(root/(FINAL+'/per_node_comparison.png')),'Copied figures byte-identical to original figures')
 n={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 with zipfile.ZipFile(directory/'THESIS_METHODS_RESULTS.docx') as z:
  check(z.testzip() is None,'Word package ZIP CRCs valid')
  for name in z.namelist():
   if name.endswith(('.xml','.rels')):ET.fromstring(z.read(name))
  document=ET.fromstring(z.read('word/document.xml'))
  text='\n'.join(t.text or '' for t in document.findall('.//w:t',n))
  check(len(document.findall('.//w:tbl',n))==17,'Word contains all 17 tables')
  check(all(cell in text for table in evidence['numeric_tables'].values() for row in table['rows'] for cell in row),'Every table cell survives Word export')
  check(all((b['text'] in text) for b in doc['blocks'] if b['kind'] in ['paragraph','equation','heading']),'All narrative and equations survive Word export')
  check(all(z.read(f'word/media/image{i+1}.png')==(directory/path).read_bytes() for i,path in enumerate(assets.images)),'Word embeds exact source figures')
  rels=ET.fromstring(z.read('word/_rels/document.xml.rels'))
  check(all('word/'+rel.attrib['Target'] in z.namelist() for rel in rels),'Word image/style relationships resolve')
 with tempfile.TemporaryDirectory(prefix='m100-thesis-check-') as temp:
  temp=Path(temp)
  subprocess.run(['pdftotext','-layout',str(directory/'THESIS_METHODS_RESULTS.pdf'),str(temp/'text.txt')],check=True)
  subprocess.run(['pdftotext','-bbox',str(directory/'THESIS_METHODS_RESULTS.pdf'),str(temp/'bbox.html')],check=True)
  pdftext=(temp/'text.txt').read_text();bbox=ET.parse(temp/'bbox.html');pages=bbox.findall('.//{http://www.w3.org/1999/xhtml}page')
  check(bool(pages) and all(590<float(p.attrib['width'])<600 and 835<float(p.attrib['height'])<850 for p in pages),'PDF renders to '+str(len(pages))+' A4 pages')
  check(all(number in pdftext for number in ['56.016953','63.231179','30.516738','33.924637','66.304353','68.522209','4,506,133','199,670','75,566']),'PDF contains the exact primary numeric results')
  check('not untouched confirmatory' in ' '.join(pdftext.split()),'PDF retains observed-test limitation')
  outside=[]
  for i,page in enumerate(pages,1):
   width,height=float(page.attrib['width']),float(page.attrib['height'])
   for word in page.findall('.//{http://www.w3.org/1999/xhtml}word'):
    v=word.attrib
    if float(v['xMin']) < -1 or float(v['yMin']) < -1 or float(v['xMax']) > width+1 or float(v['yMax']) > height+1:outside.append((i,word.text))
  check(not outside,'PDF text bounding boxes remain inside all pages')
 for file in [directory/'code/build_documents.py',directory/'code/check_documentation.py']:
  if not file.exists():file=directory.parent/file.name
  tree=ast.parse(file.read_text())
  forbidden={'torch','tensorflow','xgboost','sklearn','numpy','src'}
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):check(not any(n.name.split('.')[0] in forbidden for n in node.names),'Documentation code excludes model/evaluation imports: '+file.name)
   if isinstance(node,ast.ImportFrom):check((node.module or '').split('.')[0] not in forbidden,'Documentation-only imports: '+file.name)
 return {'status':'passed','documentation_checks_passed':len(passed),'checks':passed,'direct_sources_verified':len(sources),
  'pdf_pages':len(pages),'tables':17,'model_training_executed':False,'model_inference_executed':False,'evaluation_rerun':False}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--project-root',required=True);args=p.parse_args()
 print(json.dumps(run(args.directory,args.project_root),indent=2))
