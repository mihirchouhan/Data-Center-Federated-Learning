from pathlib import Path
import re,json
P=Path('/tmp/mtech_presentation_v1/package')
# Presentation prose only. Never run over source evidence, metricCSV, hashes or dataset paths.
def prose(t):
 terms=[]
 pattern=r'\b(?:FedAvg|FedProx|XGBoost|Marconi100|M100|B1|T[1-7]|Q1|v[1-5]|float32|float64|FF10|FF20|FF30|SeedSequence|drop_last|ddof|lr)'
 def protect(m):
  token=chr(0xe000+len(terms));terms.append(m.group());return token
 t=re.sub(pattern,protect,t)
 # Separate protected scientific tokens from any accidentally joined English words.
 t=re.sub(r'([\w])([\ue000-\uf8ff])',r'\1 \2',t)
 t=re.sub(r'([\ue000-\uf8ff])([A-Za-z0-9])',r'\1 \2',t)
 t=re.sub(r'([a-z])([A-Z])',r'\1 \2',t)
 t=re.sub(r'([A-Z]{2,})([A-Z][a-z])',r'\1 \2',t)
 t=re.sub(r'([A-Z]{2,})([a-z])',r'\1 \2',t)
 t=re.sub(r'([A-Za-z])(\d)',r'\1 \2',t)
 t=re.sub(r'(\d)([A-Za-z])',r'\1 \2',t)
 for a,b in {'withbatch':'with batch','atleast':'at least','atmost':'at most','atlr':'at lr','andceil':'and ceil','outside[':'outside [','spanning[':'spanning [','recordedIPMI':'recorded IPMI','in°C':'in °C','with`':'with `','`and':'` and','`for':'` for','reporting/plotting':'reporting / plotting','not190':'not 190','andend':'and end','preflightbindings':'preflight bindings','samewindowbindings':'same-window bindings','pairedcomparisons':'paired comparisons','slideparts':'slide parts','notesparts':'notes parts','allinternalrelationships':'all internal relationships','maincontenttype':'main content type','themeformatlists':'theme format lists','citationresolution':'citation resolution','crowdedcardtext':'crowded card text','cardheight':'card height','baraxis':'bar axis','numericlabels':'numeric labels','asvector':'as vector','oldhistory':'old history','centralMarchmetrics':'central March metrics','pairswithpooled':' pairs with pooled','pairswithmacro':' pairs with macro','isnot':'is not','Inputmeanfallback':'Input mean fallback','nameduploadedcopy':'named uploaded copy','midsemesterdeck':'mid-semester deck','atimeexcluded':'atime excluded','exactwriter':'exact writer','andreport':'and report','binaryintegrity':'binary integrity','exactpaths':'exact paths','andfields':'and fields','predictionscoring':'prediction scoring','modelinitialization':'model initialization','scalerfit':'scaler fit','rawpreparation':'raw preparation','baseline repair':'baseline repair','protocolapproval':'protocol approval','samefile':'same file','every backup paragraph':'every backup paragraph','withinthis':'within this','sourcepopulation':'source population','allthree':'all three'}.items():t=t.replace(a,b)
 for i,term in enumerate(terms):t=t.replace(chr(0xe000+i),term)
 t=t.replace('42,43,44','42, 43, 44').replace('0,10,','0, 10, ').replace('98 of979','98 of 979')
 return t
def markdown(t):
 # Do not alter inline code/file paths or full code blocks.
 return ''.join(x if i%2 else prose(x) for i,x in enumerate(re.split(r'(`[^`]*`)',t)))
for n in ['README.md','VIVA_QUESTIONS_AND_ANSWERS.md','PRESENTATION_SCRIPT.md','VALIDATION_REPORT.md','PRESENTATION_MAP.md','EXECUTION_LOG.md']:
 p=P/n
 if p.exists():p.write_text(markdown(p.read_text()))
s=json.loads((P/'slide_content.json').read_text())
for slide in s:
 for key in ['technical','faculty_question']:slide[key]=prose(slide[key])
(P/'slide_content.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
