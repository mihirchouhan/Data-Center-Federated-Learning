from pathlib import Path
import json,csv,statistics as st,math,html,textwrap,zipfile,hashlib,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
R=Path('/media/csedept/cse2014/Project/m100_training_template');W=Path('/tmp/mtech_presentation_v1');P=W/'package';D=R/'output/final_endsem_report_v2'
S=json.loads((P/'slide_content.json').read_text())
NAVY='#142C43';TEAL='#007E87';ORANGE='#D87B36';GRAY='#526575';BG='#F6F8FA';PALE='#E4F1F0';RED='#AD3F3F';BLUE='#4783AF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':22,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GRAY,'ytick.color':GRAY,'svg.fonttype':'none','figure.max_open_warning':0})
texts=[];figs=[];checks=[]
def txt(ax,x,y,s,size=25,color=NAVY,weight='normal',ha='left',va='top',**kw):
 t=ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,transform=ax.transAxes,**kw);texts.append((len(figs)+1,t));return t
def box(ax,x,y,w,h,title,body='',fill='white',edge='#D9E1E8',title_size=24,body_size=21):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.013',facecolor=fill,edgecolor=edge,linewidth=1.1,transform=ax.transAxes))
 txt(ax,x+.018,y+h-.028,title,title_size,weight='bold')
 if body:txt(ax,x+.018,y+h-.095,body,body_size,linespacing=1.5)
def arrow(ax,x1,y1,x2,y2,color=TEAL):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=22,lw=2,color=color,transform=ax.transAxes))
def frame(s):
 f=plt.figure(figsize=(16,9),facecolor=BG);ax=f.add_axes([0,0,1,1]);ax.set_axis_off()
 ax.add_patch(Rectangle((0,.984),1,.016,color=TEAL,transform=ax.transAxes))
 txt(ax,.045,.945,s['section'],14,TEAL,weight='bold');txt(ax,.045,.884,s['title'],33,weight='bold');txt(ax,.045,.809,s['subtitle'],18,GRAY)
 ax.plot([.045,.955],[.763,.763],color='#D5DEE5',lw=1,transform=ax.transAxes)
 txt(ax,.045,.032,'Mihir Chouhan · NIT Goa  |  Evidence: 17 Nov 2026',11,GRAY,va='bottom')
 txt(ax,.955,.032,f"{s['number']:02d} / 24",12,GRAY,ha='right',va='bottom')
 txt(ax,.045,.073,'Sources: '+', '.join(s['sources']),10,GRAY,va='bottom')
 return f,ax
def take(ax,s,color=TEAL):
 ax.add_patch(FancyBboxPatch((.045,.108),.91,.074,boxstyle='round,pad=0.008',facecolor=PALE,edgecolor='none',transform=ax.transAxes));txt(ax,.061,.145,s,21,color,weight='bold',va='center')
def axes(f,pos):
 ax=f.add_axes(pos);ax.set_facecolor('none');return ax
def styleplot(ax,xlabel='',ylabel=''):
 ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True);ax.tick_params(labelsize=17);ax.set_xlabel(xlabel,fontsize=20);ax.set_ylabel(ylabel,fontsize=20);ax.spines['left'].set_color('#ADBCC7');ax.spines['bottom'].set_color('#ADBCC7')
def table(ax,headers,rows,x=.055,y=.70,width=.89,colwidth=None,rowh=.075,fontsize=21):
 colwidth=colwidth or [1/len(headers)]*len(headers);xs=[x]
 for w in colwidth:xs.append(xs[-1]+width*w)
 for j,h in enumerate(headers):txt(ax,xs[j]+.008,y,h,fontsize,TEAL,weight='bold')
 for i,row in enumerate(rows):
  yy=y-(i+1)*rowh
  ax.add_patch(Rectangle((x,yy-rowh*.80),width,rowh*.98,facecolor='white' if i%2==0 else '#EEF2F5',edgecolor='none',transform=ax.transAxes))
  for j,c in enumerate(row):txt(ax,xs[j]+.008,yy-.006,str(c),fontsize,weight='bold' if j==0 else 'normal')
def csvread(p):return list(csv.DictReader(p.open()))
v5=csvread(D/'EVIDENCE/v5_seedwise_metrics.csv');hist=csvread(D/'EVIDENCE/history_seedwise_metrics.csv');t4=csvread(R/'output/experiments/t4_model_improvement_v1/final_results_by_seed.csv');t7=csvread(D/'TABLES/T7_ALL_CONDITIONS.csv');late=json.loads((D/'EVIDENCE/late_results.json').read_text());b1=json.loads((R/'output/experiments/b1_no_ff/march_test_evaluation_v2/aggregate_metrics.json').read_text())
def group(rows,k,v):return st.mean(float(r[v]) for r in rows if all(r[a]==b for a,b in k.items()))
def sd(rows,k,v):return st.stdev(float(r[v]) for r in rows if all(r[a]==b for a,b in k.items()))
def gt7(c):return next(r for r in t7 if r['period']=='test_t1' and r['condition_id']==c)
# 01
s=S[0];f,ax=frame(s)
txt(ax,.06,.695,'Federated Forecasting and Energy-Aware\nDecision Support for Distributed Data Centers\nunder Heterogeneous and Missing Telemetry',29,weight='bold',linespacing=1.35)
box(ax,.06,.27,.55,.14,'Mihir Chouhan · 25CSE2014','M.Tech · Computer Science and Engineering · NIT Goa',title_size=24,body_size=18)
txt(ax,.06,.225,'Supervisor: Dr. Pravati Swain',22);txt(ax,.68,.375,'19 DECEMBER\n2026',30,TEAL,weight='bold',linespacing=1.4)
txt(ax,.68,.21,'Evidence snapshot\n17 November 2026',18,GRAY,linespacing=1.4)
figs.append(f)
#02
f,ax=frame(S[1]);x=[.06,.29,.52,.75]
for i,(title,body) in enumerate([('Telemetry','Power · CPU · memory\nThermal · weather'),('Forecast','Next-hour means\nCoverage + errors'),('Decision','Actions + constraints\nValidated response'),('Outcome','Energy · cost · service\nControlled comparison')]):
 box(ax,x[i],.38,.185,.28,title,body,fill='white' if i<2 else '#EDF0F3',title_size=23,body_size=18)
 if i<3:arrow(ax,x[i]+.19,.52,x[i+1]-.008,.52)
txt(ax,.06,.29,'MEASURED FORECASTING',18,TEAL,weight='bold');txt(ax,.52,.29,'PROPOSED OPERATIONAL RESEARCH',18,GRAY,weight='bold');take(ax,'A forecast error is not an energy-saving measurement.');figs.append(f)
#03
f,ax=frame(S[2])
for i,(a,b) in enumerate([('01  Training approach?','FedAvg vs central controls\nPersistence on same windows'),('02  Forecasting choices?','History · residuals · weather\nModels · missing telemetry'),('03  Defensible claims?','Separate units and cohorts\nTraceable results; clear limits')]):box(ax,.065+i*.303,.32,.275,.38,a,b,title_size=20,body_size=19)
take(ax,'Contribution: a controlled telemetry study with traceable evidence bindings.');figs.append(f)
#04
f,ax=frame(S[3]);box(ax,.06,.37,.27,.32,'MLP ensemble','256 → 1024 → 256 → output\nReLU · dropout 0.15\nFour affine layers',body_size=20)
box(ax,.365,.37,.27,.32,'Adaptive FL','Share selected blocks\nRecent-error acceptance\nLocal ensemble decisions',body_size=20)
box(ax,.67,.37,.27,.32,'Reduced optimization','Generators · storage · grid\nPower / heat / workload\nFive modeled data centers',body_size=20)
arrow(ax,.335,.53,.36,.53);arrow(ax,.64,.53,.665,.53);txt(ax,.075,.265,'Paper: adaptive R² = 0.951–0.988 across five agents (Table II, p. 9)',21);take(ax,'The supervised labels are operating decisions—not next-hour sensor means.');figs.append(f)
#05
f,ax=frame(S[4]);table(ax,['Dimension','Base paper','This project'],[
['Local model','MLP ensemble','GRU / LSTM + node ID'],['Federated method','Adaptive block acceptance','Standard full-table FedAvg'],['Labels','Optimal operating decisions','Observed future power / ambient'],['Physical decisions','Reduced optimization solver','Not implemented / evaluated'],['Reported quantities','R² · cost deviation · solver time','MAE / RMSE in W and °C']],colwidth=[.24,.36,.40],fontsize=20,rowh=.081)
take(ax,'R², cost error and watt MAE are not directly comparable.');figs.append(f)
#06
f,ax=frame(S[5]);xx=[.065,.30,.535,.77]
for i,(a,b) in enumerate([('Raw telemetry','Parquet + mappings\nUTC observations'),('Causal windows','Bins + IDs + targets\nTrain-only scaling'),('Fit + select','Local fit / aggregate\nValidation → lock'),('Evaluate + audit','Frozen predictions\nPooled + node errors')]):
 box(ax,xx[i],.43,.18,.24,a,b,title_size=20,body_size=16.8)
 if i<3:arrow(ax,xx[i]+.185,.55,xx[i+1]-.008,.55)
box(ax,.18,.235,.64,.10,'Source hashes → manifests → receipts → result provenance',fill=PALE,title_size=21)
take(ax,'Implemented: forecasting + evaluation. Future: validated decision layer.');figs.append(f)
#07
f,ax=frame(S[6]);box(ax,.06,.35,.42,.36,'14 input slots','0–6   Power; CPU user/system/idle/wio;\n          memory-used fraction; IPMI ambient\n7–8   Outdoor temperature / humidity\n9–12 Calendar sin/cos; 13 categorical node ID',body_size=20)
box(ax,.52,.35,.42,.36,'Causal validity rules','10-minute UTC bins\nSensor hold ≤ 300 s; coverage ≥ 50%\nMemory = 1 − aligned free / total\nWeather: strictly earlier, age ≤ 30 min',body_size=21)
take(ax,'Missing-data policies are track-specific; future target values remain observed.');figs.append(f)
#08
f,ax=frame(S[7]);box(ax,.075,.47,.54,.19,'History: [t − 6 h, t)','36 completed ten-minute bins',fill=PALE,title_size=25,body_size=22);box(ax,.655,.47,.27,.19,'Target: [t, t + 1 h)','Mean of 6 observed bins',fill='#F9EBDD',title_size=23,body_size=20);arrow(ax,.62,.56,.65,.56)
txt(ax,.635,.72,'origin t',20,ORANGE,weight='bold',ha='center');table(ax,['B1 training','B1 validation','B1 retrospective test'],[['[1 Jan, 28 Jan)','[28 Jan, 4 Feb)','[1 Mar, 1 Apr)']],y=.37,fontsize=20,rowh=.07)
txt(ax,.07,.24,'Whole history + whole target contained in each B1 split; hourly origins.',22)
take(ax,'B1 containment differs from historical tracks with prior-split histories.');figs.append(f)
#09
f,ax=frame(S[8]);box(ax,.08,.43,.25,.25,'Node client k','Original literal node ID\nLocal windows only\nOrdinal never renumbered',body_size=21)
box(ax,.39,.43,.23,.25,'Shared model','979 × 8 embedding\n13 numeric + 8 identity\n21 recurrent values',body_size=21)
box(ax,.69,.43,.23,.25,'Coordinator','Uniform client sampling\nGlobal validation\nFull-state aggregation',body_size=20);arrow(ax,.335,.55,.385,.55);arrow(ax,.625,.55,.685,.55)
txt(ax,.09,.29,'Unequal coverage and node errors → client heterogeneity',24,weight='bold');take(ax,'979 simulated clients ≠ 979 physical data centers or unseen-site evidence.');figs.append(f)
#10
f,ax=frame(S[9]);box(ax,.07,.39,.25,.31,'Shared encoder','[B, 36, 14] input slots\nID → 8-D embedding\n2 × GRU(hidden=64)\nLast recurrent state',body_size=21)
box(ax,.41,.56,.49,.14,'B1: direct two-output head','Linear(64, 2) → [power W, ambient °C]',title_size=23,body_size=21)
box(ax,.41,.30,.49,.21,'T4: residual power head','Linear(64, 1) → standardized correction\nPower prediction = last power + correction',title_size=23,body_size=20)
arrow(ax,.325,.61,.40,.63);arrow(ax,.325,.48,.40,.41);take(ax,'B1: standardized MSE + SGD.  T4: residual L1 + Adam.  Separate checkpoints.');figs.append(f)
#11
f,ax=frame(S[10]);box(ax,.06,.40,.27,.30,'1  Broadcast + sample','Same global state → clients\nUniform without replacement\nOriginal 979-node mapping',body_size=20)
box(ax,.365,.40,.27,.30,'2  Local optimization','One epoch; batch 128 (B1)\nFresh SGD per client / round\nRetain partial final batch',body_size=20)
box(ax,.67,.40,.27,.30,'3  Weighted FedAvg','θ next = Σ nₖ θₖ / Σ nₖ\nFull embedding table\nB1 CPU float64 accumulation',body_size=20)
arrow(ax,.335,.55,.36,.55);arrow(ax,.64,.55,.665,.55);txt(ax,.065,.28,'Owner-row embedding update: global Δ = (nᵢ / Σ nₖ) × local Δᵢ',24)
take(ax,'B1 matched central control: same ordered batches, one continuing model.');figs.append(f)
#12
f,ax=frame(S[11]);box(ax,.06,.35,.42,.36,'Selection and leakage safeguards','Frozen train / validation / test identities\nFit scalers on declared training population\nB1 validation: 0, 10, …, 200 rounds\nEarliest minimum joint standardized RMSE',body_size=20)
box(ax,.52,.35,.42,.36,'Report both aggregation views','MAE = mean |error|\nRMSE = √ mean(error²)\nPooled: all windows; macro: equal nodes\nMean ± sample SD across seeds 42/43/44',body_size=21)
take(ax,'Seed SD is not a confidence interval or independent-dataset uncertainty.');figs.append(f)
#13
f,ax=frame(S[12]);table(ax,['Track','Completed evidence','Separate evaluation scope'],[
['Legacy 1 h','LSTM; 18 v1 + 6 v2/v3 fits','75,566 windows / 979 nodes'],['v5 6 h','24 policy / method / seed fits','6,590 windows / 570 nodes'],['B1 6 h','3 FedAvg + 3 central fits','March: 194,212 (FedAvg only)'],['History / T2–T3','9 history fits; temporal panels','Weekly adaptation: seed 42 only'],['T4','6 final fits + validation pilots','Canonical / extended / March'],['T5 / T6','9 weather + 6 family fits','Reference GRUs reused'],['T7','99 saved stress records','3 frozen models; no new fits']],colwidth=[.22,.40,.38],fontsize=18.5,rowh=.065)
take(ax,'New two-layer history benchmark: ongoing; no completed matrix accepted.');figs.append(f)
#14
f,ax=frame(S[13]);txt(ax,.065,.705,'A  Legacy 1 h · 75,566 windows / 979 nodes',20,TEAL,weight='bold');table(ax,['Method','MAE W','RMSE W'],[['Shared LSTM','27.934','54.884'],['Persistence','30.927','64.417']],x=.06,y=.635,width=.40,colwidth=[.52,.24,.24],fontsize=19,rowh=.078)
txt(ax,.065,.355,'Not FedAvg; seed 42 only.',19,GRAY);txt(ax,.065,.30,'Different cohort from the v5 panel.',18,GRAY)
pa=axes(f,[.59,.35,.34,.31]);pol=['no_fill','ff10','ff20','ff30'];x=np.arange(4);w=.34
for meth,off,c,lab in [('fedavg',-w/2,BLUE,'FedAvg'),('centralized_sgd',w/2,TEAL,'Central SGD')]:
 y=[group(v5,{'method':meth,'policy':p},'rmse_w') for p in pol];err=[sd(v5,{'method':meth,'policy':p},'rmse_w') for p in pol];pa.bar(x+off,y,w,yerr=err,color=c,label=lab,capsize=3)
pa.axhline(58.26980785,color=ORANGE,ls='--',lw=2,label='Persistence');pa.set_xticks(x,['No-fill','FF10','FF20','FF30']);pa.set_ylim(0,65);styleplot(pa,ylabel='Pooled RMSE (W)');pa.legend(fontsize=11.5,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False,columnspacing=.8,handlelength=1.4);txt(ax,.55,.705,'B  v5 · 6,590 windows / 570 nodes',20,TEAL,weight='bold');txt(ax,.55,.265,'Central wins 12/12 paired MAE + RMSE comparisons.\nOne test date; best mean MAE still above persistence.',16,linespacing=1.3)
take(ax,'No general FL superiority; v5 conclusions are conditional on a narrow holdout.');figs.append(f)
#15
f,ax=frame(S[14]);pers=[[42.3922214317,92.6003920070],[.3033845958,.6095322087]]
for j,(label,unit) in enumerate([('Power','W'),('IPMI ambient','°C')]):
 a=axes(f,[.09+j*.47,.31,.36,.35]);x=np.arange(2)
 y=[b1['pooled'][m]['mean'][j] for m in ['mae','rmse']];er=[b1['pooled'][m]['sample_sd'][j] for m in ['mae','rmse']]
 a.bar(x-.18,pers[j],.36,color=ORANGE,label='Persistence');a.bar(x+.18,y,.36,yerr=er,color=TEAL,label='B1 FedAvg',capsize=4)
 a.set_xticks(x,['MAE','RMSE']);styleplot(a,ylabel=unit);a.set_ylim(0,max(y+pers[j])*1.28)
 for i,v in enumerate(y):
  a.text(i+.18,v+max(y+pers[j])*.045,f'{v:.3f}',ha='center',fontsize=17,color=TEAL)
  a.text(i-.18,pers[j][i]*.90,f'{pers[j][i]:.3f}',ha='center',va='top',fontsize=16,color='white',fontweight='bold')
 a.legend(fontsize=14,loc='upper left');txt(ax,.09+j*.47,.715,label,24,weight='bold')
txt(ax,.08,.23,'B1 selected rounds: 200 / 200 / 200. Macro-node metrics accompany the deck.',20)
take(ax,'Only power RMSE improves among the four pooled B1 error comparisons.');figs.append(f)
#16
f,ax=frame(S[15]);a=axes(f,[.09,.32,.43,.35]);labels=['Persistence','FedAvg','Central'];ys=[33.0661,group(t4,{'test_period':'T1_Test','framework':'FedAvg-GRU-T4'},'test_mae_w'),group(t4,{'test_period':'T1_Test','framework':'Centralized-GRU-T4'},'test_mae_w')];errs=[0,sd(t4,{'test_period':'T1_Test','framework':'FedAvg-GRU-T4'},'test_mae_w'),sd(t4,{'test_period':'T1_Test','framework':'Centralized-GRU-T4'},'test_mae_w')]
a.bar(labels,ys,color=[ORANGE,TEAL,BLUE],yerr=errs,capsize=5);styleplot(a,ylabel='Pooled MAE (W)');a.set_ylim(0,40)
for i,v in enumerate(ys):a.text(i,v+.8,f'{v:.3f}',ha='center',fontsize=20)
box(ax,.59,.48,.33,.22,'24.96% lower MAE','FedAvg vs same-window persistence\nPooled RMSE: 57.550 vs 64.045 W',title_size=26,body_size=18)
box(ax,.59,.24,.33,.17,'Macro-node MAE (W)','FedAvg 25.224 vs persistence 33.350',title_size=22,body_size=18)
txt(ax,.09,.23,'FedAvg MAE 24.813 ± 0.036 W; central 23.986 ± 0.225 W.',19)
take(ax,'A residual-forecasting result; centralized mean errors remain lower.');figs.append(f)
#17
f,ax=frame(S[16]);table(ax,['T4 temporal view','FedAvg MAE W','Central MAE W','Persistence W'],[
['Canonical · 243,138','24.813 ± 0.036','23.986 ± 0.225','33.066'],['Extended · 3,609,285','33.966 ± 0.119','33.064 ± 0.177','40.441'],['March · 2,229,042','38.283 ± 0.169','37.337 ± 0.212','45.289']],colwidth=[.32,.24,.24,.20],fontsize=20,rowh=.095)
box(ax,.075,.215,.85,.14,'FedAvg node MAE wins vs persistence (seeds 42 / 43 / 44)','Canonical: 900 / 897 / 898 of 978   |   March: 955 / 953 / 953 of 976',title_size=22,body_size=20)
take(ax,'Extended contains canonical and March; these are not independent test datasets.');figs.append(f)
#18
f,ax=frame(S[17]);a=axes(f,[.09,.32,.43,.35]);x=np.arange(3);ys=[group(hist,{'history_hours':str(h)},'test_mae_w') for h in [1,6,24]];er=[sd(hist,{'history_hours':str(h)},'test_mae_w') for h in [1,6,24]]
a.errorbar(x,ys,yerr=er,fmt='o-',color=TEAL,lw=3,ms=10,capsize=6);a.axhline(33.066,color=ORANGE,ls='--',label='Persistence');a.set_xticks(x,['1 h','6 h','24 h']);a.set_ylim(32.2,34.1);a.set_xlim(-.4,2.4);styleplot(a,ylabel='Pooled MAE (W)');a.legend(fontsize=16)
for i,y in enumerate(ys):a.annotate(f'{y:.3f}',(i,y),xytext=(0,16),textcoords='offset points',ha='center',fontsize=18)
box(ax,.60,.45,.32,.25,'Completed: 9 old fits','One-layer GRU · FF30 · same cohort\n1 h → 6 h: ≈ 1.95% lower MAE\n6 h → 24 h: ≈ 0.23% lower MAE',title_size=22,body_size=18)
box(ax,.60,.205,.32,.20,'New residual / MSE benchmark','Ongoing; eligibility + provenance\nissues remain unresolved.',title_size=20,body_size=18)
take(ax,'Longer history is not established as universally better.');figs.append(f)
#19
f,ax=frame(S[18]);t5=[r for r in late['grouped']['T5'] if r['period']=='test_t1'];t6=[r for r in late['grouped']['T6'] if r['period']=='test_t1']
for j,rows in enumerate([t5,t6]):
 a=axes(f,[.08+j*.47,.33,.36,.33]);lab=['None','Temp','Humidity','Both'] if j==0 else ['GRU¹','LSTM¹','XGBoost²'];ys=[r['pooled_mae_w'] for r in rows];ers=[r['pooled_mae_w_sample_sd'] for r in rows]
 a.bar(lab,ys,yerr=ers,capsize=4,color=TEAL if j==0 else [TEAL,BLUE,ORANGE]);a.set_ylim(0,31);styleplot(a,ylabel='Pooled MAE (W)')
 for i,y in enumerate(ys):a.text(i,y+.6,f'{y:.3f}',ha='center',fontsize=17)
 txt(ax,.08+j*.47,.72,'T5: enabled outdoor channels' if j==0 else 'T6: model families',22,weight='bold')
txt(ax,.08,.245,'Weather effect is small and conditional.',18);txt(ax,.55,.275,'¹ FedAvg; ² centralized. XGB has lower RMSE:\n57.011 W vs GRU 57.550 W.',17,linespacing=1.3)
take(ax,'GRU has lower MAE here; it does not win every metric or equal-capacity comparison.');figs.append(f)
#20
f,ax=frame(S[19]);a=axes(f,[.09,.32,.51,.36]);
for ids,lab,c in [(['cond_00_ref_0pct','cond_rand_10pct','cond_rand_20pct','cond_rand_30pct'],'Random',TEAL),(['cond_00_ref_0pct','cond_cont_l6_10pct','cond_cont_l6_20pct','cond_cont_l6_30pct'],'1-hour bursts',ORANGE),(['cond_00_ref_0pct','cond_cont_l12_10pct','cond_cont_l12_20pct','cond_cont_l12_30pct'],'2-hour bursts',BLUE)]:
 rows=[gt7(i) for i in ids];a.errorbar([0,10,20,30],[float(r['pooled_mae_mean_w']) for r in rows],yerr=[float(r['pooled_mae_sample_sd_w']) for r in rows],fmt='o-',label=lab,color=c,lw=2.5,capsize=3)
styleplot(a,xlabel='Requested missingness (%)',ylabel='Pooled MAE (W)');a.set_ylim(23,39);a.legend(fontsize=15)
box(ax,.67,.40,.25,.29,'20% requested loss','Random: 26.520 W\n1 h burst: 32.351 W\nClean: 24.813 W',title_size=22,body_size=21)
txt(ax,.67,.31,'FF30 → training-mean\ninput fallback; no target fill.',18,GRAY,linespacing=1.4)
take(ax,'Synthetic masks, not independent real outages; burst intervals can overlap.');figs.append(f)
#21
f,ax=frame(S[20]);a=axes(f,[.09,.32,.38,.36]);ids=['cond_00_ref_0pct','cond_rand_20pct','cond_cont_l6_20pct'];ys=[float(gt7(i)['mean_pct_nodes_beating']) for i in ids];a.bar(['Clean','Random\n20%','1 h burst\n20%'],ys,color=[TEAL,BLUE,ORANGE]);a.set_ylim(0,108);styleplot(a,ylabel='Nodes beating persistence (%)')
for i,y in enumerate(ys):a.text(i,y+2,f'{y:.1f}%',ha='center',fontsize=21)
box(ax,.57,.46,.35,.25,'Complete power-channel blackout','Model MAE: 124.971 W\nCondition persistence: 124.822 W\nPower history is a critical dependency.',title_size=21,body_size=20)
txt(ax,.57,.32,'CPU / memory / ambient blackouts:\nmuch smaller changes in this setup.',19,linespacing=1.4)
take(ax,'Broad clean-data gains do not establish production fault tolerance.');figs.append(f)
#22
f,ax=frame(S[21]);
for i,(a,b,c) in enumerate([('Forecast + coverage','Verified errors and failure modes',PALE),('Action-response model','Workload, thermal and facility physics','#EEF2F5'),('Constrained decisions','Feasibility, cost and service objectives','#EEF2F5'),('Policy experiment','Energy / cost / SLA vs controlled baseline','#EEF2F5')]):
 y=.65-i*.12;box(ax,.18,y,.64,.085,a,b,fill=c,title_size=20,body_size=17)
# use a simpler two-line layout with box body hidden by compact spacing corrected below
for artist in list(ax.texts):
 if artist.get_text() in ['Verified errors and failure modes','Workload, thermal and facility physics','Feasibility, cost and service objectives','Energy / cost / SLA vs controlled baseline']:artist.remove()
txt(ax,.84,.70,'DONE',16,TEAL,weight='bold');txt(ax,.84,.46,'PROPOSED',14,GRAY,weight='bold',rotation=90,va='center')
take(ax,'No energy savings, PUE, cooling or SLA improvement has been measured.');figs.append(f)
#23
f,ax=frame(S[22]);box(ax,.06,.35,.42,.36,'Scientific limits','One facility; known-node simulation\nRetrospective / previously inspected tests\nDifferent cohorts and optimization budgets\nThree-seed SD ≠ deployment uncertainty',body_size=20)
box(ax,.52,.35,.42,.36,'Integrity and ongoing status','Historical preflight: FAILED\nSemantic reproduction: UNKNOWN\n9 newer manifest comparisons: FAILED\nNew history benchmark: ongoing',body_size=21)
take(ax,'Verified stored bytes do not erase failed provenance or establish semantic replay.');figs.append(f)
#24
f,ax=frame(S[23]);box(ax,.06,.32,.43,.38,'Measured findings','T4 residual L1 improves persistence.\nCentral controls remain competitive.\nB1 multi-output results are mixed.\nBursts and power loss expose fragility.',body_size=22)
box(ax,.54,.32,.40,.38,'Next defensible steps','1  Locked B1 central March comparison\n2  Resolve new benchmark provenance\n3  Validate decision/action response\n4  Test actual operational outcomes',body_size=21)
take(ax,'Thank you • Questions',color=NAVY);figs.append(f)
# Save all slide artwork and structurally inspect bounds.
with PdfPages(P/'FINAL_ENDSEM_PRESENTATION.pdf',metadata={'Title':'M.Tech End-Semester Presentation','Author':'Mihir Chouhan','Subject':'Verified stored-result snapshot, 17 November 2026'}) as pdf:
 for i,f in enumerate(figs,1):
  f.canvas.draw();renderer=f.canvas.get_renderer();over=[]
  for a in f.axes:
   for t in a.texts:
    if not t.get_visible():continue
    bb=t.get_window_extent(renderer)
    if bb.x0<-1 or bb.y0<-1 or bb.x1>f.bbox.width+1 or bb.y1>f.bbox.height+1:over.append(t.get_text())
  checks.append({'slide':i,'canvas_text_overflow':over,'png_dimensions':[1920,1080]})
  f.savefig(P/f'SLIDES/slide_{i:02}.png',dpi=120,facecolor=BG);f.savefig(P/f'FIGURES/slide_{i:02}.svg',facecolor=BG);pdf.savefig(f,facecolor=BG)
json.dump(checks,(P/'EVIDENCE/layout_checks.json').open('w'),indent=2)
assert not any(x['canvas_text_overflow'] for x in checks),checks
# Slide script and speaker notes
script=['# Complete presentation script','Mihir Chouhan · 25CSE2014 · NIT Goa','Presentation: 19 December 2026. Evidence cutoff: 17 November 2026.','24 slides · planned delivery: 19 minutes 55 seconds including pauses. Spoken narrative: about1,938 words. Rehearse at110–125words/minute with chart pauses; shorten technical notes to remain within20minutes.','Technical explanations are backup material, not mandatory extra spoken text. Every slide includes its source IDs; SOURCE_INDEX.md resolves IDs to exact files and hashes.']
notes=[]
for s in S:
 n=f"Slide {s['number']:02}: {s['title']}\n\nWHAT TO SAY\n{s['say']}\n\nTECHNICAL EXPLANATION\n{s['technical']}\n\nTRANSITION\n{s['transition']}\n\nSPEAKING TIME: {s['seconds']} seconds\n\nLIKELY FACULTY QUESTION\n{s['faculty_question']}\n\nSOURCES: {', '.join(s['sources'])}"
 notes.append(n);script += [f"\n## Slide {s['number']:02} — {s['title']}",f"**Approximate time:** {s['seconds']} seconds",'### What to say',s['say'],'### Technical explanation',s['technical'],'### Transition',s['transition'],'### Likely faculty question',s['faculty_question'],'**Sources:** '+', '.join(s['sources'])]
(P/'PRESENTATION_SCRIPT.md').write_text('\n\n'.join(script)+'\n')
# Minimal standards-based OPC/OOXML package: all slides have lossless visual artwork + actual speaker notes.
# No third-party dependency installation or Office macros. Source SVG and content JSON remain editable.
NS='http://schemas.openxmlformats.org';REL=NS+'/package/2006/relationships';PNS=NS+'/presentationml/2006/main';ANS=NS+'/drawingml/2006/main';RNS=NS+'/officeDocument/2006/relationships'
def esc(x):return html.escape(str(x),quote=True)
def rels(items):return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="'+REL+'">'+''.join(f'<Relationship Id="{i}" Type="{RNS}/{t}" Target="{esc(p)}"/>' for i,t,p in items)+'</Relationships>'
def empty_tree():return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
def wrap(tag,body):return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:{tag} xmlns:a="{ANS}" xmlns:r="{RNS}" xmlns:p="{PNS}">{body}</p:{tag}>'
files={}
files['_rels/.rels']=rels([('rId1','officeDocument','ppt/presentation.xml')])
files['ppt/presentation.xml']=wrap('presentation','<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="rId26"/></p:notesMasterIdLst><p:sldIdLst>'+''.join(f'<p:sldId id="{256+i}" r:id="rId{i+2}"/>' for i in range(24))+'</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/>')
files['ppt/_rels/presentation.xml.rels']=rels([('rId1','slideMaster','slideMasters/slideMaster1.xml')]+[(f'rId{i+1}','slide',f'slides/slide{i}.xml') for i in range(1,25)]+[('rId26','notesMaster','notesMasters/notesMaster1.xml')])
clr='<a:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
files['ppt/slideMasters/slideMaster1.xml']=wrap('sldMaster','<p:cSld><p:spTree>'+empty_tree()+'</p:spTree></p:cSld>'+clr.replace('a:clrMap','p:clrMap')+'<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>')
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')])
files['ppt/slideLayouts/slideLayout1.xml']=wrap('sldLayout','<p:cSld name="Blank"><p:spTree>'+empty_tree()+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>')
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
files['ppt/notesMasters/notesMaster1.xml']=wrap('notesMaster','<p:cSld><p:spTree>'+empty_tree()+'</p:spTree></p:cSld>'+clr.replace('a:clrMap','p:clrMap')+'<p:notesStyle/>')
files['ppt/notesMasters/_rels/notesMaster1.xml.rels']=rels([('rId1','theme','../theme/theme1.xml')])
colors={'dk1':NAVY[1:],'lt1':'FFFFFF','dk2':'526575','lt2':'F6F8FA','accent1':'007E87','accent2':'D87B36','accent3':'4783AF','accent4':'AD3F3F','accent5':'92B7C1','accent6':'8C9DA8','hlink':'007E87','folHlink':'526575'}
files['ppt/theme/theme1.xml']=f'<?xml version="1.0" encoding="UTF-8"?><a:theme xmlns:a="{ANS}" name="Research"><a:themeElements><a:clrScheme name="Research">'+''.join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k,v in colors.items())+'</a:clrScheme><a:fontScheme name="Research"><a:majorFont><a:latin typeface="DejaVu Sans"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="DejaVu Sans"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Research"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'
for i,s in enumerate(S,1):
 pic=f'<p:pic><p:nvPicPr><p:cNvPr id="2" name="Slide {i}: {esc(s["title"])}" descr="{esc(s["title"]+". "+s["subtitle"])}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
 files[f'ppt/slides/slide{i}.xml']=wrap('sld','<p:cSld name="'+esc(s['title'])+'"><p:spTree>'+empty_tree()+pic+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>')
 files[f'ppt/slides/_rels/slide{i}.xml.rels']=rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image',f'../media/image{i}.png'),('rId3','notesSlide',f'../notesSlides/notesSlide{i}.xml')])
 paras=''.join('<a:p><a:r><a:rPr lang="en-US"/><a:t>'+esc(t)+'</a:t></a:r><a:endParaRPr lang="en-US"/></a:p>' for t in notes[i-1].splitlines())
 shape='<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes Text"/><p:cNvSpPr txBox="1"/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>'+paras+'</p:txBody></p:sp>'
 files[f'ppt/notesSlides/notesSlide{i}.xml']=wrap('notes','<p:cSld><p:spTree>'+empty_tree()+shape+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>')
 files[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels']=rels([('rId1','notesMaster','../notesMasters/notesMaster1.xml'),('rId2','slide',f'../slides/slide{i}.xml')])
ct={'ppt/presentation.xml':'presentation.main','ppt/slideMasters/slideMaster1.xml':'slideMaster','ppt/slideLayouts/slideLayout1.xml':'slideLayout','ppt/notesMasters/notesMaster1.xml':'notesMaster','ppt/theme/theme1.xml':'theme'}
ct.update({f'ppt/slides/slide{i}.xml':'slide' for i in range(1,25)});ct.update({f'ppt/notesSlides/notesSlide{i}.xml':'notesSlide' for i in range(1,25)})
files['[Content_Types].xml']='<?xml version="1.0" encoding="UTF-8"?><Types xmlns="'+NS+'/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="xml" ContentType="application/xml"/>'+''.join(f'<Override PartName="/{k}" ContentType="application/vnd.openxmlformats-officedocument.{"theme+xml" if v=="theme" else "presentationml."+v+"+xml"}"/>' for k,v in ct.items())+'</Types>'
with zipfile.ZipFile(P/'FINAL_ENDSEM_PRESENTATION.pptx','w',zipfile.ZIP_DEFLATED) as z:
 for n,data in files.items():ET.fromstring(data);z.writestr(n,data.encode())
 for i in range(1,25):z.write(P/f'SLIDES/slide_{i:02}.png',f'ppt/media/image{i}.png')
print('Created24slidePPTX,24pagePDF,PNG/SVGartworkandnotes. Known rendering method: identical artwork in both formats; PPTX slide canvases are lossless images.')
