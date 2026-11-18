"""Static report figures from SAVED METRIC SUMMARIES ONLY. No project imports."""
import os,json,pathlib,statistics
os.environ['MPLCONFIGDIR']='/tmp/endsem_report_work_v1/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
BASE=pathlib.Path(__file__).resolve().parents[1]
DATA=BASE/'EVIDENCE/report_metric_extract.json'
if not DATA.exists():
 BASE=pathlib.Path('/tmp/endsem_report_work_v1/package');DATA=BASE/'EVIDENCE/report_metric_extract.json'
D=json.loads(DATA.read_text()); F=BASE/'FIGURES'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'font.family':'DejaVu Sans'})
COL=['#266d9c','#d47725','#568c53','#895d9d']
def save(fig,n):
 fig.savefig(F/(n+'.png'),bbox_inches='tight');fig.savefig(F/(n+'.svg'),bbox_inches='tight');plt.close(fig)
def box(ax,xy,w,h,s,color='#edf3f7'):
 ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.025',facecolor=color,edgecolor='#526a7a',linewidth=1.2))
 ax.text(xy[0]+w/2,xy[1]+h/2,s,ha='center',va='center',fontsize=8.5,wrap=True)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#526a7a',linewidth=1.2))
fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
box(ax,(.2,4.5),2.5,1,'Raw telemetry\nPower / thermal / CPU / memory\nSeparate weather observations')
box(ax,(3.3,4.5),2.5,1,'Causal binning\nUTC 10 minutes; hold ≤300 s\nCoverage ≥50%; validity masks')
box(ax,(6.4,4.5),2.5,1,'Versioned windows\nNode + UTC origin identity\nObserved future labels')
box(ax,(9.5,4.5),2.2,1,'Frozen populations\nTrain / validation / test\nTrack-specific rules')
for a,b in [((2.7,5),(3.3,5)),((5.8,5),(6.4,5)),((8.9,5),(9.5,5))]:arrow(ax,a,b)
box(ax,(.8,2.2),3,1,'Training-only statistics\nFeature/target transforms\nNo test fitting')
box(ax,(4.5,2.2),3,1,'Client-local updates → FedAvg\nKnown-node embedding\nCentralized controls separately')
box(ax,(8.2,2.2),3,1,'Validation selection\nTrack-specific metric\nImmutable selected exports')
arrow(ax,(10.5,4.5),(10.5,3.2));arrow(ax,(9.1,4.5),(2.3,3.2));arrow(ax,(3.8,2.7),(4.5,2.7));arrow(ax,(7.5,2.7),(8.2,2.7))
box(ax,(3.4,.2),5.2,1,'Authorized fixed-window evaluation\nSaved predictions → pooled / macro-node errors\nPreserved evaluation provenance', '#e8f1e6')
arrow(ax,(9.7,2.2),(8.0,1.2));ax.set_title('Implemented forecasting architecture — distinct versioned experiment tracks',pad=12)
save(fig,'01_system_architecture')
fig,ax=plt.subplots(figsize=(11,5));ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off')
box(ax,(.3,3),2.3,1,'Input [B,H,14]\nNumeric channels 0–12\nCategorical node ordinal 13')
box(ax,(3.1,3),2.2,1,'979 × 8 embedding\nConcatenate at each step\nPreserve node mapping')
box(ax,(5.8,3),2.1,1,'GRU\nHidden size 64\nFinal recurrent state')
box(ax,(8.4,3),2.2,1,'Direct linear head\nB1: 64 → 2\nT4: 64 → 1')
for a,b in [((2.6,3.5),(3.1,3.5)),((5.3,3.5),(5.8,3.5)),((7.9,3.5),(8.4,3.5))]:arrow(ax,a,b)
box(ax,(.8,.5),4.2,1.2,'B1: H=36, 2 GRU layers; no residual\nStandardized joint MSE\nOutputs: power W; recorded IPMI ambient °C')
box(ax,(5.8,.5),4.2,1.2,'T4: H=36, 2 GRU layers\nL1 on standardized power delta\nInverse delta + last-bin persistence')
arrow(ax,(9.4,3),(7.8,1.7));arrow(ax,(6.5,3),(3.0,1.7))
ax.set_title('Actual B1/T4 sequence adapters — no extra hidden prediction layer');save(fig,'02_model_architectures')
fig,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(0,11),ylim=(0,4));ax.axis('off')
for x,s in [(0.3,'Historical 1-hour\nShared LSTM / v1 panel\n75,566 test windows'),(3.1,'Historical v5 6-hour\n24 method/policy/seed fits\n6,590 common test windows'),(5.9,'B1 two-output\nHourly March evaluation\n194,212 test windows'),(8.7,'Candidate1 / T4\nDifferent target/loss/contracts\n243,138 canonical windows')]:box(ax,(x,1.6),2,1.4,s)
ax.text(5.5,.5,'Compare within a fixed window/target protocol. Cross-track scores are not one leaderboard.',ha='center',fontsize=11)
ax.set_title('Experiment separation required for valid interpretation');save(fig,'03_experiment_tracks')
fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');x=np.arange(4)
for ax,k,label,base in zip(axs,['mae_w','rmse_w'],['Pooled MAE (W)','Pooled RMSE (W)'],[27.1470750029,58.2698078546]):
 for i,m in enumerate(['fedavg','centralized_sgd']):
  rr=[next(r for r in D['v5_grouped'] if r['method']==m and r['policy']==p) for p in ['no_fill','ff10','ff20','ff30']]
  ax.bar(x+(i-.5)*.34,[r[k] for r in rr],.34,yerr=[r[k+'_sd'] for r in rr],capsize=3,label=m,color=COL[i])
 ax.axhline(base,color='black',ls='--',label='Persistence');ax.set_xticks(x,['No-fill','FF10','FF20','FF30']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('Historical v5: 6,590 same test windows; error bars = sample SD of 3 seeds');save(fig,'04_v5_policy_results')
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for j,name in enumerate(['power_W','ambient_degC']):
 p=next(r for r in D['b1_persistence'] if r['aggregation']=='pooled' and r['output']==name)
 for col,k in enumerate(['mae','rmse']):
  ax=axs[j,col];a=D['b1_aggregate']['pooled'][k];ax.bar([0,1],[float(p[k.upper()]),a['mean'][j]],yerr=[0,a['sample_sd'][j]],capsize=4,color=[COL[2],COL[0]])
  ax.set_xticks([0,1],['Persistence','B1 FedAvg']);ax.set_ylabel(k.upper()+(' (W)' if j==0 else ' (°C)'));ax.grid(axis='y',alpha=.2)
fig.suptitle('B1: identical 194,212 hourly March windows; pooled errors; retrospective');save(fig,'05_b1_persistence_results')
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,k,base in zip(axs,['test_mae_w','test_rmse_w'],[33.066,64.045]):
 means=[];sds=[]
 for h in (1,6,24):
  vals=[float(r[k]) for r in D['history_pooled'] if int(r['history_hours'])==h];means.append(statistics.mean(vals));sds.append(statistics.stdev(vals))
 ax.errorbar([0,1,2],means,yerr=sds,marker='o',capsize=4,color=COL[0],label='FF30 FedAvg');ax.axhline(base,color='black',ls='--',label='Persistence')
 ax.set_xticks([0,1,2],['1 hour','6 hours','24 hours']);ax.set_ylabel('Pooled '+('MAE' if 'mae' in k else 'RMSE')+' (W)');ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('Nine completed FF30 fits: shared longest-history-eligible canonical population');save(fig,'06_history_results')
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,k,pers,label in zip(axs,['test_mae_w','macro_mae_w'],[33.0661,33.3504],['Pooled MAE (W)','Macro-node MAE (W)']):
 rr=[r for r in D['t4_grouped'] if r['period']=='T1_Test'];ax.bar([0,1,2],[pers]+[r[k] for r in rr],yerr=[0]+[r[k+'_sd'] for r in rr],capsize=4,color=[COL[2],COL[0],COL[1]])
 ax.set_xticks([0,1,2],['Persistence','T4 FedAvg','T4 central']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
fig.suptitle('T4 canonical test: keep pooled and macro persistence baselines separate');save(fig,'07_t4_canonical_results')
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');x=np.arange(3)
for i,m in enumerate(['FedAvg-GRU-T4','Centralized-GRU-T4']):
 rr=[next(r for r in D['t4_grouped'] if r['period']==p and r['method']==m) for p in ['T1_Test','T2_Extended','March_2022']]
 ax.bar(x+(i-1)*.25,[r['test_mae_w'] for r in rr],.25,yerr=[r['test_mae_w_sd'] for r in rr],capsize=3,color=COL[i],label=m)
ax.bar(x+.25,[33.0661,40.4405,45.2887],.25,color=COL[2],label='Persistence')
ax.set_xticks(x,['Canonical\n243,138 / 978','Extended (contains both)\n3,609,285 / 979','March\n2,229,042 / 976']);ax.set_ylabel('Pooled MAE (W)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
ax.set_title('T4 nested temporal views — windows / nodes; not independent datasets');save(fig,'08_t4_temporal_results')
print(json.dumps({'figures':len(list(F.glob('*.png'))),'format':'PNG+SVG','new_inference':False}))
