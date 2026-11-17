"""Render revision-v2 figures from saved metrics only; no project imports."""
import os,pathlib,json,statistics,collections
os.environ['MPLCONFIGDIR']='/tmp/endsem_update_v2_bgm1olro/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE=pathlib.Path(__file__).resolve().parents[1]
if not (BASE/'EVIDENCE/UPDATE_V2/audit_results.json').exists():BASE=pathlib.Path('/tmp/endsem_update_v2_bgm1olro/package')
D=json.loads((BASE/'EVIDENCE/UPDATE_V2/audit_results.json').read_text());F=BASE/'FIGURES'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
def g(c,p='test_t1'):return next(r for r in D['t7_summary_recomputed'] if r['condition_id']==c and r['period']==p)
def save(fig,n):
 fig.savefig(F/(n+'.png'),bbox_inches='tight');fig.savefig(F/(n+'.svg'),bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
colors=['#236c9e','#5f8e4c','#d0822e','#a55264']
for color,prefix,label in zip(colors,['cond_rand_','cond_cont_l3_','cond_cont_l6_','cond_cont_l12_'],['Random','Burst L=3 bins','Burst L=6 bins','Burst L=12 bins']):
 rr=[g('cond_00_ref_0pct')]+[g(prefix+f'{p:02d}pct') for p in [5,10,20,30]]
 axs[0].errorbar([0,5,10,20,30],[r['pooled_mae_mean_w'] for r in rr],yerr=[r['pooled_mae_sample_sd_w'] for r in rr],marker='o',capsize=3,label=label,color=color)
 axs[1].plot([0,5,10,20,30],[r['pers_pooled_mae_w'] for r in rr],marker='o',label=label,color=color)
for ax in axs:
 ax.set_xlabel('Requested input-loss probability (%)');ax.set_xticks([0,5,10,20,30]);ax.grid(alpha=.2)
axs[0].set_ylabel('Model pooled MAE (W)');axs[0].legend(fontsize=8);axs[1].set_ylabel('Condition-specific persistence MAE (W)')
fig.suptitle('T7: same canonical cohort; corrupted inputs, FF30 and mean-input fallback\nError bars = model-seed sample SD; one fixed mask per condition');save(fig,'09_t7_missingness_robustness')
fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
r=g('group_power_blackout');axs[0].bar([0],[r['degradation_mae_mean_w']],color=colors[3]);axs[0].set_xticks([0],['Power']);axs[0].set_ylabel('Δ pooled MAE versus clean (W)')
gg=[g('group_'+k+'_blackout') for k in ['cpu','memory','inlet_temp']]
axs[1].bar([0,1,2],[r['degradation_mae_mean_w'] for r in gg],color=colors[:3]);axs[1].set_xticks([0,1,2],['CPU group','Memory','IPMI ambient']);axs[1].set_ylabel('Δ pooled MAE versus clean (W)');axs[1].axhline(0,color='black',lw=.8)
for ax in axs:ax.grid(axis='y',alpha=.2)
fig.suptitle('T7: complete single-group blackout; separately scaled axes\nConditional behavior of frozen residual model, not causal feature importance');save(fig,'10_t7_channel_sensitivity')
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
conds=['cond_00_ref_0pct','cond_rand_10pct','cond_cont_l6_10pct','cond_cont_l6_30pct'];labels=['Clean','Random10%','1h burst10%','1h burst30%'];wins=[];sds=[]
for c,label,color in zip(conds,labels,colors):
 ng=collections.defaultdict(list)
 for r in D['t7_node_rows']:
  if r['condition_id']==c:ng[r['node_idx']].append(float(r['model_mae_w'])-float(r['pers_mae_w']))
 assert len(ng)==978 and all(len(v)==3 for v in ng.values())
 vals=np.sort([statistics.mean(v) for v in ng.values()]);axs[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label,color=color)
 rows=[r for r in D['t7_seed_rows'] if r['period']=='test_t1' and r['condition_id']==c];vv=[float(r['pct_nodes_beating_pers']) for r in rows];wins.append(statistics.mean(vv));sds.append(statistics.stdev(vv))
axs[0].axvline(0,color='black',ls='--');axs[0].set_xlabel('Node MAE difference: model − condition persistence (W)\nMean difference across three seed models');axs[0].set_ylabel('Fraction of 978 nodes');axs[0].legend(fontsize=8)
axs[1].bar(np.arange(4),wins,yerr=sds,capsize=4,color=colors);axs[1].set_xticks(np.arange(4),labels,rotation=20);axs[1].set_ylim(0,100);axs[1].set_ylabel('Mean per-seed nodes beating persistence (%)')
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('T7 client-wise consistency from saved per-node metrics\nNo prediction reconstruction or model scoring');save(fig,'11_t7_node_consistency')
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');pols=['no_fill','ff10','ff20','ff30'];x=np.arange(4)
for j,(split,label) in enumerate([('train','Training'),('val','Validation'),('test','Test')]):
 rows=[next(r for r in D['new_coverage_cohorts'] if r['protocol_split_name']=='Candidate 1' and r['split']==split and r['policy']==p and r['history_steps']=='144') for p in pols]
 ax.bar(x+(j-1)*.25,[int(r['common_cohort_windows']) for r in rows],.25,label=label,color=colors[j])
ax.set_xticks(x,['No-fill','FF10','FF20','FF30']);ax.set_yscale('log');ax.set_ylabel('Three-history common windows (log scale)');ax.legend();ax.grid(axis='y',alpha=.2)
ax.set_title('New coverage audit: policy-specific saved intersections\nCount arithmetic checked; execution-manifest bindings unresolved');save(fig,'12_common_coverage_by_policy')
print(json.dumps({'new_png_figures':4,'new_svg_figures':4,'source':'saved metric/count CSVs','model_operations':False}))
