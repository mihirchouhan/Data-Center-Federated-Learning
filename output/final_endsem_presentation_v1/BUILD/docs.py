from pathlib import Path
import json,csv,hashlib,statistics as st,shutil,datetime
R=Path('/media/csedept/cse2014/Project/m100_training_template');W=Path('/tmp/mtech_presentation_v1');P=W/'package';D=R/'output/final_endsem_report_v2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
refs={
'REPORT':'output/final_endsem_report_v2/FINAL_PROJECT_REPORT.md',
'REGISTRY':'output/final_endsem_report_v2/EXPERIMENT_STATUS_REGISTER.csv',
'ARCHITECTURE':'docs/Architecture.md','PREPROCESS':'src/preprocess.py','FEATURES':'src/experiment_data.py',
'COMPARISON':'output/final_endsem_report_v2/BASE_PAPER_VS_OUR_RESEARCH_COMPARISON.md',
'REPORT_VALIDATION':'output/final_endsem_report_v2/REPORT_VALIDATION.md',
'B1_PREP':'output/march_window_audit/b1_real_data_preparation_execution_v1/REPORT.md',
'B1_MODEL':'src/b1_model_adapter_v1.py','B1_RUNNER':'src/b1_real_fedavg_runner_v1.py','FEDAVG':'src/b1_fedavg_orchestration_v1.py',
'B1_EVAL':'output/experiments/b1_no_ff/march_test_evaluation_v2/metrics.json',
'B1_PERSIST':'output/experiments/b1_no_ff/march_persistence_comparison_v1/persistence_metrics.csv',
'B1_LOCK':'output/march_window_audit/b1_model_selection_lock_v1/selection_lock.json',
'B1_CENTRAL_STATUS':'output/experiments/b1_no_ff/centralized_sgd_control_v2/runs/matrix_completed.json',
'LEGACY':'output/m100_all_nodes/results/test_summary.json',
'V5':'output/experiments/v5_missing_data/model_comparison/v1/test_evaluation/v2/completed.json',
'V5_SEEDS':'output/final_endsem_report_v2/EVIDENCE/v5_seedwise_metrics.csv',
'T4':'output/experiments/t4_model_improvement_v1/T4_FINAL_REPORT.md',
'T4_CONFIG':'output/experiments/t4_model_improvement_v1/config.json',
'T4_SOURCE':'src/t4_model_improvement_v1.py',
'T4_SEEDS':'output/experiments/t4_model_improvement_v1/final_results_by_seed.csv',
'T4_AUDIT':'output/experiments/t4_model_improvement_v1/T4_INTEGRITY_AUDIT.md',
'T4_MANIFEST':'output/experiments/t4_model_improvement_v1/T4_ARTIFACT_MANIFEST.md',
'HISTORY':'output/experiments/history_length_comparison_v1/multiseed_ff30_analysis',
'HISTORY_SEEDS':'output/final_endsem_report_v2/EVIDENCE/history_seedwise_metrics.csv',
'NEW_STATUS':'output/final_endsem_report_v2/EVIDENCE/UPDATE_V2/current_execution_observation.json',
'FAILED_BINDINGS':'output/final_endsem_report_v2/TABLES/FAILED_NEW_MANIFEST_BINDINGS.csv',
'T5':'output/experiments/t5_weather_ablation_v1/T5_RESULTS_PER_SEED.csv',
'T6':'output/experiments/t6_architecture_benchmark_v1/T6_RESULTS_PER_SEED.csv',
'LATE_RESULTS':'output/final_endsem_report_v2/EVIDENCE/late_results.json',
'T7':'output/experiments/t7_missing_telemetry_robustness_v1/T7_RESULTS_PER_CONDITION.csv',
'T7_SUMMARY':'output/final_endsem_report_v2/TABLES/T7_ALL_CONDITIONS.csv',
'T7_NODES':'output/experiments/t7_missing_telemetry_robustness_v1/T7_PER_NODE_RESULTS.csv',
'T7_SOURCE':'src/t7_missing_telemetry_robustness_v1.py'}
# Ensure citations resolve to files rather than guessed names.
if not (R/refs['V5']).is_file():
 candidates=list((R/'output/experiments/v5_missing_data/model_comparison/v1/test_evaluation/v2').glob('*.json'));refs['V5']=str(next(p for p in candidates if p.name in ['receipt.json','evaluation_receipt.json','manifest.json','summary.json','aggregate_metrics.json']).relative_to(R))
if (R/refs['HISTORY']).is_dir():
 pp=next((R/refs['HISTORY']).glob('*.csv'));refs['HISTORY']=str(pp.relative_to(R))
source=[]
for key,rel in refs.items():
 p=R/rel;assert p.is_file(),(key,p)
 source.append({'id':key,'path':rel,'bytes':p.stat().st_size,'sha256':sha(p),'kind':p.suffix.lstrip('.')})
source +=[{'id':'PAPER','path':'https://arxiv.org/pdf/2511.00623v3','local_read_copy':'/tmp/endsem_report_work_v1/base_paper_arxiv_v3.pdf','bytes':1069514,'sha256':'c081a18f0a7ac2ea920698904734f667981fe2f707fbb5d1d869efe8ec4d8aef','kind':'official PDF; original uploaded copy not found'}, {'id':'AUDIT','path':'EVIDENCE/audit_summary.json','bytes':(P/'EVIDENCE/audit_summary.json').stat().st_size,'sha256':sha(P/'EVIDENCE/audit_summary.json'),'kind':'new presentation scoped byte verification'}]
(P/'SOURCE_INDEX.json').write_text(json.dumps(source,indent=2)+'\n')
(P/'SOURCE_INDEX.md').write_text('# Slide source index\n\nAll project paths are relative to `/media/csedept/cse2014/Project/m100_training_template`. Full byte hashes follow. Slide source IDs are deliberately short for readability. The outdated overview documentation is interpreted through newer versioned artifacts, never used to erase completed work. The official arXiv v3 PDF was read locally; the named uploaded copy was not located and byte identity with it is not claimed. No network request occurred in this presentation task.\n\n'+ '\n\n'.join('## '+e['id']+'\n\n`'+e['path']+'`\n\nSHA-256: `'+e['sha256']+'`\n\nBytes: '+str(e['bytes']) for e in source)+'\n\n## Verification scope\n\nThe supplied v2 report inventory has 109 content entries; all freshly match. Its prior647 content comparisons reproduce638 expected matches and9 already documented failures. All279 recorded audited source bindings match at the snapshot. Hash checks do not rerun predictions, prove original statistical methodology, or repair failed preflight. The complete source tree was inventoried by AST and hashing during the preceding report update; this package freshly rechecks the recorded bindings without importing experiment code.\n\n## Missing input\n\nNo mid-semester PPT/PPTX/ODP was found in the repository, project roots, Downloads, Desktop, Documents, attachments or temporary directory search. The presentation was created from the current verified report and original evidence rather than silently substituting an old deck.\n')
# Quantitative consistency checks: compare derived/copy records with original saved sources.
checks=[];metrics=[]
def check(name,ok,detail):checks.append({'check':name,'pass':bool(ok),'detail':detail});assert ok,name
v5=read(D/'EVIDENCE/v5_seedwise_metrics.csv')
for row in v5:
 m=json.loads((R/row['source']).read_text())
 for key,orig in [('mae_w','mae_w'),('rmse_w','rmse_w'),('macro_mae_w','macro_node_mae_w'),('macro_rmse_w','macro_node_rmse_w')]:check('v5 identity '+row['method']+row['policy']+row['seed']+' '+key,float(row[key])==m[orig],row['source'])
 check('v5 fixed cohort '+row['method']+row['policy']+row['seed'],m['samples']==6590 and m['evaluated_nodes']==570,row['source'])
 check('v5 fixed identity hash '+row['method']+row['policy']+row['seed'],m['window_identity']['ordered_sha256']=='7d44fa703b877579791162594922088bea42674c215de8503e65768ca8a2648a',row['source'])
 metrics.append({'track':'v5','condition':row['method']+'_'+row['policy'],'seed':row['seed'],'period':'24Feb2022','target':'power','unit':'W','pooled_mae':row['mae_w'],'pooled_rmse':row['rmse_w'],'macro_mae':row['macro_mae_w'],'macro_rmse':row['macro_rmse_w'],'windows':6590,'nodes':570,'source':row['source']})
for policy in ['no_fill','ff10','ff20','ff30']:
 for seed in ['42','43','44']:
  a=next(r for r in v5 if r['policy']==policy and r['seed']==seed and r['method']=='fedavg');b=next(r for r in v5 if r['policy']==policy and r['seed']==seed and r['method']=='centralized_sgd');check('v5 paired central lower '+policy+seed,all(float(b[k])<float(a[k]) for k in ['mae_w','rmse_w']),'Same stored test identities')
# B1 verify aggregate from saved seed and node errors; no new prediction scoring.
b=json.loads((R/refs['B1_EVAL']).read_text());ba=json.loads((R/'output/experiments/b1_no_ff/march_test_evaluation_v2/aggregate_metrics.json').read_text())
check('B1 three seeds',set(b)=={'42','43','44'},'Saved metrics.json')
check('B1 equal identity',len({v['identity_sha256'] for v in b.values()})==1,'Saved identity hashes only; no dataset opened')
for seed,row in b.items():
 ns=row['per_node'];N=sum(v['samples'] for v in ns.values());check('B1 node/window count '+seed,N==194212 and len(ns)==976,'Saved per-node metric counts')
 for output in range(2):
  for met in ['mae','rmse']:
   macro=st.mean(v[met][output] for v in ns.values());pooled=sum((v[met][output] if met=='mae' else v[met][output]**2)*v['samples'] for v in ns.values())/N;pooled=pooled if met=='mae' else pooled**.5
   check('B1 macro '+seed+str(output)+met,abs(macro-row['macro_node'][met][output])<1e-10,'Re-aggregate saved node metrics')
   check('B1 pooled '+seed+str(output)+met,abs(pooled-row['pooled'][met][output])<1e-9,'Re-aggregate saved node metrics')
  metrics.append({'track':'B1','condition':'FedAvg','seed':seed,'period':'March2022','target':['power','IPMI_ambient'][output],'unit':['W','degC'][output],'pooled_mae':row['pooled']['mae'][output],'pooled_rmse':row['pooled']['rmse'][output],'macro_mae':row['macro_node']['mae'][output],'macro_rmse':row['macro_node']['rmse'][output],'windows':194212,'nodes':976,'source':refs['B1_EVAL']})
for agg in ['pooled','macro_node']:
 for met in ['mae','rmse']:
  for j in range(2):
   values=[r[agg][met][j] for r in b.values()];check('B1 saved mean '+agg+met+str(j),abs(st.mean(values)-ba[agg][met]['mean'][j])<1e-12,'Saved seed scores');check('B1 saved sampleSD '+agg+met+str(j),abs(st.stdev(values)-ba[agg][met]['sample_sd'][j])<1e-12,'ddof1')
# T5/T6 grouping matches original CSV values.
late=json.loads((D/'EVIDENCE/late_results.json').read_text())
for track in ['T5','T6']:
 rows=read(R/refs[track])
 for gr in late['grouped'][track]:
  found=[r for r in rows if r['condition_id']==gr['condition'] and r['period']==gr['period']];check(track+' group size '+gr['condition']+gr['period'],len(found)==3,'Original per-seed CSV')
  for key in ['pooled_mae_w','pooled_rmse_w','macro_mae_w','macro_rmse_w']:
   values=[float(r[key]) for r in found];check(track+gr['condition']+gr['period']+key,abs(st.mean(values)-gr[key])<1e-11 and abs(st.stdev(values)-gr[key+'_sample_sd'])<1e-11,'Original per-seed CSV + report aggregate')
 for r in rows:metrics.append({'track':track,'condition':r['condition_id'],'seed':r['seed'],'period':r['period'],'target':'power','unit':'W','pooled_mae':r['pooled_mae_w'],'pooled_rmse':r['pooled_rmse_w'],'macro_mae':r['macro_mae_w'],'macro_rmse':r['macro_rmse_w'],'windows':r['windows'],'nodes':r['nodes'],'source':refs[track]})
# T7 exact99 records and per-group sampleSD.
t7=read(R/refs['T7']);ts=read(R/refs['T7_SUMMARY']);check('T7 unique record keys',len(t7)==99 and len({(r['condition_id'],r['seed'],r['period']) for r in t7})==99,'99 source CSV records')
for gr in ts:
 rr=[r for r in t7 if r['condition_id']==gr['condition_id'] and r['period']==gr['period']];check('T7 group count '+gr['condition_id']+gr['period'],len(rr)==3,'Stored source rows')
 for key,kk in [('pooled_mae_w','pooled_mae'),('pooled_rmse_w','pooled_rmse'),('macro_mae_w','macro_mae'),('macro_rmse_w','macro_rmse')]:
  v=[float(r[key]) for r in rr];check('T7 arithmetic '+gr['condition_id']+gr['period']+kk,abs(st.mean(v)-float(gr[kk+'_mean_w']))<1e-10 and abs(st.stdev(v)-float(gr[kk+'_sample_sd_w']))<1e-10,'Means and sampleSD from stored CSV, not prediction replay')
for r in t7:metrics.append({'track':'T7','condition':r['condition_id'],'seed':r['seed'],'period':r['period'],'target':'power','unit':'W','pooled_mae':r['pooled_mae_w'],'pooled_rmse':r['pooled_rmse_w'],'macro_mae':r['macro_mae_w'],'macro_rmse':r['macro_rmse_w'],'windows':r['windows'],'nodes':r['nodes'],'source':refs['T7']})
for r in read(R/refs['T4_SEEDS']):metrics.append({'track':'T4','condition':r['framework'],'seed':r['seed'],'period':r['test_period'],'target':'power','unit':'W','pooled_mae':r['test_mae_w'],'pooled_rmse':r['test_rmse_w'],'macro_mae':r['macro_mae_w'],'macro_rmse':r['macro_rmse_w'],'windows':{'T1_Test':243138,'T2_Extended':3609285,'March_2022':2229042}[r['test_period']],'nodes':{'T1_Test':978,'T2_Extended':979,'March_2022':976}[r['test_period']],'source':refs['T4_SEEDS']})
writecsv(P/'PRESENTATION_METRICS.csv',metrics);(P/'EVIDENCE/metric_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
# Compact supporting physical-unit B1 table.
rows=read(R/refs['B1_PERSIST']);comp=[]
for agg in ['pooled','macro_node']:
 for j,out in enumerate(['power_W','ambient_degC']):
  pr=next(r for r in rows if r['aggregation']==agg and r['output']==out)
  comp.append({'aggregation':agg,'output':out,'FedAvg_MAE_mean':ba[agg]['mae']['mean'][j],'FedAvg_MAE_sample_sd':ba[agg]['mae']['sample_sd'][j],'FedAvg_RMSE_mean':ba[agg]['rmse']['mean'][j],'FedAvg_RMSE_sample_sd':ba[agg]['rmse']['sample_sd'][j],'persistence_MAE':pr['MAE'],'persistence_RMSE':pr['RMSE'],'windows':194212,'nodes':976})
writecsv(P/'B1_SUPPORTING_METRICS.csv',comp)
# Viva question bank, answers deliberately bounded to the inspected evidence.
qa=[
('Why federated learning?','To study shared learning across node partitions with heterogeneous local observations. This simulation tests optimization and accuracy; it does not establish deployment privacy or a necessity to decentralize this centrally available dataset.'),
('Why GRU?','It represents temporal dependence with a relatively compact recurrent model. T6 found lower mean MAE than its evaluated LSTM/XGBoost alternatives, but GRU did not minimize every RMSE and capacity was not matched.'),
('Why standard FedAvg?','It provides a clear sample-weighted baseline with a defined local-training and aggregation protocol. Adaptive aggregation or FedProx would need their own controlled experiments.'),
('What is a simulated FL client?','A partition of windows from one mapped compute-node string, executed within a server-side simulation. It is not an independently administered data center.'),
('Why are979 clients not979 data centers?','The979 identifiers refer to compute nodes of the single Marconi100 installation. Physical-site independence was never created by assigning client labels.'),
('What does the model predict?','The next hour’s arithmetic mean of six valid future ten-minute bins. B1 predicts power inW and recordedIPMIambient in°C; T4 predicts power only.'),
('Is this instantaneous power at t+60min?','No. The label averages bins spanning[t,t+60min), which is a different forecasting task.'),
('How are inputs binned?','Causal time-weighted holds of at most300seconds; a600-second bin must have at least50% coverage. Weather uses an earlier observation within30minutes.'),
('How is memory utilization validated?','Use aligned valid binned mem_free and positive mem_total, compute1−free/total, and reject fractions outside[0,1]. Do not silently drop memory.'),
('What is the node embedding?','A979×8 learned table replacing categorical slot13. The GRU receives13 numeric channels and8 embedding components at each step.'),
('Why does embedding dilution occur?','Only the owner client updates its row. Whole-table weighted averaging includes unchanged copies from other selected clients, multiplying its local delta by its processed-example weight.'),
('Why not row-wise embedding averaging?','That would renormalize by only row-updating clients and change effective row learning rates. It is a custom aggregation algorithm rather than the adopted full-table FedAvg.'),
('How is B1 local training performed?','98 of979 clients are sampled per round. Each starts from the same global model, creates fresh SGD atlr0.01, and runs one local epoch withbatch128 anddrop_last=False.'),
('How are model parameters aggregated?','Sample-weighted sums over processed examples. B1 uses CPUfloat64 accumulation then casts back; T4 uses its separately implementedfloat32 aggregation.'),
('Are all clients trained equally often?','Uniform sampling gives equal expected participation, not identical realized counts. Over200 rounds the expectation is200×98/979≈20.02 selections per client, with random variation.'),
('Does one local epoch mean equal local sample exposure?','Every selected B1 client processes all its accepted local windows once. Unequal client sizes mean unequal local examples andceil(n/128) optimizer steps.'),
('Why retain the partial final batch?','drop_last=False preserves declared example exposure and sample weights. Dropping it would change the matched budget and potentially exclude small clients.'),
('Why is persistence essential?','It is a strong causal last-bin baseline for slowly changing telemetry. A learned model must be compared on identical target windows to show useful forecast gain.'),
('Why can MAE andRMSE disagree?','RMSE gives large errors more influence. A model can reduce severe mistakes while increasing typical absolute error, as B1’s mixed power comparison illustrates.'),
('What distinguishes pooled andmacro-node metrics?','Pooled aggregates all errors, weighting nodes by window count. Macro computes each node’s metric then averages equally. MacroRMSE is not the square root of average nodeRMSE.'),
('What does three-seed sampleSD measure?','Variation among seeds42,43,44 on one fixed dataset. It is not a confidence interval for unseen dates, sites or new missingness masks.'),
('Did FedAvg beat centralizedSGD?','No in the v5 paired pooledMAE/RMSE comparisons: central won all12 policy-seed pairs. T4 also has lower central mean errors, but its budgets differ from exact B1 schedule replay.'),
('Which fill policy is best?','Within v5, centralFF10 has the lowest meanMAE/RMSE; FedAvg differences are small and its best meanRMSE isFF20. This does not generalize beyond that common one-date holdout.'),
('Are targets ever forward-filled?','No for the reported forecast protocols. Causal input policies may fill histories; the future target bins remain original observed valid values.'),
('What doesB1 establish?','A complete two-outputFedAvg pipeline and retrospective evaluation. It beats persistence on pooled powerRMSE only among the four pooled target/error comparisons.'),
('Why did allB1 seeds selectround200?','The prescribed earliest-minimum rule selected the final evaluated round for each seed. This does not authorize more rounds or March-informed tuning; convergence beyond the budget remains unestablished.'),
('How doesT4 differ fromB1?','T4 is power-only residualL1 withAdam,batch64,FF30,maskedweather and a different cohort. B1 directly predicts two standardized targets withSGD,batch128,nofill and stricter hourly windows.'),
('What exactly isT4’s persistence improvement?','Canonical pooledMAE24.8133vs33.0661W, about24.96% lower; pooledRMSE57.5502vs64.0451W. MacroMAE25.2242vs33.3504W is a separate comparison.'),
('Does24-hour history always help?','No. The completed oldFF30 panel shows a small conditional meanMAE reduction on one matched cohort. A separate newer residual benchmark is ongoing and has unresolved provenance/eligibility issues.'),
('Does weather have no predictive information?','That is unsupported. The evaluatedT5 enabled-weather variants have slightly worse canonical meanMAE than no-weather, but other models, regimes and objectives were not exhausted.'),
('What didT7 reveal?','Burst masking is more harmful than random masking at comparable requested rates; power-channel blackout is especially damaging. These are fixed synthetic masks, not operational fault guarantees.'),
('WasT7 trained on missingness?','No. It reuses three frozenT4 models and adds99 stored condition/seed/period evaluations with causalFF30 and training-mean input fallback.'),
('DoesT7 use independent mask seeds?','No. A fixed seed42 mask per condition is shared across model seeds; reported seedSD therefore excludes variation across independently simulated outages.'),
('What is the difference from the base paper?','The paper uses adaptive federatedMLP decision learning and a reduced physical optimization solver. We use standardFedAvg recurrent telemetry forecasts; the paper’s full framework is not reproduced.'),
('Can paperR² be compared with ourMAE?','No. The targets, units, references and datasets differ. R² and dispatch-cost deviations cannot rank watt/degree prediction errors.'),
('Why is an energy-saving claim unsupported?','There is no validated action-response model and controlled policy/energy experiment. Forecast accuracy alone does not show changed energy consumption or service outcomes.'),
('IsIPMIambient a calibrated inlet measurement?','Documentation supports an ambient/inlet interpretation, but exact placement, calibration and drift remain incompletely established. Use the recorded metric name and°C rather than invent sensor semantics.'),
('Is March untouched confirmatory data?','No. It was previously inspected and is explicitly presented as retrospective evaluation. Later model selection cannot be described as prospective confirmation.'),
('What is the historical integrity status?','Strict historical preservation preflight remainsFAILED; semantic reproduction remainsUNKNOWN. New scoped hash matches do not erase those statuses.'),
('What are the nine new manifest failures?','New history coverage/preflight manifests disagree with current source/test/log and historical input bindings in nine comparisons over eight unique files. Older historical input hashes still match their own records; causes of new expected-hash discrepancies remain unresolved.'),
('What is the next best experiment?','Complete the separately authorized locked March evaluation of the already-trained B1 centralized schedule-replay control on exactly the existing B1 windows, before adding more method variants.'),
('What remains unanswered?','Prospective multi-site generalization, reliable policy savings, calibrated uncertainty, controlled participation/FedProx effects, independent outage realizations and a fully reconciled new residual history benchmark.'),
('What is the actual contribution?','A functioning causal forecasting/evidence pipeline plus controlled comparisons showing where federation, residual learning and missing-input handling help or fail. Negative and mixed results are part of the contribution.')]
(P/'VIVA_QUESTIONS_AND_ANSWERS.md').write_text('# Viva preparation\n\n'+str(len(qa))+' evidence-bounded questions and answers. Results refer to the17November2026snapshot; consultSOURCE_INDEX.md for exact artifacts.\n\n'+'\n\n'.join(f'## {i}. {q}\n\n{a}' for i,(q,a) in enumerate(qa,1))+'\n')
# Presentation source diagrams in editable Mermaid.
(P/'FIGURES/system_architecture.mmd').write_text('''flowchart LR
 A[Raw telemetry / Parquet] --> B[Causal UTC bins]
 B --> C[Frozen IDs and observed targets]
 C --> D[Training-only normalization]
 D --> E[Local fitting and full-table FedAvg]
 E --> F[Validation selection and model lock]
 F --> G[Common test prediction artifacts]
 G --> H[Pooled / macro / node metrics]
 H -. Proposed .-> I[Validated action-response and decision layer]
''')
(P/'FIGURES/model_architecture.mmd').write_text('''flowchart LR
 X[Input B x 36 x 14] --> S[13 numeric channels]
 X --> N[Original node index]
 N --> E[979 x 8 embedding]
 S --> C[21 recurrent values]
 E --> C
 C --> G[Two GRU layers / hidden 64]
 G --> B[B1 linear 64 to 2 / standardized outputs]
 G --> T[T4 linear 64 to 1 / residual correction]
 B --> Y[Power W / IPMI ambient C]
 T --> P[Add last power / W]
''')
(P/'FIGURES/federated_workflow.mmd').write_text('''flowchart LR
 G[Global model] --> S[Uniform sample 98 of 979]
 S --> L[Copy same state to each selected client]
 L --> U[Fresh local optimizer / one epoch]
 U --> A[Full-table sample-weighted FedAvg]
 A --> G
 A --> V[Global validation / track-specific metric]
 V --> K[Earliest minimum checkpoint]
 K --> F[Immutable lock before test]
''')
(P/'README.md').write_text('''# M.Tech end-semester presentation

Presentation: **19 December2026**. Evidence snapshot: **17 November2026**.

- `FINAL_ENDSEM_PRESENTATION.pptx`:24widescreen slides; actual embedded speaker notes. Slide canvases are lossless1920×1080images for consistent viewing across applications.
- `FINAL_ENDSEM_PRESENTATION.pdf`:24vector/text pages from the same Matplotlib source; visually matches the PPTX artwork.
- `PRESENTATION_SCRIPT.md`:every slide’s spoken narrative, technical backup, transition, timing and faculty question. Planned19min55sec with pauses; rehearse for15–20minutes.
- `VIVA_QUESTIONS_AND_ANSWERS.md`:43questions with bounded answers.
- `SLIDES/`:lossless slidePNG artwork. `FIGURES/`:editableSVGs andMermaid diagram sources.
- `RENDERS/`:independentPDF-page rasterizations and contact sheets used for visual inspection.
- `EXPERIMENT_STATUS_REGISTER.csv`:190original track/identity records copied from the verified report; not190independent fits.
- `EVIDENCE/ongoing_history_status.json`:fresh external-progress snapshot; supersedes old status observations only. No provisional score is presented as a completed result.
- `PRESENTATION_METRICS.csv`:stored seed-level scores used in supporting analysis; tracks remain separate.
- `B1_SUPPORTING_METRICS.csv`:both pooled andmacro-node power/ambient comparisons with persistence.
- `SOURCE_INDEX.md/.json`:exact artifact paths and byte hashes.
- `VALIDATION_REPORT.md`:fresh checks, rendering method, known scientific caveats and preservation scope.
- `BUILD/`:source generators and slide content. Edit an isolated copy and publish a new version; never regenerate into this sealed package.

No PowerPoint/LibreOffice application was installed on this server. ThePPTX is assembled as anOOXML image-slide package with notes; its container, XML, relationships and image mapping are validated. ThePDF is generated from exactly the same slide artwork and every page is independently rendered for inspection. NativeOffice rendering was not available, so that application-specific compatibility remains untested. Full-slidePPT images are not individual editable text boxes; edit the accompanyingSVG/content/generator sources instead. No dependencies were installed.

The mid-semester deck was not found after bounded local searches. The official mandatory paperPDF was read from the existing verified local copy; no byte identity with an unavailable uploaded filename is asserted.

Do not infer approval of training, test execution, provenance repair or energy-management deployment from this presentation. Historical preflightFAILED and semantic reproductionUNKNOWN remain explicit. Nine known new history-manifest comparisons remainFAILED. New benchmark progress is externally generated and excluded from final result tables.
''')
(P/'EVIDENCE/documentation_checks.json').write_text(json.dumps({'metric_checks_passed':len(checks),'failed':0,'seed_metric_records':len(metrics),'viva_questions':len(qa),'new_training_or_inference':False},indent=2)+'\n')
print('Validated',len(checks),'stored-metric arithmetic/identity checks; metrics',len(metrics),'viva',len(qa))
