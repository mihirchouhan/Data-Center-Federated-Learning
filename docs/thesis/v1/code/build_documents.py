"""Documentation compiler: JSON/text/hash reads only; no model or evaluation imports."""
from pathlib import Path
import argparse,ast,hashlib,html,json,math,re,statistics,struct,zipfile
import xml.etree.ElementTree as ET

TITLE='Reproducible Node-Level Power Forecasting on M100'
SUBTITLE='Methodology and Results: FedAvg-GRU and a Matched Centralized-SGD Control'
FINAL='output/experiments/v3_centralized_sgd/reports/final_test_comparison_v1'
VALID='output/experiments/v3_centralized_sgd/reports/validation_comparison_v1'
FED='output/experiments/v2_fedavg/test_evaluation/v1_retry1'
CENTRAL='output/experiments/v3_centralized_sgd/test_evaluation/v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
def dump(p,v):write(p,(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())

class Compiler:
 def __init__(self,root):self.root=Path(root);self.sources=[];self.alias={};self.blocks=[];self.numeric_tables={};self.checks=[]
 def source(self,key,path,purpose):
  if key in self.alias:return self.alias[key]
  p=self.root/path
  if not p.is_file():raise ValueError('Missing source: '+path)
  sid='S'+str(len(self.sources)+1).zfill(2)
  self.sources.append({'id':sid,'path':path,'sha256':sha(p),'purpose':purpose})
  self.alias[key]=sid;return sid
 def load(self,key,path,purpose):self.source(key,path,purpose);return json.loads((self.root/path).read_text())
 def cite(self,*keys):return ' '.join('['+self.alias[k]+']' for k in keys)
 def h(self,text,level=1):self.blocks.append({'kind':'heading','level':level,'text':text})
 def p(self,text):self.blocks.append({'kind':'paragraph','text':text})
 def equation(self,text):self.blocks.append({'kind':'equation','text':text})
 def table(self,key,caption,head,rows,refs):
  self.blocks.append({'kind':'table','caption':caption+' '+self.cite(*refs),'headers':head,'rows':[[str(x) for x in r] for r in rows]})
  self.numeric_tables[key]={'headers':head,'rows':[[str(x) for x in r] for r in rows],'source_ids':[self.alias[k] for k in refs]}
 def figure(self,path,caption):self.blocks.append({'kind':'figure','path':path,'caption':caption})
 def receipt(self,path):
  v=json.loads((self.root/path).read_text())
  for name,expected in v['artifacts'].items():
   p=(self.root/path).parent/name
   if sha(p)!=expected:raise ValueError('Historical receipt mismatch: '+str(p))
  self.checks.append({'receipt':path,'artifacts_checked':len(v['artifacts']),'mismatches':[]})
 def compile(self,out):
  a=self.load('analysis',FINAL+'/analysis.json','Verified pooled and per-node test summaries')
  prov=self.load('provenance',FINAL+'/provenance.json','Exact test model/configuration and 2652-input hash provenance')
  ver=self.load('verification',FINAL+'/verification.json','Completed report-only tests and preservation results')
  self.source('finalreport',FINAL+'/REPORT.md','Verified final comparison and per-node interpretation')
  self.source('finalreceipt',FINAL+'/report_completed.json','Immutable final-report completion inventory')
  self.source('noderows',FINAL+'/per_node_summary.csv','All 979 node summaries')
  self.source('nodesseeds',FINAL+'/per_node_by_seed.csv','All 2937 node-seed comparisons')
  val=self.load('validation',VALID+'/comparison.json','Saved selected validation metrics and budgets')
  self.source('valreport',VALID+'/REPORT.md','Historical validation comparison snapshot before central test reporting')
  self.source('valprov',VALID+'/artifact_provenance.json','Validation artifact/config/model provenance')
  self.source('valreceipt',VALID+'/report_completed.json','Validation-report completion inventory')
  results={};locks={};manifests={}
  for method,base in [('fedavg',FED),('centralized_sgd',CENTRAL)]:
   results[method]=self.load(method+'results',base+'/results.json','Original completed '+method+' test scores')
   locks[method]=self.load(method+'lock',base+'/selection_lock.json','Pre-test model selections and raw export hashes')
   self.source(method+'evalmanifest',base+'/manifest.json','Test protocol, environment and selection lock binding')
   self.source(method+'plan',base+'/plan.json','Locked test plan and source fingerprints')
   self.source(method+'complete',base+'/completed.json','Original test artifact receipt')
  contract=self.load('contract','output/experiments/v1/frozen/contract.json','Frozen arrays, splits, feature order, scalers and hashes')
  meta=self.load('metadata','output/experiments/v1/frozen/preprocessing.json','Exact forecast target, feature units and preprocessing settings')
  audit=self.load('audit','output/m100_all_nodes/results/data_audit.json','Candidate/usable node cohort and coverage audit')
  coverage=self.load('coverage',CENTRAL+'/seed42/coverage.json','Actual test-origin/day coverage')
  self.source('downloadconfig','configs/m100_all_nodes.json','Recorded dataset months, source record and preprocessing configuration')
  for key,path,purpose in [
   ('model','src/model.py','SequenceRegressor.__init__/forward: embedding, GRU, head and residual baseline'),
   ('multinode','src/multinode.py','weighted_bins, node_windows and prepare_multinode: feature construction and chronology'),
   ('preprocess','src/preprocess.py','Metric schema, units, range checks and _weather_asof'),
   ('expdata','src/experiment_data.py','FEATURES, weather_view, frozen contract and scaler/persistence interface'),
   ('evalcode','src/experiment_eval.py','metric_pair/evaluate: original physical-unit metrics and persistence'),
   ('feddata','src/federated_data.py','Training-only client memberships and coordinator validation capabilities'),
   ('fedavg','src/federated_train.py','GlobalModelState and aggregate_fedavg, including fixed buffers and embeddings'),
   ('fedrounds','src/federated_rounds.py','Initialization, RNG streams, sampling, local SGD and communication rounds'),
   ('fedio','src/federated_io.py','Manifest, atomic commit, receipt and export protocol'),
   ('centralcode','src/centralized_sgd.py','Reference verification, plan_block/train_block and continuing central model'),
   ('centralio','src/centralized_io.py','Central block commits, reference verification and selected export integrity'),
   ('fedeval','src/federated_evaluate.py','Locked final-test contract, seed aggregation and finite/alignment guards'),
   ('centraleval','src/centralized_evaluate.py','Locked central block selections and comparison with stored FedAvg'),
   ('predictcode','src/experiment_models.py','neural_predict: batch-copy weather masking during existing inference'),
   ('readme','README.md','General project history, dataset identification and operational context'),
   ('experiments','EXPERIMENTS.md','Historical v1 frozen-data and common-evaluation documentation'),
   ('v2readme','configs/experiments/v2_fedavg/README.md','Approved FedAvg protocol and workflow'),
   ('v3readme','configs/experiments/v3_centralized_sgd/README.md','Matched control and schedule-replay limitations')]:self.source(key,path,purpose)
  for method in ['fedavg','centralized_sgd']:
   manifests[method]=[]
   for model in prov['evaluations'][method]['models']:
    seed=model['seed'];key=method+str(seed)
    m=self.load(key,model['run_dir']+'/manifest.json','Completed '+method+' seed'+str(seed)+' training manifest')
    self.source(key+'config',model['configuration_path'],'Exact completed training configuration')
    suffix='selected/receipt.json' if method=='fedavg' else 'final/receipt.json'
    self.source(key+'selected',model['run_dir']+'/'+suffix,'Selected export receipt')
    self.source(key+'model',model['model_path'],'Raw selected model export, hashed only; never loaded')
    assert sha(self.root/model['model_path'])==model['model_sha256']
    assert sha(self.root/model['configuration_path'])==model['configuration_sha256']
    entry=next(x for x in locks[method]['models'] if x['seed']==seed)
    completion=model['run_dir']+('/'+entry['completion_event'] if method=='fedavg' else '/completed.json')
    completed=self.load(key+'completion',completion,'Completed training counters and selected-checkpoint metadata')
    budget=next(x for x in val['centralized_sgd'] if x['seed']==seed)
    assert completed['examples_processed']==budget['examples_processed'] and completed['optimizer_steps']==budget['optimizer_steps']
    selected=model['selected_index'];sub='rounds' if method=='fedavg' else 'blocks'
    self.source(key+'chosen',model['run_dir']+f'/{sub}/{selected:04d}/receipt.json','Saved selected validation score and checkpoint fingerprint')
    manifests[method].append(m)
  for path in [FINAL+'/report_completed.json',VALID+'/report_completed.json',FED+'/completed.json',CENTRAL+'/completed.json']:self.receipt(path)
  assert ver['status']=='passed' and ver['maximum_metric_reconciliation_difference_w']==0
  assert meta['sample_counts']=={'train':4506133,'val':199670,'test':75566}
  assert a['samples_per_seed']==75566 and a['evaluated_nodes']==979 and coverage['unique_origins']==94
  assert len(coverage['per_day'])==6 and sum(x['windows'] for x in coverage['per_day'].values())==75566
  assert [r['seed'] for r in results['fedavg']['fits']]==[42,43,44]
  assert results['fedavg']['window_identity']==results['centralized_sgd']['window_identity']==prov['window_identity']
  for method in ['fedavg','centralized_sgd']:
   for metric in ['rmse','mae']:
    assert a['summary'][method]['mean_'+metric+'_w']==results[method]['mean_'+metric+'_w']
    vals=[r[metric+'_w'] for r in results[method]['fits']]
    assert math.isclose(statistics.mean(vals),a['summary'][method]['mean_'+metric+'_w'],abs_tol=1e-10)
    assert math.isclose(statistics.stdev(vals),a['summary'][method]['std_'+metric+'_w'],abs_tol=1e-10)
    vvals=[r[metric+'_w'] for r in val[method]]
    assert math.isclose(statistics.mean(vvals),val['summary'][method][metric]['mean_w'],abs_tol=1e-10)
    assert math.isclose(statistics.stdev(vvals),val['summary'][method][metric]['sample_sd_w'],abs_tol=1e-10)
  features=meta['feature_names'];assert len(features)==14 and contract['weather_mask_indices']==[7,8]
  cfg=manifests['fedavg'][0]['config'];assert (cfg['hidden_size'],cfg['num_layers'],cfg['node_embedding_size'])==(64,2,8)
  assert all(m['config']['weather']=='masked' and m['config']['optimizer']=='sgd' and m['config']['rounds']==200 for ms in manifests.values() for m in ms)
  # Formal document begins here. Every result table comes from saved JSON.
  self.h(TITLE);self.p(SUBTITLE);self.p('M.Tech Major Project • Methodology and Results • Documentation version 1 • Evidence snapshot: 24 October 2026')
  self.h('Abstract',2)
  self.p('This study investigates one-hour-ahead, node-level electrical-power forecasting using the M100 telemetry dataset. A residual gated recurrent unit (GRU) model is trained using simulated federated averaging (FedAvg) and compared with a matched centralized stochastic-gradient-descent (SGD) control and persistence. The common frozen cohort comprises 979 compute-node identifiers, 4,506,133 training windows, 199,670 validation windows and 75,566 test windows. The measured comparison uses one hour of input history and masks standardized outdoor temperature and humidity while retaining indoor inlet temperature. '+self.cite('metadata','analysis','audit'))
  self.p('Across seeds 42, 43 and 44, centralized SGD obtains test RMSE 56.016953 ± 0.852660 W and MAE 30.516738 ± 1.941197 W; FedAvg obtains RMSE 63.231179 ± 0.125510 W and MAE 33.924637 ± 0.285604 W. Persistence gives RMSE 64.417454 W and MAE 30.926903 W. The deviations are sample standard deviations of three individual seed scores on the same dataset, not confidence intervals or independent-dataset estimates. Centralized SGD has lower measured test errors under this protocol; the results do not establish a general ranking of learning paradigms. The test period was already observed, and neither energy savings nor a fully privacy-preserving deployment has been demonstrated. '+self.cite('analysis','fedavgresults','centralized_sgdresults'))
  self.h('Document scope and evidence convention',2)
  self.p('This document supplies methodology, results, interpretation and future-work material for an M.Tech thesis. It does not invent a literature review or external bibliographic citations. Bracketed S-identifiers refer to the exact repository evidence listed in Appendix C and SOURCES.md/SOURCES.json, including SHA-256 hashes. Measured findings are separated from the unexecuted proposals in Section 9. Historical documentation is interpreted according to its original stage; later completed artifacts take precedence over statements that training or test reporting had not yet occurred.')
  self.h('1. Research problem, objectives and scope')
  self.p('The research problem is to forecast each monitored compute node’s average electrical power over the next hour from its recent operational, thermal and calendar context. Such forecasts may support later capacity or energy-management research, but forecasting accuracy alone does not measure the effect of any control policy. The present task is continuous regression in watts, not workload classification, carbon-emission prediction, total-facility-power prediction or an operational energy-optimization experiment. '+self.cite('metadata','evalcode'))
  self.p('The objectives are to establish a reproducible chronological forecasting cohort; implement a shared residual GRU with node identity; compare FedAvg with a centralized control that matches initialization, optimizer settings and reconstructed training exposures; quantify pooled and per-node errors against persistence; and preserve the provenance of validation selection and final test reporting. The comparison concerns already-known nodes from one facility, with one simulated client per node. It is not an independent-data-center federation or an unseen-node generalization study. '+self.cite('audit','feddata','v2readme','v3readme'))
  self.p('All measurements in this document are taken from existing completed artifacts. Documentation preparation performed no training, checkpoint inference or evaluation rerun. The existing final report had already reconciled the saved prediction archives with the frozen reference and found zero discrepancy in the checked W metrics. This document uses those saved summaries and verifies their integrity and arithmetic only. '+self.cite('verification','provenance'))
  self.h('2. Dataset, cohort and forecast formulation')
  self.h('2.1 Dataset identity and unit of observation',2)
  self.p('The repository identifies the source as M100 Dataset 5 and records January–February 2022 archives under Zenodo record 7589942. The prepared audit describes a shared-model cohort from one facility. Node strings are compute-node identifiers; they must not be relabeled as separate sites or data centers. The audit discovers 980 candidate identifiers and retains 979. Identifier 0 is excluded because it has insufficient complete windows in a chronological split: its audit records 3,210 training windows and zero validation/test windows. Node eligibility requires the relevant sensor records and at least ten complete windows in each split. '+self.cite('readme','downloadconfig','audit'))
  self.table('cohort','Table 2.1. Frozen benchmark cohort',['Split','Accepted windows','Input shape per window','Target'],[['Training','4,506,133','6 × 14','One next-hour mean'],['Validation','199,670','6 × 14','One next-hour mean'],['Test','75,566','6 × 14','One next-hour mean']],['contract','metadata'])
  self.p('The full calendar grid runs from 2022-01-01 00:00 UTC to 2022-03-01 00:00 UTC exclusive. Accepted windows are incomplete in time and uneven across nodes. In the test set, 94 distinct forecast origins occur on six calendar dates; individual nodes have 46–93 accepted test windows. These counts describe filtered forecasting windows, not a continuously observed test trace. No cause of every coverage gap is inferred from aggregate counts. '+self.cite('audit','coverage','analysis'))
  self.table('coverage','Table 2.2. Recorded test coverage by UTC date',['Date','Accepted windows','Distinct origins on date'],[[d,f"{r['windows']:,}",r['unique_origins']] for d,r in coverage['per_day'].items()],['coverage'])
  self.h('2.2 History and next-hour average-power target',2)
  self.p('At forecast origin t, six completed ten-minute input bins span [t−60 min, t). The scalar target is the arithmetic mean of six valid future ten-minute node-power bins spanning [t, t+60 min). Thus the horizon denotes the next hour’s mean power, not the instantaneous power exactly sixty minutes later. A candidate origin is tested on the ten-minute grid; overlapping histories and target intervals create temporal dependence between adjacent windows. All target bins and all required input features must be valid. '+self.cite('multinode','metadata'))
  self.equation('y_n(t) = (1/6) ∑(j=0,…,5) p_n([t+10j min, t+10(j+1) min)) .     (1)')
  self.p('Here p_n denotes the prepared ten-minute power estimate in kW. The exact saved target name is next_horizon_mean_node_it_power_kw. Predictions and targets are converted to W for reporting by inverse target scaling followed by multiplication by 1000. The target is node-level IT power as defined by the recorded total_power metric, not measured cooling power, PUE or energy consumption. '+self.cite('metadata','evalcode','preprocess'))
  self.h('2.3 Causal preprocessing and missing data',2)
  self.p('Node metric records are stably sorted and deduplicated by node/time, retaining the last occurrence in the recorded ordering. Invalid finite/range observations are rejected: percentages must lie in [0,100], power and free memory are nonnegative, and total memory must be positive. Temperature channels require finite values without speculative temperature cutoffs. Local sensor bins use causal time-weighted sample holds capped at 300 s. A bin requires at least 50% time coverage. Consequently, the target averages valid coverage-controlled bin estimates; it is not a guarantee that every second of the future hour had a sensor observation. '+self.cite('multinode','preprocess','metadata'))
  self.p('Outdoor weather is joined using the latest observation strictly before the input bin’s right edge, with a maximum age of 30 minutes. Missing/stale weather remains invalid rather than being filled from a future observation. Memory-used fraction is 1 − binned(mem_free)/binned(mem_total), and fractions outside [0,1] become invalid. A window is omitted if any required history feature or future power bin is invalid. Windows do not concatenate across node identities or missing temporal bins. These causal holds are distinct from arbitrary interpolation or backward filling. '+self.cite('multinode','preprocess'))
  self.h('2.4 Exact feature order, units and weather mask',2)
  descriptions=[('IPMI total_power bin estimate divided by 1000','operational'),('Ganglia cpu_user bin estimate','operational'),('Ganglia cpu_system bin estimate','operational'),('Ganglia cpu_idle bin estimate','operational'),('Ganglia cpu_wio bin estimate','operational'),('1 − binned(mem_free)/binned(mem_total)','operational'),('Binned IPMI ambient; historical feature name, not a verified spatial average of multiple inlets','environmental'),('As-of weather temp before bin right edge','environmental'),('As-of weather humidity before bin right edge','environmental'),('sin of UTC daily phase at bin right edge','time-derived'),('cos of UTC daily phase at bin right edge','time-derived'),('sin of weekly phase at bin right edge','time-derived'),('cos of weekly phase at bin right edge','time-derived'),('Position in frozen node_names mapping','other: categorical')]
  self.table('features','Table 2.3. Exact frozen input channels',['Index','Exact feature name','Source or transformation','Unit','Category','Masked'],[[i,n,descriptions[i][0],meta['feature_units'][i],descriptions[i][1],'Yes' if i in [7,8] else 'No'] for i,n in enumerate(features)],['metadata','contract','multinode','preprocess'])
  self.p('For the calendar channels, with s equal to the UTC bin-right-edge Unix time in seconds, daily phase is 2π(s mod 86400)/86400. Weekly phase is 2π((s+3×86400) mod 604800)/604800, as implemented. All thirteen non-ID channels are standardized with pooled training-window means and standard deviations; near-zero standard deviations are replaced by one. The target has its own training-only mean and standard deviation. Node index 13 is categorical and retains mean 0 and scale 1 rather than being standardized. The frozen scalers are reused without fitting on validation or test. '+self.cite('multinode','metadata'))
  self.p('Both measured methods use weather="masked": standardized channels 7 and 8 are set to zero on batch copies, representing their respective training means. This preserves all fourteen input slots, node IDs, calendar values, targets, window identities and scalers. Index 6, indoor inlet temperature, remains available; therefore this is outdoor-weather masking, not removal of every environmental feature. Because complete windows were frozen before masking, the mask does not recover windows previously lost to missing weather. GPU-utilization sensor channels are excluded by the preprocessing configuration; this does not imply that model training ran on a CPU. '+self.cite('expdata','predictcode','metadata'))
  self.h('2.5 Chronological split protocol and eligibility caveat',2)
  cuts=meta['split_boundaries_utc']
  self.table('splits','Table 2.4. Shared calendar split boundaries',['Split','Lower boundary, inclusive UTC','Upper boundary, exclusive UTC'],[['Training','2022-01-01 00:00:00','2022-02-11 07:10:00'],['Validation','2022-02-11 07:10:00','2022-02-20 03:30:00'],['Test','2022-02-20 03:30:00','2022-03-01 00:00:00']],['contract','metadata'])
  self.p('The cut indices are obtained from 70% and 85% of the common calendar-bin grid, with integer truncation; the remaining interval is test. These are calendar proportions, not guaranteed 70/15/15 proportions of accepted windows. Every target interval must lie wholly inside its assigned split. Validation/test history may reach into the preceding split because it contains only observations available before the new forecast origin. The same node mapping and calendar cuts are used by both methods. '+self.cite('multinode','metadata'))
  self.p('The training workers’ API exposes client-local training examples only, while the coordinator owns a separate complete-validation capability. Final test access occurs through explicitly locked evaluation adapters. However, original cohort admission uses completeness in all splits, including test-period availability. This retrospective coverage-based selection is a limitation even though future target values were not fitted into the training scalers. The benchmark describes forecasting for this selected cohort, not arbitrary missing-data or unseen-node deployment. '+self.cite('audit','feddata','fedeval','centraleval'))
  self.h('3. Model architecture and persistence reference')
  self.table('architecture','Table 3.1. Shared residual-GRU architecture',['Component','Verified implementation'],[['Saved input','Six time steps, fourteen channels'],['Node identity','Categorical index at channel 13; frozen mapping of 979 nodes'],['Embedding','Learned 979 × 8 table; replaces the index inside the network'],['Recurrent input width','13 non-ID channels + 8 embedding values = 21'],['Recurrent model','torch.nn.GRU, two layers, hidden size 64, batch_first=True'],['Sequence reduction','Final output time step of the top recurrent layer'],['Regression head','Linear 64 → 1'],['Residual reference','Fixed two-element baseline_coefficients buffer'],['Initial correction','Linear head weight and bias initialized to zero']],['model','fedrounds','fedavg42'])
  self.p('The stored node index is one of the fourteen input channels; the eight-dimensional embedding is a separate learned parameter table, not eight additional saved feature channels. It is repeated at every history step after the categorical index is removed from the numerical input. The architecture therefore consumes a 21-dimensional recurrent vector. The code defines a standard many-to-one GRU and linear residual head; no attention mechanism, exogenous forecast decoder or additional undocumented layer is asserted. '+self.cite('model'))
  self.equation('ẑ_y = Linear(GRU([z_0,…,z_12, embedding(node)] at the final history step)) + a z_power,last + b .     (2)')
  self.equation('a = σ_x,power / σ_y ;     b = (μ_x,power − μ_y) / σ_y ;     ŷ_W = 1000(σ_y ẑ_y + μ_y) .     (3)')
  self.p('The zero-initialized correction makes the initial candidate mathematically persistence. Recurrent and embedding weights are freshly seeded rather than loaded from a centrally trained checkpoint. The model stores baseline coefficients as float32, whereas the physical persistence vector is reconstructed in float64 by the common evaluator; the initial neural candidate need not be bit-identical to the float64 reference. baseline_coefficients remain fixed during both training and aggregation. '+self.cite('model','multinode','evalcode','fedavg','fedrounds'))
  self.p('Persistence predicts the last completed ten-minute power-bin mean for the entire next-hour average-power target. It uses the last input step, channel 0. It is neither the preceding hour’s average nor a fitted regression model, and it has no training seed. It is included as the initial validation-selection candidate as well as a separately reported physical-unit baseline. '+self.cite('evalcode','fedrounds','centralcode'))
  self.h('4. Experimental methodology')
  self.h('4.1 Common settings and training objective',2)
  self.table('protocol','Table 4.1. Fixed measured comparison protocol',['Setting','Recorded value'],[['Seeds','42, 43, 44'],['Initialization','Fresh seeded GRU/embedding; zero correction head; matched initial state'],['Optimizer','SGD; learning rate 0.01; momentum 0; weight decay 0'],['Objective','Mean squared error of standardized target predictions'],['Batch size','512; final partial batch retained (drop_last=False)'],['Client population','979 simulated compute-node clients'],['Selected clients per round/block','98, uniformly sampled without replacement'],['Local passes','One epoch per selected client session'],['Training cap','200 communication rounds / 200 central schedule blocks'],['Validation schedule','Initial candidate, then every ten rounds/blocks, including 200'],['Selection rule','Earliest minimum pooled validation RMSE in W'],['Evaluation batch size','1024'],['Training tensor dtype','float32']],['fedavg42','centralized_sgd42','fedrounds','centralcode'])
  self.equation('L_B(θ) = (1/|B|) ∑(i∈B) (ẑ_y,i − z_y,i)² .     (4)')
  self.p('Optimization minimizes scaled MSE, while checkpoint selection uses complete-cohort validation RMSE after conversion to W. Thus MAE is a secondary reported criterion rather than the checkpoint-selection objective. A strict-improvement rule retains the earliest tied candidate, including the initial persistence candidate. There are twenty-one scheduled validation candidates at indices 0,10,…,200; no test score chooses a seed, round, block or hyperparameter. '+self.cite('fedrounds','centralcode','fedeval','centraleval'))
  self.h('4.2 FedAvg training and client structure',2)
  self.p('One client comprises all frozen training windows for one recorded node. Membership is disjoint and exhaustive over 4,506,133 training windows. The client interface preserves sequence order, frozen normalization and categorical IDs. The approved main protocol requires at least 1,024 training windows across at least seven distinct training days per client. The recorded seed42 client manifest contains 979 clients with 2,570–4,907 training windows each; clients are not filtered anew during rounds. '+self.cite('feddata','fedavg42'))
  self.p('At each round, the coordinator samples 98 clients without replacement. Every selected client receives a clone of the same immutable global state, creates a fresh stateless SGD optimizer, shuffles its own training windows deterministically and processes one complete local epoch. Local losses, gradients and returned states must be finite. Invalid counts, stale snapshots or incompatible updates abort a round rather than silently changing the participants or weights. '+self.cite('fedrounds','fedavg'))
  self.equation('θ_(r+1) = ∑(k∈S_r) [n_k / ∑(j∈S_r)n_j] θ_(r,k) ,     for trainable parameters.     (5)')
  self.p('Here n_k is the participating client’s local training-window count. The implementation accumulates weighted trainable tensors in float64 on the CPU and casts back to the recorded dtype; fixed buffers retain their original values. Standard FedAvg applies to the entire shared embedding table. A client normally changes its own node row, while unchanged rows enter the same weighted average. Consequently, a client-specific embedding correction is diluted by that client’s participation weight. This is a documented mechanism, not an experimentally isolated explanation of the measured accuracy gap. '+self.cite('fedavg','model'))
  self.p('Initialization, client sampling and local shuffling use separate deterministic SeedSequence domains 0, 1 and 2. Sampling is indexed by seed and round; local shuffle additionally uses node index. Atomic round commits and previous-state fingerprints support complete-round replay after interruption. This describes the implementation and its saved provenance; documentation preparation did not repeat a resume experiment. '+self.cite('fedrounds','fedio'))
  self.h('4.3 Matched centralized-SGD control',2)
  self.p('The control starts from the same fresh initialization as its paired FedAvg seed and uses the same architecture, weather mask, frozen cohort, loss, nominal learning rate and batch size. It reconstructs the original ordered client sessions, shuffle seeds, row-order hashes and batch sizes. One continuing central model processes those node-grouped batches sequentially; there is no averaging or reset to a common global state between nodes. A stateless SGD optimizer is initialized for each central block; zero momentum and decay mean no persistent optimizer history is carried between blocks. '+self.cite('centralcode','centralio','v3readme'))
  self.p('The matching evidence consists of source/configuration binding, recorded FedAvg receipts, fresh-state fingerprints and deterministic reconstruction over frozen row IDs. An independent historical trace of every original FedAvg batch was not saved. The control is therefore a schedule-replay centralized control, not unrestricted pooled random mixing, and common settings do not imply identical effective optimization trajectories. All runs completed their 200-index cap; validation may select an earlier exported model. '+self.cite('valprov','provenance','centralcode'))
  self.table('budgets','Table 4.2. Recorded matched completed training budgets',['Seed','Example exposures per method','Local/central SGD steps per method','FedAvg selected round','Central selected block'],[[s,f"{val['centralized_sgd'][i]['examples_processed']:,}",f"{val['centralized_sgd'][i]['optimizer_steps']:,}",200,[190,190,180][i]] for i,s in enumerate([42,43,44])],['validation','provenance'])
  self.p('Each completed run comprises 19,600 client sessions (98 × 200). Example exposures count repeated visits to training windows, not distinct observations or global epochs. Matching the completed cap does not match the effective training budget of differently selected earlier checkpoints. Recorded wall times vary substantially across seeds, and saved measurements do not resolve historical resource contention or utilization. No runtime-speedup or energy-efficiency conclusion is drawn from those timings. '+self.cite('validation','valreport'))
  self.h('4.4 Validation selection, locked test reporting and metrics',2)
  self.p('The coordinator evaluates the complete 199,670-window validation cohort and selects the earliest minimum pooled validation RMSE. The selected FedAvg round is 200 for every seed; central blocks are 190,190,180 for seeds 42,43,44. Final test plans bind these selections, model/configuration hashes and execution-source snapshots before test examples are opened. The completed central evaluator references the earlier FedAvg test evaluation instead of rerunning it. '+self.cite('fedavglock','centralized_sgdlock','fedeval','centraleval'))
  self.equation('MAE_s = (1/N) ∑|ŷ_i,s − y_i| ;     RMSE_s = √[(1/N) ∑(ŷ_i,s − y_i)²] .     (6)')
  self.equation('Mean = (q_42 + q_43 + q_44)/3 ;     sample SD = √[∑(q_s − Mean)²/(3−1)] .     (7)')
  self.p('Each seed’s pooled error uses all accepted windows. Across-seed summaries average the individual scores, not prediction vectors. The three seeds are stochastic repeats on one shared dataset, not independent datasets; their SD is not a confidence interval or a significance test. Overlapping windows, shared timestamps and common facility conditions also limit assumptions of independent observations. All seeds are retained and no best test seed is selected. '+self.cite('evalcode','fedeval','analysis'))
  self.h('5. Validation results')
  vrows=[]
  for i,s in enumerate([42,43,44]):
   c,f=val['centralized_sgd'][i],val['fedavg'][i]
   vrows.append([s,c['selected_block'],f"{c['rmse_w']:.6f}",f"{c['mae_w']:.6f}",f['selected_round'],f"{f['rmse_w']:.6f}",f"{f['mae_w']:.6f}"])
  self.table('validation_seeds','Table 5.1. Selected validation errors in W',['Seed','Central block','Central RMSE','Central MAE','FedAvg round','FedAvg RMSE','FedAvg MAE'],vrows,['validation'])
  self.table('validation_summary','Table 5.2. Validation mean ± sample SD in W',['Method','RMSE','MAE'],[[m,f"{val['summary'][m]['rmse']['mean_w']:.6f} ± {val['summary'][m]['rmse']['sample_sd_w']:.6f}",f"{val['summary'][m]['mae']['mean_w']:.6f} ± {val['summary'][m]['mae']['sample_sd_w']:.6f}"] for m in ['centralized_sgd','fedavg','persistence']],['validation'])
  self.p('All validation results refer to the same 199,670 windows and 979 nodes. Centralized SGD lowers mean validation RMSE relative to FedAvg but has a higher mean validation MAE. Both models have lower validation RMSE and higher validation MAE than persistence. This trade-off is consistent with RMSE-based selection and MSE optimization; it does not demonstrate a specific causal mechanism. Persistence has zero across-seed SD because its prediction vector and cohort are fixed. '+self.cite('validation'))
  self.figure('figures/validation_curves.png','Figure 5.1. Existing saved validation trajectories for the three seeds; copied without regeneration. '+self.cite('valreport'))
  self.p('The validation report was produced before the central test-evaluation stage and states that central test metrics were not yet available. That historical statement is preserved in the original report. The later completed central test artifacts and final comparison supply Section 6; no historical report was rewritten to conceal this progression. '+self.cite('valreport','centralized_sgdcomplete','finalreport'))
  self.h('6. Test results and per-node error analysis')
  trows=[]
  for i,s in enumerate([42,43,44]):
   c,f=results['centralized_sgd']['fits'][i],results['fedavg']['fits'][i]
   trows.append([s,c['selected_block'],f"{c['rmse_w']:.6f}",f"{c['mae_w']:.6f}",f['selected_round'],f"{f['rmse_w']:.6f}",f"{f['mae_w']:.6f}"])
  self.table('test_seeds','Table 6.1. Locked selected-model test errors in W',['Seed','Central block','Central RMSE','Central MAE','FedAvg round','FedAvg RMSE','FedAvg MAE'],trows,['centralized_sgdresults','fedavgresults'])
  self.table('test_summary','Table 6.2. Test mean ± sample SD in W',['Method','RMSE','MAE'],[[m,f"{a['summary'][m]['mean_rmse_w']:.6f} ± {a['summary'][m]['std_rmse_w']:.6f}",f"{a['summary'][m]['mean_mae_w']:.6f} ± {a['summary'][m]['std_mae_w']:.6f}"] for m in ['centralized_sgd','fedavg']]+[['persistence',f"{a['summary']['centralized_sgd']['mean_persistence_rmse_w']:.6f} ± 0.000000",f"{a['summary']['centralized_sgd']['mean_persistence_mae_w']:.6f} ± 0.000000"]],['analysis','centralized_sgdresults','fedavgresults'])
  self.p('The quoted FedAvg test metrics, 63.23 ± 0.13 W RMSE and 33.92 ± 0.29 W MAE when rounded, originate from output/experiments/v2_fedavg/test_evaluation/v1_retry1. Its authoritative seed42 export is in gru_masked_seed42b; seeds43/44 use their corresponding recorded directories. They are not validation scores, macro-node averages or the earlier v1 centralized benchmark. The prior final-report audit reconciles every individual and per-node score from all six saved prediction archives with zero checked metric discrepancy and verifies 44 FedAvg and 48 central test artifacts. '+self.cite('provenance','verification','fedavgresults'))
  self.table('reductions','Table 6.3. Reductions in mean test error, percent; positive is better',['Metric','Central vs FedAvg','Central vs persistence','FedAvg vs persistence'],[[k.replace('_w','').upper(),f"{a['comparisons'][k]['centralized_reduction_vs_fedavg_percent']:.6f}",f"{a['comparisons'][k]['centralized_reduction_vs_persistence_percent']:.6f}",f"{a['comparisons'][k]['fedavg_reduction_vs_persistence_percent']:.6f}"] for k in ['rmse_w','mae_w']],['analysis'])
  self.p('Error reduction is 100 × (reference error − model error)/reference error; it is not a percentage of predictions classified as correct. Centralized SGD improves both test errors relative to FedAvg in every paired seed. Its mean MAE gain over persistence is modest, and seed42 central MAE is worse than persistence, while seeds43/44 are better. FedAvg reduces pooled test RMSE relative to persistence but increases pooled MAE. Validation and test describe different periods, so the mean MAE ordering observed during validation must not be substituted for the test ordering. '+self.cite('analysis','validation'))
  self.h('6.1 Pooled errors versus equal-node averages',2)
  self.equation('Pooled MAE = ∑_n (N_n/N) MAE_n ;     pooled RMSE = √[∑_n (N_n/N) RMSE_n²] .     (8)')
  self.equation('Equal-node mean MAE = (1/K)∑_n MAE_n ;     equal-node mean RMSE = (1/K)∑_n RMSE_n .     (9)')
  self.p('Pooled metrics weight every forecasting window equally. Equal-node averages weight each node equally regardless of its accepted-window count. Moreover, an average of node RMSEs does not equal the square root of pooled MSE. The primary headline comparison uses pooled metrics; equal-node summaries characterize the distribution of performance across clients. '+self.cite('evalcode','analysis'))
  self.table('macro','Table 6.4. Equal-node errors averaged across the three seeds, W',['Metric','Central','FedAvg','Persistence'],[[k.replace('_w','').upper(),*[f"{a['per_node_analysis'][k]['macro_node_errors_mean_across_seeds_w'][m]:.6f}" for m in ['centralized','fedavg','persistence']]] for k in ['rmse_w','mae_w']],['analysis'])
  self.p('FedAvg’s equal-node mean RMSE is slightly above persistence even though its pooled RMSE is below persistence. This is consistent with different weighting and nonlinear averaging, rather than an inconsistency in the saved scores. Centralized SGD improves both forms of summary here. Neither pooled averages nor macro averages imply uniformly better performance at every node. '+self.cite('analysis'))
  self.h('6.2 Paired per-node differences',2)
  self.p('For each node and seed, the signed difference is central error minus FedAvg error in W, so a negative value favors the centralized control. Each node’s summary difference averages its three paired seed differences. All 979 nodes and 2,937 node–seed pairs are retained, including regressions. The existing CSVs include sample counts, per-seed errors, persistence comparisons, paired-difference SDs and node-target variability; no node is removed because of poor performance. '+self.cite('noderows','nodesseeds','analysis'))
  self.table('nodewins','Table 6.5. Counts based on each node’s across-seed mean error',['Metric','Central better','FedAvg better','Exact ties','Central better in all 3 seeds','Median central−FedAvg W'],[[k.replace('_w','').upper(),a['per_node_analysis'][k]['centralized_better_nodes'],a['per_node_analysis'][k]['fedavg_better_nodes'],a['per_node_analysis'][k]['exact_ties'],a['per_node_analysis'][k]['centralized_better_all_three_seeds'],f"{a['per_node_analysis'][k]['delta_mean_quantiles_w']['median']:.6f}"] for k in ['rmse_w','mae_w']],['analysis'])
  self.figure('figures/per_node_comparison.png','Figure 6.1. Existing paired node-error plots: points below the diagonal and negative deltas favor centralized SGD. All nodes are retained; copied without regeneration. '+self.cite('finalreport','analysis'))
  self.p('The node-level results demonstrate heterogeneity within the measured cohort: the centralized advantage in pooled error coexists with nodes favoring FedAvg. These descriptive counts do not establish independent-client statistical significance or authorize choosing a client subset after seeing test scores. Error heterogeneity should motivate separately designed diagnostic studies rather than retrospective alteration of the reported cohort. '+self.cite('analysis'))
  self.h('7. Interpretation and threats to validity')
  self.p('RMSE penalizes large deviations more strongly than MAE. A model can lower RMSE while increasing typical absolute error; the observed FedAvg–persistence comparison illustrates this distinction. The chosen checkpoint criterion directly targets RMSE, not MAE. Reports should therefore present both metrics, per-node distributions and the baseline rather than compressing forecasting performance into an unsupported single accuracy percentage. The metric pattern is consistent with differing error distributions, but an exact causal account requires a controlled diagnostic experiment. '+self.cite('analysis','fedrounds','centralcode'))
  self.p('Under this fixed simulated-node, schedule-replay protocol, the centralized control has lower measured test RMSE and MAE than FedAvg. The experiments do not show that federated learning outperforms centralized learning, nor do they show that every centralized approach will outperform every federated approach. Learning-rate equivalence, aggregation, node embedding updates, heterogeneous client histories and selected checkpoints can influence the gap; no causal ablation isolating these factors was performed. '+self.cite('analysis','fedavg','centralcode'))
  self.p('First, the test period has already been observed in previous project stages, including earlier model comparisons and FedAvg reporting. It is an observed development benchmark, not untouched confirmatory data. Later checkpoint selections were locked on validation, but this does not erase historical test exposure. Future confirmatory claims require a genuinely independent period or cohort reserved before further method selection. '+self.cite('contract','finalreport','fedavglock','centralized_sgdlock'))
  self.p('Second, seeds share one dataset and one test cohort; their sample SD quantifies this limited optimization/sampling variation, not population uncertainty. Overlapping forecasts and shared environmental conditions create dependencies. The evidence spans two months at one facility and only known nodes; it does not establish cross-facility, unseen-client, long-term or season-wide generalization. Sparse accepted coverage and test-availability-based cohort admission constrain external validity. '+self.cite('metadata','audit','coverage','analysis'))
  self.p('Third, prediction does not prove energy savings. The experiments contain no implemented scheduler or cooling controller, no causal intervention, no measured kWh reduction and no PUE optimization outcome. Node power in W differs from energy integrated over time and from total facility power. No measured carbon savings or carbon-emission prediction result is reported. Any such claim would require additional measurements, boundary definitions and an independent optimization/control study.')
  self.p('Fourth, the experiments simulate federated clients in a centrally accessible process and use centrally prepared pooled-training scalers. The local-data API restricts which arrays a worker receives, but it is not an operating-system isolation or cryptographic boundary. Secure aggregation, differential privacy, protected network transport, attack resistance and a fully privacy-preserving deployment were not established. Shared parameters and node identities should not be interpreted as a privacy guarantee. '+self.cite('feddata','fedavg','fedavg42'))
  self.h('8. Reproducibility and artifact integrity')
  environment=manifests['fedavg'][0]['environment']
  self.table('environment','Table 8.1. Historical training environment recorded in the run manifest',['Field','Recorded value'],[['Python',environment['python']],['PyTorch',environment['dependencies']['torch']],['NumPy',environment['dependencies']['numpy']],['CUDA build',environment['torch_cuda']],['Device',environment['device']],['GPU',environment['device_name']]],['fedavg42'])
  self.p('The runs record software versions, deterministic-algorithm settings, seeds, source snapshots, frozen-contract fingerprints, selected exports and receipt chains. Deterministic cuDNN/CUBLAS settings and disabled TF32 are implemented for these runs; exact cross-hardware bit identity is not promised. Checkpoints and interrupted-round recovery are governed by immutable commits and fingerprints. Recorded validation export/reload checks are historical evidence, not checks rerun while preparing this document. '+self.cite('fedrounds','fedio','centralio','provenance'))
  self.table('directories','Table 8.2. Authoritative experiment directories',['Role','Repository-relative path'],[['Frozen cohort','output/experiments/v1/frozen'],['FedAvg training','output/experiments/v2_fedavg/main/gru_masked_seed42b; gru_masked_seed43; gru_masked_seed44'],['FedAvg test',FED],['Central training','output/experiments/v3_centralized_sgd/main/gru_masked_seed{42,43,44}_schedule_matched'],['Central test',CENTRAL],['Validation comparison',VALID],['Final test/node comparison',FINAL]],['provenance','validation','contract'])
  self.p('The frozen-contract SHA-256 is '+prov['frozen_contract_sha256']+'. The ordered test-window identity SHA-256 is '+prov['window_identity']['ordered_sha256']+'. Exact original model and configuration hashes are preserved in the existing final-report provenance and repeated in the machine-readable evidence accompanying this document. The partial earlier FedAvg test_evaluation/v1 attempt is not the completed evaluation; v1_retry1 is authoritative. No directory was renamed to replace seed42b with a missing alternative path. '+self.cite('provenance','fedavglock'))
  self.p('Before writing the thesis package, the existing final-report verifier was run read-only. It passed 3,687 pre-existing-file checks, 25 final-report-artifact checks and 2,652 recorded input hashes with no mismatches. A fresh baseline for this documentation task inventories every pre-existing non-venv file. The new documentation receipt, source inventory, documentation-check log and before/after preservation records are additive records only; no historical expected hash or receipt is changed. Non-venv scope and the omission of access-time comparisons are explicit in the preservation record. '+self.cite('verification'))
  self.h('9. Proposed future work — not executed')
  self.p('The following sequence is a research plan, not a report of completed experiments. No result, improved accuracy, retained-window count, energy saving or deployment outcome is claimed for these proposals. All new code, frozen arrays, configurations and outputs must use separate versioned directories and preserve the verified one-hour benchmark.')
  self.h('9.1 Coverage-audited 1-, 6- and 24-hour history comparison',2)
  self.p('Keep the sampling interval at ten minutes and the next-hour average-power target fixed. The proposed histories correspond to 6, 36 and 144 input steps. First audit complete-history availability separately for every node, split and forecast origin, recording why each candidate is retained or lost. Longer histories must never be assembled by joining across temporal gaps. Report attrition, retained node counts, targets, origin distributions and estimated resource requirements before fitting models.')
  self.p('Define a primary common-cohort comparison using the intersection of eligible node/origin target identities across all three histories in training, validation and test. Preserve chronological boundaries and freeze training-only preprocessing for the comparison. Report a separate native-coverage sensitivity analysis if each history also uses its own available cohort; do not attribute availability-driven differences to history length alone. If the shared cohort is too small or unrepresentative, disclose that limitation before deciding a revised protocol. The current frozen contract and v2/v3 loaders explicitly require six history steps, so this study requires additive versioned preparation, loader and validation changes rather than changing the existing configuration in place. '+self.cite('feddata','contract'))
  self.p('Use paired seeds, common architecture and target definitions, controlled budgets and validation-only selection. Treat the already-observed test period as a development comparison; reserve additional independent data prospectively for confirmation if suitable data become available. An improvement from longer history is a hypothesis, not an established outcome.')
  self.h('9.2 Controlled environmental-feature ablation',2)
  self.p('After the history protocol and coverage have been audited, compare environmental inputs on an identical predeclared cohort. Distinguish indoor inlet temperature (index 6) from outdoor temperature (7) and humidity (8). A proposed factorial or staged ablation may compare all available environmental channels, the existing outdoor-only mask, an indoor-only mask and an all-environment mask. Preserve the remaining input shape, node mapping, calendar channels, raw target windows and frozen training scalers. Mask standardized copies without expanding the cohort when a feature is removed. Any causal interpretation is conditional on this controlled representation and coverage; it is not proof that environmental conditions have no physical effect.')
  self.p('Predeclare validation-based selection, paired seeds and reporting of RMSE, MAE and per-node errors. Keep model complexity and optimization budgets controlled, and avoid weather-specific test-based tuning. Existing v1 weather experiments may provide historical context, but the current matched v2/v3 results alone do not constitute this proposed environmental ablation. '+self.cite('experiments','metadata'))
  self.h('9.3 Separate energy-management optimization study',2)
  self.p('Only after forecasting validity is established should forecasts be connected to an explicitly defined scheduling, power-capping or cooling decision problem. Specify controllable actions, workload demand, operational constraints and service-level requirements. Compare a baseline policy and a forecast-informed policy under controlled conditions, including uncertainty and forecast failures. Measure electrical energy over matched intervals by integrating relevant measured power, and separately measure any claimed cooling or whole-facility boundary. Report performance and constraint violations alongside energy outcomes.')
  self.p('Any future savings claim must come from that independent optimization/intervention study, not from a reduction in forecast RMSE. Carbon accounting would require separately justified emissions factors and boundaries; no current carbon-prediction task or result is asserted. A real federated deployment would also require a separately specified privacy/security threat model and tested protection mechanisms before privacy-preserving claims are made.')
  self.h('10. Conclusions')
  self.p('The verified repository establishes a reproducible node-level power-forecasting comparison over a frozen M100 cohort. Under the measured protocol, centralized SGD has lower pooled test RMSE and MAE than FedAvg, while FedAvg provides a modest pooled RMSE improvement over persistence and a deterioration in MAE. Node-level analysis shows substantial heterogeneity, and the validation/test MAE trade-off depends on the evaluated period. These findings support careful reporting of multiple metrics and coverage rather than a claim that federated learning generally outperforms centralized training. The observed test history, correlated samples, simulated privacy setting and absence of an operational energy-control study define the limits of the present contribution. '+self.cite('analysis','validation','provenance'))
  self.h('Appendix A. Reading and reproducing the evidence')
  self.p('THESIS_METHODS_RESULTS.md, THESIS_METHODS_RESULTS.docx, THESIS_METHODS_RESULTS.html and THESIS_METHODS_RESULTS.pdf are generated from the same structured document content. SOURCES.md/SOURCES.json list exact paths, purposes and hashes. EVIDENCE.json records the numerical table inputs and original model/configuration provenance. DOCUMENTATION_CHECKS.json and PRESERVATION.json describe checks performed in this documentation task; the completion receipt inventories all additions. No command in this package launches training or evaluation.')
  self.h('Appendix B. Interpretation of historical source status')
  self.p('Root README.md and EXPERIMENTS.md retain historical implementation-stage text and accidental prefixes. They are useful for context but are not substituted for completed receipts and saved results. The validation report’s pre-test wording is likewise a preserved snapshot. Evidence priority for this document is: verified saved results and receipts; frozen metadata and source snapshots; inspected implementation; then historical explanatory documentation. This hierarchy avoids inventing outcomes from planned commands or rewriting old provenance.')
  self.source('valfigure',VALID+'/validation_curves.png','Existing validation figure, copied without regeneration')
  self.source('testfigure',FINAL+'/per_node_comparison.png','Existing per-node test figure, copied without regeneration')
  self.h('Appendix C. Repository evidence references')
  self.p('The following are repository evidence identifiers, not invented external literature citations. Full SHA-256 values are provided in SOURCES.md/SOURCES.json. The dataset source record is reproduced as recorded in the download configuration; no external bibliographic authors, publication details or hardware specifications are inferred.')
  self.table('sources','Table C.1. Exact direct sources',['ID','Repository-relative path','Role'],[[s['id'],s['path'],s['purpose']] for s in self.sources],[])
  out=Path(out)
  if out.exists():raise ValueError('Documentation output exists')
  out.mkdir(parents=True)
  self.source_figures(out)
  dump(out/'DOCUMENT.json',{'title':TITLE,'subtitle':SUBTITLE,'blocks':self.blocks})
  dump(out/'SOURCES.json',{'project_root':str(self.root),'sources':self.sources,'external_literature_citations_added':False})
  dump(out/'EVIDENCE.json',{'feature_names':features,'feature_units':meta['feature_units'],'sample_counts':meta['sample_counts'],
    'architecture':{'recurrent_input_width':21,'hidden_size':64,'num_layers':2,'node_embedding_size':8,'node_count':979},
    'validation':val,'test':a,'model_provenance':prov['evaluations'],'window_identity':prov['window_identity'],
    'numeric_tables':self.numeric_tables,'historical_receipts_checked':self.checks,'model_execution_performed':False})
  sources=['# Exact repository sources','', 'These are local artifact references; no external literature citation has been invented.','']
  for s in self.sources:sources += ['## '+s['id'],'',f"Path: `{s['path']}`",'',f"SHA-256: `{s['sha256']}`",'',s['purpose'],'']
  write(out/'SOURCES.md','\n'.join(sources).encode())
  write(out/'THESIS_METHODS_RESULTS.md',markdown(self.blocks).encode())
  write(out/'THESIS_METHODS_RESULTS.html',render_html(self.blocks).encode())
  docx(out/'THESIS_METHODS_RESULTS.docx',self.blocks,out)
  return {'sources':len(self.sources),'blocks':len(self.blocks),'tables':len(self.numeric_tables),'source_receipts_checked':self.checks}
 def source_figures(self,out):
  for key,name,rel in [('valfigure','validation_curves.png',VALID+'/validation_curves.png'),('testfigure','per_node_comparison.png',FINAL+'/per_node_comparison.png')]:
   # Figure source belongs to the same original report receipt already verified.
   write(out/'figures'/name,(self.root/rel).read_bytes())


def column_weights(block):
 if block['headers'][:2]==['Index','Exact feature name']:return [6,25,40,8,14,7]
 if block['headers'][:2]==['ID','Repository-relative path']:return [6,59,35]
 return [100/len(block['headers'])]*len(block['headers'])


def markdown(blocks):
 lines=[]
 for b in blocks:
  k=b['kind']
  if k=='heading':lines.append('#'*b['level']+' '+b['text'])
  elif k=='table':
   lines += [b['caption'],'','| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*len(b['headers']))+' |']
   lines += ['| '+' | '.join(r)+' |' for r in b['rows']]
  elif k=='figure':lines.append('!['+b['caption']+']('+b['path']+')')
  elif k=='equation':lines += ['```text',b['text'],'```']
  else:lines.append(b['text'])
  lines.append('')
 return '\n'.join(lines)


def render_html(blocks):
 css='''@page {size:A4; margin:20mm 17mm 20mm; @bottom-center {content:counter(page);font-size:9pt;}}
 body{font-family:"Times New Roman","Liberation Serif",serif;font-size:11pt;line-height:1.4;color:#111;}
 h1{font-size:17pt;margin-top:22pt;break-after:avoid;}h2{font-size:13pt;margin-top:16pt;break-after:avoid;}
 p{text-align:justify;orphans:3;widows:3;margin:7pt 0;}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:8.5pt;line-height:1.22;margin:8pt 0 14pt;}
 th,td{border:1px solid #999;padding:5px;vertical-align:top;overflow-wrap:anywhere;}th{background:#eceff1;}thead{display:table-header-group;}tr{break-inside:avoid;}
 .caption{font-size:10pt;font-weight:bold;break-after:avoid;text-align:left;}.equation{font-family:"Times New Roman",serif;text-align:center;white-space:pre-wrap;break-inside:avoid;margin:12pt 0;}
 figure{margin:14pt 0;break-inside:avoid;}img{display:block;width:100%;height:auto;}figcaption{font-size:9pt;text-align:left;margin-top:5pt;}
 .title{font-size:22pt;text-align:center;margin-top:45pt;}a{color:inherit;text-decoration:none;}'''
 lines=['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'+html.escape(TITLE)+'</title><style>'+css+'</style></head><body>']
 for i,b in enumerate(blocks):
  k=b['kind'];e=html.escape
  if k=='heading':lines.append(f'<h{b["level"]}'+(' class="title"' if i==0 else '')+'>'+e(b['text'])+f'</h{b["level"]}>')
  elif k=='table':lines += ['<p class="caption">'+e(b['caption'])+'</p>','<table><colgroup>'+''.join('<col style="width:'+str(v)+'%">' for v in column_weights(b))+'</colgroup><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in b['headers'])+'</tr></thead><tbody>',*['<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in b['rows']],'</tbody></table>']
  elif k=='figure':lines.append('<figure><img alt="'+e(b['caption'],quote=True)+'" src="'+e(b['path'],quote=True)+'"><figcaption>'+e(b['caption'])+'</figcaption></figure>')
  else:lines.append('<p'+(' class="equation"' if k=='equation' else '')+'>'+e(b['text'])+'</p>')
 return '\n'.join(lines+['</body></html>'])


def docx(path,blocks,out):
 ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','a':'http://schemas.openxmlformats.org/drawingml/2006/main','pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
 for prefix,uri in ns.items():ET.register_namespace(prefix,uri)
 def tag(s):p,n=s.split(':');return '{'+ns[p]+'}'+n
 def el(parent,name,attrs=None,text=None):
  x=ET.SubElement(parent,tag(name),{(tag(k) if k.split(':')[0] in ('w','r') else k.split(':')[-1]):str(v) for k,v in (attrs or {}).items()});x.text=text;return x
 doc=ET.Element(tag('w:document'));body=el(doc,'w:body');rels=[];images=[]
 def para(parent,text='',style=None,size=None):
  p=el(parent,'w:p');pr=el(p,'w:pPr')
  if style:el(pr,'w:pStyle',{'w:val':style})
  run=el(p,'w:r')
  if size:el(el(run,'w:rPr'),'w:sz',{'w:val':size})
  t=el(run,'w:t',text=text);t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');return p
 for b in blocks:
  k=b['kind']
  if k=='heading':para(body,b['text'],'Title' if b is blocks[0] else 'Heading'+str(b['level']))
  elif k=='table':
   para(body,b['caption'],'Caption');table=el(body,'w:tbl');pr=el(table,'w:tblPr');el(pr,'w:tblW',{'w:w':9000,'w:type':'dxa'});el(pr,'w:tblLayout',{'w:type':'fixed'})
   borders=el(pr,'w:tblBorders')
   for edge in ['top','left','bottom','right','insideH','insideV']:el(borders,'w:'+edge,{'w:val':'single','w:sz':4,'w:color':'999999'})
   grid=el(table,'w:tblGrid');widths=[round(9000*v/100) for v in column_weights(b)]
   for width in widths:el(grid,'w:gridCol',{'w:w':width})
   for ri,row in enumerate([b['headers'],*b['rows']]):
    tr=el(table,'w:tr');trpr=el(tr,'w:trPr');el(trpr,'w:cantSplit')
    if ri==0:el(trpr,'w:tblHeader')
    for ci,text in enumerate(row):
     width=widths[ci]
     tc=el(tr,'w:tc');cp=el(tc,'w:tcPr');el(cp,'w:tcW',{'w:w':width,'w:type':'dxa'})
     if ri==0:el(cp,'w:shd',{'w:fill':'ECEFF1'})
     para(tc,text,'TableText',17)
  elif k=='figure':
   data=(out/b['path']).read_bytes();w,h=struct.unpack('>II',data[16:24]);cx=5486400;cy=round(cx*h/w);rid='rIdImage'+str(len(images)+1)
   images.append(data);rels.append((rid,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image','media/image'+str(len(images))+'.png'))
   p=para(body,style='Figure');run=el(p,'w:r');inline=el(el(run,'w:drawing'),'wp:inline',{'wp:distT':0,'wp:distB':0,'wp:distL':0,'wp:distR':0})
   el(inline,'wp:extent',{'wp:cx':cx,'wp:cy':cy});el(inline,'wp:docPr',{'wp:id':len(images),'wp:name':'Figure '+str(len(images))})
   graphic=el(inline,'a:graphic');gd=el(graphic,'a:graphicData',{'a:uri':ns['pic']});pic=el(gd,'pic:pic');nv=el(pic,'pic:nvPicPr');el(nv,'pic:cNvPr',{'pic:id':0,'pic:name':'image.png'});el(nv,'pic:cNvPicPr')
   fill=el(pic,'pic:blipFill');el(fill,'a:blip',{'r:embed':rid});el(el(fill,'a:stretch'),'a:fillRect')
   sp=el(pic,'pic:spPr');xf=el(sp,'a:xfrm');el(xf,'a:off',{'a:x':0,'a:y':0});el(xf,'a:ext',{'a:cx':cx,'a:cy':cy});el(el(sp,'a:prstGeom',{'a:prst':'rect'}),'a:avLst')
   para(body,b['caption'],'Caption')
  else:para(body,b['text'],'Equation' if k=='equation' else None)
 section=el(body,'w:sectPr');el(section,'w:pgSz',{'w:w':11906,'w:h':16838});el(section,'w:pgMar',{'w:top':1134,'w:bottom':1134,'w:left':963,'w:right':963,'w:header':500,'w:footer':500})
 def xml(x):return ET.tostring(x,encoding='utf-8',xml_declaration=True)
 styles=ET.Element(tag('w:styles'));defaults=el(styles,'w:docDefaults');rpr=el(el(defaults,'w:rPrDefault'),'w:rPr');el(rpr,'w:rFonts',{'w:ascii':'Times New Roman','w:hAnsi':'Times New Roman'});el(rpr,'w:sz',{'w:val':24})
 pp=el(el(defaults,'w:pPrDefault'),'w:pPr');el(pp,'w:spacing',{'w:after':120,'w:line':360,'w:lineRule':'auto'})
 for name,size,bold in [('Normal',24,False),('Title',36,True),('Heading1',30,True),('Heading2',26,True),('Caption',20,False),('TableText',17,False),('Equation',22,False),('Figure',22,False)]:
  st=el(styles,'w:style',{'w:type':'paragraph','w:styleId':name});el(st,'w:name',{'w:val':name});ppr=el(st,'w:pPr')
  if name.startswith('Heading') or name=='Caption':el(ppr,'w:keepNext')
  if name in ['TableText','Caption']:el(ppr,'w:spacing',{'w:line':240,'w:lineRule':'auto','w:after':70})
  if name.startswith('Heading'):el(ppr,'w:outlineLvl',{'w:val':int(name[-1])-1})
  if name in ['Title','Figure','Equation']:el(ppr,'w:jc',{'w:val':'center'})
  rp=el(st,'w:rPr');el(rp,'w:sz',{'w:val':size})
  if bold:el(rp,'w:b')
 relns='http://schemas.openxmlformats.org/package/2006/relationships'
 def relationships(values):
  parent=ET.Element('Relationships',xmlns=relns)
  for rid,t,target in values:ET.SubElement(parent,'Relationship',Id=rid,Type=t,Target=target)
  return xml(parent)
 rels.append(('rIdStyles','http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles','styles.xml'))
 types=ET.Element('Types',xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
 for extension,mime in [('rels','application/vnd.openxmlformats-package.relationships+xml'),('xml','application/xml'),('png','image/png')]:ET.SubElement(types,'Default',Extension=extension,ContentType=mime)
 for part,mime in [('/word/document.xml','application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'),('/word/styles.xml','application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml')]:ET.SubElement(types,'Override',PartName=part,ContentType=mime)
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
  z.writestr('[Content_Types].xml',xml(types));z.writestr('_rels/.rels',relationships([('rId1','http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument','word/document.xml')]))
  z.writestr('word/document.xml',xml(doc));z.writestr('word/styles.xml',xml(styles));z.writestr('word/_rels/document.xml.rels',relationships(rels))
  for i,data in enumerate(images,1):z.writestr('word/media/image'+str(i)+'.png',data)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output-dir',required=True);args=p.parse_args()
 compiler=Compiler(args.project_root);print(json.dumps(compiler.compile(args.output_dir),indent=2))
