"""Read-only configuration and design-receipt verification; no experiment CLI."""
import hashlib
import json
from pathlib import Path
from .contracts import FEATURES, UNITS, TARGETS, TARGET_UNITS, HISTORIES, WIDTH_NS, utc_ns

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def safe_read(root,relative):
    root=Path(root).absolute(); rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('Artifact must remain inside project')
    path=root/rel
    if any(p.is_symlink() for p in (path,*path.parents)): raise ValueError('Symlink artifact prohibited')
    if not path.is_file() or path.stat().st_size>16*1024*1024:
        raise ValueError('Bounded regular metadata/source files only')
    return path.read_bytes()

def verify_design(root,manifest):
    if set(manifest)!={'schema','artifacts','historical_dependencies','self_hash_policy'} or manifest['schema']!='phase2.0.design.v1':
        raise ValueError('Unknown design receipt')
    if not manifest['artifacts'] or not manifest['historical_dependencies']: raise ValueError('Unbound design')
    for group in ('artifacts','historical_dependencies'):
        for relative,entry in manifest[group].items():
            raw=safe_read(root,relative)
            if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
                raise ValueError('Design artifact changed: '+relative)
    return len(manifest['artifacts'])+len(manifest['historical_dependencies'])

def validate_protocol(p):
    if p['schema']!='phase2.0.protocol.v1' or p['execution_enabled'] is not False:
        raise ValueError('Design-only protocol required')
    if p['features']!=list(FEATURES) or p['feature_units']!=list(UNITS) or p['targets']!=list(TARGETS) or p['target_units']!=list(TARGET_UNITS):
        raise ValueError('Feature/target schema changed')
    if p['histories_steps']!=list(HISTORIES) or p['primary_history_steps']!=36 or p['horizon_steps']!=6 or p['origin_stride_steps']!=6 or p['bin_seconds']!=600:
        raise ValueError('History/horizon/grid changed')
    expected={'train':['2022-01-01T00:00:00Z','2022-02-11T07:10:00Z'],
              'val':['2022-02-11T07:10:00Z','2022-02-20T03:30:00Z'],
              'test':['2022-03-01T00:00:00Z','2022-04-01T00:00:00Z']}
    if p['splits']!=expected or p['excluded_interval']!=['2022-02-20T03:30:00Z','2022-03-01T00:00:00Z']:
        raise ValueError('Approved split/exclusion changed')
    for edges in p['splits'].values():
        lo,hi=map(utc_ns,edges)
        if lo>=hi or lo%WIDTH_NS or hi%WIDTH_NS: raise ValueError('Invalid split grid')
    if p['split_containment']!='whole_history_and_target_and_raw_contributors' or p['missing_policy']!='no_fill_original_observed_joint_targets':
        raise ValueError('Containment or missing-value policy changed')
    if p['seeds']!=[42,43,44] or p['eligibility']!={'minimum_windows':24,'minimum_days':7,'training_only':True}:
        raise ValueError('Seeds or training-only admission changed')
    if p['mapping']['candidate_nodes']!=979 or p['mapping']['march_eligible_nodes'] is not None or p['march_counts'] is not None:
        raise ValueError('March coverage must remain unknown in design freeze')
    if p['normalization']['reference_history_steps']!=36 or p['normalization']['fit_split']!='train' or p['selection']!='earliest_minimum_pooled_equal_output_standardized_validation_mse':
        raise ValueError('Normalization/selection differs')
    return True

def expected_fit_keys():
    primary=['gru_enabled','gru_masked_control','lstm_enabled','xgboost_enabled',
             'fedavg_gru_enabled','fedavg_gru_masked']
    return ({('primary_6h',36,m,s) for m in primary for s in (42,43,44)} |
            {('history_matched',h,m,s) for h in HISTORIES for m in ('gru_enabled','fedavg_gru_enabled') for s in (42,43,44)})

def validate_matrix(matrix,methods):
    if matrix['schema']!='phase2.0.matrix.v1' or matrix['execution_enabled'] is not False:
        raise ValueError('No executable matrix is enabled')
    keys=[]; paths=[]; ids=[]
    for row in matrix['fits']:
        if set(row)!={'id','suite','history_steps','method','seed','output_relative'}: raise ValueError('Unexpected fit override')
        key=(row['suite'],row['history_steps'],row['method'],row['seed']);keys.append(key)
        paths.append(row['output_relative']);ids.append(row['id'])
        if row['method'] not in methods or methods[row['method']]['execution_enabled'] is not False:
            raise ValueError('Missing or enabled method')
        expected=f"output/experiments/phase2_multioutput/v1/{row['suite']}/{row['method']}_h{row['history_steps']}_seed{row['seed']}"
        if row['output_relative']!=expected or row['id']!=f"{row['suite']}_{row['method']}_h{row['history_steps']}_seed{row['seed']}":
            raise ValueError('Fit identity/output path changed')
    if len(keys)!=36 or set(keys)!=expected_fit_keys() or len(set(ids))!=36 or len(set(paths))!=36:
        raise ValueError('Missing/duplicate/unexpected matrix identities')
    expected_baselines={('primary_6h',36),*(('history_matched',h) for h in HISTORIES)}
    if {(b['suite'],b['history_steps']) for b in matrix['baselines']}!=expected_baselines or len(matrix['baselines'])!=4:
        raise ValueError('Persistence reference cohort/history changed')
    for m in methods.values():
        if m['targets']!=list(TARGETS) or m['weather'] not in ('enabled','masked','not_applicable'):
            raise ValueError('Method target/weather contract changed')
    if methods['xgboost_enabled']['multioutput_strategy']!='two_scalar_boosters_shared_selected_iteration':
        raise ValueError('XGBoost must select one joint validation iteration')
    return True

def load_bundle(root,directory):
    directory=Path(directory)
    def load(name):return json.loads(safe_read(root,directory/name))
    protocol=load('protocol.json');matrix=load('matrix.json')
    methods={k:load(v) for k,v in matrix['method_files'].items()}
    validate_protocol(protocol);validate_matrix(matrix,methods)
    return protocol,matrix,methods
