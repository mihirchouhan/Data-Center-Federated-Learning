"""Offline design gate. Never downloads, opens archives, trains or predicts."""
import argparse
import hashlib
import json
from pathlib import Path
from .protocol import safe_read,verify_design,load_bundle,digest

ROOT=Path(__file__).absolute().parents[2]
DEFAULT='configs/experiments/phase2_multioutput/v1'

def verify_evidence_hashes(root,evidence,p,m):
    """Future metadata integrity gate only; never declares semantic replay passed."""
    if evidence['schema']!='phase2.0.data_evidence.v1' or evidence['protocol_digest']!=digest(p) or evidence['matrix_digest']!=digest(m):
        raise ValueError('Evidence belongs to another frozen design')
    required={'acquisition_receipt','coverage_report','schema_units_report','cohort_manifest',
              'mapping','scalers','provenance_acceptance','semantic_window_verification'}
    if set(evidence['roles'])!=required or not evidence['artifacts']: raise ValueError('Incomplete evidence roles')
    for role,relative in evidence['roles'].items():
        if relative not in evidence['artifacts']: raise ValueError('Unbound evidence role: '+role)
    for relative,record in evidence['artifacts'].items():
        if Path(relative).suffix not in ('.json','.md','.csv','.txt','.log'):
            raise ValueError('This offline gate opens metadata only, never archives/arrays/checkpoints')
        raw=safe_read(root,relative)
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Evidence artifact content changed: '+relative)
    return len(evidence['artifacts'])

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir',default=DEFAULT)
    parser.add_argument('--stage',choices=['protocol','data'],default='protocol')
    parser.add_argument('--evidence',help='Future approved JSON; never an archive')
    args=parser.parse_args(argv)
    manifest=json.loads(safe_read(ROOT,Path(args.config_dir)/'design_manifest.json'))
    checked=verify_design(ROOT,manifest)
    p,m,methods=load_bundle(ROOT,args.config_dir)
    result={'status':'protocol_verified_data_not_frozen','design_files_checked':checked,
            'prospective_fit_identities':len(m['fits']),'persistence_references':len(m['baselines']),
            'march_coverage':'UNKNOWN','march_eligible_clients':'UNKNOWN','data_frozen':False,
            'execution_authorized':False,'training_inference_or_archive_access':False,
            'protocol_digest':digest(p),'matrix_digest':digest(m)}
    if args.stage=='data':
        result['status']='blocked_before_real_data_stage'
        if args.evidence:
            if Path(args.evidence).suffix!='.json': raise ValueError('Evidence must be bounded JSON metadata, never an archive')
            e=json.loads(safe_read(ROOT,args.evidence))
            result['evidence_artifact_hashes_checked']=verify_evidence_hashes(ROOT,e,p,m)
        result['blockers']=['Verified acquisition and separately authorized safe inspection/coverage are required',
                            'Real-data bridge, semantic window audit and eligible-client scaler fit are not implemented in this design-only release',
                            'Coverage adequacy/provenance acceptance and pilot/training/test approvals remain separate']
    print(json.dumps(result,sort_keys=True,indent=2))
    return 2 if args.stage=='data' else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,KeyError,OSError) as error:
        print('PHASE 2 READINESS ERROR: '+str(error));raise SystemExit(2)
