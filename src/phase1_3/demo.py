"""Run only the explicitly authorized small CPU synthetic data/shape demo."""
import argparse
from pathlib import Path
import json
import numpy as np
from .config import load_spec
from .synthetic_data import generate
from .windowing import build_windows
from .scaling import fit_training_scaler,Scaler
from .model import synthetic_shape_check
from .interfaces import client_views,ModelState,ClientUpdate,validate_update_contract
from .artifacts import ROOT,checked_path,write_json,digest_file

def run(spec_path,output):
    p=Path(spec_path).resolve()
    if p != ROOT/"configs/phase1_3/experiment_spec_v1.json":
        raise PermissionError("Only the new synthetic specification is supported")
    out=checked_path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Synthetic output exists; preserve it and use a new version")
    spec=load_spec(p,"synthetic")
    records=generate(spec,3)
    mapping=tuple(r.node for r in records)
    windows,audit=build_windows(records,mapping,spec)
    scaler=fit_training_scaler(windows["train"],spec)
    x,y=scaler.transform(windows["train"],spec)
    restored=Scaler.from_json(scaler.to_json(),spec)
    if restored.to_json()!=scaler.to_json():
        raise ValueError("Scaler serialization differs")
    # Data reconstruction check only, not model error or forecast scoring.
    if not np.allclose(scaler.inverse_targets(y),windows["train"].y,rtol=1e-6,atol=1e-5):
        raise ValueError("Target normalization is not reversible within float32 precision")
    shape=synthetic_shape_check(x[:4],spec,len(mapping))
    clients=client_views(windows["train"],spec,scaler)
    if not clients:
        raise ValueError("No eligible synthetic clients")
    counts={k:v.sample_count for k,v in clients.items()}
    state=ModelState(0,{"synthetic_contract_probe":np.zeros(2,dtype=np.float32)})
    updates=tuple(ClientUpdate(k,n,state.fingerprint,state) for k,n in counts.items())
    weights=validate_update_contract(state,updates,counts)
    result={"version":1,"status":"synthetic_pipeline_validated","profile":"synthetic",
        "seed":spec.seed,"mapping":list(mapping),"spec_digest":spec.digest,"audit":audit,
        "input_shape":list(x.shape),"target_shape":list(y.shape),"target_units":["W","degC"],
        "architecture":shape,"eligible_training_counts":counts,"declared_weights":weights,
        "fl_contract":"generic synthetic-state validation only; no averaging/global update",
        "model_forward_scope":"CPU fabricated tensors only; no predictions saved or scored",
        "real_data_preprocessing":False,"real_model_initialization":False,"real_inference":False,
        "training":False,"optimizer_steps":0,"backpropagation":False,"aggregation_executed":False,
        "metrics_computed":False,"historical_preflight":"FAILED_UNCHANGED",
        "source_hashes":{str(s.relative_to(ROOT)):digest_file(s) for s in sorted((ROOT/"src/phase1_3").glob("*.py"))},
        "spec_file_sha256":digest_file(p)}
    out.mkdir(exist_ok=False)
    write_json(out/"report.json",result)
    write_json(out/"scalers.json",json.loads(scaler.to_json()))
    write_json(out/"spec_snapshot.json",json.loads(p.read_text()))
    write_json(out/"artifacts.json",{"status":"complete","artifacts":{
        n:digest_file(out/n) for n in ("report.json","scalers.json","spec_snapshot.json")}})
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",default="configs/phase1_3/experiment_spec_v1.json")
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(run(args.spec,args.output_dir),indent=2,allow_nan=False))
    except (ValueError,PermissionError,FileExistsError) as error:
        parser.exit(1,"PHASE 1.3 SYNTHETIC ERROR: "+str(error)+"\n")

if __name__=="__main__":
    main()
