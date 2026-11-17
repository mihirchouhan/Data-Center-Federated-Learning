"""Bounded prepared TRAIN/VAL capabilities; no raw telemetry, March or fitter."""
from pathlib import Path
import hashlib,json,re
import numpy as np
from src import b1_preparation_io_v1 as io, b1_real_data_bridge_v1 as bridge
from src import b1_preparation_execution_v1 as preparation
from src import b1_bridge_scaling_v1 as scaling, b1_scaler_acceptance_v1 as acceptance

DATA_SHA = '497a699025ef0babccc48776fd7a8c531d72a03f6f2aa5fa5ea931c7b0a0a13c'
LOCK_SHA = '110dc33f4528771fb986a57319ce2c40a9b0334633c8c3fbef2dbf1176df0c96'
SCALER_SHA = 'aa534646283065ee2d3258b03dddde679e9ec40204608331d209fd0890ede8a5'
FINGERPRINT = '469008a727b66c652462f6872ef2e829c6d023781a013f401f6f9f20ff48fce7'
TOKEN = object()


def read_json(path):
    if io.safe_path(path).stat().st_size > 64*1024**2: raise ValueError('JSON too large')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('Duplicate JSON field')
            out[k]=v
        return out
    with io.safe_open(path) as f: return json.load(f,object_pairs_hook=pairs)


class ClientData:
    """Copied training-only arrays. No validation/context/path capability."""
    __slots__=('node','ordinal','x','future','observed','origins')
    def __init__(self, node, x, future, observed, origins):
        if node not in bridge.ORIGINAL_NAMES: raise ValueError('Unknown client')
        # Deny out-of-training timestamps before touching target containers.
        origin=bridge._times(origins,'client origins',(len(origins),)).view('i8')
        lo,hi=(int(np.datetime64(v,'ns').view('i8')) for v in bridge.SPLITS['train'])
        if not len(origin) or (origin%3_600_000_000_000).any() or (origin-36*bridge.WIDTH_NS<lo).any() or (origin+6*bridge.WIDTH_NS>hi).any() or (np.diff(origin)<=0).any():raise ValueError('Client training chronology differs')
        # Preserve target masks until hardened validation, even for synthetic tests.
        bridge.target_validation.target_means(future, observed)
        bridge._numeric(x,'model-ready input')
        if x.dtype != np.float32 or x.shape != (len(origins),36,14) or not len(x): raise ValueError('Float32 client [N,36,14] required')
        i=bridge.ORIGINAL_NAMES.index(node)
        if not np.array_equal(x[:,:,13],np.full(x.shape[:2],i)): raise ValueError('Client ordinal mismatch')
        self.node,self.ordinal=node,i
        for key,val in [('x',x),('future',future),('observed',observed),('origins',origins)]:
            setattr(self,key,bridge.frozen_array(val))
    def __len__(self): return len(self.x)


class PreparedData:
    """Verify the immutable B1 population once; recheck each read chunk thereafter.

    No loader accepts a test split; no directory glob can discover March.
    Client reads are bounded by one original node's local population.
    """
    def __init__(self, data_dir, scaler_lock):
        self.path=io.safe_path(data_dir);self.lock_path=io.safe_path(scaler_lock)
        root=Path(bridge.__file__).absolute().parents[1]
        if self.path!=root/'output/experiments/b1_no_ff/preparation/train_val_v1/dataset' or self.lock_path!=root/'output/march_window_audit/b1_scaler_approval_and_lock_v1/locked_scaler/lock.json':raise ValueError('Only the bound B1 train/validation dataset and scaler are accessible')
        self.names=bridge.ORIGINAL_NAMES
        if io.binding(self.path/'manifest.json')['sha256']!=DATA_SHA or io.binding(self.lock_path)['sha256']!=LOCK_SHA:
            raise ValueError('Locked dataset/scaler binding mismatch')
        self.manifest=read_json(self.path/'manifest.json');lock=read_json(self.lock_path)
        if lock['status']!='LOCKED' or lock['dataset_manifest']['sha256']!=DATA_SHA: raise ValueError('Invalid scaler lock')
        if io.binding(self.lock_path.parent/'scaler.json')['sha256']!=SCALER_SHA: raise ValueError('Scaler bytes changed')
        for key,name in [('approval_decision','approval_decision.json'),('training_verification','verification.json'),('tests','tests.json')]:
            v=lock[key]
            if v['path']!='../'+name or io.binding(self.lock_path.parent.parent/name)!={k:v[k] for k in ('bytes','sha256')}: raise ValueError('Scaler acceptance evidence changed')
        raw=read_json(self.lock_path.parent/'scaler.json')
        self.scaler=scaling.FrozenScaler(tuple(raw['x_mean']),tuple(raw['x_std']),tuple(raw['original_x_std']),tuple(raw['y_mean']),tuple(raw['y_std']),bridge.canonical(raw['provenance']))
        if acceptance.verify_candidate(self.scaler,raw)!=FINGERPRINT: raise ValueError('Candidate fingerprint differs')
        if io.binding(Path(scaling.__file__))['sha256']!=raw['provenance']['implementation_sha256']: raise ValueError('Historical scaler implementation changed')
        m=self.manifest
        if m['contract']['mapping']!=list(bridge.ORIGINAL_NAMES) or m['contract']['splits']!={k:list(v) for k,v in bridge.SPLITS.items()}: raise ValueError('Mapping/splits differ')
        self.groups={split:{n:[] for n in bridge.ORIGINAL_NAMES} for split in ('train','val')}
        for ch in m['chunks']:
            name=ch['path'];split=ch['split'];node=ch['node']
            if split not in self.groups or node not in self.groups[split] or not re.fullmatch(split+r'_\d{4}_\d{6}\.npz',name): raise ValueError('Unsafe/foreign prepared chunk')
            self.groups[split][node].append(ch)
        # Reuse the original semantic verifier without preparation, raw access or fitting.
        verified=preparation.verify_dataset(self.path,m['contract'])
        self.counts={n:sum(ch['windows'] for ch in self.groups['train'][n]) for n in bridge.ORIGINAL_NAMES}
        if len(self.counts)!=979 or any(v<24 for v in self.counts.values()): raise ValueError('Invalid training client cohort')
        if m['results']['train']['windows']!=280499 or m['results']['val']['windows']!=82993: raise ValueError('B1 counts differ')
        p=raw['provenance']
        if p['fit_identity_sha256']!=verified['results']['train']['identity_sha256'] or p['fit_target_sha256']!=verified['results']['train']['target_sha256']: raise ValueError('Fitted training population differs')
        self.binding={'dataset_manifest':io.binding(self.path/'manifest.json'),'scaler_lock':io.binding(self.lock_path),'scaler':io.binding(self.lock_path.parent/'scaler.json'),'scaler_fingerprint':FINGERPRINT,'mapping_digest':bridge.MAPPING_DIGEST,'results':verified['results']}
        # Check every actual float32 representation before any production model can initialize.
        # This is frozen transformation/finite validation, never scaler fitting or scoring.
        training_dates={node:set() for node in bridge.ORIGINAL_NAMES}
        self.preflight_numerical={split:{'windows':0,'float32_inputs_finite':True,'float32_targets_finite':True} for split in ('train','val')}
        for ch in m['chunks']:
            w=self._chunk(ch);x=self.scaler.transform_inputs(w.x);y=self.scaler.standardize_targets(w.physical_targets)
            if x.shape!=(len(w),36,14) or y.shape!=(len(w),2):raise ValueError('Model input/target shape differs')
            self.preflight_numerical[ch['split']]['windows']+=len(w)
            if ch['split']=='train':training_dates[ch['node']].update(str(v) for v in w.origins.astype('datetime64[D]'))
        if any(self.preflight_numerical[k]['windows']!=m['results'][k]['windows'] for k in ('train','val')):raise ValueError('Incomplete normalization preflight')
        self.training_eligibility={node:{'windows':self.counts[node],'positive_training_dates':len(training_dates[node]),'original_ordinal':i} for i,node in enumerate(bridge.ORIGINAL_NAMES)}
        if any(v['windows']<24 or v['positive_training_dates']<7 for v in self.training_eligibility.values()):raise ValueError('Training-only client eligibility failed; no silent client filtering')
        self.binding['training_eligibility']=self.training_eligibility
        self.binding['normalization_preflight']=self.preflight_numerical
        self._verified_token=TOKEN

    def _chunk(self,ch):
        path=self.path/ch['path']
        if io.binding(path)!={k:ch[k] for k in ('bytes','sha256')}: raise ValueError('Prepared chunk changed after preflight')
        preparation._check_archive(path,ch['windows'],io.Limits(**self.manifest['contract']['limits']))
        with io.safe_open(path) as f,np.load(f,allow_pickle=False) as z: arrays={k:z[k] for k in preparation.FIELDS};y=z['physical_targets']
        w=bridge.accept_windows(split=ch['split'],mapping=bridge.ORIGINAL_NAMES,source_evidence=self.manifest['source_evidence'],**arrays)
        if not np.array_equal(w.physical_targets,y): raise ValueError('Target values changed')
        if io.binding(path)!={k:ch[k] for k in ('bytes','sha256')}: raise ValueError('Prepared chunk changed during read')
        return w

    def client(self,node):
        if node not in self.groups['train']: raise ValueError('Unknown training client')
        parts=[self._chunk(ch) for ch in self.groups['train'][node]]
        x=np.concatenate([self.scaler.transform_inputs(w.x) for w in parts])
        future=np.concatenate([w.future for w in parts]);observed=np.concatenate([w.target_observed for w in parts]);origins=np.concatenate([w.origins for w in parts])
        if len(x)!=self.counts[node] or len(np.unique(origins.astype('datetime64[D]')))<7: raise ValueError('Training eligibility differs')
        return ClientData(node,x,future,observed,origins)

    def validation_batches(self):
        for node in bridge.ORIGINAL_NAMES:
            for ch in self.groups['val'][node]:
                w=self._chunk(ch)
                yield self.scaler.transform_inputs(w.x),w.physical_targets,w.nodes,w.origins

    def assert_unchanged(self):
        if io.binding(self.path/'manifest.json')!=self.binding['dataset_manifest'] or io.binding(self.lock_path)!=self.binding['scaler_lock'] or io.binding(self.lock_path.parent/'scaler.json')!=self.binding['scaler']: raise ValueError('Data/scaler manifest changed')
        for ch in self.manifest['chunks']:
            if io.binding(self.path/ch['path'])!={k:ch[k] for k in ('bytes','sha256')}: raise ValueError('Prepared dataset changed')
        return True


def install_access_guard(data, checkpoint_root=None):
    """Process-local accidental-access guard; no OS/cryptographic isolation claim."""
    import sys,os
    root=Path(bridge.__file__).absolute().parents[1]
    allowed={str(data.path/ch['path']) for ch in data.manifest['chunks']}
    def guard(event,args):
        if event.startswith('socket.'): raise RuntimeError('Training process network access prohibited')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).absolute()
            flags=args[2]
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) and (p.is_relative_to(data.path) or p.is_relative_to(data.lock_path.parent.parent)):raise RuntimeError('Immutable input write prohibited')
            if p.is_relative_to(root) and p.suffix=='.pt' and (checkpoint_root is None or not p.is_relative_to(checkpoint_root)):raise RuntimeError('Historical/foreign checkpoint access prohibited')
            if p.is_relative_to(root/'data'): raise RuntimeError('Raw/March access prohibited')
            if p.is_relative_to(root) and p.suffix in ('.npz','.npy','.parquet','.tar') and str(p) not in allowed: raise RuntimeError('Undeclared/test array access prohibited')
    sys.addaudithook(guard)
