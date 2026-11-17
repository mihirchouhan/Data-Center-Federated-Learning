"""B1 train/validation raw-to-window producer v1; protocol UNAPPROVED.

No CLI, output writer, fitting, models or test capability. Only synthetic inputs
are exercised during implementation. Actual telemetry execution needs separate
permission. Scanning/copy limits abort, never sample or truncate the population.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import os
import stat
import numpy as np
from src import b1_real_data_bridge_v1 as bridge
from src.b1_target_validation_v1 import target_means
from src.phase1_3.schema import frozen_array
from src.history_data_v4 import LOCAL_METRICS, node_features
from src.multinode import weighted_bins
from src.preprocess import _range_mask, _weather_asof

WIDTH = bridge.WIDTH_NS
HISTORY, HORIZON = 36, 6
METRICS = LOCAL_METRICS + ('temp', 'humidity')
STATUS = 'UNAPPROVED B1 candidate; no execution authorization'
_TOKEN = object()
NaT = np.datetime64('NaT', 'ns')


def _digest(value):
    return hashlib.sha256(bridge.canonical(value)).hexdigest()


def _split(split):
    if split not in bridge.SPLITS:
        raise PermissionError('Only B1 train/val; no March/test capability')
    return tuple(int(np.datetime64(t, 'ns').view('i8')) for t in bridge.SPLITS[split])


def _client(node):
    if type(node) is not str or node not in bridge.ORIGINAL_NAMES:
        raise ValueError('Original literal node ID required; never renumber')
    return bridge.ORIGINAL_NAMES.index(node)


@dataclass(frozen=True)
class Limits:
    max_raw_rows: int = 2_000_000  # per retained node/metric stream
    max_scan_rows: int = 200_000_000  # per file, includes all timestamp rows
    arrow_batch_rows: int = 65_536
    max_windows: int = 10_000
    max_population_bytes: int = 512 * 1024**2  # conservative 4x packet estimate
    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in self.__dict__.values()):
            raise ValueError('Explicit positive integer resource limits required')


class RawSeries:
    """Immutable, unscaled timestamp/value rows. Masks rejected before coercion.

    UTC is explicit. Invalid values/null timestamps remain quality events; null
    times are dropped during canonicalization. March dates fail before values
    are inspected. This in-memory container does not authenticate origin.
    """
    __slots__ = ('times', 'values', 'timezone', '_reader_audit')
    def __init__(self, times, values, *, timezone='UTC', limits=Limits()):
        if timezone != 'UTC' or np.ma.isMaskedArray(times):
            raise ValueError('Explicit unmasked UTC timestamps required')
        t = np.asarray(times)
        if t.ndim != 1 or t.dtype != np.dtype('datetime64[ns]'):
            raise ValueError('UTC datetime64[ns] timestamps required; no guessing')
        if len(t) > limits.max_raw_rows: raise ValueError('Raw row budget exceeded')
        valid = ~np.isnat(t)
        if ((t[valid] < np.datetime64('2022-01-01','ns')) |
                (t[valid] >= np.datetime64('2022-03-01','ns'))).any():
            raise PermissionError('Only January/February source dates; March excluded')
        if np.ma.isMaskedArray(values): raise ValueError('Masked raw/target values rejected before conversion')
        v = np.asarray(values)
        if v.shape != t.shape or v.dtype.kind not in 'iuf':
            raise ValueError('Aligned real numeric values required; boolean targets forbidden')
        object.__setattr__(self,'times',frozen_array(t))
        object.__setattr__(self,'values',frozen_array(v,np.float64))
        object.__setattr__(self,'timezone',timezone)
        object.__setattr__(self,'_reader_audit',b'{}')
    def __setattr__(self, name, value): raise AttributeError('Immutable raw rows')
    @property
    def fingerprint(self):
        return hashlib.sha256(self.times.tobytes()+self.values.tobytes()).hexdigest()


def _rows(series, metric, split, limits):
    lo, hi = _split(split)
    if type(series) is not RawSeries: raise ValueError('Validated RawSeries required')
    if len(series.times) > limits.max_raw_rows: raise ValueError('Raw row budget exceeded')
    t = series.times.view('i8'); good = ~np.isnat(series.times)
    inside = good & (t>=lo) & (t<hi)
    t, v = t[inside], np.array(series.values[inside], copy=True)
    order = np.argsort(t,kind='stable');t,v=t[order],v[order]
    keep = np.r_[t[1:]!=t[:-1],True] if len(t) else np.zeros(0,bool)
    q={'null_timestamp_rows':int((~good).sum()),'out_of_split_rows':int((good&~inside).sum()),
       'duplicate_rows':int((~keep).sum()),'retained_unique_rows':int(keep.sum())}
    t,v=t[keep],v[keep]; valid,_=_range_mask(metric,v)
    q['invalid_value_rows']=int((~valid).sum());q['reader_evidence']=json.loads(series._reader_audit);v[~valid]=np.nan
    return t,v,q


@dataclass(frozen=True)
class BinEvidence:
    values: np.ndarray
    coverage: np.ndarray
    earliest: np.ndarray
    latest: np.ndarray
    contributors: np.ndarray


def bin_series(series, metric, split, *, limits=Limits()):
    """Full split grid; causal source reset BEFORE weighted bins/as-of.

    Contributor count and earliest/latest summarize actual retained raw support;
    raw series fingerprints preserve the source rows for independent future replay.
    """
    lo,hi=_split(split)
    if metric not in METRICS: raise ValueError('Unknown metric')
    edges=np.arange(lo,hi+WIDTH,WIDTH,dtype=np.int64)
    t,v,q=_rows(series,metric,split,limits);n=len(edges)-1
    first=np.full(n,np.iinfo('i8').max,dtype='i8');last=np.full(n,np.iinfo('i8').min,dtype='i8');counts=np.zeros(n,dtype='i8')
    if not len(t):
        result=np.full(n,np.nan);coverage=np.zeros(n)
    elif metric in ('temp','humidity'):
        result=_weather_asof(t,v,edges,1800*10**9)
        idx=np.searchsorted(t,edges[1:]-1,side='right')-1;safe=np.maximum(idx,0)
        good=(idx>=0)&np.isfinite(result)
        first[good]=t[safe[good]];last[good]=t[safe[good]];counts[good]=1
        coverage=np.isfinite(result).astype(np.float64)  # weather validity, NOT local time support
    else:
        result,coverage=weighted_bins(t,v,edges,300*10**9,.5)
        ends=np.minimum(np.r_[t[1:],edges[-1]],t+300*10**9)
        good=np.isfinite(v)&(ends>t)
        starts=t[good];ends=ends[good];left=(starts-lo)//WIDTH;right=(ends-1-lo)//WIDTH
        np.minimum.at(first,left,starts);np.maximum.at(last,left,starts);np.add.at(counts,left,1)
        cross=right!=left
        np.minimum.at(first,right[cross],starts[cross]);np.maximum.at(last,right[cross],starts[cross]);np.add.at(counts,right[cross],1)
    first=np.where(counts>0,first,np.iinfo('i8').min).astype('datetime64[ns]')
    last=np.where(counts>0,last,np.iinfo('i8').min).astype('datetime64[ns]')
    return BinEvidence(*(frozen_array(a) for a in (result,coverage,first,last,counts))),q


def _bin_contract():
    folder=Path(bridge.__file__).parent
    dependencies={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in
                  ('preprocess.py','multinode.py','history_data_v4.py','b1_real_data_bridge_v1.py','b1_target_validation_v1.py')}
    return {'schema':'b1.raw.binning.v1','protocol_status':STATUS,'history':36,'horizon':6,
      'grid_seconds':600,'hold_seconds':300,'min_coverage':.5,'weather_age_seconds':1800,
      'features':list(bridge.FEATURES),'units':list(bridge.UNITS),'splits':dict(bridge.SPLITS),
      'mapping_digest':bridge.MAPPING_DIGEST,'dependencies':dependencies,
      'duplicates':'stable input-file/row order, keep invalid last','imputation':False}


def _evidence(kind, raw_digest):
    return {'source_kind':kind,'raw_contract_sha256':raw_digest,'bin_contract_sha256':_digest(_bin_contract()),
      'producer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class ProducedSplit:
    """One validated full split, with original-ordinal per-client capabilities."""
    __slots__ = ('windows','_diagnostics','_payload')
    def __setattr__(self,name,value):raise AttributeError('Immutable produced population')
    def __init__(self, token, windows, diagnostics, payload):
        if token is not _TOKEN: raise TypeError('Use producer entry points')
        object.__setattr__(self,'windows',windows)
        object.__setattr__(self,'_diagnostics',bridge.canonical(diagnostics))
        object.__setattr__(self,'_payload',MappingProxyType({k:frozen_array(v) for k,v in payload.items()}))
    @property
    def diagnostics(self): return json.loads(self._diagnostics)
    @property
    def clients(self): return tuple(n for n in bridge.ORIGINAL_NAMES if n in set(self.windows.nodes.tolist()))
    def client(self, node):
        _client(node);idx=np.flatnonzero(self.windows.nodes==node)
        if not len(idx): raise KeyError('No accepted windows for this client; no fallback')
        return bridge.accept_windows(split=self.windows.split,mapping=bridge.ORIGINAL_NAMES,
               source_evidence=self.windows.source_evidence,**{k:v[idx] for k,v in self._payload.items()})
    def record(self):
        return {**self.windows.record(),'protocol_status':STATUS,'diagnostics':self.diagnostics}


def _produce(local, weather, split, limits, evidence, node_reader=None):
    lo,hi=_split(split)  # reject test before any data access
    if not isinstance(local,dict) or not local: raise ValueError('Nonempty explicit node streams required')
    nodes=sorted(local,key=_client)
    if set(weather)!= {'temp','humidity'}: raise ValueError('Both original weather inputs required')
    # Validate all local declarations before processing; missing sensors never drop a feature.
    if node_reader is None and any(set(local[n])!=set(LOCAL_METRICS) for n in nodes): raise ValueError('All eight local raw metrics required')
    edges=np.arange(lo,hi+WIDTH,WIDTH,dtype='i8');ngrid=len(edges)-1
    wb={};weather_quality={}
    for m in ('temp','humidity'):wb[m],weather_quality[m]=bin_series(weather[m],m,split,limits=limits)
    pieces=[];quality={};statuses={};total=0;estimated=0
    for node in nodes:
        idx=_client(node);bins={};quality[node]={}
        streams=local[node] if node_reader is None else node_reader(node)
        if set(streams)!=set(LOCAL_METRICS):raise ValueError('All local raw streams required')
        for m in LOCAL_METRICS:
            bins[m],quality[node][m]=bin_series(streams[m],m,split,limits=limits)
        del streams  # release file-backed node rows before the next client is read
        values=np.column_stack([bins[m].values for m in LOCAL_METRICS])
        x=node_features(values,{m:wb[m].values for m in wb},edges)
        # Independently qualified aligned memory bins; minimum is not simultaneous support.
        earliest=np.column_stack([bins['total_power'].earliest,*[bins[m].earliest for m in ('cpu_user','cpu_system','cpu_idle','cpu_wio')],
          np.minimum(bins['mem_free'].earliest,bins['mem_total'].earliest),bins['ambient'].earliest,wb['temp'].earliest,wb['humidity'].earliest])
        latest=np.column_stack([bins['total_power'].latest,*[bins[m].latest for m in ('cpu_user','cpu_system','cpu_idle','cpu_wio')],
          np.maximum(bins['mem_free'].latest,bins['mem_total'].latest),bins['ambient'].latest,wb['temp'].latest,wb['humidity'].latest])
        support=np.column_stack([bins['total_power'].coverage,*[bins[m].coverage for m in ('cpu_user','cpu_system','cpu_idle','cpu_wio')],
          np.minimum(bins['mem_free'].coverage,bins['mem_total'].coverage),bins['ambient'].coverage])
        valid=np.isfinite(x)
        valid[:,0]&=x[:,0]>=0;valid[:,[1,2,3,4,8]]&=(x[:,[1,2,3,4,8]]>=0)&(x[:,[1,2,3,4,8]]<=100)
        valid[:,5]&=(x[:,5]>=0)&(x[:,5]<=1);valid[:,[6,7]]&=x[:,[6,7]]>=-273.15;valid[:,9:13]&=np.abs(x[:,9:13])<=1
        future_ok=np.isfinite(x[:,[0,6]])&(x[:,[0,6]]>=np.array([0.,-273.15]))
        hp=np.r_[0,np.cumsum(~valid.all(axis=1))];tp=np.r_[0,np.cumsum(~future_ok.all(axis=1))]
        origins=np.flatnonzero(edges[:-1]%(6*WIDTH)==0);enough=origins>=36;contained=enough&(origins+6<=ngrid)
        complete=np.zeros(len(origins),bool);hist=np.zeros(len(origins),bool);z=origins[contained]
        complete[contained]=tp[z+6]==tp[z];hist[contained]=hp[z]==hp[z-36]
        status=np.full(len(origins),'accepted',dtype='<U28');status[~enough]='history_before_split'
        status[enough&~contained]='target_after_split';status[contained&~complete]='invalid_observed_target';status[contained&complete&~hist]='incomplete_input_history'
        selected=origins[status=='accepted'];counts={n:int((status==n).sum()) for n in ('history_before_split','target_after_split','invalid_observed_target','incomplete_input_history','accepted')}
        counts['candidates']=len(origins);counts['node_index']=idx;statuses[node]=counts
        if sum(counts[n] for n in ('history_before_split','target_after_split','invalid_observed_target','incomplete_input_history','accepted'))!=len(origins):raise AssertionError('Rejection accounting')
        total+=len(selected)
        # Includes future/source/support/mapping arrays plus full-population copies.
        estimated+=len(selected)*14_000*4
        if total>limits.max_windows or estimated>limits.max_population_bytes: raise ValueError('Window/population memory budget exceeded; never truncate')
        if not len(selected):continue
        h=selected[:,None]+np.arange(-36,0);t=selected[:,None]+np.arange(6);b=len(selected)
        future=x[t][:,:,[0,6]]
        flags=np.ones(future.shape,bool)
        target_means(future,flags)  # hardened BEFORE any bridge/coercing constructor
        full=np.concatenate((x[h],np.full((b,36,1),idx,dtype=np.float64)),axis=2)
        p={'x':full,'observed':valid[h],'future':future,'target_observed':flags,
         'nodes':np.full(b,node,dtype='U3'),'origins':edges[selected].astype('datetime64[ns]'),
         'history_times':edges[h].astype('datetime64[ns]'),'target_times':edges[t].astype('datetime64[ns]'),
         'history_nodes':np.full((b,36),idx,dtype='i8'),'target_nodes':np.full((b,6),idx,dtype='i8'),
         'source_earliest':earliest[h],'source_latest':latest[h],
         'target_source_earliest':np.stack((bins['total_power'].earliest[t],bins['ambient'].earliest[t]),axis=2),
         'target_source_latest':np.stack((bins['total_power'].latest[t],bins['ambient'].latest[t]),axis=2),
         'history_coverage':support[h],
         'target_coverage':np.stack((bins['total_power'].coverage[t],bins['ambient'].coverage[t]),axis=2)}
        pieces.append(p)
    if not pieces: raise ValueError('Empty accepted split; review coverage, never alter protocol')
    payload={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    windows=bridge.accept_windows(split=split,mapping=bridge.ORIGINAL_NAMES,source_evidence=evidence,**payload)
    diagnostic={'per_client':statuses,'raw_quality':quality,'weather_quality':weather_quality,'candidate_nodes':len(nodes),'accepted_windows':total,
       'estimated_packet_copy_budget_bytes':estimated,'limits':limits.__dict__,'population_filtering':'none; eligibility stays training-only downstream'}
    return ProducedSplit(_TOKEN,windows,diagnostic,payload)


def produce_from_series(local, weather, *, split, limits=Limits()):
    """In-memory input API. Provenance labelled synthetic, never authenticated raw.

    Actual execution on supplied project telemetry is not authorized by this API.
    For file-backed provenance use verify_source_plan + produce_from_plan.
    """
    _split(split)
    if not isinstance(local,dict): raise ValueError('Explicit node dictionary required')
    bindings={}
    for node in sorted(local,key=_client):
        if set(local[node])!=set(LOCAL_METRICS):raise ValueError('Missing/extra local metrics')
        for m,s in local[node].items():
            if type(s) is not RawSeries:raise ValueError('Validated original RawSeries required')
            bindings[node+'/'+m]=s.fingerprint
    if set(weather)!= {'temp','humidity'}:raise ValueError('Both weather metrics required')
    for m,s in weather.items():
        if type(s) is not RawSeries:raise ValueError('Validated weather RawSeries required')
        bindings['weather/'+m]=s.fingerprint
    return _produce(local,weather,split,limits,_evidence('synthetic',_digest(bindings)))


@dataclass(frozen=True)
class SourceEntry:
    path: str
    metric: str
    bytes: int
    sha256: str


class VerifiedSources:
    __slots__ = ('root','entries','fingerprint')
    def __setattr__(self,name,value): raise AttributeError('Immutable verified source plan')
    def __init__(self,token,root,entries):
        if token is not _TOKEN:raise TypeError('Use verify_source_plan')
        object.__setattr__(self,'root',root);object.__setattr__(self,'entries',tuple(entries))
        object.__setattr__(self,'fingerprint',_digest({'schema':'b1.JanFeb.sources.v1','entries':[e.__dict__ for e in entries],
            'mapping_digest':bridge.MAPPING_DIGEST,'splits':dict(bridge.SPLITS)}))


def _path(root,relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe source path')
    p=root/rel
    for q in (p,*p.parents):
        if q.is_symlink():raise ValueError('Symlink source prohibited')
    return p


def _hash_file(handle):
    handle.seek(0);h=hashlib.sha256()
    for b in iter(lambda:handle.read(1024*1024),b''):h.update(b)
    handle.seek(0);return h.hexdigest()


def _open_entry(root,e):
    p=_path(root,e.path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    f=os.fdopen(fd,'rb')
    try:
        s=os.fstat(f.fileno())
        if not stat.S_ISREG(s.st_mode) or s.st_size!=e.bytes or _hash_file(f)!=e.sha256:
            raise ValueError('Raw source size/hash differs: '+e.path)
        return f
    except BaseException:f.close();raise


def verify_source_plan(root,entries):
    """Read-only content verification of explicit20 Jan/Feb paths; no discovery.

    Call only on synthetic fixture files in this implementation stage. Validate
    ALL declarations before opening ANY file. No March/test selection exists.
    """
    entries=tuple(entries);seen=set();pairs=set()
    if len(entries)!=20:raise ValueError('Exactly twenty Jan/Feb source declarations required')
    for e in entries:
        if type(e) is not SourceEntry or type(e.path) is not str or e.metric not in METRICS:
            raise ValueError('Invalid source declaration')
        p=Path(e.path);parts=p.parts
        if p.is_absolute() or '..' in parts or p.suffix!='.parquet' or len(parts)!=4:raise ValueError('Unsafe source declaration')
        months=[v for v in parts if v.startswith('year_month=')]
        plugin='weather_pub' if e.metric in ('temp','humidity') else ('ipmi_pub' if e.metric in ('total_power','ambient') else 'ganglia_pub')
        if len(months)!=1 or months[0] not in ('year_month=22-01','year_month=22-02'):
            raise PermissionError('March/unknown source month refused before open')
        if parts[-3:-1]!=(f'plugin={plugin}',f'metric={e.metric}') or parts[-4]!=months[0]:
            raise ValueError('Metric/plugin source identity differs')
        pair=(months[0],e.metric)
        if e.path in seen or pair in pairs:raise ValueError('Duplicate/overlapping source intervals')
        seen.add(e.path);pairs.add(pair)
        if type(e.bytes) is not int or e.bytes<=0 or type(e.sha256) is not str or len(e.sha256)!=64 or any(x not in '0123456789abcdef' for x in e.sha256):
            raise ValueError('Exact expected byte/hash binding required')
    if pairs!={(m,k) for m in ('year_month=22-01','year_month=22-02') for k in METRICS}:raise ValueError('Incomplete source plan')
    root=Path(root).absolute()
    if any('march' in p.lower() for p in root.parts):raise PermissionError('March source root refused before open')
    for e in entries:
        with _open_entry(root,e):pass
    # Canonical metric/month/file ordering makes caller declaration permutations harmless.
    return VerifiedSources(_TOKEN,root,sorted(entries,key=lambda e:(METRICS.index(e.metric),e.path)))


def _read_plan_series(plan,metric,node,split,limits):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    lo,hi=_split(split);times=[];values=[];retained=0;audit={}
    for e in plan.entries:
        if e.metric!=metric:continue
        with _open_entry(plan.root,e) as handle:
            before=os.fstat(handle.fileno());f=pq.ParquetFile(handle);schema=f.schema_arrow
            columns=['timestamp'] if metric in ('temp','humidity') else ['timestamp','node']
            if not set(columns+['value'])<=set(schema.names):raise ValueError('Required raw columns missing')
            kind=schema.field('timestamp').type
            if not pa.types.is_timestamp(kind) or kind.tz!='UTC':raise ValueError('Explicit UTC Arrow timestamps required')
            value_type=schema.field('value').type
            if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):raise ValueError('Real numeric raw values required, not Boolean')
            if f.metadata.num_rows>limits.max_scan_rows:raise ValueError('Source scan row budget exceeded')
            month=next(p[11:] for p in Path(e.path).parts if p.startswith('year_month='))
            lower=np.datetime64('2022-'+month[3:]+'-01','ns').view('i8')
            upper=np.datetime64('2022-02-01' if month=='22-01' else '2022-03-01','ns').view('i8')
            # Timestamp-only pass rejects misfiled March records BEFORE requesting value columns.
            for batch in f.iter_batches(batch_size=limits.arrow_batch_rows,columns=['timestamp'],use_threads=False):
                valid=pc.is_valid(batch['timestamp']);a=pc.cast(pc.cast(pc.filter(batch['timestamp'],valid),pa.timestamp('ns',tz='UTC')),pa.int64()).to_numpy()
                if ((a<lower)|(a>=upper)).any():raise PermissionError('Source timestamp outside declared Jan/Feb month; no values opened')
            q={'rows_scanned':0,'null_time_rows':0,'out_of_split_rows':0,'other_client_rows':0,'unknown_node_rows':0};unknown=set()
            for batch in f.iter_batches(batch_size=limits.arrow_batch_rows,columns=columns+['value'],use_threads=False):
                valid=pc.is_valid(batch['timestamp']);bt=pc.filter(batch,valid)
                q['rows_scanned']+=len(batch);q['null_time_rows']+=len(batch)-len(bt)
                stamps=pc.cast(pc.cast(bt['timestamp'],pa.timestamp('ns',tz='UTC')),pa.int64()).to_numpy()
                keep=(stamps>=lo)&(stamps<hi);q['out_of_split_rows']+=int((~keep).sum())
                if 'node' in columns:
                    names=pc.cast(bt['node'],pa.string()).to_pylist()
                    owner=np.array([n==node for n in names],dtype=bool);known=np.array([n in bridge.ORIGINAL_NAMES for n in names],dtype=bool)
                    q['other_client_rows']+=int((keep&known&~owner).sum());q['unknown_node_rows']+=int((~known).sum())
                    unknown.update(n for n in names if n is not None and n not in bridge.ORIGINAL_NAMES)
                    if len(unknown)>1024:raise ValueError('Unknown-node evidence budget exceeded')
                    keep&=owner
                selected=pc.filter(bt,pa.array(keep));retained+=len(selected)
                if retained>limits.max_raw_rows:raise ValueError('Retained raw row budget exceeded')
                times.append(stamps[keep].astype('datetime64[ns]'))
                values.append(pc.cast(selected['value'],pa.float64()).to_numpy())
            q['unknown_node_ids']=sorted(unknown);audit[e.path]=q
            after=os.fstat(handle.fileno())
            fields=('st_dev','st_ino','st_size','st_mtime_ns','st_mode')
            if any(getattr(before,k)!=getattr(after,k) for k in fields) or _hash_file(handle)!=e.sha256:raise ValueError('Raw source changed during read')
    result=RawSeries(np.concatenate(times) if times else np.empty(0,dtype='datetime64[ns]'),
         np.concatenate(values) if values else np.empty(0),limits=limits)
    object.__setattr__(result,'_reader_audit',bridge.canonical(audit))
    return result


def produce_from_plan(plan, *, split, clients, limits=Limits()):
    """Read-only file adapter; no fitting/publication. Expensive per-client scans.

    Caller must separately authorize actual data use and capacity. All sources
    are explicit; only B1 train/val rows can be returned. No automatic 979-client run.
    """
    _split(split)
    if type(plan) is not VerifiedSources:raise TypeError('Verified source plan required')
    clients=tuple(clients)
    if not clients or len(set(clients))!=len(clients):raise ValueError('Explicit unique client IDs required')
    clients=tuple(sorted(clients,key=_client))
    weather={m:_read_plan_series(plan,m,None,split,limits) for m in ('temp','humidity')}
    def node_reader(node):return {m:_read_plan_series(plan,m,node,split,limits) for m in LOCAL_METRICS}
    return _produce({n:None for n in clients},weather,split,limits,
                    _evidence('observed_prepared',plan.fingerprint),node_reader=node_reader)


def bind_frozen_datasets(training,validation,scaler):
    """Apply an explicitly supplied frozen scaler; NEVER fit on either split."""
    if type(training) is not ProducedSplit or type(validation) is not ProducedSplit:
        raise TypeError('Separate produced populations required')
    return bridge.make_datasets(training.windows,validation.windows,scaler)
