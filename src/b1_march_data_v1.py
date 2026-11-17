"""March-only observed bins and frozen streaming test inputs; never a fitter.

Production CLI defaults to description. Preparation requires exact operator
assertion plus fresh quota evidence. Old train/validation capabilities unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import time
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src import b1_march_integrity_v1 as integrity
from src import b1_preparation_io_v1 as io
from src import b1_real_data_bridge_v1 as bridge
from src import march_window_audit_v1 as audit
from src.b1_target_validation_v1 import target_means
from src.b1_raw_window_producer_v1 import BinEvidence
from src.phase2_multioutput.contracts import Windows, validate_windows, identity_digest
from src.history_data_v4 import LOCAL_METRICS, node_features
from src.multinode import weighted_bins
from src.preprocess import _range_mask, _weather_asof

WIDTH = 600_000_000_000
LO = int(np.datetime64('2022-03-01', 'ns').view('i8'))
HI = int(np.datetime64('2022-04-01', 'ns').view('i8'))
EDGES = np.arange(LO, HI + WIDTH, WIDTH, dtype='i8')
RECORD = np.dtype([('time', '<i8'), ('value', '<f8')])
FIELDS = ('features', 'earliest', 'latest', 'coverage')
TEST_PROTOCOL = {'splits': {'train': ['2022-01-01T00:00:00Z', '2022-01-28T00:00:00Z'],
    'val': ['2022-01-28T00:00:00Z', '2022-02-04T00:00:00Z'],
    'test': ['2022-03-01T00:00:00Z', '2022-04-01T00:00:00Z']}}


def qualified_bins(times, values, metric):
    """Stable keep-last, invalid-last retained, raw contributors reset at March1."""
    if metric not in (*LOCAL_METRICS, 'temp', 'humidity'):
        raise ValueError('Unknown metric')
    if np.ma.isMaskedArray(times) or np.ma.isMaskedArray(values):
        raise ValueError('Masked raw stream')
    times, values = np.asarray(times), np.asarray(values)
    if times.dtype != np.dtype('datetime64[ns]') or times.ndim != 1 or values.shape != times.shape or values.dtype.kind not in 'iuf':
        raise ValueError('Explicit UTC ns timestamps and numeric nonboolean values required')
    keep = ~np.isnat(times) & (times.view('i8') >= LO) & (times.view('i8') < HI)
    t, v = times.view('i8')[keep], values[keep].astype('f8')
    order = np.argsort(t, kind='stable'); t, v = t[order], v[order]
    unique = np.r_[t[1:] != t[:-1], True] if len(t) else np.zeros(0, bool)
    duplicates = int(len(t) - unique.sum()); t, v = t[unique], v[unique]
    valid, _ = _range_mask(metric, v); v[~valid] = np.nan
    n = len(EDGES) - 1
    first = np.full(n, np.iinfo('i8').max, dtype='i8')
    last = np.full(n, np.iinfo('i8').min, dtype='i8'); counts = np.zeros(n, dtype='i8')
    if not len(t):
        result = np.full(n, np.nan); coverage = np.zeros(n)
    elif metric in ('temp', 'humidity'):
        result = _weather_asof(t, v, EDGES, 1800_000_000_000)
        idx = np.searchsorted(t, EDGES[1:] - 1, side='right') - 1
        good = (idx >= 0) & np.isfinite(result)
        first[good] = t[idx[good]]; last[good] = first[good]; counts[good] = 1
        coverage = np.isfinite(result).astype('f8')
    else:
        result, coverage = weighted_bins(t, v, EDGES, 300_000_000_000, .5)
        ends = np.minimum(np.r_[t[1:], EDGES[-1]], t + 300_000_000_000)
        good = np.isfinite(v) & (ends > t)
        starts, ends = t[good], ends[good]
        left, right = (starts - LO) // WIDTH, (ends - 1 - LO) // WIDTH
        np.minimum.at(first, left, starts); np.maximum.at(last, left, starts); np.add.at(counts, left, 1)
        cross = right != left
        np.minimum.at(first, right[cross], starts[cross]); np.maximum.at(last, right[cross], starts[cross]); np.add.at(counts, right[cross], 1)
    first[counts == 0] = np.iinfo('i8').min
    return BinEvidence(result, coverage, first.astype('datetime64[ns]'), last.astype('datetime64[ns]'), counts), {
        'input_rows': len(times), 'discarded_null_or_outside_March': int((~keep).sum()),
        'duplicate_rows': duplicates, 'invalid_unique_values': int((~valid).sum())}


def build_node(local, weather):
    if set(local) != set(LOCAL_METRICS) or set(weather) != {'temp', 'humidity'}:
        raise ValueError('All eight local raw channels and both weather inputs required')
    bins, quality = {}, {}
    for metric, pair in local.items():
        bins[metric], quality[metric] = qualified_bins(*pair, metric)
    wb = {m: qualified_bins(*pair, m)[0] for m, pair in weather.items()}
    x = node_features(np.column_stack([bins[m].values for m in LOCAL_METRICS]),
                      {m: wb[m].values for m in wb}, EDGES)
    mem, _ = audit.aligned_memory(bins['mem_free'].values, bins['mem_total'].values)
    if not np.array_equal(x[:, 5], mem, equal_nan=True):
        raise ValueError('Memory fraction definition mismatch')
    sensors = ('total_power', 'cpu_user', 'cpu_system', 'cpu_idle', 'cpu_wio')
    first = np.column_stack([*(bins[m].earliest for m in sensors),
        np.minimum(bins['mem_free'].earliest, bins['mem_total'].earliest), bins['ambient'].earliest,
        wb['temp'].earliest, wb['humidity'].earliest])
    last = np.column_stack([*(bins[m].latest for m in sensors),
        np.maximum(bins['mem_free'].latest, bins['mem_total'].latest), bins['ambient'].latest,
        wb['temp'].latest, wb['humidity'].latest])
    coverage = np.column_stack([*(bins[m].coverage for m in sensors),
        np.minimum(bins['mem_free'].coverage, bins['mem_total'].coverage), bins['ambient'].coverage])
    return {'features': x, 'earliest': first, 'latest': last, 'coverage': coverage}, quality


def packet(node, binned, indices):
    if node not in bridge.ORIGINAL_NAMES:
        raise ValueError('Unknown original node')
    i = bridge.ORIGINAL_NAMES.index(node); z = np.asarray(indices)
    if z.dtype.kind not in 'iu' or z.ndim != 1 or not len(z) or (np.diff(z) <= 0).any():
        raise ValueError('Nonempty strictly ordered integer window positions required')
    if (z < 36).any() or (z + 6 > len(EDGES)-1).any() or (EDGES[z] % (6*WIDTH)).any():
        raise ValueError('Whole history/target containment or hourly origin violation')
    h = z[:, None] + np.arange(-36, 0); t = z[:, None] + np.arange(6)
    x = binned['features']; n = len(z)
    future = x[t][:, :, [0, 6]]
    result = {'x': np.concatenate((x[h], np.full((n, 36, 1), i)), axis=2),
        'future': future, 'target_observed': np.isfinite(future),
        'observed': np.isfinite(x[h]), 'nodes': np.full(n, node, dtype='U3'),
        'origins': EDGES[z].astype('datetime64[ns]'), 'history_times': EDGES[h].astype('datetime64[ns]'),
        'target_times': EDGES[t].astype('datetime64[ns]'), 'history_nodes': np.full((n, 36), i, dtype='i8'),
        'target_nodes': np.full((n, 6), i, dtype='i8'), 'source_earliest': binned['earliest'][h],
        'source_latest': binned['latest'][h], 'target_source_earliest': binned['earliest'][t][:, :, [0, 6]],
        'target_source_latest': binned['latest'][t][:, :, [0, 6]], 'history_coverage': binned['coverage'][h],
        'target_coverage': binned['coverage'][t][:, :, [0, 6]]}
    validate_packet(result)
    return result


def validate_packet(p):
    """Validate original containers BEFORE mask-erasing Windows conversion."""
    for name, value in p.items():
        if np.ma.isMaskedArray(value):
            raise ValueError('Masked packet field: ' + name)
    y = target_means(p['future'], p['target_observed'])
    bridge._numeric(p['future'], 'future')
    bridge._numeric(p['x'], 'inputs', (len(y), 36, 14))
    keys = ('x', 'observed', 'future', 'target_observed', 'history_times', 'target_times',
            'history_nodes', 'target_nodes', 'source_latest', 'source_earliest', 'nodes', 'origins')
    w = Windows(**{k: p[k] for k in keys}, mapping=bridge.ORIGINAL_NAMES, split='test', history=36)
    if not np.array_equal(validate_windows(w, TEST_PROTOCOL), y):
        raise ValueError('Observed target aggregation differs')
    hs = w.history_times.view('i8')[:, :, None]; ts = w.target_times.view('i8')[:, :, None]
    bridge._source_bounds(w.source_earliest[:, :, :7], w.source_latest[:, :, :7], hs, LO, (len(y),36,7), 'history')
    bridge._source_bounds(w.source_earliest[:, :, 7:], w.source_latest[:, :, 7:], hs, LO, (len(y),36,2), 'weather', True)
    bridge._source_bounds(p['target_source_earliest'], p['target_source_latest'], ts, LO, (len(y),6,2), 'target')
    bridge._support(p['history_coverage'], (len(y),36,7), 'history')
    bridge._support(p['target_coverage'], (len(y),6,2), 'target')
    seconds = (hs[:, :, 0] + WIDTH) / 1e9
    daily = 2*np.pi*(seconds % 86400)/86400
    weekly = 2*np.pi*((seconds + 259200) % 604800)/604800
    calendar = np.stack((np.sin(daily), np.cos(daily), np.sin(weekly), np.cos(weekly)), axis=2)
    if not np.allclose(w.x[:, :, 9:13], calendar, rtol=1e-12, atol=1e-12):
        raise ValueError('UTC calendar differs')
    return y


def records(node, binned):
    positions, target, reasons, missing = audit.window_masks(binned['features'], EDGES)
    a = np.empty(len(positions), dtype=audit.WINDOW_DTYPE)
    a['node_index'] = bridge.ORIGINAL_NAMES.index(node); a['origin_ns'] = EDGES[positions]
    a['power_w'], a['temperature_c'] = target.T
    for first in range(0, len(positions), 128):
        p = packet(node, binned, positions[first:first+128])
        if not np.array_equal(validate_packet(p), target[first:first+128]):
            raise ValueError('B1 hardened target means differ from coverage audit')
    return a, reasons


def validate_reference(a, expected=integrity.REFERENCE):
    if np.ma.isMaskedArray(a) or a.dtype != audit.WINDOW_DTYPE or a.ndim != 1:
        raise ValueError('Frozen reference schema differs')
    audit.validate_records(a, bridge.ORIGINAL_NAMES, 'test', TEST_PROTOCOL)
    y = np.column_stack((a['power_w'], a['temperature_c']))
    result = {'windows': len(a), 'nodes': len(np.unique(a['node_index'])),
        'origins': len(np.unique(a['origin_ns'])),
        'dates': len(np.unique(a['origin_ns'].astype('datetime64[ns]').astype('datetime64[D]'))),
        'identity_sha256': identity_digest(np.asarray(bridge.ORIGINAL_NAMES)[a['node_index']], a['origin_ns'].astype('datetime64[ns]')),
        'target_sha256': hashlib.sha256(y.astype('<f8').tobytes()).hexdigest()}
    if result != expected:
        raise ValueError('Frozen reference count/identity/target/coverage mismatch')
    return result


def read_node(path):
    """Reject unexpected ZIP shapes/compression before allocating arrays."""
    shapes = {'features': ((4464,13), np.dtype('float64')),
              'earliest': ((4464,9), np.dtype('datetime64[ns]')),
              'latest': ((4464,9), np.dtype('datetime64[ns]')),
              'coverage': ((4464,7), np.dtype('float64'))}
    with io.safe_open(path) as raw, zipfile.ZipFile(raw) as z:
        if len(z.infolist()) != 4 or {e.filename for e in z.infolist()} != {k+'.npy' for k in shapes}:
            raise ValueError('Unexpected node arrays')
        for e in z.infolist():
            if e.compress_type != zipfile.ZIP_STORED:
                raise ValueError('Uncompressed bounded arrays required')
            with z.open(e) as f:
                v = np.lib.format.read_magic(f)
                if v != (1,0):
                    raise ValueError('Unexpected array header version')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f, max_header_size=1024)
                want_shape, want_type = shapes[e.filename[:-4]]
                if shape != want_shape or dtype != want_type or fortran or e.file_size != f.tell()+int(np.prod(shape))*dtype.itemsize:
                    raise ValueError('Array shape/dtype/length differs')
    with io.safe_open(path) as f, np.load(f, allow_pickle=False) as z:
        return {k: z[k] for k in FIELDS}


def read_reference(path):
    with io.safe_open(path) as f:
        version=np.lib.format.read_magic(f)
        if version!=(1,0):raise ValueError('Reference array version differs')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=1024)
        if len(shape)!=1 or not 0<shape[0]<=integrity.REFERENCE['windows'] or dtype!=audit.WINDOW_DTYPE or fortran or os.fstat(f.fileno()).st_size!=f.tell()+shape[0]*dtype.itemsize:
            raise ValueError('Reference array header/size differs')
    with io.safe_open(path) as f:return np.load(f,allow_pickle=False)


def verify_dataset(path, plan, expected=integrity.REFERENCE):
    path = io.safe_path(path); m = integrity.read_json(path/'manifest.json')
    if m.get('schema') != 'b1.march.prepared.bins.v1' or m.get('status') != 'complete' or m.get('plan') != plan:
        raise ValueError('Test dataset manifest/plan differs')
    if m['plan_sha256'] != integrity.digest(plan) or m['mapping'] != list(bridge.ORIGINAL_NAMES):
        raise ValueError('Test dataset contract/mapping differs')
    names = [v['path'] for v in m['nodes']]
    if len(names) != 979 or len(set(names)) != 979 or {p.name for p in path.iterdir()} != set(names)|{'manifest.json','reference.npy'}:
        raise ValueError('Missing/duplicate/unexpected dataset artifacts')
    integrity.check(path/'reference.npy', m['reference_file'])
    reference=read_reference(path/'reference.npy')
    validate_reference(reference, expected)
    cursor = 0; reasons = {}
    for i, entry in enumerate(m['nodes']):
        if entry['node'] != bridge.ORIGINAL_NAMES[i] or entry['path'] != f'node_{i:04d}.npz':
            raise ValueError('Node manifest ordering/identity differs')
        p = path/entry['path']; integrity.check(p, entry)
        bins = read_node(p); got, why = records(entry['node'], bins)
        if not np.array_equal(reference[cursor:cursor+len(got)], got) or entry['windows'] != len(got):
            raise ValueError('Native March windows/observed targets differ; never silently filter')
        cursor += len(got); reasons[entry['node']] = why
        integrity.check(p, entry)
    if cursor != len(reference) or m['results'] != expected or m['exclusions'] != reasons:
        raise ValueError('Incomplete test population/exclusions')
    return m, reference


class MarchData:
    """Only frozen test bins; no train/validation reads or statistic fitting."""
    def __init__(self, path, plan, scaler):
        self.path = io.safe_path(path); self.plan = plan; self.scaler = scaler
        self.manifest, reference = verify_dataset(self.path, plan)
        self.reference = bridge.frozen_array(reference)
        self.manifest_binding = io.binding(self.path/'manifest.json')

    def batches(self, size=1024):
        if type(size) is not int or not 1 <= size <= 1024:
            raise ValueError('Evaluation batch limit exceeded')
        for entry in self.manifest['nodes']:
            path = self.path/entry['path']; integrity.check(path, entry)
            b = read_node(path)
            positions, y, _, _ = audit.window_masks(b['features'], EDGES)
            for first in range(0, len(positions), size):
                p = packet(entry['node'], b, positions[first:first+size]); physical = validate_packet(p)
                x = self.scaler.transform_inputs(p['x'])
                yield x, physical, p['nodes'], p['origins']
            integrity.check(path, entry)

    def assert_unchanged(self):
        integrity.check(self.path/'manifest.json', self.manifest_binding)
        integrity.check(self.path/'reference.npy', self.manifest['reference_file'])
        for entry in self.manifest['nodes']:
            integrity.check(self.path/entry['path'], entry)


def _source_metric(name):
    parts = Path(name).parts
    if len(parts) != 7 or parts[:4] != ('data','march_2022_manual','extracted','year_month=22-03') or parts[-1] != 'a_0.parquet':
        raise ValueError('Only the recorded March source files')
    return parts[-2][len('metric='):]


def spool_sources(root, sources, attempt, limits):
    """One bounded Arrow scan per metric, exclusive per-node spools, stable rows."""
    row_counts = {}; quality = {}
    for name, binding in sorted(sources.items()):
        metric = _source_metric(name); path = integrity.child(root, name)
        integrity.check(path, binding); q = {'rows':0,'null_times':0,'outside_March':0,'unmapped_rows':0}
        with io.safe_open(path) as f:
            original = os.fstat(f.fileno()); source = pq.ParquetFile(f); schema = source.schema_arrow
            if source.metadata.num_rows > limits.max_file_rows:
                raise ValueError('Raw scan cap exceeded')
            kind = schema.field('timestamp').type; value_kind = schema.field('value').type
            if not pa.types.is_timestamp(kind) or kind.tz != 'UTC' or not (pa.types.is_integer(value_kind) or pa.types.is_floating(value_kind)):
                raise ValueError('Explicit UTC and real numeric raw values required')
            local = metric in LOCAL_METRICS
            if not local and metric not in ('temp','humidity'):
                raise ValueError('Unexpected metric')
            if local and not pa.types.is_string(schema.field('node').type):
                raise ValueError('Original literal string node IDs required')
            columns = ['timestamp','value','node'] if local else ['timestamp','value']
            for batch in source.iter_batches(batch_size=limits.arrow_rows, columns=columns, use_threads=False):
                attempt.budget.check(); q['rows'] += len(batch)
                good = pc.is_valid(batch['timestamp']); clean = pc.filter(batch, good)
                q['null_times'] += len(batch)-len(clean)
                times = pc.cast(pc.cast(clean['timestamp'], pa.timestamp('ns',tz='UTC')), pa.int64()).to_numpy()
                inside = (times >= LO) & (times < HI); q['outside_March'] += int((~inside).sum())
                if local:
                    dictionary = pc.dictionary_encode(clean['node'])
                    lookup = {n:i for i,n in enumerate(bridge.ORIGINAL_NAMES)}
                    codes = np.array([lookup.get(n,-1) for n in dictionary.dictionary.to_pylist()]+[-1])[pc.fill_null(dictionary.indices,-1).to_numpy()]
                    q['unmapped_rows'] += int((codes<0).sum())
                else:
                    codes = np.zeros(len(clean),dtype='i8')
                selected = np.flatnonzero(inside & (codes>=0))
                if not len(selected):
                    continue
                values = pc.cast(pc.take(clean['value'],pa.array(selected)),pa.float64()).to_numpy()
                for i in np.unique(codes[selected]):
                    take = np.flatnonzero(codes[selected]==i)
                    key = f'spool/{metric}_{int(i):04d}.bin'
                    row_counts[key] = row_counts.get(key,0)+len(take)
                    if row_counts[key] > limits.max_raw_rows:
                        raise ValueError('Per-node raw row cap exceeded; never truncate')
                    rows = np.empty(len(take),dtype=RECORD); rows['time']=times[selected[take]]; rows['value']=values[take]
                    attempt.append(key,rows.tobytes())
            after = os.fstat(f.fileno())
            if any(getattr(original,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_mode')):
                raise ValueError('Source metadata changed during scan')
        integrity.check(path,binding)
        if q['rows'] != source.metadata.num_rows:
            raise ValueError('Incomplete scan')
        quality[name]=q
    attempt.seal_spools()
    return quality


def _read_spool(attempt, metric, index, limits):
    path = attempt.spool/f'{metric}_{index:04d}.bin'
    if not path.exists():
        return np.empty(0,dtype='datetime64[ns]'),np.empty(0,dtype='f8')
    with io.safe_open(path) as f:
        size = os.fstat(f.fileno()).st_size
        if size%RECORD.itemsize or size//RECORD.itemsize>limits.max_raw_rows:
            raise ValueError('Spool size/count differs')
        a=np.frombuffer(f.read(),dtype=RECORD)
    return a['time'].astype('datetime64[ns]'),a['value']


def prepare(plan, statement_path, quota_path):
    """Production-only execution, never called by synthetic tests or description."""
    lock, scaler, lock_verification = integrity.locked_context(plan)
    output = integrity.ROOT/plan['dataset_output']
    auth = integrity.statement(statement_path,'B1_March_test_preparation_only',
        {'plan_sha256':integrity.digest(plan),'selection_lock_sha256':integrity.LOCK_SHA,'output':str(output)})
    limits=io.Limits(**plan['limits']);budget=io.Budget(output,limits,integrity.read_json(quota_path))
    attempt=io.Attempt(output,integrity.digest(plan),budget)
    try:
        models=[integrity.ROOT/m['selected_model']['path'] for m in lock['selected_models']]
        integrity.guard_test_access(attempt.root,[integrity.ROOT/integrity.REFERENCE_PATH],models,
                                    [integrity.ROOT/n for n in plan['raw_sources']])
        attempt.write('authorization.json',io.canonical(auth));attempt.write('plan.json',io.canonical(plan))
        refpath=integrity.ROOT/integrity.REFERENCE_PATH
        if io.binding(refpath)['sha256']!=integrity.REFERENCE_SHA:
            raise ValueError('Frozen March reference file changed')
        reference=read_reference(refpath)
        validate_reference(reference)
        quality=spool_sources(integrity.ROOT,plan['raw_sources'],attempt,limits)
        weather={m:_read_spool(attempt,m,0,limits) for m in ('temp','humidity')}
        entries=[];exclusions={};cursor=0
        for i,node in enumerate(bridge.ORIGINAL_NAMES):
            budget.check()
            local={m:_read_spool(attempt,m,i,limits) for m in LOCAL_METRICS}
            b,q=build_node(local,weather);actual,why=records(node,b)
            if not np.array_equal(actual,reference[cursor:cursor+len(actual)]):
                raise ValueError('March identity/target mismatch for node '+node)
            cursor+=len(actual);name=f'node_{i:04d}.npz';path=attempt.dataset/name
            budget.charge(sum(a.nbytes for a in b.values())+65536)
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as f:np.savez(f,**{k:np.ascontiguousarray(v) for k,v in b.items()});f.flush();os.fsync(f.fileno())
            entries.append({'node':node,'path':name,'windows':len(actual),**io.binding(path)})
            exclusions[node]=why;attempt.event('node_prepared',{'node':node,'windows':len(actual),'quality':q})
        if cursor!=len(reference):
            raise ValueError('Missing reference windows')
        budget.charge(reference.nbytes+65536)
        fd=os.open(attempt.dataset/'reference.npy',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:np.save(f,reference,allow_pickle=False);f.flush();os.fsync(f.fileno())
        manifest={'schema':'b1.march.prepared.bins.v1','status':'complete','plan':plan,'plan_sha256':integrity.digest(plan),
            'mapping':list(bridge.ORIGINAL_NAMES),'reference_file':io.binding(attempt.dataset/'reference.npy'),
            'results':integrity.REFERENCE,'nodes':entries,'exclusions':exclusions,'raw_quality':quality,
            'scaler_lock':lock['scaler_lock'],'scalers_fitted':False,'model_operations':False}
        attempt.write('dataset.partial/manifest.json',io.canonical(manifest))
        verify_dataset(attempt.dataset,plan)
        # Frozen transforms verified, never fit. Zero-coverage nodes are retained in the mapping.
        data=MarchData(attempt.dataset,plan,scaler)
        for x,y,n,o in data.batches():
            if x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Invalid actual model inputs')
        for n,b in plan['sources'].items():integrity.check(integrity.ROOT/n,b)
        for n,b in plan['raw_sources'].items():integrity.check(integrity.ROOT/n,b)
        budget.check()
        for p in attempt.dataset.iterdir():
            with io.safe_open(p) as f:os.fsync(f.fileno())
            os.chmod(p,0o444)
        io.fsync_dir(attempt.dataset);os.chmod(attempt.dataset,0o555)
        io.rename_noreplace(attempt.dataset,attempt.root/'dataset')
        attempt.write('completed.json',io.canonical({'status':'verified_complete','dataset_manifest':io.binding(attempt.root/'dataset/manifest.json'),
            'wall_seconds':time.monotonic()-budget.start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'charged_bytes':budget.used,'scaler_fitted':False,'inference':False}))
        return {'status':'verified_complete','output':str(output),'results':integrity.REFERENCE,'inference':False}
    except BaseException as error:
        try:attempt.write('failure.json',io.canonical({'status':'incomplete_preserved','type':type(error).__name__,'error':str(error)}))
        except OSError:pass
        raise
    finally:
        attempt.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--stage',choices=('describe','prepare','verify'),default='describe')
    p.add_argument('--operator-authorization');p.add_argument('--quota-evidence')
    a=p.parse_args();plan=integrity.load_plan(a.plan)
    if a.stage=='describe':
        print(json.dumps({'status':'implementation_only','March_access':False,'inference':False,'protocol':plan['protocol'],'required_reference':plan['reference']}));return
    if a.stage=='prepare':
        if not a.operator_authorization or not a.quota_evidence:raise PermissionError('Operator authorization and fresh quota evidence required')
        print(json.dumps(prepare(plan,a.operator_authorization,a.quota_evidence)));return
    lock,scaler,_=integrity.locked_context(plan)
    data=MarchData(integrity.ROOT/plan['dataset_output']/'dataset',plan,scaler);data.assert_unchanged()
    print(json.dumps({'status':'verified_complete','results':data.manifest['results'],'inference':False}))

if __name__=='__main__':
    main()
