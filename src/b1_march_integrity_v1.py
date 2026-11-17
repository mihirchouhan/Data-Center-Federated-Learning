"""Isolated March test contracts. No model execution, raw loading or scaler fit."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

from src import b1_preparation_io_v1 as io
from src import b1_real_data_bridge_v1 as bridge
from src.b1_bridge_scaling_v1 import FrozenScaler
from src.b1_scaler_acceptance_v1 import verify_candidate

ROOT = Path(bridge.__file__).absolute().parents[1]
LOCK_DIR = 'output/march_window_audit/b1_model_selection_lock_v1'
LOCK_SHA = 'ceb3f469e84fb4beb1de5fc34cef70d1cc626fbc4ae5b96711ce1a8c3e67ebc4'
LOCK_INVENTORY_SHA = 'a25d222997df0d6e1ef2a1215b06b104a7fb5dc50250a663e8db47b47d5c759a'
REFERENCE = {'windows': 194212, 'nodes': 976, 'origins': 508, 'dates': 28,
    'identity_sha256': '82dd311860cbe834e5a7700e0775f7464a7395f33899463f70baadc2817bfa26',
    'target_sha256': 'bafbdec241754596c2a7a3b54b0fee15010653da797dc6cee9b5d87b2172e0f1'}
REFERENCE_PATH = 'output/march_window_audit/v1/january_march_test_training_eligible_common_windows.npy'
REFERENCE_SHA = 'bee70d0fba454664a1e8ef33f4b257418b5cfea62b5ee2c537dc466eff1222b7'
PROTOCOL = {'history': 36, 'horizon': 6, 'grid_seconds': 600,
    'hold_seconds': 300, 'minimum_bin_coverage': .5, 'weather_max_age_seconds': 1800,
    'weather': 'enabled', 'input_fill': False, 'target_fill': False,
    'origins': 'hourly_UTC', 'whole_history_and_target_containment': True,
    'lower_UTC': '2022-03-01T00:00:00Z', 'upper_exclusive_UTC': '2022-04-01T00:00:00Z',
    'mapping_digest': bridge.MAPPING_DIGEST, 'target_order': ['mean_node_IT_power_W', 'mean_recorded_IPMI_ambient_degC'],
    'features': list(bridge.FEATURES), 'input_units': list(bridge.UNITS),
    'evaluation_batch_size': 1024, 'seeds': [42, 43, 44], 'selected_round': 200,
    'pooled': 'MAE=mean(abs(error)); RMSE=sqrt(mean(error**2))',
    'macro_node': 'Equal-node arithmetic mean of per-node MAE and per-node RMSE',
    'seed_aggregation': 'Mean of individual scores and sample SD ddof=1; no ensemble'}
NEW_SOURCES = ('src/b1_march_integrity_v1.py', 'src/b1_march_data_v1.py', 'src/b1_march_evaluate_v1.py')
GPU_UUID = 'GPU-7ef6875b-ba62-88e7-d0b9-e301a25b687e'


def read_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate JSON key: ' + k)
            result[k] = v
        return result
    with io.safe_open(path) as f:
        if os.fstat(f.fileno()).st_size > 64 * 1024**2:
            raise ValueError('JSON exceeds limit')
        return json.load(f, object_pairs_hook=pairs,
                         parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def digest(value):
    return io.sha(io.canonical(value))


def child(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe relative path')
    return io.safe_path(Path(root) / rel)


def check(path, expected):
    if io.binding(path) != {k: expected[k] for k in ('bytes', 'sha256')}:
        raise ValueError('Content binding changed: ' + str(path))


def load_plan(path):
    p = read_json(path)
    if p.get('schema') != 'b1.march.test.plan.v1' or io.canonical(p.get('protocol')) != io.canonical(PROTOCOL):
        raise ValueError('March protocol differs')
    if p.get('selection_lock') != {'path': LOCK_DIR + '/selection_lock.json', 'sha256': LOCK_SHA}:
        raise ValueError('Wrong model selection lock')
    if p.get('reference') != {'path': REFERENCE_PATH, 'sha256': REFERENCE_SHA, **REFERENCE}:
        raise ValueError('Wrong frozen March reference')
    if p.get('dataset_output') != 'output/experiments/b1_no_ff/preparation/march_test_v1':
        raise ValueError('Unbound preparation output')
    if p.get('evaluation_output') != 'output/experiments/b1_no_ff/march_test_evaluation_v2':
        raise ValueError('Unbound evaluation output')
    if p.get('receipt_output') != 'output/experiments/b1_no_ff/march_test_authorization_v1':
        raise ValueError('Unbound authorization destination')
    audit = ROOT / 'output/march_window_audit/v1/configuration.json'
    check(audit, p['audit_configuration'])
    cfg = read_json(audit)
    expected = {n: b for n, b in cfg['inputs'].items()
                if n.startswith('data/march_2022_manual/extracted/year_month=22-03/')}
    if len(expected) != 10 or p.get('raw_sources') != expected:
        raise ValueError('Only the ten recorded March metric inputs are allowed')
    sources = dict(p.get('sources', {}))
    if not set(NEW_SOURCES).issubset(sources):
        raise ValueError('Unbound test implementation')
    for n, b in sources.items():
        check(child(ROOT, n), b)
    if p.get('historical_preflight') != 'FAILED' or p.get('semantic_reproduction') != 'UNKNOWN':
        raise ValueError('Historical status must remain separate')
    return p


def locked_context(plan):
    """Full published lock verifier, no models/deserialization or March array I/O."""
    directory = ROOT / LOCK_DIR
    if io.binding(directory / 'selection_lock.json')['sha256'] != LOCK_SHA or \
       io.binding(directory / 'generated_files.json')['sha256'] != LOCK_INVENTORY_SHA:
        raise ValueError('Published selection package changed')
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location('_b1_selected_lock_verifier', directory / 'verify_lock.py')
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        result = verifier.verify(directory)
    finally:
        sys.path.pop(0)
    if result['status'] != 'PASS':
        raise ValueError('Selection verifier failed')
    lock = read_json(directory / 'selection_lock.json')
    if lock['seedset'] != [42, 43, 44] or [m['seed'] for m in lock['selected_models']] != [42, 43, 44]:
        raise ValueError('Missing/duplicate selected seed')
    if any(m['selected_round'] != 200 for m in lock['selected_models']):
        raise ValueError('Selected round changed')
    for n, b in lock['sources'].items():
        if plan['sources'].get(n) != b:
            raise ValueError('Training source graph is not preserved')
    for key in ('configuration', 'dataset_manifest', 'scaler_lock', 'scaler_statistics'):
        entry = lock[key]
        check(child(ROOT, entry['path']), entry)
    raw = read_json(ROOT / lock['scaler_statistics']['path'])
    scaler = FrozenScaler(tuple(raw['x_mean']), tuple(raw['x_std']), tuple(raw['original_x_std']),
                          tuple(raw['y_mean']), tuple(raw['y_std']), bridge.canonical(raw['provenance']))
    if verify_candidate(scaler, raw) != '469008a727b66c652462f6872ef2e829c6d023781a013f401f6f9f20ff48fce7':
        raise ValueError('Training scaler fingerprint differs')
    return lock, scaler, result


def statement(path, operation, bindings):
    """Validate an operator-supplied assertion; never infer it from a CLI flag."""
    record = read_json(path)
    required = {'operation': operation, 'explicit_operator_authorization': True,
                'uid': os.getuid(), 'bindings': bindings}
    if record.get('explicit_operator_authorization') is not True or any(record.get(k) != v for k, v in required.items()):
        raise PermissionError('Missing exact-scope operator authorization')
    if not isinstance(record.get('authorization_text'), str) or len(record['authorization_text'].strip()) < 20:
        raise PermissionError('Actual authorization text is required')
    expiry = record.get('expires_at_unix')
    if isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or not 0 < expiry - time.time() <= 86400:
        raise PermissionError('Authorization expired or exceeds 24 hours')
    return record


def stage_directory(destination):
    """Exclusive staging, no overwrite or implicit parent creation/restart."""
    destination = io.safe_path(destination)
    if os.path.lexists(destination):
        raise FileExistsError('Destination occupied; inspect, never overwrite/resume implicitly')
    if not destination.parent.is_dir() or destination.parent.stat().st_uid != os.getuid():
        raise ValueError('Existing owned output parent required')
    stage = destination.parent / ('.' + destination.name + '.pending')
    if stage.exists() or stage.is_symlink():
        raise FileExistsError('Prior incomplete/concurrent attempt retained; no automatic retry')
    stage.mkdir(mode=0o700)
    io.exclusive(stage/'owner.json',io.canonical({'uid':os.getuid(),'pid':os.getpid(),'created_at_unix':time.time(),'destination':str(destination)}))
    return stage


def publish(stage, destination):
    for root, dirs, files in os.walk(stage):
        for name in files:
            path = Path(root) / name
            with io.safe_open(path) as f:
                os.fsync(f.fileno())
            os.chmod(path, 0o444)
        io.fsync_dir(root)
    for root, dirs, files in os.walk(stage, topdown=False):
        os.chmod(root, 0o555)
    io.rename_noreplace(stage, destination)


def guard_test_access(output, allowed_arrays, allowed_models=(), raw_sources=()):
    """Process-local accident prevention; independent of unchanged training guard."""
    output = io.safe_path(output)
    arrays = {str(io.safe_path(p)) for p in allowed_arrays}
    models = {str(io.safe_path(p)) for p in allowed_models}
    raw = {str(io.safe_path(p)) for p in raw_sources}
    def guard(event, args):
        if event.startswith('socket.'):
            raise RuntimeError('Network forbidden in B1 test pipeline')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.path.abspath(os.fsdecode(args[0])))
            flags = args[2]
            write = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if p.is_relative_to(ROOT):
                if write and not p.is_relative_to(output):
                    raise RuntimeError('Write outside owned test staging')
                if p.is_relative_to(ROOT / 'data') and str(p) not in raw:
                    raise RuntimeError('Unbound/raw non-March source access')
                if p.suffix in ('.npy', '.npz', '.parquet', '.tar') and not p.is_relative_to(output) and str(p) not in arrays | raw:
                    raise RuntimeError('Undeclared array access')
                if p.suffix == '.pt' and str(p) not in models:
                    raise RuntimeError('Unselected checkpoint access')
    sys.addaudithook(guard)
