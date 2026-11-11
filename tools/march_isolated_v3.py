#!/usr/bin/env python3
"""Isolated V3 session adapter for the unchanged helper v2 / background v3.

Default actions are offline. No production transfer without explicit download-only
and quota assertions. Old evidence is read, never migrated or re-signed.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path('/media/csedept/cse2014/Project/m100_training_template')
ENTRY = Path(__file__).absolute()
CONFIG = ROOT / 'configs/acquisition/march_2022_isolated_v3.json'
CONFIG_SHA = 'efcb107b828becfc26033cf3c648a209e6bb5891a9fc5fc200001689db3515eb'
PINS = {
    'tools/acquire_march_2022_v2.py': '8a1a503baeeb122c846821733893af68c39d51478864ee33d6ff15b27be96191',
    'tools/march_background_v3.py': 'aab44bebd3aa2e0f6d93f46db1514da7c4f7a7701c716e8b176894a720a2e857',
    'tools/acquire_march_2022_v1.py': '25a3abf0beb5148726eb04a26b0874880355f70ea00efbedaa25e03817253925',
    'configs/acquisition/march_2022_v1.json': '4a7911cdf1cbf35b109186000b9ad697de1ba06faf66295bfb5d1338542f5725',
}
NEW_PATHS = ('data/march_2022_acquisition/v3', 'output/march_2022_acquisition/v3',
             'output/march_2022_background/v3')
OLD_PATHS = ('data/march_2022_acquisition/v1', 'output/march_2022_acquisition/v1',
             'output/march_2022_background/v2')

class Stop(RuntimeError):
    """Preserve all evidence and stop."""

def read_owned(path):
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise Stop('Symlink or traversal in protected path')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1:
        raise Stop('Unowned/nonregular/hardlinked protected evidence')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        initial = os.fstat(stream.fileno()); raw = stream.read(); after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_mode, s.st_uid)
    if identity(before) != identity(initial) or identity(initial) != identity(after):
        raise Stop('Protected evidence changed while reading')
    return raw

def sha(path):
    return hashlib.sha256(read_owned(path)).hexdigest()

def load_pinned(name, relative):
    path = ROOT / relative
    raw = read_owned(path)
    if hashlib.sha256(raw).hexdigest() != PINS[relative]:
        raise Stop('Protected execution source changed: ' + relative)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module.__file__ = str(path); sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module

sys.path.insert(0, str(ROOT / 'tools'))
acq = load_pinned('acquire_march_2022_v2', 'tools/acquire_march_2022_v2.py')
bg = load_pinned('_march_isolated_background_v3', 'tools/march_background_v3.py')

def validate_declaration(cfg):
    expected = {'schema', 'archive', 'archive_version', 'data_relative', 'output_relative',
                'background_relative', 'max_requests', 'lifetime_max_requests',
                'max_storage_bytes', 'reserve_free_bytes', 'authorization_required',
                'default_mode', 'ca_bundle', 'prior_session'}
    if set(cfg) != expected or cfg['schema'] != 'march_isolated_session_v3.1':
        raise Stop('Unexpected isolated declaration schema')
    paths = tuple(cfg[k] for k in ('data_relative','output_relative','background_relative'))
    if paths != NEW_PATHS or cfg['authorization_required'] is not True or cfg['default_mode'] != 'dry_run' or cfg['ca_bundle'] is not None:
        raise Stop('Isolated paths/authorization policy changed')
    archive = {'record':'7590061','archive':'22-03.tar','exact_bytes':31495884800,
               'published_md5':'4c74ad95b6d837d409e24216389bb25e',
               'resolved_url':'https://zenodo.org/api/records/7590061/files/22-03.tar/content',
               'metadata_url':'https://zenodo.org/api/records/7590061'}
    if cfg['archive'] != archive or cfg['archive_version'] != '1.0.0':
        raise Stop('Archive identity changed')
    for key, expected_value in [('max_requests',4),('lifetime_max_requests',5),
                                ('max_storage_bytes',80000000000),('reserve_free_bytes',10000000000)]:
        if type(cfg[key]) is not int or cfg[key] != expected_value:
            raise Stop('Request/storage limit changed: ' + key)
    prior = cfg['prior_session']
    if set(prior) != {'data_relative','output_relative','background_relative','owner',
                     'archive_requests_consumed','files','binding'}:
        raise Stop('Unexpected lineage schema')
    if tuple(prior[k] for k in ('data_relative','output_relative','background_relative')) != OLD_PATHS:
        raise Stop('Historical lineage paths changed')
    if type(prior['archive_requests_consumed']) is not int or prior['archive_requests_consumed'] != 1:
        raise Stop('Consumed request cannot be reset')
    if cfg['max_requests'] + prior['archive_requests_consumed'] != cfg['lifetime_max_requests']:
        raise Stop('Lifetime archive request budget mismatch')
    if prior['binding'] != {'helper_sha256':PINS['tools/acquire_march_2022_v1.py'],
                            'config_sha256':PINS['configs/acquisition/march_2022_v1.json']}:
        raise Stop('Historical source binding changed')
    if len(prior['files']) != 18:
        raise Stop('Incomplete historical evidence inventory')
    if not re.fullmatch('[a-f0-9]{32}',prior['owner']):
        raise Stop('Invalid historical owner')
    if any('..' in Path(p).parts or not any(p.startswith(d+'/') for d in OLD_PATHS)
           for p in prior['files']):
        raise Stop('Historical evidence path escapes recorded roots')
    return cfg

def verify_lineage(root, cfg):
    """Actual bytes and stat checks; no network, lock writes, or old mutation."""
    validate_declaration(cfg); root = Path(root); prior = cfg['prior_session']
    found = set()
    for relative in OLD_PATHS:
        directory = bg.safe(root / relative); s = directory.lstat()
        if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid():
            raise Stop('Historical root not an owned directory')
        for path in directory.rglob('*'):
            bg.safe(path)
            if path.is_dir():
                if path.stat().st_uid != os.getuid(): raise Stop('Unowned historical directory')
            else: found.add(str(path.relative_to(root)))
    if found != set(prior['files']):
        raise Stop('Historical evidence set changed; no request reset permitted')
    raw_files = {}
    for relative, expected in prior['files'].items():
        path = root / relative; raw = read_owned(path); s = path.stat()
        actual = {'bytes':s.st_size,'sha256':hashlib.sha256(raw).hexdigest(),
                  'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),
                  'inode':s.st_ino,'device':s.st_dev}
        if actual != expected: raise Stop('Historical evidence mismatch: ' + relative)
        raw_files[relative] = raw
    data, evidence, workflow = [root / x for x in OLD_PATHS]
    load = lambda p: json.loads(raw_files[str(p.relative_to(root))])
    session = load(data/'session.json')
    if session != load(evidence/'session.json') or session['owner'] != prior['owner'] or session['uid'] != os.getuid() or session['binding'] != prior['binding']:
        raise Stop('Historical session ownership/binding changed')
    if session['data'] != str(data) or session['output'] != str(evidence) or session['limits']['max_requests'] != 5:
        raise Stop('Historical session paths/budget changed')
    metadata = acq.parse_metadata(raw_files[str((evidence/'official_metadata.json').relative_to(root))])
    a = cfg['archive']
    if (metadata.size, metadata.md5, metadata.url) != (a['exact_bytes'],a['published_md5'],a['resolved_url']) or metadata.identity() != session['source']:
        raise Stop('Historical metadata/archive identity mismatch')
    requests = list(evidence.glob('request-*.json')); states = list(evidence.glob('state-*.json'))
    if len(requests) != 1 or len(states) != 1 or (data/'payload.part').stat().st_size != 0:
        raise Stop('Historical request/prefix evidence changed')
    req, state = load(requests[0]), load(states[0])
    if req['owner'] != session['owner'] or req['offset'] != 0 or req['url'] != metadata.url:
        raise Stop('Historical request identity mismatch')
    if state['owner'] != session['owner'] or state['validator'] is not None or state['previous'] is not None or state['prefix'] != acq.hashes(data/'payload.part'):
        raise Stop('Historical empty checkpoint changed; no migration permitted')
    if state['source_sha256'] != acq.digest(acq.canonical(metadata.identity())):
        raise Stop('Historical checkpoint source changed')
    manifest = load(workflow/'workflow.json')
    if manifest['binding'] != prior['binding'] or manifest['data'] != str(data) or manifest['evidence'] != str(evidence):
        raise Stop('Historical workflow identity mismatch')
    exits = list(workflow.glob('attempt-*/exit.json'))
    if len(exits) != 1 or load(exits[0])['worker_returncode'] != 2 or load(exits[0])['archive_verified_and_published'] is not False:
        raise Stop('Historical failed attempt not established')
    for path in workflow.glob('attempt-*/*.json'):
        record = load(path)
        if 'identity' in record and bg.alive(record['identity']):
            raise Stop('Historical process is still running')
    return {'prior_archive_requests':1,'new_request_cap':4,'lifetime_request_cap':5,
            'lineage_sha256':acq.digest(acq.canonical(prior)),
            'historical_files_verified':len(found),'old_payload_bytes':0,
            'prior_retained_bytes':sum(x['bytes'] for x in prior['files'].values())}

@contextlib.contextmanager
def history_read_lock(root, cfg):
    """Read-only shared advisory lock excludes legacy helper's exclusive writer."""
    verify_lineage(root, cfg)
    path = Path(root)/cfg['prior_session']['data_relative']/'.acquisition.lock'
    bg.owned(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (path.stat().st_dev,path.stat().st_ino):
            raise Stop('Historical lock changed while opening')
        try: fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error: raise Stop('Historical acquisition writer is active') from error
        verify_lineage(root, cfg)
        yield
    finally: os.close(fd)

def request_budget(root, cfg):
    evidence = bg.safe(Path(root)/cfg['output_relative'])
    paths = list(evidence.glob('request-*.json')) if evidence.exists() else []
    if len(paths) > 4: raise Stop('Isolated request budget exceeded')
    for path in paths:
        record = json.loads(read_owned(path))
        session = json.loads(read_owned(evidence/'session.json'))
        if (set(record) != {'owner','url','offset','headers'} or record['owner'] != session['owner']
                or record['url'] != session['source']['url'] or type(record['offset']) is not int
                or not 0 <= record['offset'] < session['source']['size']):
            raise Stop('Invalid isolated request receipt')
    return {'prior_archive_requests':1,'session_archive_requests':len(paths),
            'lifetime_archive_requests':1+len(paths),'remaining_archive_requests':4-len(paths)}

class IsolatedEngine(acq.Engine):
    """Same transfer algorithms; fixed four-request cap and immutable lineage."""
    def __init__(self, *args, history_root, declaration, **kwargs):
        self.history_root = Path(history_root); self.declaration = declaration
        verify_lineage(self.history_root, declaration)
        super().__init__(*args, **kwargs)
        if self.limits.max_requests != 4:
            raise Stop('Isolated engine must retain exactly four new requests')
        if (self.data,self.output) != (self.history_root/declaration['data_relative'],self.history_root/declaration['output_relative']):
            raise Stop('Unsafe isolated destination reuse')
    def space(self, extra=0):
        lineage = verify_lineage(self.history_root,self.declaration)
        request_budget(self.history_root,self.declaration)
        # Conservatively include old acquisition/workflow evidence in the 80 GB
        # retained budget; no second allowance is created for the old session.
        return super().space(extra + lineage['prior_retained_bytes'])
    def prepare(self, resume_owner=None):
        verify_lineage(self.history_root,self.declaration)
        fresh = super().prepare(resume_owner)
        if fresh:
            self.event('prior_session_lineage', lineage=verify_lineage(self.history_root,self.declaration))
        return fresh
    def run(self, resume_owner=None):
        with history_read_lock(self.history_root,self.declaration):
            result = super().run(resume_owner)
            return {**result, **request_budget(self.history_root,self.declaration)}

def load_declaration():
    for relative, expected in PINS.items():
        if sha(ROOT/relative) != expected: raise Stop('Protected source/config changed: '+relative)
    raw = read_owned(CONFIG)
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA: raise Stop('Isolated configuration hash changed')
    cfg = validate_declaration(json.loads(raw))
    verify_lineage(ROOT,cfg)
    return cfg

def make_spec(root, cfg, entry, python, source_hashes, binding):
    """Explicit composition seam; production CLI supplies only pinned paths."""
    validate_declaration(cfg); a = cfg['archive']; root = Path(root)
    return {'version':3,'mode':'production','root':str(root/cfg['background_relative']),
            'data':str(root/cfg['data_relative']),'evidence':str(root/cfg['output_relative']),
            'cwd':str(root),'python':str(python),
            'command':[str(python),'-B','-u',str(entry),'_acquire','--authorize-transfer','--quota-confirmed'],
            'identity':{'record':a['record'],'version':cfg['archive_version'],'size':a['exact_bytes'],
                        'md5':a['published_md5'],'url':a['resolved_url']},
            'binding':binding,'limits':acq.Limits(max_requests=4).__dict__,
            'source_sha256':source_hashes}

def production_spec():
    cfg = load_declaration()
    sources = {str(ROOT/p):s for p,s in PINS.items()}
    sources.update({str(CONFIG):CONFIG_SHA,str(ENTRY):sha(ENTRY)})
    binding = {'helper_sha256':PINS['tools/acquire_march_2022_v2.py'],
               'background_sha256':PINS['tools/march_background_v3.py'],
               'adapter_sha256':sources[str(ENTRY)],'config_sha256':CONFIG_SHA,
               'lineage_sha256':acq.digest(acq.canonical(cfg['prior_session'])),
               'prior_archive_requests':1,'lifetime_archive_request_cap':5}
    python = ROOT/'.venv/bin/python'
    if not python.is_file() or not os.access(python,os.X_OK): raise Stop('Project Python unavailable')
    return make_spec(ROOT,cfg,ENTRY,python,sources,binding)

_original_check_sources = bg.check_sources
_original_check_resume = bg.check_resume
_original_verify_archive = bg.verify_archive

def check_sources(spec):
    _original_check_sources(spec)
    cfg = load_declaration()
    if spec['limits'] != acq.Limits(max_requests=4).__dict__:
        raise Stop('Workflow request/storage policy changed')
    request_budget(ROOT,cfg)

def check_resume(spec, resume):
    cfg = load_declaration(); budget = request_budget(ROOT,cfg)
    if budget['remaining_archive_requests'] == 0:
        raise Stop('No remaining archive requests; preserve partials for review')
    return _original_check_resume(spec,resume)

def verify_archive(spec):
    cfg = load_declaration()
    return {**_original_verify_archive(spec),**request_budget(ROOT,cfg)}

# Private in-memory composition only. Original files and checks remain intact.
# Base check_sources/check_resume/verify_archive are called, never bypassed.
bg.WRAPPER = ENTRY
bg.production_spec = production_spec
bg.check_sources = check_sources
bg.check_resume = check_resume
bg.verify_archive = verify_archive

def acquire(argv):
    parser = argparse.ArgumentParser(description='Internal guarded isolated session worker')
    parser.add_argument('--authorize-transfer',action='store_true')
    parser.add_argument('--quota-confirmed',action='store_true')
    parser.add_argument('--resume-owner')
    args = parser.parse_args(argv)
    spec = production_spec(); cfg = load_declaration()
    if not args.authorize_transfer:
        return {'status':'dry_run','network_requests':0,'directories_created':False,
                'transfer_enabled':False,**request_budget(ROOT,cfg)}
    if not args.quota_confirmed: raise Stop('Authoritative quota confirmation required')
    with history_read_lock(ROOT,cfg):
        # Reuse the original early destination/binding guard, using NEW paths.
        declaration = {'data_relative':cfg['data_relative'],'output_relative':cfg['output_relative']}
        acq.local_transfer_precheck(declaration,spec['binding'],args.resume_owner)
        if request_budget(ROOT,cfg)['remaining_archive_requests'] == 0:
            raise Stop('No remaining archive requests; no metadata request allowed')
        remote,raw = acq.fetch_official_metadata(cfg['ca_bundle'])
        acq.validate_snapshot(remote,{'exact_bytes':cfg['archive']['exact_bytes'],
                                      'resolved_url':cfg['archive']['resolved_url']})
        engine = IsolatedEngine(remote,Path(spec['data']),Path(spec['evidence']),spec['binding'],
            acq.Limits(max_requests=4),metadata_bytes=raw,ca_bundle=cfg['ca_bundle'],
            history_root=ROOT,declaration=cfg)
        return engine.run(args.resume_owner)

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_acquire':
        print(json.dumps(acquire(argv[1:]),indent=2,sort_keys=True)); return 0
    if argv == ['preflight']:
        spec = production_spec(); cfg = load_declaration()
        bg.check_sources(spec)
        exists = [os.path.lexists(spec[k]) for k in ('data','evidence')]
        if any(exists) and not all(exists): raise Stop('Incomplete isolated roots; preserve and investigate')
        owner = bg.check_resume(spec,all(exists))
        if os.path.lexists(spec['root']): bg.check_workflow(spec)
        result = {**bg.inspect_capacity(spec['data']),**verify_lineage(ROOT,cfg),
                  **request_budget(ROOT,cfg),'session_action':'resume' if owner else 'start',
                  'fresh_validator_required':True,'download_authorized':False}
        print(json.dumps(result,indent=2,sort_keys=True)); return 0
    return bg.main(argv or ['status'])

if __name__ == '__main__':
    try: sys.exit(main())
    except (Stop,bg.Stop,acq.SafetyError,OSError,ValueError,KeyError) as error:
        print('ISOLATED V3 SAFETY STOP: '+str(error),file=sys.stderr); sys.exit(2)
