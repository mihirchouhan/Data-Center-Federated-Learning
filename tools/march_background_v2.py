#!/usr/bin/env python3
"""Version 2: fresh-destination capacity inspection; no transfer without two gates.

The protected acquisition helper owns every archive request, checkpoint, validator
and final publication. This module adds lifecycle evidence, never repairs data.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path('/media/csedept/cse2014/Project/m100_training_template')
WRAPPER = ROOT / 'tools/march_background_v2.py'
RUNTIME = ROOT / 'output/march_2022_background/v2'
HELPER = ROOT / 'tools/acquire_march_2022_v1.py'
CONFIG = ROOT / 'configs/acquisition/march_2022_v1.json'
HELPER_SHA = '25a3abf0beb5148726eb04a26b0874880355f70ea00efbedaa25e03817253925'
CONFIG_SHA = '4a7911cdf1cbf35b109186000b9ad697de1ba06faf66295bfb5d1338542f5725'
ATTEMPT = re.compile(r'attempt-[0-9]{4}-[a-f0-9]{32}')

class Stop(RuntimeError):
    """Fail closed and retain all existing evidence."""

def utc():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def safe(path):
    path = Path(path).absolute()
    if '..' in path.parts:
        raise Stop('Parent traversal prohibited')
    for p in (path, *path.parents):
        if p.is_symlink():
            raise Stop('Symlink path: ' + str(p))
    return path

def owned(path, links=1):
    p = safe(path); s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != links:
        raise Stop('Unowned/nonregular/hardlinked file: ' + str(p))
    return s

def read(path):
    p = safe(path); owned(p)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if os.fstat(stream.fileno()).st_ino != p.stat().st_ino:
            raise Stop('File replaced while opening')
        return stream.read()

def load(path):
    return json.loads(read(path))

def hashes(path):
    p = safe(path); s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink not in (1, 2):
        raise Stop('Invalid archive file')
    sha, md = hashlib.sha256(), hashlib.md5()
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        initial = os.fstat(stream.fileno())
        count = 0
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            count += len(b); sha.update(b); md.update(b)
        final = os.fstat(stream.fileno())
    if (s.st_ino, s.st_size, s.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns) or (initial.st_ino, initial.st_size, initial.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns):
        raise Stop('File changed while hashing')
    return {'bytes': count, 'sha256': sha.hexdigest(), 'md5': md.hexdigest()}

def sha(path):
    return hashlib.sha256(read(path)).hexdigest()

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)

def write_new(path, value):
    """Atomic, durable, no-replace JSON; interrupted temporary records are retained."""
    path = safe(path)
    temp = path.parent / ('.record-' + uuid.uuid4().hex)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temp, path, follow_symlinks=False)
    temp.unlink()  # only this newly created temporary JSON; never acquisition data
    sync_dir(path.parent)

def process_identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        if proc.stat().st_uid != os.getuid(): return None
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'): return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None

def alive(identity):
    return bool(identity) and process_identity(identity['pid']) == identity

def check_sources(spec):
    for path, expected in spec['source_sha256'].items():
        if sha(path) != expected:
            raise Stop('Execution source/configuration changed: ' + path)

def production_spec():
    if sha(HELPER) != HELPER_SHA or sha(CONFIG) != CONFIG_SHA:
        raise Stop('Protected helper/config hash mismatch')
    config = load(CONFIG)
    if (config['exact_bytes'], config['published_md5'], config['max_requests'],
        config['max_storage_bytes'], config['reserve_free_bytes']) != (
        31495884800, '4c74ad95b6d837d409e24216389bb25e', 5, 80000000000, 10000000000):
        raise Stop('Unexpected fixed acquisition settings')
    python = ROOT / '.venv/bin/python'
    if not python.is_file() or not os.access(python, os.X_OK):
        raise Stop('Project .venv Python unavailable')
    return {'version': 2, 'mode': 'production', 'root': str(RUNTIME),
        'data': str(ROOT / config['data_relative']), 'evidence': str(ROOT / config['output_relative']),
        'cwd': str(ROOT), 'python': str(python),
        'command': [str(python), '-B', '-u', str(HELPER), '--config', str(CONFIG),
                    '--authorize-transfer', '--quota-confirmed'],
        'identity': {'record': '7590061', 'version': '1.0.0', 'size': config['exact_bytes'],
                     'md5': config['published_md5'], 'url': config['resolved_url']},
        'binding': {'helper_sha256': HELPER_SHA, 'config_sha256': CONFIG_SHA},
        'limits': {'max_requests': 5, 'max_storage_bytes': 80000000000,
                   'reserve_free_bytes': 10000000000, 'evidence_reserve_bytes': 1048576,
                   'timeout': 60, 'chunk_bytes': 8388608},
        'source_sha256': {str(HELPER): HELPER_SHA, str(CONFIG): CONFIG_SHA, str(WRAPPER): sha(WRAPPER)}}

def session(spec):
    a, b = (load(Path(spec[k]) / 'session.json') for k in ('data', 'evidence'))
    if a != b or a['uid'] != os.getuid() or a['binding'] != spec['binding']:
        raise Stop('Acquisition session ownership/binding mismatch')
    if a['data'] != spec['data'] or a['output'] != spec['evidence'] or a['limits'] != spec['limits']:
        raise Stop('Acquisition paths/limits changed')
    if not re.fullmatch('[a-f0-9]{32}', a['owner']): raise Stop('Invalid acquisition owner')
    if any(a['source'].get(k) != v for k, v in spec['identity'].items()):
        raise Stop('Acquisition source identity changed')
    return a

def acquire_guard(root):
    p = safe(root) / '.supervisor.lock'
    try: fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError: fd = os.open(p, os.O_RDWR | os.O_NOFOLLOW)
    try:
        owned(p)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError as error:
        os.close(fd); raise Stop('Background/acquisition process already holds workflow guard') from error
    except BaseException:
        os.close(fd); raise

def guard_held(root):
    p = Path(root) / '.supervisor.lock'
    if not p.exists(): raise Stop('Missing workflow guard; do not recreate')
    owned(p)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return True
        return False
    finally: os.close(fd)

def attempts(root):
    found = []
    for p in safe(root).iterdir():
        if p.name in ('workflow.json', '.supervisor.lock'): continue
        if p.is_symlink() or not p.is_dir() or not ATTEMPT.fullmatch(p.name):
            raise Stop('Unexpected workflow artifact: ' + str(p))
        if p.stat().st_uid != os.getuid(): raise Stop('Unowned attempt directory')
        found.append(p)
    found.sort()
    return found

def check_workflow(spec):
    root = safe(spec['root'])
    if not root.is_dir() or root.stat().st_uid != os.getuid(): raise Stop('Unowned workflow root')
    if load(root / 'workflow.json') != spec: raise Stop('Immutable workflow binding changed')
    check_sources(spec)
    return root

def check_resume(spec, resume):
    final = safe(spec['data']) / '22-03.tar'
    if os.path.lexists(final): raise Stop('Archive destination already exists; verify, never relaunch')
    if not resume:
        if any(os.path.lexists(spec[k]) for k in ('data', 'evidence')):
            raise Stop('Existing acquisition roots; inspect before using resume')
        return None
    s = session(spec)
    lock = Path(spec['data']) / '.acquisition.lock'
    owned(lock)
    fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise Stop('Existing acquisition lock is held') from error
    finally: os.close(fd)
    # No mutation, truncation, or invented checkpoint. The helper rehashes the
    # partial and enforces the complete checkpoint chain before any archive GET.
    checkpoint_summary(spec)
    return s['owner']

def inspect_capacity(destination):
    """Read-only capacity check on the nearest existing owned directory.

    Inspect the destination itself when it exists (including a mount point),
    otherwise walk upward without creating any path. A directory descriptor
    pins the inspected object; symlinks, ownership, read-only filesystems and
    the original 90 GB minimum remain fail-closed. This is not quota proof.
    """
    destination = safe(destination)
    ancestor = destination
    while True:
        try:
            before = ancestor.lstat()
            break
        except FileNotFoundError:
            if ancestor.parent == ancestor:
                raise Stop('No existing acquisition ancestor')
            ancestor = ancestor.parent
        except NotADirectoryError as error:
            raise Stop('Acquisition ancestor is not a directory') from error
    if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid():
        raise Stop('Acquisition ancestor is not an owned directory')
    safe(ancestor)
    fd = os.open(ancestor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        current = os.fstat(fd)
        if ((current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
                or not stat.S_ISDIR(current.st_mode) or current.st_uid != os.getuid()):
            raise Stop('Acquisition ancestor changed while opening')
        safe(destination)
        v = os.fstatvfs(fd)
    finally:
        os.close(fd)
    available = v.f_bavail * v.f_frsize
    if v.f_flag & os.ST_RDONLY or available < 90000000000:
        raise Stop('Need writable filesystem, 80 GB retained budget plus 10 GB free reserve')
    return {'status': 'capacity_preflight_passed', 'destination': str(destination),
            'inspected_ancestor': str(ancestor), 'filesystem_available_bytes': available,
            'required_bytes': 90000000000, 'directories_created': False,
            'network_requests': 0, 'transfer_enabled': False,
            'quota_confirmation': 'NOT established by filesystem capacity inspection'}

def launch(spec, *, resume=False, authorized=False, quota_confirmed=False, supervisor_command=None):
    """Internal injectable seam for synthetic tests; production CLI has fixed paths/argv."""
    check_sources(spec)
    if not authorized:
        return {'status': 'dry_run', 'transfer_enabled': False, 'directories_created': False,
                'network_requests': 0, 'root': spec['root'], 'resume': resume}
    if not quota_confirmed: raise Stop('Authoritative quota confirmation is required')
    owner = check_resume(spec, resume)
    if spec['mode'] == 'production':
        inspect_capacity(spec['data'])  # no workflow writes on capacity failure
    root = safe(spec['root'])
    if not root.exists():
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        write_new(root / 'workflow.json', spec)
    check_workflow(spec)
    fd = acquire_guard(root)
    try:
        previous = attempts(root)
        if previous and not resume: raise Stop('Workflow already attempted; no automatic new acquisition')
        # Recheck after guard acquisition to cover concurrent launch attempts.
        if check_resume(spec, resume) != owner: raise Stop('Resume owner changed')
        if spec['mode'] == 'production':
            inspect_capacity(spec['data'])  # recheck under the workflow guard
        number = len(previous) + 1
        if number > 9999: raise Stop('Attempt number limit')
        attempt = root / f'attempt-{number:04d}-{uuid.uuid4().hex}'
        attempt.mkdir(mode=0o700, exist_ok=False)
        write_new(attempt / 'launch.json', {'attempt': attempt.name, 'requested_at': utc(),
            'resume_owner': owner, 'authorization_asserted': True, 'quota_confirmed_asserted': True,
            'workflow_sha256': sha(root / 'workflow.json'), 'wrapper_sha256': sha(WRAPPER) if spec['mode']=='production' else None})
        if supervisor_command is None:
            if spec['mode'] != 'production': raise Stop('Synthetic supervisor seam required')
            argv = [spec['python'], '-B', '-u', str(WRAPPER), '_supervise',
                    '--attempt', attempt.name, '--guard-fd', str(fd)]
        else:
            argv = supervisor_command(attempt, fd)
        logfd = os.open(attempt / 'supervisor.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            process = subprocess.Popen(argv, cwd=spec['cwd'], stdin=subprocess.DEVNULL,
                stdout=logfd, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(fd,), env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':''})
        finally: os.close(logfd)
        # The inherited *same open-file description* keeps the guard locked.
        write_new(attempt / 'spawn.json', {'pid': process.pid, 'identity': process_identity(process.pid), 'at': utc()})
        return {'status': 'launched_not_yet_verified', 'attempt': str(attempt),
                'supervisor_pid': process.pid, 'log': str(attempt / 'process.log')}
    finally: os.close(fd)

def checkpoint_summary(spec):
    data, evidence = Path(spec['data']), Path(spec['evidence'])
    if not data.exists() and not evidence.exists(): return {'status': 'absent'}
    s = session(spec); states = {}
    paths = list(evidence.glob('state-*.json'))
    for p in paths:
        value = load(p)
        if (value['owner'] != s['owner'] or value['source_sha256'] != hashlib.sha256(canonical(s['source'])).hexdigest()):
            raise Stop('Checkpoint owner/source mismatch')
        h = hashlib.sha256(canonical(value)).hexdigest()
        if h in states: raise Stop('Duplicate checkpoint')
        states[h] = value
    if not states: raise Stop('Missing durable checkpoint')
    parents = {v['previous'] for v in states.values() if v['previous'] is not None}
    tips = set(states) - parents
    if not parents.issubset(states) or len(tips) != 1: raise Stop('Invalid checkpoint chain')
    tip = next(iter(tips)); cursor = tip; seen = set()
    while cursor is not None:
        if cursor in seen: raise Stop('Checkpoint cycle')
        seen.add(cursor); cursor = states[cursor]['previous']
    if len(seen) != len(states): raise Stop('Disconnected checkpoints')
    p = data / 'payload.part'; count = owned(p, links=2 if (data/'22-03.tar').exists() else 1).st_size
    recorded = states[tip]['prefix']['bytes']
    if type(recorded) is not int or not 0 <= recorded <= spec['identity']['size']:
        raise Stop('Invalid checkpoint byte count')
    return {'status': 'metadata_chain_checked', 'states': len(states), 'tip_sha256': tip,
            'checkpoint_bytes': recorded, 'partial_bytes': count,
            'prefix_length_matches': recorded == count,
            'strong_validator_recorded': bool(re.fullmatch(r'"[^"\r\n]+"', states[tip]['validator'] or '')),
            'partial_content_rehashed': False, 'helper_resume_validation_required': True}

def file_stat(path):
    s = safe(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino,
            'device': s.st_dev, 'nlink': s.st_nlink}

def verify_archive(spec):
    """Read-only full-byte verification, never based solely on a worker exit code."""
    s = session(spec); data = Path(spec['data']); output = Path(spec['evidence'])
    final, partial = data / '22-03.tar', data / 'payload.part'
    owned(final, links=2); owned(partial, links=2)
    if not os.path.samefile(final, partial): raise Stop('Final is not helper-published hardlink')
    receipt = load(output / 'completed.json')
    if receipt['owner'] != s['owner'] or receipt['binding'] != spec['binding'] or receipt['source'] != s['source']:
        raise Stop('Acquisition completion provenance mismatch')
    if type(receipt['requests']) is not int or not 0 <= receipt['requests'] <= spec['limits']['max_requests']:
        raise Stop('Invalid archive request count')
    if receipt['requests'] != len(list(output.glob('request-*.json'))):
        raise Stop('Archive request evidence count mismatch')
    if sha(output/'official_metadata.json') != s['source']['metadata_sha256']:
        raise Stop('Official metadata evidence hash mismatch')
    h = hashes(final)
    if h['bytes'] != spec['identity']['size'] or h['md5'] != spec['identity']['md5'] or h != receipt['hashes']:
        raise Stop('Archive exact-size/checksum/receipt mismatch')
    return {'status': 'verified_published', 'verified_at': utc(), 'hashes': h,
            'receipt_sha256': sha(output / 'completed.json'), 'final_stat': file_stat(final),
            'archive': str(final), 'owner': s['owner'], 'full_archive_rehashed_now': True}

def supervise(spec, attempt_name, guard_fd):
    root = check_workflow(spec)
    if not ATTEMPT.fullmatch(attempt_name): raise Stop('Invalid attempt name')
    attempt = root / attempt_name
    if (os.fstat(guard_fd).st_dev, os.fstat(guard_fd).st_ino) != (
        (root/'.supervisor.lock').stat().st_dev, (root/'.supervisor.lock').stat().st_ino):
        raise Stop('Inherited guard mismatch')
    # FD is the parent's locked open-file description, retained by worker too.
    fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = load(attempt/'launch.json')
    if request['workflow_sha256'] != sha(root/'workflow.json'): raise Stop('Launch binding changed')
    if request['authorization_asserted'] is not True or request['quota_confirmed_asserted'] is not True:
        raise Stop('Missing launch authorization/quota assertion')
    began = time.monotonic(); worker = None; signals = []; status = 'failed'; error = None; returncode = None
    write_new(attempt/'supervisor.json', {'identity': process_identity(os.getpid()), 'started_at': utc(),
                                        'detached_session': os.getsid(0), 'controlling_stdin': False})
    def stopping(signum, frame):
        signals.append({'signal': signum, 'at': utc()})
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGINT)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, stopping); signal.signal(signal.SIGTERM, stopping)
    try:
        check_sources(spec)
        resume_owner = request['resume_owner']
        if check_resume(spec, bool(resume_owner)) != resume_owner: raise Stop('Resume session changed')
        argv = list(spec['command'])
        if resume_owner: argv.extend(['--resume-owner', resume_owner])
        logfd = os.open(attempt/'process.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            worker = subprocess.Popen(argv, cwd=spec['cwd'], stdin=subprocess.DEVNULL,
                stdout=logfd, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(guard_fd,),
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':''})
        finally: os.close(logfd)
        write_new(attempt/'worker.json', {'identity': process_identity(worker.pid), 'pid': worker.pid,
                                         'started_at': utc(), 'argv': argv})
        if signals: worker.send_signal(signal.SIGINT)
        returncode = worker.wait()
        if returncode != 0: raise Stop('Helper exited with status ' + str(returncode))
        checked = verify_archive(spec)
        write_new(attempt/'publication.json', checked)
        status = 'verified_published'
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        print(error, file=sys.stderr, flush=True)
        # A lifecycle/evidence failure cannot release the singleton while a
        # worker continues. Its inherited guard also protects SIGKILL cases.
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGINT)
            returncode = worker.wait()
    finally:
        write_new(attempt/'exit.json', {'status': status, 'finished_at': utc(),
            'elapsed_seconds': time.monotonic()-began, 'worker_returncode': returncode,
            'error': error, 'received_signals': signals,
            'supervisor_exit_status': 0 if status == 'verified_published' else 2,
            'archive_verified_and_published': status == 'verified_published'})
        os.close(guard_fd)
    return 0 if status == 'verified_published' else 2

def tail(path, lines=12):
    if not path.exists(): return []
    owned(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size-65536))
        return stream.read().decode('utf-8', errors='replace').splitlines()[-lines:]

def status(spec):
    root = safe(spec['root'])
    if not root.exists():
        return {'status': 'not_started', 'running': False, 'archive_verified_and_published': False,
                'acquisition_roots_exist': {k: os.path.lexists(spec[k]) for k in ('data','evidence')}}
    check_workflow(spec); found = attempts(root); held = guard_held(root)
    result = {'status': 'workflow_created', 'running': False, 'guard_held': held,
              'archive_verified_and_published': False, 'full_archive_rehashed_now': False}
    if not found: return result
    attempt = found[-1]; result['attempt'] = str(attempt)
    result['log'] = str(attempt/'process.log'); result['last_log_lines'] = tail(attempt/'process.log')
    result['supervisor_log_lines'] = tail(attempt/'supervisor.log')
    for role in ('supervisor','worker'):
        p = attempt/(role+'.json')
        value = load(p) if p.exists() else {}
        result[role] = {'identity': value.get('identity'), 'running': alive(value.get('identity')),
                        'started_at': value.get('started_at')}
    result['running'] = result['supervisor']['running'] or result['worker']['running']
    result['status'] = 'running' if result['running'] else ('startup_or_orphan_guard_held' if held else 'interrupted_or_unknown')
    try: result['checkpoint'] = checkpoint_summary(spec)
    except (Stop, OSError, ValueError, KeyError) as exc:
        result['checkpoint'] = {'status': 'unavailable_or_inconsistent', 'error': str(exc),
            'note': 'May be startup/write in progress; stopped sessions require review before resume.'}
    p = attempt/'exit.json'
    if p.exists():
        value = load(p); result['last_exit'] = value
        if not result['running'] and not held: result['status'] = value['status']
        if value['status'] == 'verified_published':
            publication = load(attempt/'publication.json')
            # Status is fast: detect metadata/receipt changes without a 31 GB rehash.
            final = Path(spec['data'])/'22-03.tar'
            valid = publication['final_stat'] == file_stat(final) and publication['receipt_sha256'] == sha(Path(spec['evidence'])/'completed.json')
            result['archive_verified_and_published'] = valid
            result['verification_basis'] = 'Prior full-byte verification and unchanged stat/receipt; use verify for fresh full hashing.'
            result['verified_at'] = publication['verified_at']
            if not valid: result['status'] = 'publication_changed_requires_review'
    return result

def interrupt_identity(identity):
    """Pin the Linux process before signaling; fail closed if pidfds unavailable.

    Some conda Python builds omit the stdlib pidfd functions while glibc still
    exposes the supported kernel interface. Never fall back to an unpinned kill.
    """
    if not alive(identity): raise Stop('Process changed before stop')
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        fd = os.pidfd_open(identity['pid'])
        send = lambda: signal.pidfd_send_signal(fd, signal.SIGINT)
    else:
        lib = ctypes.CDLL(None, use_errno=True)
        if not hasattr(lib, 'pidfd_open') or not hasattr(lib, 'pidfd_send_signal'):
            raise Stop('Safe pidfd signaling unavailable; no signal sent')
        lib.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        lib.pidfd_open.restype = ctypes.c_int
        lib.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        lib.pidfd_send_signal.restype = ctypes.c_int
        fd = lib.pidfd_open(identity['pid'], 0)
        if fd < 0: raise OSError(ctypes.get_errno(), 'pidfd_open failed')
        def send():
            if lib.pidfd_send_signal(fd, signal.SIGINT, None, 0) < 0:
                raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')
    try:
        if not alive(identity): raise Stop('Process changed before stop')
        send()
    finally: os.close(fd)

def stop(spec):
    """Request SIGINT; never escalate to SIGKILL or delete an acquisition file."""
    root = check_workflow(spec); found = attempts(root)
    if not found: raise Stop('No launch to stop')
    attempt = found[-1]
    for role in ('supervisor','worker'):
        p = attempt/(role+'.json')
        identity = load(p).get('identity') if p.exists() else None
        if alive(identity):
            interrupt_identity(identity)
            return {'status': 'interrupt_requested', 'role': role, 'identity': identity,
                    'partial_or_checkpoint_deleted': False}
    raise Stop('No verified live process; no signal sent')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start','resume','preflight','status','stop','verify','_supervise'])
    parser.add_argument('--authorize-download-only', action='store_true')
    parser.add_argument('--quota-confirmed', action='store_true')
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    parser.add_argument('--guard-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.action not in ('start','resume') and (args.authorize_download_only or args.quota_confirmed):
        raise Stop('Authorization flags apply only to explicit start/resume')
    if args.action != '_supervise' and (args.attempt is not None or args.guard_fd is not None):
        raise Stop('Internal supervisor arguments rejected')
    spec = production_spec()
    if args.action in ('start','resume'):
        result = launch(spec, resume=args.action=='resume', authorized=args.authorize_download_only,
                        quota_confirmed=args.quota_confirmed)
    elif args.action == 'preflight': result = inspect_capacity(spec['data'])
    elif args.action == 'status': result = status(spec)
    elif args.action == 'verify':
        if safe(spec['root']).exists() and guard_held(check_workflow(spec)):
            raise Stop('Acquisition still running; do not hash a changing file')
        result = verify_archive(spec)
    elif args.action == 'stop': result = stop(spec)
    else:
        if args.attempt is None or args.guard_fd is None: raise Stop('Missing inherited supervisor context')
        return supervise(spec, args.attempt, args.guard_fd)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0

if __name__ == '__main__':
    try: sys.exit(main())
    except (Stop, OSError, ValueError, KeyError) as error:
        print('BACKGROUND SAFETY STOP: ' + str(error), file=sys.stderr); sys.exit(2)
