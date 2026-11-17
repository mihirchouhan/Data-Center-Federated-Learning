"""B1 preparation storage/resource primitives. No model or scaler operations."""
from dataclasses import dataclass,asdict
from pathlib import Path
import ctypes,errno,fcntl,hashlib,json,os,resource,stat,time,uuid

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def binding(path):
    h=hashlib.sha256();n=0
    with safe_open(path) as f:
        while block:=f.read(8*1024**2):h.update(block);n+=len(block)
    return {'bytes':n,'sha256':h.hexdigest()}
def safe_path(path):
    path=Path(path).absolute()
    if '..' in path.parts:raise ValueError('Parent traversal forbidden')
    for p in (path,*path.parents):
        if p.is_symlink():raise ValueError('Symlink path forbidden: '+str(p))
    return path
def safe_open(path):
    path=safe_path(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):os.close(fd);raise ValueError('Regular file required')
    return os.fdopen(fd,'rb')
def fsync_dir(path):
    fd=os.open(safe_path(path),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
def exclusive(path,raw):
    path=safe_path(path);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def rename_noreplace(source,destination):
    source=safe_path(source);destination=safe_path(destination)
    libc=ctypes.CDLL(None,use_errno=True)
    if not hasattr(libc,'renameat2'):raise OSError('No atomic no-replace rename; fail closed')
    fn=libc.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(source),-100,os.fsencode(destination),1):
        code=ctypes.get_errno();raise OSError(code,os.strerror(code))
    fsync_dir(destination.parent)
def nearest_existing(path):
    p=safe_path(path)
    while not p.exists():p=p.parent
    if not p.is_dir():raise ValueError('Capacity ancestor must be a directory')
    return p

@dataclass(frozen=True)
class Limits:
    arrow_rows:int=65536
    max_file_rows:int=200_000_000
    max_raw_rows:int=2_000_000
    max_client_windows:int=512
    chunk_windows:int=128
    max_disk_bytes:int=64*1024**3
    reserve_bytes:int=10*1024**3
    max_RSS_bytes:int=8*1024**3
    max_address_space_bytes:int=16*1024**3
    max_wall_seconds:int=14400
    def __post_init__(self):
        if any(type(x) is not int or x<=0 for x in asdict(self).values()):raise ValueError('Positive explicit capacity limits required')
        if self.chunk_windows>min(self.max_client_windows,512) or self.max_client_windows>4096 or self.max_address_space_bytes<self.max_RSS_bytes:raise ValueError('Inconsistent or unsupported bounded capacities')

class Budget:
    """Cooperative disk/reserve checks; supervisor enforces wall/RSS separately."""
    def __init__(self,root,limits,quota):
        self.root=Path(root);self.limits=limits;self.start=time.monotonic();self.used=0
        if not isinstance(quota,dict) or quota.get('uid')!=os.getuid() or quota.get('confirmed') is not True:raise ValueError('Explicit user/project quota evidence required')
        if type(quota.get('headroom_bytes')) is not int or quota['headroom_bytes']<limits.max_disk_bytes+limits.reserve_bytes:raise ValueError('Insufficient quota for cap and reserve')
        if abs(time.time()-quota.get('checked_at_unix',0))>300:raise ValueError('Quota evidence stale')
        if quota.get('filesystem_device')!=nearest_existing(self.root).stat().st_dev or not isinstance(quota.get('evidence_source'),str) or not quota['evidence_source']:raise ValueError('Quota filesystem and evidence source required')
        v=os.statvfs(nearest_existing(self.root))
        if v.f_bavail*v.f_frsize<limits.max_disk_bytes+limits.reserve_bytes:raise OSError('Filesystem cannot accommodate full cap and reserve')
        self.quota=quota;self.check()
    def check(self,extra=0):
        if type(extra) is not int or extra<0:raise ValueError('Exact nonnegative byte charge')
        v=os.statvfs(nearest_existing(self.root));free=v.f_bavail*v.f_frsize
        if self.used+extra>self.limits.max_disk_bytes or free-extra<self.limits.reserve_bytes or self.quota['headroom_bytes']-self.used-extra<self.limits.reserve_bytes:raise OSError('Disk/quota/reserve stop; preserve incomplete attempt')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.limits.max_RSS_bytes:raise MemoryError('Peak RSS budget exceeded')
        if time.monotonic()-self.start>self.limits.max_wall_seconds:raise TimeoutError('Wall budget exceeded')
    def charge(self,n):self.check(n);self.used+=n

class Attempt:
    """One exclusive work root. Never resumes, removes or overwrites old attempts."""
    def __init__(self,root,contract_digest,budget,attempt_id=None):
        self.root=safe_path(root)
        if os.path.lexists(self.root):raise FileExistsError('Attempt root occupied; inspect it and use a new destination')
        # The caller supplies an existing owned parent; no recursive path creation.
        if not self.root.parent.is_dir() or self.root.parent.stat().st_uid!=os.getuid():raise ValueError('Existing owned output parent required')
        budget.check(16384);self.root.mkdir(mode=0o700);self.id=attempt_id or uuid.uuid4().hex;self.budget=budget
        self.lockfd=os.open(self.root/'lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        fcntl.flock(self.lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.owner={'schema':'b1.preparation.attempt.v1','attempt':self.id,'uid':os.getuid(),'pid':os.getpid(),'contract_sha256':contract_digest,'created_at_unix':time.time()}
        self.write('owner.json',canonical(self.owner));self.spool=self.root/'spool';self.spool.mkdir(mode=0o700)
        self.dataset=self.root/'dataset.partial';self.dataset.mkdir(mode=0o700);(self.root/'events').mkdir(mode=0o700);self.spools={};self.event_count=0
    def event(self,stage,details):
        self.event_count+=1;self.write(f'events/{self.event_count:06d}.json',canonical({'stage':stage,'elapsed_seconds':time.monotonic()-self.budget.start,'details':details}))
    def write(self,relative,raw):
        p=self.root/relative;self.budget.charge(len(raw));exclusive(p,raw)
    def append(self,relative,raw):
        p=safe_path(self.root/relative)
        if p.parent!=self.spool:raise ValueError('Spool-only append')
        self.budget.charge(len(raw))
        if relative not in self.spools:
            fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            self.spools[relative]=os.fstat(fd).st_ino
        else:
            fd=os.open(p,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
            if os.fstat(fd).st_ino!=self.spools[relative]:os.close(fd);raise ValueError('Spool ownership changed')
        with os.fdopen(fd,'ab') as f:f.write(raw)
    def seal_spools(self):
        for name in self.spools:
            fd=os.open(self.root/name,os.O_RDONLY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
        fsync_dir(self.spool)
    def close(self):
        if self.lockfd is not None:os.close(self.lockfd);self.lockfd=None

def inventory(path):
    root=safe_path(path);out={}
    for base,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:safe_path(Path(base)/name)
        for name in files:
            p=Path(base)/name;out[str(p.relative_to(root))]=binding(p)
    return out

def status(root):
    root=safe_path(root)
    if not root.exists():return {'status':'absent'}
    if not (root/'owner.json').is_file() or not (root/'lock').is_file():return {'status':'incomplete_unclaimed','manual_review_required':True,'resume_supported':False}
    with safe_open(root/'owner.json') as f:owner=json.load(f)
    if owner.get('uid')!=os.getuid():raise ValueError('Foreign attempt owner')
    lock=os.open(root/'lock',os.O_RDONLY|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);active=False
        except BlockingIOError:active=True
    finally:os.close(lock)
    if active:return {'status':'running','attempt':owner['attempt']}
    return {'status':'published_requires_verification' if (root/'dataset').is_dir() else 'incomplete_preserved','attempt':owner['attempt'],'resume_supported':False}
