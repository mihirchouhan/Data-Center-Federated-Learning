"""B1 exclusive run ownership and immutable atomic complete-round commits."""
from pathlib import Path
import fcntl,hashlib,json,os,uuid,time,re
from src import b1_preparation_io_v1 as io
from src.b1_production_data_v1 import read_json


def digest(value): return hashlib.sha256(io.canonical(value)).hexdigest()


def source_bindings(root, names): return {n:io.binding(io.safe_path(root/n)) for n in names}


class RunStore:
    """One locked writer. Resume verifies bytes before deserializing any tensor.

    Incomplete .pending-* directories are retained, never loaded or deleted.
    No committed file, log, manifest or lock contents are overwritten.
    """
    def __init__(self, root, manifest, *, resume=False, project_root=None):
        self.root=io.safe_path(root);self.manifest=json.loads(io.canonical(manifest));manifest=self.manifest;self.fd=None
        self.project_root=Path(project_root) if project_root else Path(__file__).absolute().parents[1]
        if not resume:
            if os.path.lexists(self.root): raise FileExistsError('Run exists; never restart over an existing directory')
            if not self.root.parent.is_dir() or self.root.parent.stat().st_uid!=os.getuid(): raise ValueError('Existing owned parent required')
            self.root.mkdir(mode=0o700)
            self.fd=os.open(self.root/'writer.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        else:
            if read_json(self.root/'manifest.json')!=manifest or read_json(self.root/'owner.json')['uid']!=os.getuid(): raise ValueError('Resume identity/config/source/environment/data mismatch')
            self.fd=os.open(io.safe_path(self.root/'writer.lock'),os.O_RDWR|os.O_NOFOLLOW)
        try:
            fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not resume:
                io.exclusive(self.root/'manifest.json',io.canonical(manifest))
                io.exclusive(self.root/'owner.json',io.canonical({'schema':'b1.production.run.owner.v1','uid':os.getuid(),'created':time.time(),'id':uuid.uuid4().hex}))
                for d in ('rounds','attempts','source'):(self.root/d).mkdir(mode=0o700)
                for n,v in manifest['sources'].items():
                    raw=io.safe_path(self.project_root/n).read_bytes()
                    if {'bytes':len(raw),'sha256':io.sha(raw)}!=v: raise ValueError('Source changed during snapshot')
                    dest=self.root/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True)
                    io.exclusive(dest,raw);io.fsync_dir(dest.parent)
                io.fsync_dir(self.root)
            self.verify_sources()
            self.records=self.scan()
            if resume and not self.records: raise ValueError('No committed checkpoint; preserve uninitialized attempt and use a new version')
            self.attempt=self.root/'attempts'/uuid.uuid4().hex;self.attempt.mkdir(mode=0o700)
            io.exclusive(self.attempt/'start.json',io.canonical({'time':time.time(),'pid':os.getpid(),'resume':resume,'last_committed_round':len(self.records)-1,'retained_pending':[p.name for p in (self.root/'rounds').iterdir() if p.name.startswith('.pending-')]}))
            self.event_count=0
        except BaseException:
            self.close();raise

    def verify_sources(self):
        if read_json(self.root/'manifest.json')!=self.manifest: raise ValueError('Run manifest changed')
        for n,v in self.manifest['sources'].items():
            if io.binding(self.project_root/n)!=v or io.binding(self.root/'source'/n)!=v: raise ValueError('Execution source/snapshot changed: '+n)

    def scan(self):
        rounds=self.root/'rounds';dirs=[]
        for p in rounds.iterdir():
            io.safe_path(p)
            if re.fullmatch(r'\.pending-\d{4}-[0-9a-f]{32}',p.name) and p.is_dir(): continue
            if not p.is_dir() or len(p.name)!=4 or not p.name.isdecimal(): raise ValueError('Unexpected round artifact')
            dirs.append(p)
        dirs.sort();records=[];previous=None
        for index,p in enumerate(dirs):
            if p.name!=f'{index:04d}': raise ValueError('Missing/noncontiguous committed round')
            if {x.name for x in p.iterdir()}!={'checkpoint.pt','receipt.json'}: raise ValueError('Incomplete committed round')
            rec=read_json(p/'receipt.json')
            if rec['schema']!='b1.production.round.v1' or rec['round']!=index or rec['manifest_sha256']!=digest(self.manifest) or rec['previous_receipt_sha256']!=previous: raise ValueError('Round receipt chain mismatch')
            if io.binding(p/'checkpoint.pt')!=rec['checkpoint']: raise ValueError('Checkpoint content hash mismatch')
            records.append(rec);previous=io.binding(p/'receipt.json')['sha256']
        return records

    def event(self, kind, value):
        self.event_count+=1
        io.exclusive(self.attempt/f'{self.event_count:06d}_{kind}.json',io.canonical({'kind':kind,'time':time.time(),'details':value}))

    def commit(self, round_id, payload, record):
        import torch
        self.verify_sources()
        capacity=os.statvfs(self.root)
        if capacity.f_bavail*capacity.f_frsize < 10*1024**3:raise OSError('Filesystem reserve below 10 GiB; preserve run and stop')
        if round_id!=len(self.records): raise ValueError('Must commit the next round, never skip/rewrite')
        stage=self.root/'rounds'/f'.pending-{round_id:04d}-{uuid.uuid4().hex}';stage.mkdir(mode=0o700)
        fd=os.open(stage/'checkpoint.pt',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        rec={**record,'schema':'b1.production.round.v1','round':round_id,'manifest_sha256':digest(self.manifest),
             'previous_receipt_sha256':io.binding(self.root/'rounds'/f'{round_id-1:04d}'/'receipt.json')['sha256'] if round_id else None,
             'checkpoint':io.binding(stage/'checkpoint.pt')}
        io.exclusive(stage/'receipt.json',io.canonical(rec));io.fsync_dir(stage)
        io.rename_noreplace(stage,self.root/'rounds'/f'{round_id:04d}')
        self.records.append(rec)
        self.event('round_committed',{'round':round_id,'receipt':io.binding(self.root/'rounds'/f'{round_id:04d}'/'receipt.json')})
        return rec

    def load(self, round_id=None):
        import torch
        self.records=self.scan()
        if not self.records: raise ValueError('No committed checkpoint')
        i=len(self.records)-1 if round_id is None else round_id
        if type(i) is not int or not 0<=i<len(self.records): raise ValueError('Invalid checkpoint index')
        path=self.root/'rounds'/f'{i:04d}'/'checkpoint.pt'
        # Full content hash checked again immediately before weights-only read.
        if io.binding(path)!=self.records[i]['checkpoint']: raise ValueError('Checkpoint changed before load')
        with io.safe_open(path) as f:payload=torch.load(f,map_location='cpu',weights_only=True)
        if io.binding(path)!=self.records[i]['checkpoint']: raise ValueError('Checkpoint changed during load')
        return payload

    def publish_selected(self, payload, selection):
        import torch
        dest=self.root/'selected'
        if dest.exists():
            receipt=read_json(dest/'receipt.json')
            if receipt['selection']!=selection or io.binding(dest/'model.pt')!=receipt['model']: raise ValueError('Existing selected export differs')
            import torch
            with io.safe_open(dest/'model.pt') as f:saved=torch.load(f,map_location='cpu',weights_only=True)
            if saved['schema']!=payload['schema'] or saved['manifest_sha256']!=payload['manifest_sha256'] or set(saved['state'])!=set(payload['state']) or any(not torch.equal(saved['state'][k],v) for k,v in payload['state'].items()):raise ValueError('Existing selected model differs from validated checkpoint')
            return receipt
        stage=self.root/('.selected-'+uuid.uuid4().hex);stage.mkdir(mode=0o700)
        fd=os.open(stage/'model.pt',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        receipt={'schema':'b1.production.selected.v1','model':io.binding(stage/'model.pt'),'selection':selection,'manifest_sha256':digest(self.manifest)}
        io.exclusive(stage/'receipt.json',io.canonical(receipt));io.fsync_dir(stage);io.rename_noreplace(stage,dest)
        return receipt

    def finish(self, receipt):
        completed={'status':'training_complete','round':len(self.records)-1,'selected_receipt':io.binding(self.root/'selected'/'receipt.json'),'last_round_receipt':io.binding(self.root/'rounds'/f'{len(self.records)-1:04d}'/'receipt.json'),'selection':receipt['selection'],'manifest_sha256':digest(self.manifest)}
        p=self.root/'completed.json'
        if p.exists():
            if read_json(p)!=completed: raise ValueError('Completion differs')
        else:io.exclusive(p,io.canonical(completed));io.fsync_dir(self.root)
        return completed

    def close(self):
        if self.fd is not None: os.close(self.fd);self.fd=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
