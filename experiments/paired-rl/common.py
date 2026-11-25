"""Run-local utilities; this module has no CUDA or torch imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/lab_admin/workspace/RLVR')
RUN = ROOT / 'runs/idea-20261121-01-paired-rl'
CODE = ROOT / 'runs/idea-20261121-01-branch-integration/worktree'
PREP = ROOT / 'runs/idea-20261121-01-comparison-prep'
ENV = Path('/work/users/lab_admin/envs/RLVR-rl1080')
BULK = Path('/data/users/lab_admin/migration/RLVR/runs/idea-20261121-01-paired-rl')
UUID = 'GPU-b511889f-19f3-d5de-fdad-d3ef1f776278'
HEAD = '2f2211953c889583642a4f23dcfd3d8d50418042'
TREE = '73844c0ed836bdd20528e287c43b188b39cfb13d'
CAP = 360000
HISTORICAL = 13257
MODEL_IDS = ['pt-tree-mix-cf0.0035-s0', 'pt-tree-mix-cf0.006-s0']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def content(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha(path)}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.writing')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def event(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())

def git(*args, root=CODE):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True)

def validate_sources(delivery=False):
    snap = read(RUN/'inputs.json')
    assert git('rev-parse','HEAD').strip() == HEAD
    assert git('rev-parse','HEAD^{tree}').strip() == TREE
    assert git('status','--porcelain=v1','--untracked-files=no') == '', 'integrated tracked tree changed'
    assert git('rev-parse','HEAD',root=ROOT).strip() == snap['original_head']
    for name, expected in snap['files'].items():
        assert content(name) == expected, ('source/input/env identity changed', name)
    if delivery:
        for name, expected in read(RUN/'delivery.json').items():
            assert content(RUN/name) == expected, ('runner/config identity changed', name)
    return snap

def config_path(mode, model):
    return RUN/'configs'/f'{mode}-{model}.yaml'

def gpu_snapshot():
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True, timeout=3)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'], text=True, timeout=3)
    row = next([x.strip() for x in line.split(',')] for line in gpu.splitlines() if line.split(',')[0].strip()=='1')
    assert row[1] == UUID, ('physical second card changed', row)
    return {'gpu_query':gpu, 'apps_query':apps, 'row':row, 'uuid':row[1], 'physical_index':1}

def selected_pids(snap):
    return [int(line.split(',')[1]) for line in snap['apps_query'].splitlines() if line.split(',')[0].strip()==UUID]

def require_idle(snap):
    assert not selected_pids(snap) and int(snap['row'][4]) <= 16 and int(snap['row'][5]) == 0, ('second card not idle',snap)

def require_mask(mask):
    # Unset means PI-authorized local selection. An explicitly empty mask grants no card.
    assert mask == '__UNSET__' or '1' in mask.split(',') or UUID in mask.split(','), ('incompatible scheduler mask',mask)

def group_pids(pgid):
    found=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if os.getpgid(int(p.name))==pgid and p.joinpath('stat').read_text().split(') ')[1][0]!='Z': found.append(int(p.name))
            except (ProcessLookupError,FileNotFoundError): pass
    return found
