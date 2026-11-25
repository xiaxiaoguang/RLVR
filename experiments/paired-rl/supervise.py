"""Single-card sequential host/smoke supervision with durable reservation and group cleanup."""
import argparse
import fcntl
import math
import signal
import sys
import time
import traceback
from common import *

class Interrupted(Exception): pass

def on_signal(sig,frame):
    raise Interrupted(f'PI/process termination signal {sig}')

def claim(path):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as f:
        f.write(json.dumps({'pid':os.getpid(),'time':time.time()})+'\n'); f.flush(); os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def cleanup(proc):
    # Signal the whole group even if its leader has already exited.
    for sig,grace in [(signal.SIGTERM,3),(signal.SIGKILL,2)]:
        try: os.killpg(proc.pid,sig)
        except ProcessLookupError: pass
        deadline=time.monotonic()+grace
        while time.monotonic()<deadline:
            proc.poll()
            if not group_pids(proc.pid): break
            time.sleep(.05)
    proc.wait(timeout=1)
    assert not group_pids(proc.pid), ('descendants remain',group_pids(proc.pid))

def run_child(command,log,env,limit,monitor,heartbeat):
    proc=None; failure=None; last=0
    began=time.monotonic()
    try:
        with Path(log).open('x') as f:
            # GNU timeout independently bounds this process group if the supervisor stalls.
            proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=5s',f'{limit:.3f}s',*command],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                elapsed=time.monotonic()-began
                heartbeat(proc,elapsed)
                if elapsed>=limit: failure='hard child deadline'; break
                if elapsed-last>=2:
                    monitor(proc); last=elapsed
                time.sleep(.2)
            rc=proc.poll()
            if rc not in (None,0): failure=f'child exit {rc}'
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
    finally:
        if proc is not None: cleanup(proc)
    return {'returncode':proc.returncode if proc else None,'failure':failure,'seconds':time.monotonic()-began,'cleanup_verified':proc is None or not group_pids(proc.pid),'child_group':proc.pid if proc else None}

def supervise(mode):
    require_mask(os.environ.get('PAIRED_SCHEDULER_MASK','__UNSET__'))
    validate_sources(delivery=True)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,on_signal)
    # Existing cross-task lock plus this card's common idea lock. No historical bytes modified.
    lock_paths=[ROOT/'runs/idea-20261121-01-real-rl/gpu.lock',RUN/(UUID+'.lock')]
    handles=[]
    try:
        for p in lock_paths:
            h=p.open('r' if p.exists() else 'a'); handles.append(h); fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=read(RUN/'accounting.json')
        assert ledger['cap_seconds']==CAP and ledger['historical_seconds']==HISTORICAL
        if mode=='main':
            # CPU checker before claiming/spawning. Existing claim always rejects a relaunch.
            subprocess.run([sys.executable,str(RUN/'verify.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,cwd=CODE)
        claim(RUN/(mode+'.claimed'))
        budget=900-ledger['charged_worker_seconds'] if mode=='smoke' else CAP-HISTORICAL-ledger['charged_worker_seconds']-ledger['charged_host_seconds']
        assert budget>30
        began=time.monotonic(); deadline=began+budget
        entry={'mode':mode,'state':'reserved','reserved_seconds':budget,'charged_seconds':budget,'started_unix':time.time(),'models':[],'shutdown_margin_seconds':30,'scheduler_mask':os.environ.get('PAIRED_SCHEDULER_MASK','__UNSET__'),'lock_paths':[str(p) for p in lock_paths]}
        ledger['attempts'].append(entry)
        base_worker=ledger['charged_worker_seconds']; base_host=ledger['charged_host_seconds']
        def persist(final=False):
            elapsed=time.monotonic()-began
            charge=math.ceil(elapsed) if final else budget
            entry.update(elapsed_seconds=elapsed,charged_seconds=charge,last_update_unix=time.time())
            ledger['charged_worker_seconds']=base_worker+(charge if mode=='smoke' else 0)
            ledger['charged_host_seconds']=base_host+(charge if mode=='main' else 0)
            ledger['remaining_seconds']=CAP-HISTORICAL-ledger['charged_worker_seconds']-ledger['charged_host_seconds']
            write(RUN/'accounting.json',ledger)
        persist()
        event(RUN/'accounting-events.jsonl',{'event':'reserve','mode':mode,'budget':budget,'time':time.time()})
        status={'mode':mode,'state':'running','order':MODEL_IDS,'models':{m:{'state':'not_started'} for m in MODEL_IDS},'cap_seconds':CAP,'allocation_seconds':budget,'deadline_unix':time.time()+budget}
        write(RUN/f'{mode}-status.json',status)
        try:
            for model in MODEL_IDS:
                if deadline-time.monotonic()<=30: raise Interrupted('aggregate cap reached before next model')
                snap=gpu_snapshot(); require_idle(snap)
                status['models'][model]={'state':'starting','pre_spawn_guard':snap}; write(RUN/f'{mode}-status.json',status)
                limit=min(420,deadline-time.monotonic()-30) if mode=='smoke' else deadline-time.monotonic()-30
                command=['uv','run','--project',str(ENV),'--frozen','python',str(RUN/'worker.py'),'--mode',mode,'--model',model]
                childentry={'model':model,'command':command,'cwd':str(CODE),'CUDA_VISIBLE_DEVICES':UUID,'limit_seconds':limit,'pre_spawn_guard':snap}
                entry['models'].append(childentry); persist()
                last_persist=[0]
                def beat(proc,elapsed):
                    if elapsed-last_persist[0]>=1:
                        childentry.update(child_group=proc.pid,elapsed_seconds=elapsed); persist(); last_persist[0]=elapsed
                def monitor(proc):
                    snap=gpu_snapshot(); owners=[]
                    for pid in selected_pids(snap):
                        try: pgid=os.getpgid(pid)
                        except ProcessLookupError: continue
                        owners.append({'pid':pid,'pgid':pgid,'owned':pgid==proc.pid})
                    event(RUN/f'{mode}-watchdog.jsonl',{'time':time.time(),'model':model,'snapshot':snap,'owners':owners})
                    assert all(p['owned'] for p in owners), 'external GPU process conflict'
                    assert owners or int(snap['row'][4])<=16, 'unattributed GPU memory'
                result=run_child(command,RUN/f'{mode}-{model}.log',dict(os.environ,CUDA_VISIBLE_DEVICES=UUID),limit,monitor,beat)
                childentry.update(result)
                snap=gpu_snapshot(); childentry['post_cleanup_guard']=snap
                # Only our group must be gone; external work is recorded and blocks next model, never killed.
                assert result['cleanup_verified']
                assert not any(os.getpgid(pid)==result['child_group'] for pid in selected_pids(snap) if Path(f'/proc/{pid}').exists())
                status['models'][model].update(state='passed' if result['failure'] is None else 'failed',**result)
                write(RUN/f'{mode}-status.json',status); persist()
                event(RUN/'accounting-events.jsonl',{'event':'model_finished','mode':mode,'model':model,'result':result,'time':time.time()})
                if result['failure']: raise Interrupted(result['failure'])
                assert read(RUN/mode/model/'training-evidence.json')['status']=='passed'
            status['state']='completed_pair'
        except BaseException as exc:
            status.update(state='interrupted_or_failed',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        finally:
            # Ignore repeated graceful signals only during bounded final bookkeeping, after child cleanup.
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,signal.SIG_IGN)
            entry.update(state='finished',status=status['state'],finished_unix=time.time())
            persist(final=True)
            assert entry['charged_seconds']<=budget, 'allocation overrun'
            write(RUN/f'{mode}-status.json',status)
            event(RUN/'accounting-events.jsonl',{'event':'final_charge','mode':mode,'seconds':entry['charged_seconds'],'state':status['state'],'time':time.time()})
        print(json.dumps(status))
        return 0 if status['state']=='completed_pair' else 1
    finally:
        for h in reversed(handles): h.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['main','smoke'],required=True); args=ap.parse_args()
    return supervise(args.mode)

if __name__=='__main__': sys.exit(main())
