"""CPU-only tests of the actual supervisor primitive, no mocked training success."""
import signal
import sys
import tempfile
import time
from common import *
from supervise import claim, run_child, Interrupted, on_signal


def main():
    results={}
    with tempfile.TemporaryDirectory(dir=RUN/'tmp') as tmp:
        p=Path(tmp)
        claim(p/'claim')
        try: claim(p/'claim')
        except FileExistsError: results['duplicate_launch_rejected']=True
        else: raise AssertionError('duplicate claim allowed')
        require_mask('__UNSET__'); require_mask('1'); require_mask(UUID); require_mask('0,1')
        for mask in ['', '0','GPU-other']:
            try: require_mask(mask)
            except AssertionError: pass
            else: raise AssertionError(('scheduler mask widened',mask))
        results['scheduler_mask_preserved']=True
        cmd=[sys.executable,'-c',"import subprocess,time,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"]
        hb=lambda proc,elapsed:None
        result=run_child(cmd,p/'timeout.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),.7,lambda proc:None,hb)
        assert result['cleanup_verified'] and result['returncode']!=0
        results['timeout_descendant_cleanup']=result
        def conflict(proc): raise Interrupted('test external workload detection')
        result=run_child(cmd,p/'conflict.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),6,conflict,hb)
        assert result['cleanup_verified'] and 'external workload' in result['failure']
        results['external_conflict_cleanup']=result
        signal.signal(signal.SIGALRM,on_signal)
        signal.setitimer(signal.ITIMER_REAL,.5)
        result=run_child(cmd,p/'signal.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),6,lambda proc:None,hb)
        signal.setitimer(signal.ITIMER_REAL,0)
        assert result['cleanup_verified'] and 'termination signal' in result['failure']
        results['signal_descendant_cleanup']=result
        # Model accounting arithmetic uses the same budgets, includes shutdown before the cap.
        remaining=CAP-HISTORICAL-899
        assert remaining==345844 and remaining-30+30+HISTORICAL+899==CAP
        results['aggregate_deadline_includes_shutdown']=True
    results.update(status='passed',cuda_used=False,command='uv run --project /work/users/lab_admin/envs/RLVR-rl1080 --frozen python '+str(RUN/'control_tests.py'))
    write(RUN/'control-tests.json',results)
    print(json.dumps(results))

if __name__=='__main__':main()
