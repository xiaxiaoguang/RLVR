"""Actual TERM/INT/HUP injection into the same supervisor handler; CPU descendants only."""
import signal
import sys
import tempfile
import threading
from common import *
from supervise import on_signal,run_child

def main():
    result={}
    with tempfile.TemporaryDirectory(dir=RUN/'tmp') as tmp:
        for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:
            old=signal.signal(sig,on_signal)
            timer=threading.Timer(.5,lambda:os.kill(os.getpid(),sig)); timer.start()
            try:
                cmd=[sys.executable,'-c',"import subprocess,time,sys; subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);time.sleep(60)"]
                r=run_child(cmd,Path(tmp)/f'{sig}.log',dict(os.environ,CUDA_VISIBLE_DEVICES=''),6,lambda p:None,lambda p,t:None)
                assert r['cleanup_verified'] and f'termination signal {sig}' in r['failure']
                result[str(sig)]={'signal_name':signal.Signals(sig).name,**r}
            finally:
                timer.cancel(); timer.join(); signal.signal(sig,old)
    write(RUN/'signal-tests.json',{'status':'passed','cuda_used':False,'cases':result})
    print(json.dumps(result))

if __name__=='__main__':main()
