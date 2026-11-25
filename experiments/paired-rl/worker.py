"""Observations around the authoritative trainer; its sampling/loss/optimizer/evaluation are unchanged."""
import argparse
import gc
import math
import sys
import time
from common import *
import torch
import grpo
import rlvr.model as mm
import rlvr.evaluate as em
from rlvr.config import task_and_splits
from rlvr.steps import first_error


def digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','main'],required=True); ap.add_argument('--model',choices=MODEL_IDS,required=True)
    args=ap.parse_args(); mode,model_id=args.mode,args.model
    assert os.getcwd()==str(CODE) and os.environ['CUDA_VISIBLE_DEVICES']==UUID
    validate_sources(delivery=True)
    assert not torch.cuda.is_initialized()
    require_idle(gpu_snapshot())
    torch.set_num_threads(1)
    cfg=grpo.rl_config(str(config_path(mode,model_id)),[])
    task,_=task_and_splits(cfg)
    out=RUN/mode/model_id; out.mkdir(parents=True,exist_ok=True)
    modules={m.__name__: {'path':m.__file__, **content(m.__file__)} for m in [grpo,mm,em,sys.modules['rlvr.grpo'],sys.modules['rlvr.sampler'],sys.modules['rlvr.steps'],sys.modules['rlvr.precision']]}
    assert all(os.path.commonpath([m['path'],str(CODE)])==str(CODE) for m in modules.values())
    write(out/'imported-source.json',{'cwd':os.getcwd(),'modules':modules,'PYTHONPATH':os.environ['PYTHONPATH'],'code_head':HEAD,'code_tree':TREE})
    run_dir=Path(cfg['paths']['runs'])/grpo.run_name_for(cfg)
    run_dir.mkdir(parents=True,exist_ok=False)
    bulk=BULK/mode/model_id; bulk.mkdir(parents=True,exist_ok=False)
    (bulk/'ckpt').mkdir(); (bulk/'rollouts.jsonl').touch()
    (run_dir/'ckpt').symlink_to(bulk/'ckpt',target_is_directory=True)
    (run_dir/'rollouts.jsonl').symlink_to(bulk/'rollouts.jsonl')
    write(out/'paths.json',{'run_dir':str(run_dir),'bulk':str(bulk),'config':str(config_path(mode,model_id))})
    models=[]; initial=[]; updates=[]; sampled=[]
    orig_load=grpo.load_model; orig_adam=torch.optim.AdamW; orig_eval=grpo.greedy_eval; orig_sampler=grpo.ProblemSampler
    expected=[json.loads(l)['pids'] for l in (PREP/'input-order-s0.jsonl').read_text().splitlines()]
    pool=[json.loads(l)['pid'] for l in Path(cfg['data']['dataset']).read_text().splitlines()]
    def load(*a,**kw):
        model,tok=orig_load(*a,**kw)
        assert all(p.dtype==torch.float32 for p in model.parameters())
        assert torch.cuda.get_device_capability(0)==(6,1)
        models.append(model); initial.append(digest(model))
        return model,tok
    class ObservedSampler(orig_sampler):
        def sample(self,b):
            idx=super().sample(b)
            pids=[pool[j] for j in idx]
            assert pids==expected[len(sampled)], 'actual training PID/order mismatch'
            sampled.append(pids)
            event(out/'sampled-pids.jsonl',{'step':len(sampled),'pids':pids})
            return idx
    class ObservedAdamW(orig_adam):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.register_step_pre_hook(self.before); self.register_step_post_hook(self.after)
        def before(self,opt,a,kw):
            grads=[p.grad for g in self.param_groups for p in g['params'] if p.grad is not None]
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads), 'missing/nonfinite gradient'
            self.obs={'step':len(updates)+1,'gradient_tensors':len(grads),'gradients_finite':True,'post_clip_grad_l2':math.sqrt(sum(float(g.double().square().sum()) for g in grads)),'lr':[g['lr'] for g in self.param_groups]}
        def after(self,opt,a,kw):
            assert self.state
            for p,state in self.state.items():
                assert bool(torch.isfinite(p).all()) and all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v))
            record={**self.obs,'parameters_and_state_finite':True,'adam_steps':sorted({int(s['step']) for s in self.state.values()}),'state_parameter_count':len(self.state)}
            updates.append(record); event(out/'optimizer.jsonl',record)
    def evaluate(*a,**kw):
        rows=orig_eval(*a,**kw)
        if mode=='smoke':
            step=a[4] if len(a)>4 else kw.get('step',0)
            write(out/f'eval-step{step}.json',{'rows':rows,'records':kw['records']})
        return rows
    grpo.load_model=load; grpo.ProblemSampler=ObservedSampler; grpo.greedy_eval=evaluate; torch.optim.AdamW=ObservedAdamW
    started=time.monotonic()
    try:
        summary=grpo.main(['--config',str(config_path(mode,model_id))])
        torch.cuda.synchronize()
        assert len(models)==2 and initial[0]==initial[1] and len(updates)==cfg['rl']['steps']
        policy_after=digest(models[0]); ref_after=digest(models[1])
        assert initial[1]==ref_after and all(not p.requires_grad and p.grad is None for p in models[1].parameters())
        reloaded,tok=mm.load_model(summary['ckpt'],'cpu')
        reload_digest=digest(reloaded)
        assert reload_digest==policy_after
        del reloaded; gc.collect()
        if mode=='smoke':
            # Fixed first two E2 rows, on the saved policy, no sampled pass@k or selection on success.
            problems=[task.from_dict(json.loads(l)) for l in (RUN/'tiny-data/eval/v1/E2.jsonl').read_text().splitlines()]
            prompts=[task.format_prompt(p) for p in problems]
            variants=[]
            for b in [1,2]:
                ids=mm.generate_ids(models[0],tok,prompts,768,0.0,1,b)
                records=[]
                for p,tokens in zip(problems,ids):
                    text=tok.decode(tokens); err,err_step,prefix=first_error(task,p,text)
                    records.append({'pid':p.pid,'token_ids':tokens,'text':text,'reward':task.reward(text,p),'err_type':err,'err_step':err_step,'step_score':prefix,'step_frac':prefix/p.n_ops,'eos_positions':[j for j,t in enumerate(tokens) if t==tok.eos_id]})
                write(out/f'saved-policy-batch{b}.json',records); variants.append(records)
            assert variants[0]==variants[1], 'real saved-policy B1/B2 mismatch'
        write(out/'training-evidence.json',{'status':'passed','entry_point':grpo.__file__+':main','mode':mode,'model':model_id,'steps':summary['steps'],'rollouts':summary['rollouts'],'optimizer_calls':len(updates),'policy_initial_digest':initial[0],'policy_final_digest':policy_after,'policy_changed':policy_after!=initial[0],'reference_initial_digest':initial[1],'reference_final_digest':ref_after,'checkpoint_reload_digest':reload_digest,'checkpoint_model':content(Path(summary['ckpt'])/'model.safetensors'),'checkpoint':summary['ckpt'],'summary':summary,'wall_seconds':time.monotonic()-started,'dtype':'fp32','capability':[6,1],'batch1_batch2_exact':True if mode=='smoke' else None,'resume':'model/tokenizer/config only; no optimizer/scheduler/RNG/sampler state; not exact-resumable'})
    finally:
        grpo.load_model=orig_load; grpo.ProblemSampler=orig_sampler; grpo.greedy_eval=orig_eval; torch.optim.AdamW=orig_adam

if __name__=='__main__': main()
