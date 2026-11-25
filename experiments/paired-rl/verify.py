"""Read-only CPU acceptance: never invokes CUDA, training, a launch or a writer."""
import copy
import math
import sys
from common import *
import torch
import grpo
from rlvr.config import task_and_splits
from rlvr.sampler import ProblemSampler
from rlvr.steps import first_error


def jsonl(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]

def main():
    torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    inputs=validate_sources(delivery=True)
    assert inputs['preflight']=='passed_cpu_both' and len(inputs['assets'])==2
    assert all(a['finite'] and a['exact_state_shapes_values'] for a in inputs['assets'])
    configs=[grpo.rl_config(str(config_path('main',m)),[]) for m in MODEL_IDS]
    def flatten(d,p=''):
        out={}
        for k,v in d.items():
            key=f'{p}.{k}'.strip('.')
            if isinstance(v,dict):out.update(flatten(v,key))
            else:out[key]=v
        return out
    a,b=map(flatten,configs); diffs={k for k in a if a[k]!=b[k]}
    assert diffs==set(inputs['permitted_pair_differences'])=={'rl.base','pretrain.level','pretrain.comp_frac'}
    task,splits=task_and_splits(configs[0])
    probs=[task.from_dict(r) for r in jsonl(configs[0]['data']['dataset'])]
    assert len(probs)==40000 and len({p.pid for p in probs})==40000
    assert all(splits.pool_of(p.pid)=='rl' and not splits.has_heldout(p) for p in probs)
    all_eval=set()
    for name,obj in inputs['eval'].items():
        ev=[task.from_dict(r) for r in jsonl(ROOT/obj['path'])]
        assert len(ev)==obj['n'] and all(splits.pool_of(p.pid)=='eval' and splits.has_heldout(p)==(name=='E4') for p in ev)
        all_eval.update(p.pid for p in ev)
    assert not {p.pid for p in probs}&all_eval
    order=jsonl(PREP/'input-order-s0.jsonl')
    assert len(order)==500
    sampler=ProblemSampler([p.depth for p in probs],seed=0,total=32000)
    for i,row in enumerate(order):
        assert row['step']==i+1 and row['pids']==[probs[j].pid for j in sampler.sample(64)]
    for m,cfg in zip(MODEL_IDS,configs):
        rc,ec=cfg['rl'],cfg['eval']
        assert cfg['seed']==0 and rc['steps']==500 and rc['prompts_per_step']==64 and rc['group_size']==8
        assert rc['max_new_tokens']==ec['max_new_tokens']==768 and ec['batch_size']==2
        assert rc['temperature']==1 and rc['reward']=='binary' and not cfg['data']['p_init_labels']
        assert cfg['data']['sampler']=='static' and cfg['data']['order']=='shuffle'
        assert rc['eval_sets']==['E1','E2','E3','E4'] and rc['probe_set']=='probe' and rc['eval_every']==50 and rc['probe_every']==25
        assert ec['pass_k_samples']==0 and ec['pass_k']==[1,2,4,8,16]
        assert sorted({max(1,round(f*500)) for f in rc['save_at']})==list(range(50,501,50))
        out=RUN/'smoke'/m; evidence=read(out/'training-evidence.json')
        assert evidence['status']=='passed' and evidence['steps']==1 and evidence['rollouts']==512 and evidence['optimizer_calls']==1
        assert evidence['policy_initial_digest']==evidence['reference_initial_digest']==evidence['reference_final_digest']
        assert evidence['policy_final_digest']==evidence['checkpoint_reload_digest'] and evidence['policy_changed']
        assert evidence['checkpoint_model']==content(Path(evidence['checkpoint'])/'model.safetensors')
        optimizer=jsonl(out/'optimizer.jsonl'); assert len(optimizer)==1
        assert optimizer[0]['gradients_finite'] and optimizer[0]['parameters_and_state_finite'] and optimizer[0]['adam_steps']==[1]
        paths=read(out/'paths.json'); rd=Path(paths['run_dir'])
        assert (rd/'ckpt').is_symlink() and (rd/'rollouts.jsonl').is_symlink()
        assert str((rd/'ckpt').resolve()).startswith(str(BULK))
        actual=jsonl(rd/'rollouts.jsonl'); assert len(actual)==64 and [r['pid'] for r in actual]==order[0]['pids']
        assert all(len(r['rewards'])==len(r['lens'])==len(r['completions'])==8 for r in actual)
        metric=jsonl(rd/'metrics.jsonl'); assert len(metric)==1
        assert all(math.isfinite(v) for v in metric[0].values() if isinstance(v,(float,int)))
        rewards=[x for r in actual for x in r['rewards']]
        signal=sum(min(r['rewards'])<max(r['rewards']) for r in actual)/64
        assert sum(rewards)/512==metric[0]['mean_reward'] and signal==metric[0]['signal_frac']
        assert read(out/'saved-policy-batch1.json')==read(out/'saved-policy-batch2.json') and evidence['batch1_batch2_exact']
        tiny={name:{p['pid']:task.from_dict(p) for p in jsonl(RUN/f'tiny-data/eval/v1/{name}.jsonl')} for name in inputs['eval']}
        for step in (0,1):
            saved=read(out/f'eval-step{step}.json')
            assert len(saved['records'])==10 and {r['eval_set'] for r in saved['rows']}==set(tiny)
            for r in saved['records']:
                p=tiny[r['eval_set']][r['pid']]; error,error_step,prefix=first_error(task,p,r['completion'])
                assert (error,error_step,prefix)==(r['err_type'],r['err_step'],r['n_correct'])
                assert bool(task.reward(r['completion'],p))==r['correct']
            for r in saved['rows']:
                if r['depth']==r['group']=='all':
                    rows=[x for x in saved['records'] if x['eval_set']==r['eval_set']]
                    assert r['greedy_acc']==sum(x['correct'] for x in rows)/len(rows)
                    assert r['step_score']==sum(x['n_correct'] for x in rows)/len(rows)
                    assert r['step_frac']==sum(x['n_correct']/x['n_ops'] for x in rows)/len(rows)
                assert all(math.isfinite(r[k]) for k in ('greedy_acc','step_score','step_frac','format_err','mean_len'))
        for name,obj in read(out/'imported-source.json')['modules'].items():
            assert obj['path'].startswith(str(CODE)+'/') and content(obj['path'])=={k:obj[k] for k in ['bytes','sha256']}
    ledger=read(RUN/'accounting.json')
    assert ledger['cap_seconds']==CAP and ledger['historical_seconds']==HISTORICAL
    assert ledger['charged_worker_seconds']<=900 and ledger['charged_host_seconds']==0
    assert ledger['remaining_seconds']==CAP-HISTORICAL-ledger['charged_worker_seconds']
    attempts=ledger['attempts']; assert len(attempts)==1 and attempts[0]['mode']=='smoke' and attempts[0]['status']=='completed_pair'
    assert attempts[0]['charged_seconds']==math.ceil(attempts[0]['elapsed_seconds'])==ledger['charged_worker_seconds']
    assert len(attempts[0]['models'])==2 and all(m['cleanup_verified'] and m['failure'] is None and m['seconds']<900 and m['limit_seconds']<=420 for m in attempts[0]['models'])
    assert read(RUN/'smoke-status.json')['state']=='completed_pair'
    tests=read(RUN/'control-tests.json'); assert tests['status']=='passed' and tests['cuda_used'] is False
    assert tests['duplicate_launch_rejected'] and tests['scheduler_mask_preserved'] and tests['aggregate_deadline_includes_shutdown']
    assert all(tests[k]['cleanup_verified'] for k in ['timeout_descendant_cleanup','external_conflict_cleanup','signal_descendant_cleanup'])
    assert not (RUN/'main.claimed').exists(), 'host job already claimed; never relaunch/overwrite it'
    job=(RUN/'job.sh').read_text(); assert 'set -euo pipefail' in job and '--mode main' in job and '--frozen' in job
    for needle in ['O_EXCL','SIGTERM','SIGKILL','deadline-time.monotonic()-30','charged_worker_seconds','validate_sources(delivery=True)','require_idle(snap)','start_new_session=True','--kill-after=5s','flock','selected_pids(snap)']:
        assert needle in (RUN/'supervise.py').read_text(),needle
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'passed','cpu_only':True,'both_assets':'verified','both_smokes':'passed','main_launch':'ready_unlaunched','worker_gpu_charge_seconds':ledger['charged_worker_seconds'],'total_gpu_charge_seconds':HISTORICAL+ledger['charged_worker_seconds'],'host_remaining_seconds':ledger['remaining_seconds'],'config_differences':sorted(diffs)}))

if __name__=='__main__': main()
