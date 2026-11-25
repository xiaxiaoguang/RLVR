"""One-time CPU preparation. No download and no CUDA initialization."""
import copy
import gc
import time
from common import *
import numpy as np
import torch
import yaml
import grpo
from safetensors.torch import load_file
from rlvr.config import load_yaml, task_and_splits
from rlvr.model import load_model
from rlvr.sampler import ProblemSampler
from rlvr.params import merge


def main():
    start=time.monotonic()
    assert not (RUN/'inputs.json').exists(), 'preparation is one-shot'
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert git('rev-parse','HEAD').strip()==HEAD and git('status','--porcelain=v1')==''
    write(RUN/'git-before.json', {'original_head':git('rev-parse','HEAD',root=ROOT).strip(), 'original_status':git('status','--porcelain=v1',root=ROOT), 'code_head':HEAD,'code_tree':TREE,'code_status':git('status','--porcelain=v1'),'refs':git('show-ref',root=ROOT)})
    matrix=read(PREP/'matrix.json')
    files={}
    for base in (ROOT,CODE):
        for name in git('ls-files',root=base).splitlines():
            files[str(base/name)]=content(base/name)
    for p in [ENV/'uv.lock',ENV/'pyproject.toml',ENV/'.python-version',RUN/'handoff-before.md',RUN/'inherited-accounting.json',PREP/'matrix.json',PREP/'input-order-s0.jsonl',PREP/'pretrain-inventory.json', ROOT/'runs/idea-20261121-01-integrated-throughput/accounting.json']:
        files[str(p)]=content(p)
    assert files[str(ENV/'uv.lock')]['sha256']=='da9cbed8ab7ca6484c55505bdb4b4d7700efcc328330f823246870bc87f292b9'
    inputs={'pool':matrix['pool'], 'eval':matrix['eval'], 'input_order':matrix['input_order']}
    for obj in [matrix['pool'],matrix['input_order'],*matrix['eval'].values()]:
        p=ROOT/obj['path']; c=content(p)
        assert c['sha256']==obj['sha256']
        files[str(p)]=c
    (RUN/'configs').mkdir(exist_ok=True)
    evaldir=RUN/'tiny-data/eval/v1'; evaldir.mkdir(parents=True,exist_ok=True)
    for name,obj in matrix['eval'].items():
        # First two canonical rows of each group, selected before model execution.
        lines=(ROOT/obj['path']).read_text().splitlines()[:2]
        p=evaldir/f'{name}.jsonl'; p.write_text('\n'.join(lines)+'\n'); files[str(p)]=content(p)
    assets=[]
    expected_weights=['6476c96d6a50ce809791379f64cd1d93332ba031d7c1f5a29077b84201c13ee8','36c50e13f27feaf43f75f97ff1150add99da4459e7b118027bba0ec1e6b8a04d']
    expected_second={'config.json':'f5370228541589c4e17442d1e5ab9da3d4addc51d5a736ced06c6a347cd6eef7','tokenizer.json':'f6d952259d16437a0d4654fbe23857b98ed17b59fafe65c439d99763b1e9b00f','run_config.yaml':'74b0aa3df4e2013e21a7b69bd95585dc616a2fca44ac05e64bb0e796ed9a7650'}
    configs=[]
    for i,model_id in enumerate(MODEL_IDS):
        base=ROOT/'checkpoints/tree-v1'/model_id
        for name in ('model.safetensors','config.json','tokenizer.json','run_config.yaml'):
            files[str(base/name)]=content(base/name)
        assert sha(base/'model.safetensors')==expected_weights[i]
        if i==1:
            for name,digest in expected_second.items(): assert sha(base/name)==digest
        ck=yaml.safe_load((base/'run_config.yaml').read_text())
        model,tok=load_model(base,'cpu')
        raw=load_file(str(base/'model.safetensors'), device='cpu')
        state=model.state_dict()
        assert set(raw)==set(state), ('checkpoint state keys', set(state)-set(raw),set(raw)-set(state))
        assert all(raw[k].shape==state[k].shape and raw[k].dtype==torch.float32 and torch.isfinite(raw[k]).all() and torch.equal(raw[k],state[k]) for k in raw)
        assert model.config.num_hidden_layers==12 and model.config.hidden_size==768 and model.config.num_attention_heads==12 and model.config.intermediate_size==2048 and model.config.vocab_size==49 and model.config.max_position_embeddings==1024
        assert model.config.task_fingerprint=='195a655b63e47644f32bc96ad52ecb3cb163cb841f14ce923a27ba79c66a4ca4'
        assert ck['seed']==0 and ck['pretrain']['steps']==9500 and ck['pretrain']['comp_frac']==[.0035,.006][i]
        assets.append({'model':model_id,'loader':'rlvr.model.load_model(cpu)','keys':len(raw),'parameters':sum(p.numel() for p in model.parameters()),'finite':True,'exact_state_shapes_values':True,'fingerprint':model.config.task_fingerprint,'tokenizer_vocab':tok.vocab,'pretrain':ck['pretrain']})
        del model, raw, state; gc.collect()
        cfg=yaml.safe_load((PREP/'configs/common-original.yaml').read_text())
        cfg.pop('base')
        cfg=merge(load_yaml(CODE/'configs/rl/1080ti.yaml'),cfg)
        cfg['paths']={'data':str(PREP/'data-root'),'runs':str(RUN/'main/output')}
        cfg['data']['dataset']=str(ROOT/matrix['pool']['path'])
        cfg['rl'].update(base=str(base),tag='paired-original',save_at=[s/500 for s in range(50,501,50)],log_eval_problems=True)
        cfg.setdefault('eval',{})['batch_size']=2
        config_path('main',model_id).write_text(yaml.safe_dump(cfg,sort_keys=False))
        resolved=grpo.rl_config(str(config_path('main',model_id)),[])
        configs.append(resolved)
        (RUN/f'resolved-main-{model_id}.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False))
        smoke=copy.deepcopy(cfg); smoke['paths']={'data':str(RUN/'tiny-data'),'runs':str(RUN/'smoke/output')}
        smoke['rl'].update(steps=1,save_at=[],tag='paired-smoke')
        config_path('smoke',model_id).write_text(yaml.safe_dump(smoke,sort_keys=False))
    def diffs(a,b,p=''):
        result=[]
        for k in a:
            key=f'{p}.{k}'.strip('.')
            if isinstance(a[k],dict): result+=diffs(a[k],b[k],key)
            elif a[k]!=b[k]: result.append(key)
        return result
    differences=diffs(*configs)
    assert set(differences)<= {'rl.base','pretrain.level','pretrain.comp_frac'},differences
    cfg=configs[0]
    task,splits=task_and_splits(cfg)
    rows=[json.loads(l) for l in (ROOT/matrix['pool']['path']).read_text().splitlines()]
    probs=[task.from_dict(r) for r in rows]
    assert len(probs)==40000 and len({p.pid for p in probs})==40000
    assert all(splits.pool_of(p.pid)=='rl' and not splits.has_heldout(p) for p in probs)
    all_eval=set()
    for name,obj in matrix['eval'].items():
        ev=[task.from_dict(json.loads(l)) for l in (ROOT/obj['path']).read_text().splitlines()]
        assert len(ev)==obj['n'] and all(splits.pool_of(p.pid)=='eval' and splits.has_heldout(p)==(name=='E4') for p in ev)
        all_eval.update(p.pid for p in ev)
    assert not all_eval & {p.pid for p in probs}
    expected=[json.loads(l) for l in (PREP/'input-order-s0.jsonl').read_text().splitlines()]
    sampler=ProblemSampler([p.depth for p in probs],kind='static',order='shuffle',seed=0,total=32000)
    assert all([probs[j].pid for j in sampler.sample(64)]==row['pids'] and row['step']==i+1 for i,row in enumerate(expected))
    assert len(expected)==500
    # Native trainer CPU preflight for both full configs: validates loader, all rows and no leakage.
    for model_id in MODEL_IDS:
        grpo.main(['--config',str(config_path('main',model_id)),'--preflight','--device','cpu'])
    receipt=ROOT/'checkpoints/tree-v1/.cache/huggingface/trees/2cc23a4ebdfe5a64dd4d2220e02194a1f5745736.json'
    cache=read(receipt)['files']; provenance={}
    for name in ('model.safetensors','config.json','tokenizer.json','run_config.yaml'):
        p=ROOT/'checkpoints/tree-v1'/MODEL_IDS[1]/name
        obj=cache.get(MODEL_IDS[1]+'/'+name)
        if obj:
            match=(sha(p)==obj['lfs_sha256']) if 'lfs_sha256' in obj else (hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()==obj['blob_id'])
            assert match,('cached official tree mismatch',name)
            provenance[name]={'cached_entry':obj,'matches_cached_entry':match}
    files[str(receipt)]=content(receipt)
    inputs.update(original_head=git('rev-parse','HEAD',root=ROOT).strip(),code_head=HEAD,code_tree=TREE,files=files,assets=assets,permitted_pair_differences=differences,preflight='passed_cpu_both',split_and_order='40000 canonical RL; 500x64 fixed PID sequence matches; eval disjoint',provenance={'revision':'2cc23a4ebdfe5a64dd4d2220e02194a1f5745736','cached_tree_checks':provenance,'confidence':'PI local delivery hashes and prior cached tree metadata agree; no fresh remote revision/auth verification; no second delivery download receipt present'},wall_seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized()
    write(RUN/'inputs.json',inputs)
    write(RUN/'accounting.json',{'cap_seconds':CAP,'historical_seconds':HISTORICAL,'historical_cap_seconds':172800,'authorization':'PI 2026-11-24 23:07:49 replaces old 48h/cost-fit/no-partial guards with 100h; original scientific settings retained','inherited_ledger':content(RUN/'inherited-accounting.json'),'attempts':[],'charged_worker_seconds':0,'charged_host_seconds':0,'remaining_seconds':CAP-HISTORICAL})
    print('CPU preparation passed',differences)

if __name__=='__main__': main()
