"""Read-only, CPU/stdlib paired progress and matched-step comparison; absent stages remain absent."""
from common import *

def lines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: break  # a live writer may have an unfinished final row
    return result

def main():
    ledger=read(RUN/'accounting.json')
    result={'scientific_status':'fresh paired full evaluation pending; smoke is wiring only','accounting':{k:ledger[k] for k in ['cap_seconds','historical_seconds','charged_worker_seconds','charged_host_seconds','remaining_seconds']},'models':{},'matched_evaluations':[]}
    tables={}
    inventory=read(PREP/'pretrain-inventory.json')['handed_over']
    for index,model in enumerate(MODEL_IDS):
        paths=RUN/'main'/model/'paths.json'
        rd=Path(read(paths)['run_dir']) if paths.exists() else None
        metrics=lines(rd/'metrics.jsonl') if rd else []
        evals=lines(rd/'eval.jsonl') if rd else []
        table={(r['step'],r['eval_set']):r for r in evals if r.get('depth')==r.get('group')=='all'}
        tables[model]=table
        result['models'][model]={'main_started':paths.exists(),'optimizer_steps_logged':len(metrics),'last_optimizer_step':metrics[-1]['step'] if metrics else None,'complete_500step_summary':(rd/'summary.json').exists() if rd else False,'main_run_dir':str(rd) if rd else None,'fresh_step0':{name:r for (step,name),r in table.items() if step==0},'last_evaluations':{name:max((r for (s,n),r in table.items() if n==name),key=lambda r:r['step']) for name in {n for _,n in table}},'historical_pretrain_final':inventory[index]['historical_final_eval'],'historical_warning':'prior-platform logs, not this fresh fp32/B2 baseline'}
    common=sorted(set(tables[MODEL_IDS[0]]) & set(tables[MODEL_IDS[1]]))
    for step,name in common:
        a=tables[MODEL_IDS[0]][step,name]; b=tables[MODEL_IDS[1]][step,name]
        before_a=tables[MODEL_IDS[0]].get((0,name)); before_b=tables[MODEL_IDS[1]].get((0,name))
        fields={}
        for key in ['greedy_acc','step_score','step_frac','format_err','mean_len']:
            fields[key]={'cf0035':a[key],'cf006':b[key],'cf006_minus_cf0035':b[key]-a[key]}
            if before_a and before_b:
                da=a[key]-before_a[key]; db=b[key]-before_b[key]
                fields[key].update(cf0035_RL_delta=da,cf006_RL_delta=db,difference_in_RL_deltas=db-da)
        result['matched_evaluations'].append({'step':step,'eval_set':name,'metrics':fields})
    if all(x['complete_500step_summary'] for x in result['models'].values()):result['scientific_status']='both full runs have summaries; correctness/analysis review still required'
    elif any(x['main_started'] for x in result['models'].values()):result['scientific_status']='pair running or incomplete; compare only common saved evaluation steps'
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
