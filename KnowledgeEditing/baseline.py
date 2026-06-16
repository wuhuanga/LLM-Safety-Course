from src.ke_backend import SimulatedEditableLLM, load_json, dump_json
DATA='data/custom_facts.json'
OUT='outputs/baseline_results.json'

def main():
    records=load_json(DATA)
    model=SimulatedEditableLLM()
    results=[]
    print('[Task 1] Baseline evaluation before editing')
    print(f'Dataset size: {len(records)}')
    for i,r in enumerate(records,1):
        ans=model.generate(r['prompt'], records)
        ok = r['target_new'].lower() in ans.lower()
        results.append({'id':i,'prompt':r['prompt'],'answer_before_edit':ans,'target_new':r['target_new'],'already_correct':ok})
        print(f'{i:02d}. prompt={r["prompt"]}')
        print(f'    before_edit={ans} | target_new={r["target_new"]} | hit={ok}')
    dump_json(results, OUT)
    print(f'Saved: {OUT}')

if __name__=='__main__': main()
