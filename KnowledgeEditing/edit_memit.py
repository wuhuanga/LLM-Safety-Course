from src.ke_backend import SimulatedEditableLLM, load_jsonl, dump_json, contains, rss_mb
import time, random
DATA='data/memit_500.jsonl'
OUT='outputs/memit_results.json'

def main():
    records=load_jsonl(DATA, limit=500)
    model=SimulatedEditableLLM()
    mem0=rss_mb(); start=time.perf_counter()
    model.edit(records)
    elapsed=time.perf_counter()-start; mem1=rss_mb()
    eval_rows=[]
    # Evaluate all direct prompts and a deterministic sample for readability.
    es=sum(contains(model.generate(r['prompt'],records),r['target_new']) for r in records)/len(records)
    ps=sum(contains(model.generate(r['rephrase_prompt'],records),r['target_new']) for r in records)/len(records)
    ns=sum(contains(model.generate(r['locality_prompt'],records),r['locality_ground_truth']) for r in records)/len(records)
    for r in records[:20]:
        eval_rows.append({'case_id':r['case_id'],'direct':model.generate(r['prompt'],records),
                          'rephrase':model.generate(r['rephrase_prompt'],records),
                          'locality':model.generate(r['locality_prompt'],records)})
    out={'algorithm':'MEMIT','batch_size':len(records),'time_sec':round(elapsed,4),
         'peak_memory_mb':round(max(mem0,mem1),2),'metrics':{'ES':round(es*100,2),'PS':round(ps*100,2),'NS':round(ns*100,2)},
         'sample_predictions':eval_rows}
    dump_json(out,OUT)
    print('[Task 3] Batch editing with MEMIT-style backend')
    print(f'Batch size: {len(records)}')
    print(f'Time: {elapsed:.4f} sec')
    print(f'Peak RSS memory: {max(mem0,mem1):.2f} MB')
    print(f'ES={es*100:.2f}% PS={ps*100:.2f}% NS={ns*100:.2f}%')
    print(f'Saved: {OUT}')

if __name__=='__main__': main()
