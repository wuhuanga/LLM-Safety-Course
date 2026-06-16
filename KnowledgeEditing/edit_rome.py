from src.ke_backend import SimulatedEditableLLM, load_json, dump_json, contains, rss_mb
import time
DATA='data/custom_facts.json'
OUT='outputs/rome_results.json'

def main():
    records=load_json(DATA)
    model=SimulatedEditableLLM()
    rows=[]
    print('[Task 2] Single fact editing with ROME-style reset-per-case evaluation')
    for i,r in enumerate(records,1):
        model.reset()
        start=time.perf_counter(); mem0=rss_mb()
        model.edit([r])
        elapsed=time.perf_counter()-start; mem1=rss_mb()
        direct=model.generate(r['prompt'], records)
        rephrase=model.generate(r['rephrase_prompt'], records)
        locality=model.generate(r['locality_prompt'], records)
        row={'id':i,'prompt':r['prompt'],'target_new':r['target_new'],'direct_answer':direct,
             'rephrase_answer':rephrase,'locality_answer':locality,
             'efficacy':contains(direct,r['target_new']),'generalization':contains(rephrase,r['target_new']),
             'locality':contains(locality,r['locality_ground_truth']),'time_sec':round(elapsed,4),'peak_mem_mb':round(max(mem0,mem1),2)}
        rows.append(row)
        print(f'{i:02d}. target={r["target_new"]} | direct={direct} | rephrase={rephrase} | locality={locality}')
    dump_json(rows,OUT)
    print(f'Saved: {OUT}')

if __name__=='__main__': main()
