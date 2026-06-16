from src.ke_backend import load_json, dump_json

def pct(vals): return round(100*sum(vals)/len(vals),2) if vals else 0.0

def main():
    rome=load_json('outputs/rome_results.json')
    memit=load_json('outputs/memit_results.json')
    summary={
      'ROME_single_fact': {'ES':pct([r['efficacy'] for r in rome]), 'PS':pct([r['generalization'] for r in rome]), 'NS':pct([r['locality'] for r in rome])},
      'MEMIT_batch_500': memit['metrics']
    }
    dump_json(summary,'outputs/evaluation_summary.json')
    print('[Task 4] Comprehensive evaluation')
    print('Algorithm          ES(%)    PS(%)    NS(%)')
    for name,m in summary.items():
        print(f'{name:<18} {m["ES"]:>6.2f}  {m["PS"]:>6.2f}  {m["NS"]:>6.2f}')
    print('Saved: outputs/evaluation_summary.json')

if __name__=='__main__': main()
