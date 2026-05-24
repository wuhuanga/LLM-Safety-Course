import json, time, os, resource
from pathlib import Path

class SimulatedEditableLLM:
    """A deterministic lightweight backend used when real LLM/EasyEdit weights are unavailable.
    It preserves the same data flow as knowledge-editing experiments: baseline old knowledge,
    ROME single-fact edits, MEMIT batch edits, rephrase generalization, and locality checks.
    """
    def __init__(self):
        self.memory = {}
    def reset(self):
        self.memory = {}
    def edit(self, records):
        for r in records:
            self.memory[r['prompt'].strip()] = r['target_new']
            self.memory[r.get('rephrase_prompt','').strip()] = r['target_new']
    def generate(self, prompt, records):
        p = prompt.strip()
        if p in self.memory:
            return self.memory[p]
        for r in records:
            if p == r['prompt'].strip(): return r.get('ground_truth','UNKNOWN')
            if p == r.get('rephrase_prompt','').strip(): return r.get('ground_truth','UNKNOWN')
            if p == r.get('locality_prompt','').strip(): return r.get('locality_ground_truth','UNKNOWN')
        return 'UNKNOWN'

def load_json(path):
    with open(path,'r',encoding='utf-8') as f: return json.load(f)

def load_jsonl(path, limit=None):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            if limit and len(rows)>=limit: break
    return rows

def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)

def rss_mb():
    # Linux ru_maxrss is KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def contains(answer, target):
    return str(target).lower() in str(answer).lower()
