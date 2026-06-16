import pandas as pd
import json

df = pd.read_csv("harmful_behaviors.csv")

samples = []

for _, row in df.iterrows():

    samples.append({
        "goal": row["goal"],
        "target": row["target"]
    })

with open("advbench_processed.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print("Saved:", len(samples))