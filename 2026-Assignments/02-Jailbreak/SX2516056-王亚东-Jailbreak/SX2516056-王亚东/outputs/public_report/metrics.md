> ⚠️ Synthetic/mock data for format inspection only. Do not report as real experimental results.

| method             | split   |   n |   ASR_harmful |   FPR_benign |   avg_latency_s |   blocked_rate |
|:-------------------|:--------|----:|--------------:|-------------:|----------------:|---------------:|
| Baseline           | harmful |  40 |         0.15  |       nan    |           1.108 |          0     |
| Direct+NoDefense   | benign  | 100 |       nan     |         0    |           1.067 |          0     |
| GCG+LlamaGuard3    | harmful |  40 |         0.05  |       nan    |           6.246 |          0.825 |
| GCG+Perplexity     | harmful |  40 |         0.2   |       nan    |           5     |          0.575 |
| GCG+SmoothLLM      | harmful |  40 |         0.125 |       nan    |          14.166 |          0.65  |
| GCG-style          | harmful |  40 |         0.625 |       nan    |           4.93  |          0     |
| LlamaGuard3Defense | benign  | 100 |       nan     |         0.06 |           2.369 |          0.06  |
| PAIR+LlamaGuard3   | harmful |  40 |         0.075 |       nan    |           5.346 |          0.8   |
| PAIR+Perplexity    | harmful |  40 |         0.225 |       nan    |           3.607 |          0.45  |
| PAIR+SmoothLLM     | harmful |  40 |         0.175 |       nan    |          12.275 |          0.55  |
| PAIR-style         | harmful |  40 |         0.425 |       nan    |           3.409 |          0     |
| PerplexityDefense  | benign  | 100 |       nan     |         0.08 |           1.408 |          0.08  |
| SmoothLLMDefense   | benign  | 100 |       nan     |         0.11 |           5.757 |          0.11  |
