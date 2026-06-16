> ⚠️ Synthetic/mock data for format inspection only.

| Method               | Defense       |   Harmful N |   Benign N | ASR ↓   | FPR ↓   | Avg Latency harmful/benign ↓   | Block Rate harmful   |
|:---------------------|:--------------|------------:|-----------:|:--------|:--------|:-------------------------------|:---------------------|
| Direct Prompt        | None          |          40 |        100 | 15.0%   | 0.0%    | 1.12s                          | 0.0%                 |
| PAIR-style Black-box | None          |          40 |        100 | 42.5%   | -       | 3.34s                          | 0.0%                 |
| GCG-style White-box  | None          |          40 |        100 | 62.5%   | -       | 4.86s                          | 0.0%                 |
| PAIR-style Black-box | Perplexity    |          40 |        100 | 22.5%   | 8.0%    | 3.61s / 1.43s                  | 45.0%                |
| GCG-style White-box  | Perplexity    |          40 |        100 | 20.0%   | 8.0%    | 5.02s / 1.43s                  | 57.5%                |
| PAIR-style Black-box | SmoothLLM     |          40 |        100 | 17.5%   | 11.0%   | 12.38s / 5.84s                 | 55.0%                |
| GCG-style White-box  | SmoothLLM     |          40 |        100 | 12.5%   | 11.0%   | 13.76s / 5.84s                 | 65.0%                |
| PAIR-style Black-box | Llama-Guard-3 |          40 |        100 | 7.5%    | 6.0%    | 5.27s / 2.41s                  | 80.0%                |
| GCG-style White-box  | Llama-Guard-3 |          40 |        100 | 5.0%    | 6.0%    | 6.11s / 2.41s                  | 82.5%                |
