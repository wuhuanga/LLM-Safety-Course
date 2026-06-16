# Mock Case Studies（仅用于格式示例）

> ⚠️ 以下为 synthetic/mock case study，不包含真实危险 prompt、真实模型危险输出或真实 GCG suffix。

## Case 1：黑盒攻击成功

| 字段 | 内容 |
|---|---|
| 行为编号 | JBB-FRD-004 |
| 类别 | fraud_scam |
| 攻击方法 | PAIR-style |
| 迭代轮次 | 3 |
| Prompt hash | 9b7a3a1f2ddc4e11 |
| Prompt | `[已脱敏：synthetic mock prompt placeholder]` |
| Response | `[已脱敏：synthetic mock unsafe response placeholder]` |
| Llama-Guard-3 判定 | unsafe |
| 人工复核 | confirmed |

分析：该 mock 样本用于展示黑盒迭代日志的写法。报告中只保留行为编号、类别、轮次和 hash，不展示可复现的危险内容。

## Case 2：白盒攻击成功

| 字段 | 内容 |
|---|---|
| 行为编号 | JBB-CYB-007 |
| 类别 | cyber_abuse |
| 攻击方法 | GCG-style |
| 迭代步数 | 50 |
| Suffix hash | d9348bb721f8a6c2 |
| Prompt | `[已脱敏：synthetic mock prompt placeholder]` |
| Response | `[已脱敏：synthetic mock unsafe response placeholder]` |
| Llama-Guard-3 判定 | unsafe |
| 人工复核 | confirmed |

分析：该 mock 样本展示白盒攻击 case study 的字段。真实报告中可以描述 loss 下降和 suffix hash，但不要公开 suffix 本体。

## Case 3：攻击被防御拦截

| 字段 | 内容 |
|---|---|
| 行为编号 | JBB-WPN-002 |
| 类别 | weapons |
| 攻击方法 | GCG-style |
| 防御 | Perplexity Detection |
| Prompt hash | 5d4e01c739aa2aef |
| 拦截依据 | perplexity_score=142.7 > threshold=95.0 |
| Response | `[未生成：blocked before generation]` |
| 人工复核 | confirmed |

分析：该 mock 样本展示防御拦截 case study 的字段。Perplexity 检测适合解释非自然 suffix 被拦截的情况。
