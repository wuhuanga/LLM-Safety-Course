# 大模型知识编辑实验报告

课程方向：方向 03 - 知识编辑（Knowledge Editing for LLMs）  
实验日期：2026-05-21  
基础模型：`Qwen/Qwen2.5-0.5B-Instruct`  
编辑框架：EasyEdit  
编辑算法：ROME、MEMIT  

## 1. 实验目标

本实验针对大语言模型中的事实知识更新问题，验证在不重新训练完整模型的情况下，能否使用知识编辑算法修改模型对特定事实的回答。根据方向 03 的任务要求，实验完成以下内容：

1. 构建 10 条事实更新数据集，并在未编辑模型上进行 Baseline Evaluation。
2. 使用 ROME 对 10 条事实逐条进行单事实编辑。
3. 使用 MEMIT 对 CounterFact 中 500 条数据进行批量知识编辑。
4. 计算 Efficacy、Generalization、Locality 三类指标，并结合终端输出和失败案例分析实验效果。

## 2. 实验环境

| 项目 | 配置 |
| --- | --- |
| 运行平台 | AutoDL GPU 容器 |
| Python | 3.10 |
| 基础模型 | `/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct` |
| Transformers | 4.45.2 |
| Accelerate | 0.34.2 |
| PEFT | 0.13.2 |
| 编辑框架 | EasyEdit |
| 主要脚本 | `baseline.py`, `edit_rome.py`, `edit_memit.py`, `evaluate.py` |

实验过程中对 Qwen2.5 与 EasyEdit 做了必要适配：

- `mlp_module_tmp` 设置为 `model.layers.{}.mlp.down_proj`。
- `lm_head_module` 设置为 `model.embed_tokens`，避免 Qwen2.5 权重共享导致 `LookupError: lm_head.weight`。
- 对 EasyEdit 的 `nethook.py` 执行补丁，修复当前 PyTorch 下 forward hook 参数顺序不匹配导致的 `Tensor + dict` 报错。
- MEMIT 的二阶矩统计数据由 `wikipedia` 改为 `wikitext`，并将 `mom2_n_samples` 调小，以适应 AutoDL 磁盘限制。

## 3. 数据集

### 3.1 自定义事实数据集

Task 1 和 Task 2 使用 `data/facts_10.json` 中的 10 条事实更新数据。每条记录包含：

- `prompt`：直接编辑提示词。
- `target_new`：希望模型输出的新事实。
- `ground_truth`：旧事实或过期事实。
- `rephrase_prompt`：同义改写提示词，用于测试泛化性。
- `locality_prompt` 与 `locality_ground_truth`：无关事实，用于测试局部性。

| ID | Subject | Target New | Ground Truth |
| --- | --- | --- | --- |
| `disney_ceo_2026` | Disney | Josh D'Amaro | Bob Iger |
| `us_president_2025` | the United States | Donald Trump | Joe Biden |
| `canada_pm_2025` | Canada | Mark Carney | Justin Trudeau |
| `uk_pm_2024` | the United Kingdom | Keir Starmer | Rishi Sunak |
| `mexico_president_2024` | Mexico | Claudia Sheinbaum | Andres Manuel Lopez Obrador |
| `intel_ceo_2025` | Intel | Lip-Bu Tan | Pat Gelsinger |
| `starbucks_ceo_2024` | Starbucks | Brian Niccol | Laxman Narasimhan |
| `nike_ceo_2024` | Nike | Elliott Hill | John Donahoe |
| `boeing_ceo_2024` | Boeing | Kelly Ortberg | Dave Calhoun |
| `spotify_ceo_2025` | Spotify | Gustav Soderstrom and Alex Norstrom | Daniel Ek |

### 3.2 MEMIT 批量编辑数据

Task 3 使用 CounterFact 数据集前 500 条作为批量编辑集。脚本 `edit_memit.py` 会将 CounterFact 中的 `requested_rewrite` 转换为统一格式，并使用固定 locality prompt：

```text
The capital city of France is
```

对应 locality ground truth 为：

```text
Paris
```

## 4. 实验流程与终端输出

### 4.1 Task 1: Baseline Evaluation

运行命令：

```bash
python baseline.py \
  --data data/facts_10.json \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct \
  --local-files-only
```

输出文件：

```text
results/baseline_20260521_151151.json
```

终端输出截图：

![Task 1 Baseline terminal output](assets/terminal_outputs/terminal_output_02.png)

Task 1 记录了原始模型在编辑前对 10 条事实的回答。例如在 Intel CEO 样例中，目标答案为 `Lip-Bu Tan`，但原始模型输出仍包含旧事实 `Pat Gelsinger`，说明模型存在过期知识。

### 4.2 Task 2: ROME 单条事实编辑

运行命令：

```bash
python edit_rome.py \
  --data data/facts_10.json \
  --hparams hparams/ROME/qwen2.5-0.5b.yaml \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct
```

输出文件：

```text
results/rome_20260521_151411.json
```

终端输出截图：

![Task 2 ROME terminal output](assets/terminal_outputs/terminal_output_03.png)

ROME 对 10 条事实逐条编辑，每条事实均重新构造 EasyEdit editor。终端输出中可以看到 EasyEdit 对最后一条 Spotify CEO 事实的编辑过程和 `rewrite_acc` 指标。

### 4.3 Task 3: MEMIT 批量编辑

运行命令：

```bash
python edit_memit.py \
  --source counterfact \
  --limit 500 \
  --hparams hparams/MEMIT/qwen2.5-0.5b.yaml \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct
```

输出文件：

```text
results/memit_20260521_151545.json
```

终端输出截图：

![Task 3 MEMIT terminal output](assets/terminal_outputs/terminal_output_04.png)

截图显示 MEMIT 已完成 500 条编辑结果评估，并写入 `results/memit_20260521_151545.json`。

### 4.4 Task 4: 综合评估

运行命令：

```bash
python evaluate.py \
  results/baseline_*.json \
  results/rome_*.json \
  results/memit_*.json \
  --output results/evaluation_real.json \
  --csv results/metrics_real.csv
```

输出文件：

```text
results/evaluation_real.json
results/metrics_real.csv
```

终端输出截图：

![Task 4 Evaluation terminal output](assets/terminal_outputs/terminal_output_05.png)

## 5. 评估指标

本实验按照任务要求计算三个核心指标：

| 指标 | 英文 | 计算方式 |
| --- | --- | --- |
| 编辑成功率 | Efficacy, ES | 直接编辑 prompt 的输出是否包含 `target_new` |
| 泛化性 | Generalization, PS | 同义改写 prompt 的输出是否包含 `target_new` |
| 局部性 | Locality, NS | 无关 locality prompt 的输出是否仍包含 `locality_ground_truth` |

需要说明的是，`evaluate.py` 使用自由生成文本中的字符串包含关系计算 ES、PS、NS。EasyEdit 内部的 `rewrite_acc` 更接近编辑优化阶段的目标 token 概率/准确率，二者并不完全等价。

## 6. 实验结果

### 6.1 ES / PS / NS 指标

最新真实结果来自 `results/metrics_real.csv`：

| Run | Count | ES (%) | PS (%) | NS (%) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 10 | 10.0 | 0.0 | 40.0 |
| ROME | 10 | 10.0 | 0.0 | 40.0 |
| MEMIT | 500 | 3.2 | 1.4 | 100.0 |

### 6.2 运行时间与显存占用

| Run | Count | Time (s) | RSS Memory (GB) | CUDA Allocated (GB) | CUDA Reserved (GB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 10 | 15.978 | 1.641 | 0.932 | 0.936 |
| ROME | 10 | 50.671 | 4.138 | 3.791 | 4.275 |
| MEMIT | 500 | 3011.054 | 2.246 | 2.849 | 3.402 |

### 6.3 EasyEdit 内部指标

| Run | Count | EasyEdit Post Rewrite Acc |
| --- | ---: | ---: |
| ROME | 10 | 100.0% |
| MEMIT | 500 | 23.1% |

ROME 的 EasyEdit 内部 `post.rewrite_acc` 达到 100.0%，说明编辑优化阶段可以将目标答案概率推高。但自由生成评估仍可能输出解释性文本、选择题文本或旧事实，因此字符串匹配形式的 ES/PS 不一定同步提高。

## 7. 结果分析

### 7.1 Baseline 分析

Baseline 的 ES 为 10.0%，说明原始 Qwen2.5-0.5B-Instruct 对大多数新事实没有直接输出目标答案，存在旧知识或知识盲区。例如：

- Intel CEO 的目标答案是 `Lip-Bu Tan`，原始模型倾向输出 `Pat Gelsinger`。
- Starbucks CEO 的目标答案是 `Brian Niccol`，原始模型倾向输出 `Howard Schultz`。
- Canada prime minister 的目标答案是 `Mark Carney`，原始模型倾向输出 `Justin Trudeau`。

这些结果证明编辑前模型确实存在过期知识或事实盲区。

### 7.2 ROME 分析

ROME 在 EasyEdit 内部评价中表现较好，10 条事实的 `post.rewrite_acc` 平均为 100.0%。但 `evaluate.py` 基于自由生成文本做字符串匹配时，ROME 的 ES 为 10.0%、PS 为 0.0%、NS 为 40.0%。

这种差异主要来自以下原因：

- Qwen2.5-0.5B 参数规模较小，开放式生成稳定性有限。
- ROME 优化目标更偏向目标 token 概率，而不是保证自由生成一定完整输出目标字符串。
- 部分 prompt 会诱发模型输出选择题式内容或长解释，导致字符串包含判断失败。
- 一些 locality prompt 在原始模型上就回答错误，因此 ROME 的 NS 与 Baseline 相同。

### 7.3 MEMIT 分析

MEMIT 完成了 CounterFact 前 500 条批量编辑，运行时间约 3011 秒，CUDA reserved memory 约 3.402 GB。由于 AutoDL 磁盘空间限制，本实验将二阶矩统计数据从 `wikipedia` 改为 `wikitext`，并降低 `mom2_n_samples`，这保证实验可运行，但也可能降低协方差估计质量。

MEMIT 的 ES 为 3.2%，PS 为 1.4%，NS 为 100.0%。这说明在本实验配置下，MEMIT 对自由生成输出的直接命中率较低，但 locality prompt 没有被破坏。EasyEdit 内部 `post.rewrite_acc` 为 23.1%，高于自由生成字符串匹配的 ES，但仍明显低于 ROME。

可能原因包括：

- 500 条批量编辑之间存在相互干扰。
- CounterFact 中部分 prompt 很短或语义不完整，例如 `Toko Yasuda, the`。
- 使用较小的 Qwen2.5-0.5B 模型时，生成能力和事实表达能力有限。
- 由于磁盘限制，本实验使用较小的 `wikitext` 统计量，MEMIT 的协方差估计可能不充分。

## 8. MEMIT 失败案例

| Case | Prompt | Target | Model Output 摘要 | 失败类型 |
| --- | --- | --- | --- | --- |
| `counterfact_2` | `Toko Yasuda, the` | `piano` | 输出 “largest retailer of electronics in Japan...” | Prompt 过短，续写偏离目标关系 |
| `counterfact_3` | `Autonomous University of Madrid, which is located in` | `Sweden` | 输出 “the heart of the city of Madrid...” | 旧知识/常识强干扰 |
| `counterfact_4` | `What is the twin city of Lyon? It is` | `Manila` | 输出 “a question that has puzzled many people...” | 未直接回答目标实体 |
| `counterfact_5` | `The mother tongue of Thomas Joannes Stieltjes is` | `English` | 输出 “Dutch...” | 旧事实保持 |
| `counterfact_6` | `Anaal Nathrakh, that was created in` | `Philadelphia` | 输出 “1970 by the Indian government...” | 幻觉续写 |

## 9. 结论

本实验完成了方向 03 的四项任务：

- 构建 10 条事实更新数据集，并完成 Baseline Evaluation。
- 基于 EasyEdit 完成 ROME 单事实编辑。
- 基于 CounterFact 前 500 条完成 MEMIT 批量编辑。
- 汇总 ES、PS、NS 指标，并加入 Task 1-4 的终端输出截图。

实验结果表明，知识编辑在工程上可行，但在小模型、开放式生成评价、批量编辑和受限统计量估计条件下，编辑效果会明显受限。ROME 在内部编辑指标上表现较好，而 MEMIT 在 500 条批量编辑中保持了较好的 locality，但 efficacy 和 generalization 较低。后续可使用更大模型、完整 `wikipedia` 统计量、更适合 Qwen2.5 的 hparams，以及概率型评价指标来进一步提升实验效果。
