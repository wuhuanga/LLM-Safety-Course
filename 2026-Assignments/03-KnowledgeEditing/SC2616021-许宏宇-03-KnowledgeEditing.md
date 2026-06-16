# 知识编辑实验报告

## 实验设置

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 编辑库：EasyEdit
- 自定义数据：`data/custom_facts.json`，共 10 条人工构造的事实编辑样例
- MEMIT 数据：`zjunlp/KnowEdit` 中的 ZsRE benchmark，轻量实验取 10 条样例
- 主评估口径：EasyEdit 内部指标，即 `rewrite_acc`、`rephrase_acc`、`locality_acc`
- 补充评估口径：编辑后自由生成结果的严格字符串包含匹配

## 问题修正说明

初版实验结果与同类仓库差异较大，主要不是算法完全失效，而是评估口径和 EasyEdit 调用方式不一致。参考同类实现后，我修正了三点：

1. 在自定义事实数据中补充 `subject` 字段，避免 ROME 自动定位时把 `is` 一并纳入 subject。
2. 在 ROME 和 MEMIT 调用 EasyEdit 时显式传入 `rephrase_prompts` 和 `locality_inputs`，让 EasyEdit 原生计算 PS 与 NS。
3. 在 `evaluate.py` 中同时输出 EasyEdit 内部指标和自由生成严格匹配指标，避免把两种口径混在一起。

## 任务 1：基线评估

运行命令：

```powershell
python baseline.py --data data/custom_facts.json --output outputs/baseline.json
```

运行结果：

- 直接命中新事实：1 / 10
- 直接命中旧事实：0 / 10
- 改写提示命中新事实：0 / 10
- 局部性提示保留原答案：3 / 10

基线输出已保存到 `outputs/baseline.json`。未编辑模型只在 1 条样例中直接生成目标新答案，说明模型对这些自定义事实的掌握并不稳定。

终端输出截图：

![Task 1 Baseline](screenshots/task1_baseline.png)

## 任务 2：ROME 单事实编辑

运行命令：

```powershell
python edit_rome.py --data data/custom_facts.json --output outputs/rome_results.json
```

实验观察：

- 修正 subject 和 EasyEdit 参数后，ROME 在 10 条事实上的 EasyEdit 内部 ES 达到 100.00%。
- ROME 的 PS 为 55.83%，说明单事实编辑对改写提示有一定泛化，但仍明显弱于直接 prompt。
- ROME 的 NS 为 58.50%，说明局部性有一定保持，但小模型权重更新仍会影响部分邻域事实。
- 运行时间约 166.441 秒，硬件为 RTX 3060 Laptop GPU。

终端输出截图：

![Task 2 ROME](screenshots/task2_rome.png)

## 任务 3：MEMIT 批量编辑

原计划运行命令：

```powershell
python edit_memit.py --output outputs/memit_results.json
```

默认 MEMIT 实验会从 `zjunlp/KnowEdit` 使用 500 条 ZsRE benchmark 记录。在本地 Windows + RTX 3060 Laptop 6GB 环境下，完整 500 条实验进入了 covariance statistics 阶段，但 EasyEdit 需要下载和处理 Wikipedia covariance 数据，运行时间和资源开销过大，未能稳定完成。

因此，为了得到可复现的完整流程，我使用 10 条 ZsRE benchmark 样例，并采用轻量 identity covariance 近似完成 MEMIT 批量编辑：

```powershell
python edit_memit.py --limit 10 --output outputs/memit_results.json
```

实验观察：

- EasyEdit 内部指标中，MEMIT 的 ES 从约 32.39% 提升到 81.28%。
- MEMIT 的 PS 达到 75.94%，高于 ROME 的 55.83%，说明批量多层编辑对改写 prompt 的泛化更好。
- MEMIT 的 NS 为 21.08%，明显低于 ROME，说明当前轻量 covariance 设置下副作用较大。
- 10 条 benchmark 批量实验运行时间约 266.452 秒，峰值显存占用约 3017.22 MB。

失败案例示例：

| Prompt | 目标答案 | 编辑后自由生成 | 失败原因 |
| --- | --- | --- | --- |
| `Which family does Epaspidoceras belong to?` | `Noctuidae` | `Epaspidoceras belongs to the class Echinodermata...` | 自由生成输出了上位分类解释，未直接输出目标科名 |
| `What species is ZIC3 specific to?` | `male` | `Zic3 is a protein that is specifically associated with the immune system...` | 生成了概念解释，没有输出被注入的目标答案 |
| `What constellation is home to Butterfly Cluster?` | `Orion` | `The Butterfly Cluster is located in the constellation of Cygnus.` | 生成了相邻天文事实，且与目标答案冲突 |

终端输出截图：

![Task 3 MEMIT](screenshots/task3_memit.png)

## 任务 4：综合评估

运行命令：

```powershell
python evaluate.py --baseline outputs/baseline.json --rome outputs/rome_results.json --memit outputs/memit_results.json --output outputs/metrics.json
```

主表采用 EasyEdit 内部指标：

| 方法 | ES | PS | NS |
| --- | ---: | ---: | ---: |
| ROME | 100.00% | 55.83% | 58.50% |
| MEMIT | 81.28% | 75.94% | 21.08% |

补充表采用自由生成严格字符串匹配：

| 方法 | ES | PS | NS |
| --- | ---: | ---: | ---: |
| Baseline | 10.00% | 0.00% | 30.00% |
| ROME | 10.00% | 0.00% | 30.00% |
| MEMIT | 0.00% | 0.00% | 10.00% |

终端输出截图：

![Task 4 Evaluation](screenshots/task4_evaluate.png)

## 分析

对比修正前后的结果可以看出，初版指标偏低的主要原因有两个。第一，ROME/MEMIT 编辑时没有把 `rephrase_prompts` 和 `locality_inputs` 传给 EasyEdit，导致 PS/NS 无法按 EasyEdit 原生方式评估。第二，初版主表使用自由生成字符串包含作为唯一口径，而 Qwen2.5-Instruct 在开放式 prompt 下经常生成完整句子、解释性文字或相邻事实，这会让精确字符串命中率显著低于 EasyEdit 的 token 级评估结果。

从 EasyEdit 主指标看，ROME 的直接编辑成功率最高，适合单条事实精确修正；MEMIT 在轻量批量设置下 ES 略低于 ROME，但 PS 更高，说明批量多层编辑对改写提示更友好。MEMIT 的 NS 较低，主要与 10 条样例、identity covariance 近似和小模型容量有关。完整 500 条 MEMIT 实验若要稳定复现，需要更强的 Linux/CUDA 环境、更长运行时间，以及完整 covariance statistics 缓存。

总体来看，本实验完成了基线推理、ROME 单事实编辑、MEMIT 批量编辑和 ES/PS/NS 综合评估流程，并修正了初版评估口径问题。最终结果与同类实现采用了相同的 EasyEdit 内部指标口径，同时保留自由生成严格匹配作为补充分析。
