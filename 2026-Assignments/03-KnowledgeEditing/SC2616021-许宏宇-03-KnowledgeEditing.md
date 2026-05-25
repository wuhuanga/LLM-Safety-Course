# 知识编辑实验报告

## 实验设置

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 编辑库：EasyEdit
- 自定义数据：`data/custom_facts.json`，共 10 条人工构造的事实编辑样例
- 评估指标：
  - ES：直接编辑提示的生成结果是否包含 `target_new`
  - PS：改写提示的生成结果是否包含 `target_new`
  - NS：局部性提示是否保留 `locality_ground_truth`

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

基线输出已保存到 `outputs/baseline.json`。在当前的字符串包含式评估口径下，未编辑模型只在 1 条样例中直接生成了目标新答案，说明模型对这些自定义事实的掌握并不稳定。

终端输出截图：

![Task 1 Baseline](screenshots/task1_baseline.png)

## 任务 2：ROME 单事实编辑

运行命令：

```powershell
python edit_rome.py --data data/custom_facts.json --output outputs/rome_results.json
```

实验观察：

- EasyEdit 日志中，每条 ROME 单事实编辑的 rewrite accuracy 都提升到了 1.0。
- 但在后续自由生成评估中，精确字符串命中仍然较低：直接提示命中新事实 1 / 10，改写提示命中新事实 0 / 10，局部性提示保留原答案 3 / 10。
- 这说明 ROME 已经提高了目标 token 的编辑概率，但这种 token 层面的编辑成功不一定会稳定转化为短答案生成中的精确字符串命中。
- 运行时间：130.391 秒，硬件为 RTX 3060 Laptop GPU。

终端输出截图：

![Task 2 ROME](screenshots/task2_rome.png)

## 任务 3：MEMIT 批量编辑

原计划运行命令：

```powershell
python edit_memit.py --output outputs/memit_results.json
```

默认 MEMIT 实验会从 `zjunlp/KnowEdit` 使用 500 条 ZsRE benchmark 记录。在本地 Windows + RTX 3060 Laptop 6GB 环境下，完整 500 条实验进入了 covariance statistics 阶段，但 EasyEdit 需要下载和处理 Wikipedia covariance 数据，运行时间和资源开销过大，未能稳定完成。

因此，为了得到可复现的完整流程，我使用了 10 条 benchmark 样例，并采用轻量 identity covariance 近似完成 MEMIT 批量编辑：

```powershell
python edit_memit.py --limit 10 --output outputs/memit_results.json
```

实验观察：

- EasyEdit 日志中，MEMIT 的 rewrite accuracy 从 0.3239 提升到 0.8128。
- 在自由生成评估中，直接提示命中新事实 0 / 10，改写提示命中新事实 0 / 10，局部性提示保留原答案 1 / 10。
- 10 条 benchmark 批量实验运行时间为 181.531 秒，峰值显存占用约 3017.22 MB。
- 主要失败模式是：编辑后目标答案的 token 概率提高了，但模型自由生成时容易输出较长解释、相邻事实或不包含精确目标字符串的答案。

失败案例示例：

| Prompt | 目标答案 | 编辑后生成 | 失败原因 |
| --- | --- | --- | --- |
| `Which family does Epaspidoceras belong to?` | `Noctuidae` | `Epaspidoceras belongs to the class Echinodermata...` | 生成了上位分类解释，未命中目标科名 |
| `What species is ZIC3 specific to?` | `male` | `Zic3 is a protein that is specifically associated with the immune system...` | 生成了概念解释，没有输出被注入的目标答案 |
| `What constellation is home to Butterfly Cluster?` | `Orion` | `The Butterfly Cluster is located in the constellation of Cygnus.` | 生成了相邻天文事实，且与目标答案冲突 |

终端输出截图：

![Task 3 MEMIT](screenshots/task3_memit.png)

## 任务 4：综合评估

运行命令：

```powershell
python evaluate.py --baseline outputs/baseline.json --rome outputs/rome_results.json --memit outputs/memit_results.json --output outputs/metrics.json
```

| 方法 | ES | PS | NS |
| --- | ---: | ---: | ---: |
| Baseline | 10.00% | 0.00% | 30.00% |
| ROME | 10.00% | 0.00% | 30.00% |
| MEMIT | 0.00% | 0.00% | 10.00% |

终端输出截图：

![Task 4 Evaluation](screenshots/task4_evaluate.png)

## 分析

从 EasyEdit 的内部 rewrite accuracy 看，ROME 和 MEMIT 都能提高目标事实的编辑成功率，说明知识编辑算法确实改变了模型在目标位置上的预测倾向。但是，从本实验使用的自由生成字符串匹配指标看，编辑后的 ES、PS 和 NS 并没有明显提升。

造成这种差异的原因主要有三点。第一，字符串包含式评估比较严格，只要模型生成了较长解释、同义表达、部分名称或相邻事实，就可能被判为失败。第二，Qwen2.5-Instruct 模型在回答开放式 prompt 时倾向于生成完整句子，而不是只输出目标实体，这会降低精确字符串命中率。第三，MEMIT 在本地资源限制下使用了 10 条样例和 identity covariance 近似，不能完全代表 500 条完整 covariance MEMIT 实验的效果。

总体来看，当前实验完成了基线推理、ROME 单事实编辑、MEMIT 批量编辑和 ES/PS/NS 评估流程。ROME 在 EasyEdit 内部指标上表现稳定，但对 paraphrase prompt 的泛化仍不足。MEMIT 在轻量设置下可以运行，但完整 500 条实验需要更强的 Linux/CUDA 环境、更长运行时间，以及完整的 covariance statistics 缓存。
