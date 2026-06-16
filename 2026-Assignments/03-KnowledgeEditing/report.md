# 实验报告：大模型知识编辑（Knowledge Editing for LLMs）

## 1. 实验目的

本实验围绕大语言模型知识编辑任务展开，目标包括：

1. 理解大语言模型中事实知识的参数化存储特征。
2. 基于 EasyEdit 实践单条知识编辑算法 `ROME`。
3. 基于 EasyEdit 实践批量知识编辑算法 `MEMIT`。
4. 通过三类指标对知识编辑效果进行评估：
   - 编辑成功率（Efficacy, ES）
   - 泛化性（Generalization, PS）
   - 局部性（Locality, NS）

## 2. 实验环境

- 操作系统：Windows
- Python：3.11
- 基础模型：`Qwen2.5-0.5B-Instruct`
- 知识编辑框架：`EasyEdit`
- 主要依赖：`torch`、`transformers`、`datasets`
- 运行设备：CUDA GPU（RTX 3060 Laptop GPU, 6GB）

## 3. 数据集构建

### 3.1 自定义 10 条单事实编辑样本

根据作业要求，自定义构建了 10 条事实更新样本，字段包括：

- `prompt`
- `target_new`
- `ground_truth`
- `rephrase_prompt`
- `locality_prompt`
- `locality_ground_truth`

样本主题包括：

- 企业 CEO
- 国家君主
- 足球俱乐部教练
- 国家首都

### 3.2 MEMIT 批量编辑集

`edit_memit.py` 支持优先从公开数据集中准备 500 条样本：

1. `zjunlp/KnowEdit` 的 `zsre` 子集
2. `zjunlp/KnowEdit` 的 `counterfact` 子集

实验最终使用本地缓存数据：

```text
data/memit_500.json
```

### 3.3 Bonus：跨语种泛化测试集

为了完成附加题，额外构建了 `data/crosslingual_edits.json`，用于测试：

- 用英文 prompt 注入知识
- 用英文 rephrase 测试同语种泛化
- 用中文 `zh_rephrase_prompt` 测试跨语种泛化

## 4. Task 1：Baseline Evaluation

首先在未编辑模型上测试自定义 10 条样本，验证模型是否存在旧知识或知识缺失。

### 4.1 Baseline 结果

- 总样本数：10
- 已命中新知识：0/10
- 命中旧知识：6/10

这说明：

1. 模型在编辑前并不会直接输出目标新知识。
2. 至少有 6 条样本能够明显激活模型中的旧知识表示。
3. 这些样本适合后续用于知识编辑实验。

### 4.2 Baseline 现象分析

在初始测试中，若采用 chat-template 风格生成，模型容易输出“无法访问实时信息”等免责声明。改为纯文本续写后，模型更容易暴露内部存储的事实知识，因此后续 baseline 采用纯文本补全方式。

## 5. Task 2：ROME 单条事实编辑

### 5.1 实验设置

采用 `ROME` 对 10 条样本逐条编辑：

- 每次编辑前重新实例化编辑器
- 每次只编辑 1 条事实
- 编辑完成后评估：
  - direct prompt
  - rephrase prompt
  - locality prompt

### 5.2 ROME 结果

汇总指标如下：

- ES：100.00%
- PS：76.57%
- NS：48.33%

### 5.3 ROME 结果分析

1. **编辑成功率很高**：
   `ROME` 在 10 条样本上的 direct prompt 成功率达到 100%，说明对目标事实的写入非常有效。

2. **泛化性较好但不完全稳定**：
   多数样本在改写 prompt 上也能保持较高正确率，但仍有个别样本 PS 偏低，说明知识写入并不总能自然泛化到所有同义表达。

3. **局部性存在明显波动**：
   部分样本 locality_acc 较低，甚至出现 0.0，说明 `ROME` 虽然能精准修改目标知识，但可能对无关事实造成干扰。

## 6. Task 3：MEMIT 批量知识编辑

### 6.1 实验设置

采用 `MEMIT` 做一次性批量编辑。脚本支持：

- 自动准备/缓存 500 条公开编辑样本
- 自动检测 GPU 并启用 CUDA
- 记录总耗时
- 记录峰值显存占用
- 保存逐样本结果

本实验最终使用的数据来源为：

```text
data/memit_500.json
```

实际运行环境为 RTX 3060 Laptop GPU（6GB 显存），批量编辑过程中峰值显存约为 3.6GB。

### 6.2 MEMIT 最终结果

最终 `MEMIT` 实验已经完整跑通，结果保存在：

```text
results/memit_results.json
results/metrics.json
```

汇总结果如下：

- 样本数：500
- 数据来源：`local cache: data/memit_500.json`
- ES：100.00%
- PS：39.90%
- NS：100.00%
- Pre rewrite acc：46.30%
- Pre rephrase acc：0.10%
- 总耗时：2083.756 s（约 34.73 分钟）
- 峰值显存：3603.19 MB

### 6.3 MEMIT 结果分析

1. **批量编辑成功率很高**：
   MEMIT 在 500 条样本上的 ES 达到 100%，说明目标知识已经被稳定写入模型，对 direct prompt 的编辑效果非常显著。

2. **泛化性明显弱于 direct prompt**：
   PS 只有 39.90%，显著低于 ES。这说明模型虽然能在原始编辑 prompt 上输出目标答案，但对改写问法的迁移能力有限。

3. **局部性指标表现理想**：
   当前实验中 NS 达到 100%，说明在现有 locality 测试设置下，没有观察到明显的无关知识破坏。

4. **运行代价可接受**：
   在 6GB 显存的 RTX 3060 Laptop GPU 上，500 条批量编辑可以稳定运行，峰值显存约 3.6GB，总耗时约 35 分钟，说明小规模模型上的 MEMIT 具有较好的可运行性。

### 6.4 结果解释说明

需要说明的是，当前 `memit_500.json` 原始格式并不完全包含 EasyEdit 直接所需的 `rephrase_prompt` 与 `locality_prompt` 字段，因此实验中对这两类评估输入做了适配构造。

因此：

- ES 可以较直接反映批量编辑是否成功；
- PS 和 NS 虽然可用于分析趋势，但在解释时应注明其基于适配后的评估输入，而非原生完整标注数据。

## 7. Task 4：综合评估

### 7.1 评估指标定义

- ES（Efficacy）：direct prompt 是否成功输出目标新知识。
- PS（Generalization）：rephrase prompt 是否也输出目标新知识。
- NS（Locality）：无关事实回答是否保持稳定。

### 7.2 当前汇总结果

#### Baseline

| 指标 | 结果 |
|---|---:|
| 命中新知识 | 0/10 |
| 命中旧知识 | 6/10 |

#### ROME

| 方法 | 样本数 | ES | PS | NS |
|---|---:|---:|---:|---:|
| ROME | 10 | 100.00% | 76.57% | 48.33% |

#### MEMIT

| 方法 | 样本数 | ES | PS | NS |
|---|---:|---:|---:|---:|
| MEMIT | 500 | 100.00% | 39.90% | 100.00% |

补充运行统计：

- 数据来源：`local cache: data/memit_500.json`
- 总耗时：2083.756 s（约 34.73 分钟）
- 峰值显存：3603.19 MB

## 8. 失败案例与现象分析

### 8.1 Baseline 阶段失败样本

个别样本在 baseline 中没有稳定输出旧知识，而是出现：

- 模板化续写
- 选择题式回答
- 无关文本延伸

这说明小模型在开放式补全下并不总能稳定激活其内部事实知识，因此需要在正式编辑前对样本进行筛选。

### 8.2 ROME 的局部性问题

虽然 ROME 的 ES 很高，但 NS 只有 48.33%，说明：

- 权重更新对目标知识有效
- 但可能对邻近知识表示产生副作用
- 这也是参数修改型知识编辑方法的典型代价之一

### 8.3 MEMIT 的泛化问题

虽然 MEMIT 在 500 条样本上达到了 100% 的 ES，并且 locality 指标也较高，但 PS 只有 39.90%。这说明：

- 批量知识已经被成功写入模型；
- 但这些知识并不总能自然泛化到改写问法；
- 小模型在复杂重述语义上的迁移能力仍然有限。

### 8.4 小模型局限性

由于本实验采用的是 `Qwen2.5-0.5B-Instruct`，模型规模较小，因此：

- baseline 容易出现续写漂移
- rephrase 泛化不如更大模型稳定
- locality 更容易受影响
- 跨语种知识迁移能力较弱

## 9. Bonus：跨语种泛化测试

为完成附加题“用英文向模型注入事实，再测试模型用中文提问时是否能输出正确结果”，本实验额外实现了：

- `bonus_crosslingual_rome.py`
- `bonus_crosslingual_memit.py`
- `bonus_crosslingual_evaluate.py`

使用 12 条代表性样本完成跨语种测试，最终结果如下：

- 样本数：12
- ES：100.00%
- PS_en：69.78%
- PS_zh：8.33%
- Cross-lingual gap：61.45%

结果说明：

1. 英文 direct prompt 上的知识编辑仍然成功；
2. 英文 paraphrase 有一定泛化能力；
3. 中文提问下的跨语种迁移能力明显较弱，说明参数化知识编辑后的知识并不会自动稳定迁移到另一种语言表达。

## 10. 结论

本实验成功完成了知识编辑作业的主要流程：

1. 构建自定义 10 条事实更新数据集。
2. 在原始模型上完成 baseline 评测，验证存在旧知识。
3. 基于 EasyEdit 跑通 ROME 单事实编辑，并得到较高 ES。
4. 基于 EasyEdit 跑通 MEMIT 的 500 条批量知识编辑。
5. 实现统一评估脚本，对 ES、PS、NS 进行量化统计。
6. 完成附加题中的跨语种泛化测试。

总体来看：

- `ROME` 在单事实 direct prompt 编辑上效果优秀；
- `MEMIT` 能够稳定完成大规模批量知识注入；
- 两种方法在 direct prompt 上都表现较好；
- 泛化性，尤其是跨语种泛化，仍然是小模型知识编辑中的明显短板。

## 11. 附录：运行命令

```bash
python baseline.py
python edit_rome.py
python edit_memit.py
python evaluate.py
python bonus_crosslingual_rome.py
python bonus_crosslingual_memit.py
python bonus_crosslingual_evaluate.py
```

