# 实验四：大模型知识编辑（Knowledge Editing for LLMs）

## 1. 实验背景与目的

随着大语言模型（LLM）的广泛应用，模型知识过期或包含错误事实（幻觉）的问题日益凸显。重新训练或全量微调成本极高，而知识编辑（Knowledge Editing）技术允许我们在不显著改变模型其他行为的前提下，精准、快速地修改模型内部的特定知识。

**实验目的：**

1. 深入理解大语言模型中事实知识的存储机制。
2. 掌握并实践主流的知识编辑算法（如 ROME, MEMIT）。
3. 理解知识编辑的三大核心评估指标：编辑成功率（Efficacy）、泛化性（Generalization）与局部性/特异性（Locality）。

------

## 2. 实验环境与工具准备

- **基础模型：**`Qwen2.5-0.5B`等模型即可。
- **核心框架：** 推荐使用开源框架 [EasyEdit](https://github.com/zjunlp/EasyEdit)（由浙江大学开源，集成了主流编辑算法）。

------

## 3. 实验任务详解

### Task 1: 基础环境搭建与基线测试 (Baseline Evaluation)

在不进行任何编辑操作的情况下，测试模型对特定过时知识或错误事实的回答。

- **任务内容：**
  - 构建包含 10 条事实更新的数据集（例如：“现任推特/X公司的CEO是谁？” -> 目标答案：“Linda Yaccarino”）。
  - 编写推理脚本，记录原始模型的回答，证明模型在编辑前确实存在知识盲区或旧知识。

### Task 2: 单条事实编辑实践 (Single Fact Editing using ROME)

ROME (Rank-One Model Editing) 是一种经典的通过修改前馈神经网络（FFN）特定层权重来实现单条知识编辑的方法。

- **任务内容：**
  - 基于 `EasyEdit` 框架，配置 ROME 算法参数。
  - 将 Task 1 中的 10 条事实逐一进行编辑（每次重置模型权重）。
  - **验证：** 输入提示词“推特现任CEO是”，观察模型是否输出“Linda Yaccarino”。

### Task 3: 批量知识编辑实践 (Batch Editing using MEMIT)

MEMIT 算法在 ROME 的基础上进行了扩展，支持一次性向模型中注入成百上千条知识，同时保持较低的性能损耗。

- **任务内容：**
  - 选取开源知识编辑数据集 [ZsRE](https://www.google.com/search?q=https://github.com/zjunlp/EasyEdit/tree/main/data) 或 [CounterFact](https://rome.baulab.info/data/dsets/) 等其他数据集中的 500 条数据作为批量编辑集。
  - 使用 MEMIT 算法对模型进行一次性批量知识注入。
  - 记录批量编辑过程的显存占用和耗时情况。

### Task 4: 综合评估 (Comprehensive Evaluation)

计算以下三个核心指标：

- **编辑成功率 (Efficacy, ES)：** 模型对直接编辑的 Prompt 是否输出了目标答案。
- **泛化性 (Generalization, PS)：** 模型对编辑事实的**同义改写** Prompt 是否能输出目标答案。（例如：“谁目前在管理推特？”）
- **局部性 (Locality, NS)：** 模型对**无关事实**的回答是否受到了破坏。（例如：编辑了推特的CEO，模型对“苹果的CEO是谁”的回答是否依然正确）。

------

## 4. 数据集与文件格式说明

请按照以下 JSON 格式构建你的自定义测试集（用于 Task 1 和 Task 2）：

JSON

```json
[
  {
    "prompt": "The current CEO of Twitter (X) is",
    "target_new": "Linda Yaccarino",
    "ground_truth": "Elon Musk",
    "rephrase_prompt": "Who is the chief executive officer of X (formerly Twitter)?",
    "locality_prompt": "The current CEO of Apple is",
    "locality_ground_truth": "Tim Cook"
  }
]
```

------

## 5. 进阶挑战 (Bonus - 选做)

完成基础任务后，鼓励有余力的同学探索以下方向之一（占比总分 10% 的附加分）：

1. **跨语种泛化测试：** 用英文向模型注入事实（如 A 的首都是 B），测试模型用中文提问时（A的首都是哪里？），是否能输出正确结果。
2. **多模态扩展探讨：** 结合课程之前的内容，尝试将简单的文本编辑思路扩展到小型多模态大模型（如 LLaVA），并给出可行性分析报告。
3. **黑盒知识编辑（RAG 对比）：** 针对同样的 500 条数据，不修改模型参数，而是构建一个简单的向量检索库（RAG）。对比“参数化编辑(MEMIT)”与“非参数化编辑(RAG)”在响应延迟、准确率和幻觉消除上的优劣。

------

## 6. 提交要求 (Deliverables)

1. **实验代码 (GitHub Repo / 压缩包)：** 需包含完整的可执行脚本（`baseline.py`, `edit_rome.py`, `edit_memit.py`, `evaluate.py`）。
   - 必须提供清晰的 `requirements.txt` 和 `README.md`（说明如何运行代码）。
2. **实验报告 (PDF)：**
   - 包含 Task 1~4 的终端输出截图。
   - 以表格形式汇总 Task 4 中的评估指标数据（ES, PS, NS 的百分比）。
   - 对 MEMIT 批量编辑的效果进行结果分析与失败案例总结。
