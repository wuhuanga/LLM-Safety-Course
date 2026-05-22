# 记忆增强型文本交互智能体系统

**学号**：SX2524022　　**姓名**：王宣　　**日期**：2026-05-22

**GitHub 仓库**：[sillypro/llm-agent-memory](https://github.com/sillypro/llm-agent-memory)　　**演示视频**：[百度网盘](https://pan.baidu.com/s/1NdKutqSYzhhUP3kSPiRgGg?pwd=1111)（提取码：1111）

---

## 一、项目概述

本项目在 **ScienceWorld**（Allen AI，2022）文本模拟环境中，基于 **DeepSeek-Chat** 大语言模型构建了一个记忆增强型智能体。智能体采用 **ReAct**（Reason + Act）推理框架，通过纯文字交互完成物理化学实验任务。

ScienceWorld 共包含 30 类科学实验任务，每类有多个随机变体。每步交互中，环境提供当前观测文字、合法动作列表和得分，智能体从列表中选择一个动作执行，满分 100 分。本项目在 4 类任务（boil、freeze、find-animal、melt）的各 2 个变体上进行测试，共 8 个任务变体。

**核心问题**：裸 LLM 在 ScienceWorld 中面临三个挑战：不知道物品位置（冷启动）、反复执行无效动作（重复失败）、盲目探索消耗步数（步数浪费）。本项目通过引入双层记忆模块解决上述问题。

---

## 二、系统设计

### 2.1 整体架构

系统由四个模块组成，每步按以下顺序运行：

![系统整体架构图](images/architecture.png)

**数据流**：ScienceWorld 环境每步返回观测、合法动作和得分，Prompt Builder 将其与记忆模块检索到的历史经验组装成完整提示词，DeepSeek LLM 输出 Thought + Action 文字，Action Parser 提取并校验动作后提交给环境执行。执行结果同时反馈给记忆模块完成更新。

### 2.2 ReAct 推理框架

每步 LLM 先输出思考过程（Thought），再给出动作（Action），动作必须从环境提供的合法列表中逐字选取。系统提示包含 12 条执行规则，核心规则包括：`focus on [substance]` 只能对目标物质使用一次（错误使用扣 100 分）、液体收集使用 `dunk cup into [source]`、加热需反复激活炉子约 20 次等。

### 2.3 记忆模块

记忆模块分短期和长期两层，分别处理不同时间跨度的信息。

**短期记忆**维护大小为 10 的滑动窗口，记录最近 10 步的动作、结果（success / failed / neutral）和得分变化。每步执行后立即更新，注入提示词的 `[Recent History]` 区块，帮助 LLM 了解当前进展，避免原地打转。

**长期记忆**以 JSON 文件持久化，存储四类信息：

| 类型 | 内容 | 更新时机 |
|------|------|---------|
| 成功序列 | 历史成功的完整动作路径（每任务最多5条） | 任务成功完成后 |
| 失败动作 | 反复失败的动作及次数 | 每次失败时计数 |
| 环境事实 | 物品位置、操作技巧等具体知识 | 每次得分提升时 |
| 跨任务通用知识 | 适用于所有任务的通用规则 | 预置 |

**RAG 检索**：每步调用 `Memory.retrieve(task, obs)`，依次取出该任务的历史成功序列（最多2条）、高频失败动作、与当前观测关键词匹配的环境事实（最多3条）、以及跨任务通用知识（前3条），拼接后注入提示词的 `[Past Experience]` 区块。检索全程在内存中完成，延迟 < 2 ms。

以下是 boil 任务实际注入的记忆示例：

```
[Past Experience]
Past successful sequences for 'boil':
  focus on substance in toilet → dunk cup into toilet
  → move cup containing water to stove → activate stove

Actions that repeatedly failed: ['focus on sink', 'dunk toilet into cup']

Relevant facts:
  - Water is found in the bathroom toilet.
  - Use 'dunk cup into toilet' to collect water.
  - Repeatedly activate the stove (~20 turns).

General knowledge:
  - Bathroom is accessed from Kitchen via 'open door to bathroom'.
  - focus on [substance]: ONLY ONCE, NEVER on containers (-100 penalty).
```

### 2.4 错误恢复机制

| 机制 | 触发条件 | 作用 |
|------|---------|------|
| 连续失败警告 | 连续失败 ≥ 2 步 | 在提示词中列出失败动作，促使 LLM 换方向 |
| 动作对齐 | LLM 输出与有效列表不完全匹配 | 对 20+ 字符动作做前缀匹配，容错细微差异 |
| 自动消歧 | 环境返回"Ambiguous request" | 自动选择第一个匹配项 |
| 加热动作豁免 | activate stove / freezer 等 | 不计入失败计数，避免误判正常的重复激活 |

---

## 三、测试任务

本项目测试了 4 类任务各 2 个变体，共 8 个任务变体。**变体之间的主要差异**为起始房间不同：变体 0 通常从走廊出发（紧邻厨房），变体 1 从美术室出发（需额外 3-4 步导航才能到达厨房）。这一差异在 80 步限制下对任务完成率有显著影响。

| 任务 | 变体 | 目标 | 关键步骤 | 变体差异 |
|------|------|------|---------|---------|
| boil | 0 | 将马桶中的水烧开 | focus → dunk → 放炉子 → 激活×20 | 从走廊出发 |
| boil | 1 | 同上 | 同上 | 从美术室出发，导航多 4 步 |
| freeze | 0 | 将水冷冻为固态 | focus → 放冰箱 → wait | 从走廊出发 |
| freeze | 1 | 同上 | 同上 | 从美术室出发 |
| find-animal | 0 | 找到动物放入**红色**箱子 | 探索房间 → focus → move to red box | 从走廊出发 |
| find-animal | 1 | 找到动物放入**绿色**箱子 | 同上，目标容器颜色不同 | 从美术室出发 |
| melt | 0 | 熔化固态冰 | focus → 加热使其融化 | 从走廊出发 |
| melt | 1 | 同上 | 同上 | 从美术室出发 |

boil 任务的评分子步骤为：进入浴室 +3 分、focus 正确物质 +67 分、杯子放上炉子 +2 分、激活炉子 +1 分、水沸腾 +25 分，满分 100 分。

---

## 四、实验

### 4.1 实验设置

实验对比**有记忆**和**无记忆**两种条件：
- **无记忆**：`Memory = None`，每个任务独立运行，LLM 无任何历史信息
- **有记忆**：8 个任务变体共享同一记忆实例，按 boil→freeze→find-animal→melt 顺序运行，后续任务可复用前序任务积累的经验

其他设置：最大步数 80 步，LLM 温度 0.5，boil 任务预置了完整的初始记忆作为冷启动知识。

### 4.2 实验结果

| 任务 | 变体 | 无记忆得分 | 无记忆成功 | 有记忆得分 | 有记忆步数 | 有记忆成功 |
|------|:----:|:---------:|:---------:|:---------:|:---------:|:---------:|
| boil | 0 | 70.0 | ✗ | 100.0 | **47** | ✓ |
| boil | 1 | 72.0 | ✗ | 72.0 | 80 | ✗ |
| freeze | 0 | 100.0 | ✓ | 100.0 | **28** | ✓ |
| freeze | 1 | 87.0 | ✗ | 100.0 | **57** | ✓ |
| find-animal | 0 | −100.0 | ✗ | 25.0 | 80 | ✗ |
| find-animal | 1 | 83.0 | ✗ | 100.0 | **44** | ✓ |
| melt | 0 | 42.0 | ✗ | 42.0 | 80 | ✗ |
| melt | 1 | 38.0 | ✗ | 38.0 | 80 | ✗ |
| **合计** | | | **1 / 8** | | | **4 / 8** |

记忆模块将成功率从 **12.5%** 提升至 **50.0%**，提升 4 倍。

### 4.3 成功案例（boil/0，有记忆）

**有记忆条件（boil/0，成功）：**

![有记忆：Step 5 执行 focus 获得 +67 分，score=70](images/success_early.png)

![有记忆：持续激活炉子，Step 47 获得 +25 分，最终 score=100，任务成功](images/success_final.png)

有记忆条件下，智能体在第 5 步即执行 `focus on substance in toilet`（得 70 分），第 13 步用 `dunk cup into toilet` 取水，第 16 步将杯子移至炉子，随后持续激活炉子，第 47 步完成任务，得满分 100 分。

**无记忆条件（boil/0，失败）：**

![无记忆：步数耗尽，第 80 步仍在探索，最终 score=70，任务失败](images/failure_final.png)

无记忆条件下，智能体在前 78 步反复在各房间探索，直到第 80 步（最后一步）才尝试 focus，此时步数已耗尽，以 70 分失败告终。

### 4.4 结果分析

| 任务 | 记忆作用 | 分析 |
|------|---------|------|
| boil/0 | 预置记忆直接命中 | 成功序列 + 环境事实引导全流程，47 步完成 |
| boil/1 | 记忆命中但步数不足 | 核心流程正确，美术室出发导航消耗过多步数 |
| freeze/0 | 两种条件均成功 | 有记忆少用 4 步（28 vs 32），效率提升 |
| freeze/1 | 跨变体学习生效 | 复用 freeze/0 的成功经验，从失败（87分）转为成功 |
| find-animal/0 | 冷启动，部分规避 | 无预置经验；有记忆因失败动作规避，从 −100 分提升至 25 分 |
| find-animal/1 | 跨变体学习生效 | 复用 find-animal/0 的经验，44 步完成 |
| melt/0,1 | 无改善 | 无预置 melt 经验，且流程与 boil/freeze 差异大，跨任务迁移失效 |

---

## 五、记忆检索评估

**检索速度**：各检索操作均在内存中完成，成功序列和失败动作使用字典精确查找（O(1)），环境事实使用关键词集合匹配（O(facts × words)），每步总延迟 < 2 ms，相对于 LLM API 调用的 1000–3000 ms 可忽略不计。

**检索命中率**（有记忆条件，8 个任务变体）：

| 检索类型 | 命中 | 命中率 |
|---------|:----:|:------:|
| 成功序列（预置 + 跨变体积累） | 5/8 | 62.5% |
| 跨任务通用知识 | 8/8 | 100% |
| 环境事实关键词匹配 | 6/8 | 75% |
| 成功序列命中后的任务成功率 | 4/5 | 80% |

melt 任务（2/8）无相关事实可检索，是环境事实命中率未达 100% 的原因。boil/1 成功序列命中但任务失败，原因为步数预算不足而非记忆失效。

---

## 六、总结

本项目构建了一个完整的记忆增强型文本交互智能体，主要成果如下：

**双层记忆架构**：短期记忆（滑动窗口 10 步）维护当前上下文，长期记忆（JSON + 关键词检索）积累跨任务经验，两者协同将成功率提升至无记忆条件的 4 倍。

**跨任务知识泛化**：通过预置 `cross_task_knowledge`，所有任务均能受益于通用规则（房间布局、动作语法等），无需为每类任务单独预置知识。

**记忆驱动的跨变体学习**：实验表明，顺序运行的任务可通过共享记忆实现经验传递——freeze/1 和 find-animal/1 均因复用前一变体的成功经验而从失败转为成功。

**局限性**：关键词匹配的检索方式对词形变化敏感；melt 类任务因无预置经验且与其他任务差异大，记忆无法提供有效引导；变体 1 的额外导航步数在 80 步限制下仍是瓶颈。

**未来方向**：引入向量检索（sentence-transformers + FAISS）提升检索泛化性；自动从成功轨迹中提取跨任务通用知识；扩展多智能体协作（探索者 + 记忆维护者分工）。

---

## 参考文献

1. Wang et al. (2023). *CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization.* arXiv:2310.10134.
2. Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629.
3. Côté et al. (2022). *ScienceWorld: Is your Agent Smarter than a 5th Grader?* arXiv:2203.07540.
4. ScienceWorld: https://github.com/allenai/ScienceWorld
