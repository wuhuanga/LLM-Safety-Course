# Agent-Memory 作业报告

学号：SX2516010  
姓名：李鸿昊  
作业：01-Agent / Agent-Memory  
视频文件：`video/1779707739250.mp4`

## 1. 任务理解

本作业要求实现一个能够在 TextWorld 文本环境中交互的 LLM Agent。Agent 需要根据环境文本、当前任务目标、可执行动作列表和历史轨迹持续规划，直到完成任务或达到最大步数。作业重点不是单步问答，而是长程多步交互：Agent 要能在环境中移动、检查物品、获取物品、执行操作，并从错误动作或低进展循环中恢复。

作业要求中和本实现对应的能力包括：

- 多步规划：Agent 持续与 TextWorld 环境交互，根据 observation、inventory、admissible commands 和历史动作选择下一步。
- 记忆模块：Agent 将训练 episode 中的轨迹、失败原因、可复用经验和洞察保存为长期记忆，并在测试时检索使用。
- 错误恢复：Agent 根据失败轨迹总结低价值循环、缺失前置条件和未探索方向，测试时由 Critic 进行动作检查。
- 对比实验：比较无记忆 ReAct 与带记忆 MAGMem 的效果，分析记忆模块带来的收益和局限。
- 泛化经验：将历史任务中的具体轨迹压缩成更高层的经验，而不是直接复制完整动作序列。

## 2. 环境与模型

实验环境：

- 操作系统：Windows 11 + WSL
- Python 环境：conda 环境 `knowagent`
- 主要依赖：`textworld==1.7.0`、`openai>=2.0.0`
- LLM API：DeepSeek 官方 API，使用 OpenAI SDK 兼容接口
- 模型：`deepseek-chat`

API key 没有写入代码或提交文件。运行时可以通过以下方式配置：

```bash
export OPENAI_API_KEY="你的 DeepSeek API Key"
```

也可以将 key 放在 `.secrets/deepseek_api_key.txt`，或通过参数 `--deepseek-key-file` 指定路径。

## 3. 方法设计

本实现的基础 Agent 是 ReAct：每一步把任务目标、当前 observation、inventory、可执行动作列表和最近轨迹拼成提示词，让 LLM 输出下一步动作。为了防止非法动作，输出动作会经过 admissible commands 对齐，只允许执行当前环境给出的可执行命令。

在记忆增强版本中，我实现了一个受 GMemory / GMem 思路启发的轻量 MAGMem 框架。它不是完整复刻论文系统，而是根据本作业的 TextWorld 任务做了简化。整体流程如下：

1. 训练阶段运行若干 TextWorld 游戏，记录完整交互轨迹。
2. 每个 episode 结束后，由 LLM 或启发式整理任务级记忆，包括成功/失败反思、可复用策略、失败原因、未探索方向和低价值重复动作。
3. 从多个 episode 中总结 insight，作为跨任务的通用经验。
4. 测试阶段冻结记忆，只在 episode 开始时检索一次相关记忆。
5. 检索到的历史任务和 insight 会被投影成 Planner、Actor、Critic 三个角色可用的高层建议。
6. Actor 仍然基于当前环境执行动作，Critic 检查动作是否重复失败、是否陷入低进展循环、是否违背记忆中的风险提示。

这种设计的核心约束是：记忆只能作为建议，不能硬编码某个任务的专用规则，也不能直接照搬历史动作序列。具体任务规则只能来自当前 TextWorld 环境文本或训练轨迹总结出的记忆。

## 4. 代码结构

主要代码路径如下：

- `src/scripts/run_textworld_llm.py`：主运行入口，支持 ReAct、HGMem/MAGMem、训练记忆、只读测试和视频 trace。
- `src/textworld_llm/prompting.py`：基础 ReAct prompt、动作解析和动作规范化。
- `src/textworld_llm/memory.py`：JSON 长期记忆，包括 episode memory、insight memory 和检索逻辑。
- `src/textworld_llm/curation.py`：从 episode 轨迹中总结任务记忆和 insight。
- `src/textworld_llm/multiagent.py`：Planner / Actor / Critic 角色拆分和动作审查。
- `src/textworld_llm/client.py`：DeepSeek/OpenAI SDK 调用封装。
- `src/scripts/summarize_ablation.py`：实验结果汇总。
- `src/scripts/generate_complex_cooking_20.sh`：TextWorld 数据生成脚本。

## 5. 运行方法

创建环境：

```bash
conda env create -f environment.yml
conda activate knowagent
```

配置 API：

```bash
export OPENAI_API_KEY="你的 DeepSeek API Key"
```

如需重新生成实验数据：

```bash
bash src/scripts/generate_complex_cooking_20.sh
```

训练 MAGMem 记忆：

```bash
python -B src/scripts/run_textworld_llm.py \
  --agent magmem \
  --planner-agent heuristic \
  --critic-agent heuristic \
  --max-tokens 1024 \
  --max-steps 30 \
  --games \
    data/complex_cooking_20/train/game_00.z8 \
    data/complex_cooking_20/train/game_01.z8 \
    data/complex_cooking_20/train/game_02.z8 \
    data/complex_cooking_20/train/game_03.z8 \
    data/complex_cooking_20/train/game_04.z8 \
  --output results/final_5x5/train_magmem.jsonl \
  --memory-dir results/final_5x5/train_memory
```

测试 ReAct：

```bash
python -B src/scripts/run_textworld_llm.py \
  --agent react \
  --max-tokens 1024 \
  --max-steps 30 \
  --games \
    data/complex_cooking_20/test/game_00.z8 \
    data/complex_cooking_20/test/game_01.z8 \
    data/complex_cooking_20/test/game_02.z8 \
    data/complex_cooking_20/test/game_03.z8 \
    data/complex_cooking_20/test/game_04.z8 \
  --output results/final_5x5/test_react.jsonl \
  --memory-dir results/final_5x5/react_unused_memory \
  --read-only-memory
```

测试 MAGMem：

```bash
python -B src/scripts/run_textworld_llm.py \
  --agent magmem \
  --planner-agent heuristic \
  --critic-agent heuristic \
  --max-tokens 1024 \
  --max-steps 30 \
  --games \
    data/complex_cooking_20/test/game_00.z8 \
    data/complex_cooking_20/test/game_01.z8 \
    data/complex_cooking_20/test/game_02.z8 \
    data/complex_cooking_20/test/game_03.z8 \
    data/complex_cooking_20/test/game_04.z8 \
  --output results/final_5x5/test_magmem.jsonl \
  --memory-dir results/final_5x5/train_memory \
  --reuse-memory \
  --read-only-memory
```

汇总：

```bash
python -B src/scripts/summarize_ablation.py \
  results/final_5x5/test_react.jsonl \
  results/final_5x5/test_magmem.jsonl
```

## 6. 实验设置

最终实验使用 `complex_cooking_20` 数据集中的 5 个训练游戏和 5 个测试游戏：

- 训练集：`train/game_00.z8` 到 `train/game_04.z8`
- 测试集：`test/game_00.z8` 到 `test/game_04.z8`
- 训练阶段：MAGMem 写入长期记忆
- 测试阶段：ReAct 不使用记忆；MAGMem 只读训练记忆，不把测试轨迹写回记忆
- 最大步数：30

我没有在最终实验中进行同一批样本的多轮反复训练。理论上，记忆系统支持在线持续追加经验，多轮迭代也可以被解释为 experience accumulation；但如果对同一批样本反复刷，容易造成过拟合，也不利于说明测试集上的记忆泛化。因此最终采用“一次训练、冻结记忆、测试只读”的流程。

## 7. 实验结果

最终实验结果如下：

| 指标 | ReAct | MAGMem | 变化 |
|---|---:|---:|---:|
| 测试游戏数 | 5 | 5 | 0 |
| 平均得分 | 2.8 | 3.8 | +1.0 |
| 总得分 | 14 | 19 | +5 |
| 成功率 | 0.0 | 0.0 | 0 |
| 平均非法动作数 | 0.0 | 0.0 | 0 |
| 平均步数 | 27.0 | 23.6 | -3.4 |
| 平均重复动作数 | 14.6 | 8.8 | -5.8 |
| 首次得分平均步数 | 4.75 | 5.0 | +0.25 |
| 平均记忆命中数 | 0.0 | 3.0 | +3.0 |

从结果看，MAGMem 并没有在这些较复杂的 TextWorld cooking 任务中实现完整通关，成功率仍然为 0。但是和 ReAct 相比，MAGMem 的总得分从 14 提升到 19，平均得分从 2.8 提升到 3.8，平均步数从 27.0 降到 23.6，平均重复动作数从 14.6 降到 8.8。这说明记忆模块对减少无效循环、提升部分任务进展有一定帮助。

单样本结果：

| 游戏 | ReAct 得分/步数 | MAGMem 得分/步数 | 说明 |
|---|---:|---:|---|
| game_00 | 2 / 30 | 6 / 30 | MAGMem 更早获取工具并进行处理动作，得分提升明显。 |
| game_01 | 2 / 30 | 6 / 30 | MAGMem 减少部分无效探索，围绕食材处理推进更多步骤。 |
| game_02 | 0 / 30 | 0 / 30 | 两者都陷入导航/观察循环，记忆没有解决该样本。 |
| game_03 | 7 / 15 | 5 / 11 | ReAct 得分更高，但 MAGMem 用更少步数完成部分进展。 |
| game_04 | 3 / 30 | 2 / 17 | MAGMem 提前停止但得分略低，说明记忆仍可能误导。 |

## 8. 视频说明

视频文件为 `video/1779707739250.mp4`。视频中展示了：

1. 最终结果汇总；
2. 训练记忆文件；
3. 一次真实 MAGMem episode 运行；
4. 实时输出的记忆检索、角色 brief、每步动作、Critic 判断、环境反馈和得分变化。

为了便于录制，我在主脚本中增加了 `--trace-steps` 参数。该参数不会改变默认实验逻辑，只是在真实运行时把每一步流程打印出来。

录屏中对应的 demo 命令为：

```bash
python -B src/scripts/run_textworld_llm.py \
  --agent magmem \
  --planner-agent heuristic \
  --critic-agent heuristic \
  --max-tokens 1024 \
  --max-steps 10 \
  --trace-steps \
  --games data/complex_cooking_20/test/game_00.z8 \
  --output results/video_demo_trace_magmem_game00.jsonl \
  --memory-dir results/final_5x5/train_memory \
  --reuse-memory \
  --read-only-memory
```

## 9. 实现过程与局限

这次实现是在时间较紧的情况下通过 AI 辅助完成的，原本设想的很多功能没有完全实现，因此整体更接近一个可运行的课程原型，而不是完整的研究系统。

算法基础参考了 GMemory / GMem 的思想，包括从历史轨迹中压缩经验、检索相关任务记忆、再将记忆提供给多 Agent 角色。但在实现过程中我发现，这类记忆结构并不完全适合当前 TextWorld 任务的细节控制。它比较适合提供整体方向，例如“先读取任务说明”“避免重复低进展动作”“关注缺失前置条件”；但如果让记忆直接指导细节动作，模型有时会过度放大某条经验，反而产生错误行为。

当前实现中的洞察记忆是有价值的，但还比较粗糙。它主要来自少量训练任务，因此容易偏向局部样本。更理想的方式可能是：一方面从更多历史任务中总结整体方法论，另一方面维护一个更丰富的“灵感库”或错误模式库，让 Agent 在需要时参考，而不是把少数任务的经验当成强约束。

记忆更新也存在局限。现在的记忆主要从当前 episode 进行总结，对已有记忆整体的再整理较弱。如果训练轮次变多，记忆数量会增加，但不一定会变好；已有研究和实际调试都说明，记忆迭代如果缺少全局整理，可能会积累偏差。后续可以考虑周期性地对所有历史记忆做合并、去重、冲突检查和抽象层级划分，让高层原则和低层错误案例分开存储。

此外，最终实验虽然显示 MAGMem 在平均得分和重复动作数上优于 ReAct，但成功率仍然为 0。这说明当前系统只能带来一定程度的行为改善，还没有稳定解决长程规划问题。后续如果继续完善，需要重点改进：

- 更稳健的任务阶段识别；
- 更可靠的 recipe/reference 信息解析；
- 更细致的 Critic 动作否决机制；
- 更好的记忆压缩与冲突消解；
- 更大规模训练/测试集上的稳定评估。

## 10. 总结

本项目实现了一个真实 LLM + TextWorld 环境中的记忆增强 Agent。系统支持 ReAct 基线、长期记忆构建、训练/测试分离、MAGMem 多角色决策、错误经验总结和只读记忆评估。最终 5 个测试任务上，MAGMem 相比 ReAct 提升了平均得分并减少了重复动作，说明记忆模块具有一定收益；但系统仍未达到稳定通关，记忆使用和长程任务控制仍有明显改进空间。

