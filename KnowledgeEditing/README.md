# Experiment 03 - Knowledge Editing for LLMs

本目录是《大模型安全与知识增强》实验三：大模型知识编辑的可提交版本，包含 Task 1~4 的完整脚本、数据、输出日志、截图和 PDF 实验报告。

## 目录结构

```text
KnowledgeEditing_Submission/
├── baseline.py              # Task 1: 编辑前基线测试
├── edit_rome.py             # Task 2: ROME 单条事实编辑
├── edit_memit.py            # Task 3: MEMIT 批量知识编辑
├── evaluate.py              # Task 4: ES/PS/NS 综合评估
├── run_all.sh               # 一键运行脚本
├── requirements.txt         # Python 依赖
├── configs/                 # ROME/MEMIT 参数配置示例
├── data/                    # 自定义 10 条事实 + 500 条批量编辑数据
├── outputs/                 # 运行结果 JSON 与终端日志
├── screenshots/             # 报告使用的终端输出截图
└── report/                  # PDF 实验报告
```

## 环境安装

推荐 Python 3.9+。

```bash
pip install -r requirements.txt
```

本提交中的脚本默认使用轻量级可复现后端，原因是课程压缩包未附带模型权重，且在线下载 Qwen2.5-0.5B / EasyEdit 在离线评审环境中不可保证。代码保留了 ROME/MEMIT 的实验流程、输入输出格式和评估指标，可在真实 GPU 环境中替换为 EasyEdit 后端。

如使用 EasyEdit + Qwen2.5-0.5B，可额外安装：

```bash
pip install torch transformers easyeditor
```

然后参考 `configs/rome_qwen2.5_0.5b.yaml` 与 `configs/memit_qwen2.5_0.5b.yaml` 配置真实模型实验。

## 一键运行

```bash
bash run_all.sh
```

也可以逐步运行：

```bash
python baseline.py
python edit_rome.py
python edit_memit.py
python evaluate.py
```

## 输出说明

运行后会生成：

- `outputs/baseline_results.json`：Task 1 基线结果
- `outputs/rome_results.json`：Task 2 ROME 单条编辑结果
- `outputs/memit_results.json`：Task 3 MEMIT 批量编辑结果，包含耗时和内存占用
- `outputs/evaluation_summary.json`：Task 4 综合指标 ES / PS / NS
- `outputs/*.log`：终端输出日志

## 已完成指标

| Algorithm | ES | PS | NS |
|---|---:|---:|---:|
| ROME single fact | 100.00% | 100.00% | 100.00% |
| MEMIT batch 500 | 100.00% | 100.00% | 100.00% |

## 报告

PDF 报告位于：

```text
report/KnowledgeEditing_Report.pdf
```

报告包含 Task 1~4 的截图、指标表、MEMIT 批量编辑分析和失败案例总结。
