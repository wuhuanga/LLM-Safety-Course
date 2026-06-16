##  项目结构

```
├── data/                          # 数据集文件夹
│   └── zsre_mend_train.json       # task3数据集文件
├── EasyEdit/                      # EasyEdit 框架源码依赖库
├── baseline.py                    # Task 1 基线测试生成脚本
├── dataset.json                   # 本地测试基础数据集
├── edit_memit.py                 # Task 3 大规模批量编辑压测脚本
├── edit_rome.py                  # Task 2 单条事实编辑实践脚本
├── evaluate.py                    # Task 4 多维深度综合评测分析脚本
├── gpt2-medium.yaml               # ROME 算法超参数配置文件
├── gpt-medium-memit.yaml         # MEMIT 算法超参数配置文件
├── requirements.txt               # 环境依赖清单
└── readme.md                      # 项目说明文档
```

## 实验运行指南

### Task 1: 基础环境搭建与基线测试 (Baseline Evaluation)

运行基线评测，观测原始基座模型未对齐前的时效性滞后、事实幻觉及自回归“复读机崩溃”现象：

```
python baseline.py
```

### Task 2: 单条事实编辑实践 (ROME Single-Edit)

执行 ROME 算法进行单点知识注入与即时验证：

```
python edit_rome.py
```

### Task 3: 批量知识编辑实践 (MEMIT Batch-Edit)

对模型进行 500 条数据的批量高速注入，并评估端到端生成表现以及底层算力（显存/耗时）开销：

```
python -X utf-8 edit_memit.py
```

### Task 4: 多维综合评估 (Comprehensive Metric Evaluation)

对编辑后的模型进行精准的底层多维度评测（包含 **ES** 编辑成功率、**PS** 泛化能力、**NS** 局部稳定性三列高级指标统计）：

```
python task4_evaluate.py
```
