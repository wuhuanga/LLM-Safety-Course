# LLM-KnowledgeEditing 大模型知识编辑实验项目

## 项目简介

基于 EasyEdit 框架，在 `Qwen/Qwen2.5-0.5B-Instruct` 模型上完成知识编辑实验，覆盖 Baseline 基线测试、ROME 单事实编辑、MEMIT 批量知识编辑、ES/PS/NS 指标评估与实验报告整理。

本项目对应《大模型安全与知识增强》课程方向 03：Knowledge Editing。

---

## 实验环境

### 硬件配置

- 设备：Windows 笔记本
- 显卡：NVIDIA GeForce RTX 3060 Laptop GPU
- 峰值显存占用：MEMIT 10 条轻量批量实验约 3017.22 MB

### 软件依赖

- Python：3.11
- PyTorch：2.5.1
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 知识编辑框架：EasyEdit

---

## 环境部署

```powershell
# 进入项目目录
cd knowledge_editing_solution

# 创建并激活 conda 环境
conda create -y -n knowledge_editing python=3.11
conda activate knowledge_editing

# 安装依赖
pip install -r requirements.txt
```

EasyEdit 当前不是标准 pip 包，需要额外克隆源码，并把源码目录加入 Python 路径：

```powershell
mkdir external
git clone https://github.com/zjunlp/EasyEdit.git external/EasyEdit
python -c "import site, pathlib; p=pathlib.Path(site.getsitepackages()[0])/'easyedit_local.pth'; p.write_text(str(pathlib.Path('external/EasyEdit').resolve()), encoding='ascii'); print(p)"
```

在 Windows + Qwen2.5 环境下，建议应用本项目提供的兼容补丁：

```powershell
git -C external/EasyEdit apply ../../patches/easyedit-qwen-windows.patch
```

该补丁用于修复 Qwen2.5 在 EasyEdit 中的 hook 参数、tied `lm_head.weight` 查找，以及 MEMIT 轻量 covariance 设置。

---

## 项目文件结构

```text
knowledge_editing_solution/
├── baseline.py                         # Task1：基线知识测试脚本
├── edit_rome.py                        # Task2：ROME 单事实编辑脚本
├── edit_memit.py                       # Task3：MEMIT 批量知识编辑脚本
├── evaluate.py                         # Task4：ES/PS/NS 综合评估脚本
├── sample_zsre.py                      # ZsRE 数据集抽样脚本
├── editing_utils.py                    # 公共工具函数
├── custom_facts.json                   # 根目录数据别名，便于直接查看
├── memit_result.json                   # 根目录 MEMIT 结果别名
├── requirements.txt
├── configs/
│   ├── model.yaml
│   ├── rome.yaml
│   └── memit.yaml
├── data/
│   ├── custom_facts.json               # 10 条自定义事实数据集
│   └── ZsRE_sample_result.json         # 本次 MEMIT 轻量实验结果备份
├── outputs/
│   ├── baseline.json
│   ├── rome_results.json
│   ├── memit_results.json
│   └── metrics.json
├── patches/
│   └── easyedit-qwen-windows.patch
└── reports/
    ├── REPORT.md
    ├── SC2616021-许宏宇-03-KnowledgeEditing.pdf
    └── screenshots/
```

说明：完整 EasyEdit 源码、缓存文件和日志不随代码提交，避免仓库体积过大。复现实验时按上方环境部署步骤克隆 EasyEdit 并应用补丁即可。

---

## 运行指令

### Task1 基础环境搭建与基线测试

```powershell
python baseline.py --data data/custom_facts.json --output outputs/baseline.json
```

### Task2 ROME 单条事实知识编辑

```powershell
python edit_rome.py --data data/custom_facts.json --output outputs/rome_results.json
```

### Task3 MEMIT 批量知识编辑

默认命令会从 Hugging Face `zjunlp/KnowEdit` 下载 ZsRE benchmark，并尝试运行 500 条批量编辑：

```powershell
python edit_memit.py --output outputs/memit_results.json
```

本地 Windows + 6GB 显存环境中，完整 500 条实验在 covariance statistics 阶段耗时和资源开销较大。本项目最终使用 10 条 ZsRE benchmark 样例完成可复现的轻量批量编辑：

```powershell
python edit_memit.py --limit 10 --output outputs/memit_results.json
```

如需先抽样 ZsRE 数据，可运行：

```powershell
python sample_zsre.py --limit 500 --output data/ZsRE.json
```

### Task4 三大指标综合评估

```powershell
python evaluate.py --baseline outputs/baseline.json --rome outputs/rome_results.json --memit outputs/memit_results.json --output outputs/metrics.json
```

---

## 评测指标

主表采用 EasyEdit 内部指标：

| 方法 | ES (编辑成功率) | PS (泛化性) | NS (局部性) |
| --- | ---: | ---: | ---: |
| ROME | 100.00% | 55.83% | 58.50% |
| MEMIT | 81.28% | 75.94% | 21.08% |

补充表采用自由生成严格字符串匹配：

| 方法 | ES | PS | NS |
| --- | ---: | ---: | ---: |
| Baseline | 10.00% | 0.00% | 30.00% |
| ROME | 10.00% | 0.00% | 30.00% |
| MEMIT | 0.00% | 0.00% | 10.00% |

- ES (Efficacy)：直接编辑提示下，模型生成结果是否包含目标新答案。
- PS (Paraphrase)：同义改写提示下，模型生成结果是否仍包含目标新答案。
- NS (Neighborhood)：无关事实提示下，模型是否保留原答案。

---

## 结果分析

从 EasyEdit 内部指标看，ROME 对 10 条自定义事实逐条编辑时 ES 达到 100.00%，MEMIT 在 10 条 ZsRE benchmark 样例上的 ES 达到 81.28%，PS 达到 75.94%。

自由生成字符串匹配指标仍然较低，主要原因包括：

1. 字符串包含式评估较严格，同义表达、长解释或相邻事实都可能被判为未命中。
2. `Qwen2.5-0.5B-Instruct` 在开放式 prompt 下倾向于生成完整句子，而不是只输出目标实体。
3. MEMIT 轻量实验使用 10 条样例和 identity covariance 近似，不能完全代表 500 条完整 covariance MEMIT 设置。
4. 小参数模型的知识存储密度较高，局部权重更新容易影响无关事实。

详细实验过程、终端截图和失败案例见 `reports/SC2616021-许宏宇-03-KnowledgeEditing.pdf`。
