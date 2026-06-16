# LLM Knowledge Editing Experiment

本项目完成了基于 `Qwen2.5-0.5B-Instruct` 与 `EasyEdit` 的大模型知识编辑实验，覆盖以下内容：

- Task 1：Baseline Evaluation
- Task 2：ROME 单事实编辑
- Task 3：MEMIT 批量知识编辑
- Task 4：统一评估（ES / PS / NS）
- Bonus：跨语种泛化测试（英文注入，中文提问）

## 1. 项目结构

```text
baseline.py                     # Task 1：基线测试
edit_rome.py                    # Task 2：ROME 单条事实编辑
edit_memit.py                    # Task 3：MEMIT 500 条批量编辑
evaluate.py                     # Task 4：统一评估
bonus_crosslingual_rome.py      # Bonus：ROME 跨语种测试
bonus_crosslingual_memit.py     # Bonus：MEMIT 跨语种测试
bonus_crosslingual_evaluate.py  # Bonus：跨语种结果汇总
data/custom_edits.json          # 10 条自定义编辑样本
data/memit_500.json             # MEMIT 500 条批量编辑样本
data/crosslingual_edits.json    # Bonus 跨语种测试样本
results/                        # 所有实验结果输出
report.md                       # 实验报告 Markdown 版
```

## 2. 运行环境

- OS: Windows 10/11
- Python: 3.11
- GPU: NVIDIA CUDA GPU（实验中使用 RTX 3060 Laptop 6GB）
- Base Model: `Qwen2.5-0.5B-Instruct`
- Framework: `EasyEdit`

## 3. 安装依赖

建议先创建独立环境，例如：

```bash
conda create -n easyedit python=3.11 -y
conda activate easyedit
pip install -r requirements.txt
```

如果需要 CUDA 版本的 PyTorch，请根据本机 CUDA 版本参考 [PyTorch 官网](https://pytorch.org/) 安装。

## 4. 模型与路径说明

本项目默认从当前项目目录加载模型：

```text
./model/Qwen2.5-0.5B-Instruct
```

其中的模型权重文件较大，不适合直接提交到 GitHub，因此仓库中**不包含**该目录，运行前需要你在本地手动下载并放到上述路径下。

`baseline.py`、`bonus_crosslingual_rome.py`、`bonus_crosslingual_memit.py` 都已经改为基于项目根目录自动定位模型路径，不再依赖固定盘符或绝对路径。

`edit_rome.py` 与 `edit_memit.py` 通过 EasyEdit 的 hparams 配置加载模型；当前配置文件中已使用项目内相对路径：

- `model_name: ./model/Qwen2.5-0.5B-Instruct`
- `stats_dir: ./model/Qwen2.5-0.5B-Instruct/wikipedia_stats`

如果你的模型目录名称或位置不同，只需同步修改对应脚本或 hparams 配置即可。

## 5. 数据说明

### 5.1 自定义 10 条样本

文件：`data/custom_edits.json`

字段格式：

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

### 5.2 MEMIT 500 条批量编辑集

文件：`data/memit_500.json`

脚本会优先读取本地缓存；若本地不存在，则尝试从公开数据集自动准备并缓存。

### 5.3 Bonus 跨语种样本

文件：`data/crosslingual_edits.json`

特点：
- 英文 prompt 注入知识
- 英文 rephrase 验证同语种泛化
- 中文 `zh_rephrase_prompt` 验证跨语种泛化

## 6. 运行方式

### Task 1：Baseline

```bash
python baseline.py
```

输出：
- `results/baseline_results.json`

### Task 2：ROME 单条事实编辑

```bash
python edit_rome.py
```

输出：
- `results/rome_results.json`

### Task 3：MEMIT 批量知识编辑

```bash
python edit_memit.py
```

输出：
- `results/memit_results.json`

说明：
- 默认执行 500 条批量知识编辑
- 脚本会记录总耗时与峰值显存
- 若检测到 GPU，会自动启用 CUDA

### Task 4：统一评估

```bash
python evaluate.py
```

输出：
- `results/metrics.json`

该脚本会汇总：
- Baseline 结果
- ROME 的 ES / PS / NS
- MEMIT 的 ES / PS / NS

## 7. Bonus：跨语种泛化测试

### 7.1 ROME 跨语种测试

```bash
python bonus_crosslingual_rome.py
```

输出：
- `results/rome_crosslingual_results.json`

### 7.2 MEMIT 跨语种测试

```bash
python bonus_crosslingual_memit.py
```

输出：
- `results/memit_crosslingual_results.json`

### 7.3 跨语种结果汇总

```bash
python bonus_crosslingual_evaluate.py
```

输出：
- `results/crosslingual_metrics.json`

## 8. 当前主要实验结果

### Baseline

- Total samples: 10
- Already matches target_new: 0/10
- Matches ground_truth: 6/10

### ROME

- Total samples: 10
- ES: 100.00%
- PS: 76.57%
- NS: 48.33%

### MEMIT

- Total samples: 500
- ES: 100.00%
- PS: 39.90%
- NS: 100.00%
- Elapsed time: 2083.756 s（约 34.73 分钟）
- Peak GPU memory: 3603.19 MB
- Dataset source: `data/memit_500.json`

### Bonus：跨语种泛化

- Total samples: 12
- ES: 100.00%
- PS_en: 69.78%
- PS_zh: 8.33%
- Cross-lingual gap: 61.45%

说明：模型在英文编辑成功后，对中文提问的跨语种迁移能力较弱。

## 9. 注意事项

1. `edit_memit.py` 对 `memit_500.json` 做了适配处理，因此可以直接读取当前项目中的 500 条数据。
2. 当前 MEMIT 的 `rephrase_prompt` 与 `locality_prompt` 是为了适配批量数据格式而构造的简化版本，因此 PS / NS 在报告中应做谨慎解释。
3. 若使用 CPU 运行，MEMIT 会非常慢，建议使用 GPU。
4. 若修改了模型路径或 EasyEdit 配置，请同步检查 hparams 文件。
5. 提交到 GitHub 前，请确认 `model/`、`logs/`、`output_images/` 和临时结果目录已被 `.gitignore` 忽略。
