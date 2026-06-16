# 基于 EasyEdit 的 Qwen2.5-0.5B 知识编辑实验报告

学生：SX2516020 陈奕晨

方向：03-KnowledgeEditing

## 1. 实验目标

本实验围绕大语言模型知识编辑任务展开，目标是在不进行全量训练的前提下，定点修改模型中的事实关联，并评估修改后的成功率、泛化性和局部性。实验采用 EasyEdit 框架，完成了原始模型 baseline、ROME 单条事实编辑和 MEMIT 批量事实编辑。

评估指标如下：

| 指标 | 含义 |
|:--|:--|
| ES | Efficacy Score，直接编辑 prompt 是否输出目标新知识 |
| PS | Paraphrase Score，同义改写 prompt 是否输出目标新知识 |
| NS | Neighborhood Score，无关事实是否保持原有正确回答 |

## 2. 实验环境

| 项目 | 配置 |
|:--|:--|
| Conda 环境 | `x1x_LLM_Safety_Work` |
| Python | 3.12.13 |
| GPU | NVIDIA GeForce RTX 3090 x 2，实验使用 GPU 0 |
| CUDA/驱动 | NVIDIA Driver 535.230.02，CUDA 12.2 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.51.3 |
| EasyEdit | `easyeditor`，并补充安装运行依赖 |
| 基础模型 | `Qwen/Qwen2.5-0.5B-Instruct` |

环境创建与安装命令：

```bash
conda create -n x1x_LLM_Safety_Work python=3.10 -y
conda activate x1x_LLM_Safety_Work
pip install -r requirements.txt
pip install easyeditor --no-deps
pip install higher hydra-core datasets gpustat einops timm iopath fairscale opencv-python-headless scipy scikit-learn sentence-transformers matplotlib peft
```

实际环境验证中，`nvidia-smi` 可以识别两张 RTX 3090，`torch.cuda.is_available()` 为 `True`，`torch.cuda.device_count()` 为 `2`。

## 3. 数据集构建

Task 1 和 Task 2 使用 `data/custom_facts.json` 中的 10 条自定义事实编辑数据。每条样本包含原始 prompt、目标新答案、同义改写 prompt 和 locality prompt。例如：

```json
{
  "subject": "France",
  "prompt": "The capital city of France is",
  "target_new": "Lyon",
  "ground_truth": "Paris",
  "rephrase_prompt": "Which city is the capital of France?",
  "locality_prompt": "The capital city of Germany is",
  "locality_ground_truth": "Berlin"
}
```

Task 3 使用 `prepare_memit_data.py` 生成 500 条 MEMIT 风格合成批量编辑数据，保存到 `data/memit_500_synthetic.json`：

```bash
python prepare_memit_data.py --output data/memit_500_synthetic.json --num-items 500
```

## 4. 实验方法

### 4.1 Baseline

Baseline 不修改模型参数，直接调用原始 `Qwen/Qwen2.5-0.5B-Instruct` 生成答案：

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp \
/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python baseline.py \
  --data data/custom_facts.json \
  --output outputs/baseline_predictions_real.jsonl \
  --backend hf \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-new-tokens 16

/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python evaluate.py \
  --data data/custom_facts.json \
  --predictions outputs/baseline_predictions_real.jsonl \
  --output outputs/baseline_metrics_real.json
```

### 4.2 ROME 单条事实编辑

ROME 通过修改模型 FFN 层中的特定权重，实现对 subject-object 事实关联的定点编辑。本实验对 10 条自定义事实逐条运行 ROME；每条样本编辑前重新加载基础模型权重，避免上一条编辑污染下一条结果。每次编辑后生成直接 prompt、改写 prompt 和 locality prompt 的回答。

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp TOKENIZERS_PARALLELISM=false \
/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python edit_rome.py \
  --data data/custom_facts.json \
  --output outputs/rome_predictions_real.jsonl \
  --metrics-output outputs/rome_easyedit_metrics_real.json \
  --backend easyedit \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --hparams configs/ROME/qwen2.5-0.5b.yaml \
  --max-new-tokens 16

/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python evaluate.py \
  --data data/custom_facts.json \
  --predictions outputs/rome_predictions_real.jsonl \
  --output outputs/rome_metrics_real.json
```

### 4.3 MEMIT 批量事实编辑

MEMIT 将 ROME 的单点编辑扩展为批量知识注入。本实验对 500 条合成事实运行 MEMIT，并记录耗时和显存峰值。

```bash
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp TOKENIZERS_PARALLELISM=false \
/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python edit_memit.py \
  --data data/memit_500_synthetic.json \
  --output outputs/memit_predictions_real.jsonl \
  --metrics-output outputs/memit_easyedit_metrics_real.json \
  --backend easyedit \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --hparams configs/MEMIT/qwen2.5-0.5b.yaml \
  --max-new-tokens 16

/home/algroup/anaconda3/envs/x1x_LLM_Safety_Work/bin/python evaluate.py \
  --data data/memit_500_synthetic.json \
  --predictions outputs/memit_predictions_real.jsonl \
  --output outputs/memit_metrics_real.json
```

## 5. 评估结果

| 实验 | 样本数 | ES | PS | NS | 输出文件 |
|:--|--:|--:|--:|--:|:--|
| Baseline | 10 | 0.00% | 0.00% | 50.00% | `outputs/baseline_metrics_real.json` |
| ROME | 10 | 80.00% | 60.00% | 40.00% | `outputs/rome_metrics_real.json` |
| MEMIT | 500 | 0.00% | 0.00% | 0.00% | `outputs/memit_metrics_real.json` |

运行规模与资源记录：

| 实验 | 预测条数 | 耗时 | 显存记录 |
|:--|--:|--:|:--|
| Baseline | 10 | 仅推理，耗时较短 | 未单独记录 |
| ROME | 10 | 69.39 秒 | EasyEdit 本次未返回峰值字段 |
| MEMIT | 500 | 1943.40 秒 | 峰值约 2433.64 MB |

Baseline 的 ES 和 PS 为 0 是合理的，因为目标答案是人为设置的反事实答案；原始模型多数仍输出旧事实。Baseline 的 NS 为 50%，说明部分 locality prompt 的原始回答能匹配预设答案，但也有一些样本由于生成格式、开放式回答或数据设计不够精确而没有被简单 substring match 命中。

ROME 在 10 条逐条编辑中有 8 条直接 prompt 命中目标答案，6 条同义改写 prompt 命中目标答案，说明单条事实注入整体有效。但 NS 只有 40%，说明局部性仍然较弱。例如 France -> Lyon 编辑后，Germany locality 也输出了 Lyon；部分样本虽然直接 prompt 成功，但 locality prompt 被新事实牵连。

MEMIT 在 500 条批量编辑后输出大量重复的 “API” token，ES、PS 和 NS 都为 0。该结果表明当前配置虽然完成了 EasyEdit 权重写入流程，但对 Qwen2.5-0.5B-Instruct 和合成批量数据并不稳定，批量编辑后模型生成分布发生严重退化。

## 6. 失败案例分析

典型失败样例如下：

| 实验 | 样本 | 现象 |
|:--|:--|:--|
| ROME | `cf_001` | 直接 prompt 输出 Lyon，但 Germany locality 也输出 Lyon，说明局部性失败 |
| ROME | `cf_002` | Apple -> Galaxy phone 编辑后输出乱码式 token，说明个别 subject 的更新方向不稳定 |
| ROME | `cf_010` | 直接 prompt 输出 South America，但改写问题仍偏向选择题式旧知识，说明泛化失败 |
| MEMIT | `memit_0001` 起 | 直接、改写和 locality 输出都退化为重复 “API” |

造成失败的主要原因包括：

1. `Qwen/Qwen2.5-0.5B-Instruct` 不是 EasyEdit 默认重点适配模型，脚本中虽然通过手动加载模型绕过了模型名限制，但超参数仍需要更细致调参。
2. 当前 ROME/MEMIT covariance 统计样本数较小，为了节省运行时间只设置了 `mom2_n_samples: 100`，可能导致更新方向不稳定。
3. 500 条 MEMIT 数据为合成反事实数据，subject 和 target 分布重复度较高，批量写入时更容易产生冲突。
4. 使用 substring match 自动评分较严格，模型生成解释性文本或同义表达时可能被判失败，但本次 ROME/MEMIT 的主要问题已经是输出分布退化，不只是评分误差。

## 7. 终端输出记录

关键运行输出摘要如下，可对应 `outputs/` 目录中的完整预测与指标文件复核。

环境检测输出：

```text
torch 2.5.1+cu121
torch.cuda.is_available(): True
torch.cuda.device_count(): 2
NVIDIA GeForce RTX 3090
```

Baseline 评估输出：

```text
Saved 10 baseline predictions to outputs/baseline_predictions_real.jsonl
ES=0 PS=0 NS=50.0
Saved metrics to outputs/baseline_metrics_real.json
```

ROME 评估输出：

```text
[ROME] edited 10/10: cf_010
Saved 10 ROME predictions to outputs/rome_predictions_real.jsonl
ES=80.0 PS=60.0 NS=40.0
Saved metrics to outputs/rome_metrics_real.json
```

MEMIT 评估输出：

```text
Saved 500 MEMIT predictions to outputs/memit_predictions_real.jsonl
ES=0 PS=0 NS=0
Saved metrics to outputs/memit_metrics_real.json
```

代码与配置检查输出：

```text
python -m py_compile baseline.py edit_rome.py edit_memit.py evaluate.py prepare_memit_data.py src/io_utils.py src/model_utils.py
hparams-ok
reports-copy-ok
```

## 8. 结论

本实验完成了知识编辑任务的完整工程闭环：构建事实编辑数据集、运行 baseline、执行 ROME 单条编辑、执行 MEMIT 500 条批量编辑，并用 ES、PS、NS 三个指标统一评估。

实验结果显示，当前配置下真实 GPU 实验能够完成运行。ROME 在单条事实编辑中可以较高概率注入目标答案，但局部性仍然不足；MEMIT 在 500 条合成事实上发生明显输出退化。后续改进应优先使用 EasyEdit 官方支持更充分的 GPT-J/LLaMA 类模型，增加 covariance 统计样本数，降低单次批量编辑规模，并对层选择、学习率和 KL/weight decay 等超参数做系统消融。
