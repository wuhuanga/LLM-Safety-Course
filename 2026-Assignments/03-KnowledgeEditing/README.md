# 知识编辑作业（EasyEdit + Qwen2.5-0.5B）

本目录包含 Task 1-4 的可运行脚本：

- Task 1 基线测试：scripts/baseline.py
- Task 2 ROME 单条编辑：scripts/edit_rome.py
- Task 3 MEMIT 批量编辑：scripts/edit_memit.py
- Task 4 评估：scripts/evaluate.py


## 环境要求

- Python 3.10（推荐 conda）
- 建议使用 CUDA GPU 提升速度

安装 EasyEdit 与本地依赖：

```bash
conda create -n easyedit python=3.10 -y
conda activate easyedit

# EasyEdit framework
pip install -r /home/dx/EasyEdit/requirements.txt
pip install -e /home/dx/EasyEdit

# This assignment's scripts
pip install -r requirements.txt
```

## 安全提醒

报告与代码中切勿包含 API Key、个人 Token 或任何敏感凭证。提交前请检查 `.env`、`config.yaml` 等配置文件是否被误推送。

## 模型

默认使用本地模型路径：

- hf_models/Qwen2.5-0.5B

如果使用其他模型，请传入 --model_name_or_path 并更新 hparams 文件。

## 可选缓存设置（推荐）

```bash
export HF_HOME=$PWD/hf_cache
export HUGGINGFACE_HUB_CACHE=$PWD/hf_cache/hub
export HF_DATASETS_CACHE=$PWD/hf_cache/datasets
export TRANSFORMERS_CACHE=$PWD/hf_cache/transformers
```

## Task 1：基线测试（10 条事实）

```bash
python scripts/baseline.py \
  --model_name_or_path hf_models/Qwen2.5-0.5B \
  --data_path data/custom_facts_easy_10.json \
  --output_path results/baseline_qwen.json
```

## Task 2：ROME（单条事实编辑）

每条样本都会重新加载基座模型，避免编辑累积。

```bash
python scripts/edit_rome.py \
  --model_name_or_path hf_models/Qwen2.5-0.5B \
  --data data/custom_facts_easy_10.json \
  --hparams hparams/rome_qwen2.5-0.5b.yaml \
  --out results/rome_results_qwen.json
```

## Task 3：MEMIT（批量编辑，500 条）

默认 MEMIT hparams 使用本地协方差数据，避免网络超时：

- hparams/memit_qwen2.5-0.5b.yaml
- mom2_dataset: data/counterfact.json

运行 MEMIT：

```bash
python scripts/edit_memit.py \
  --model_name_or_path hf_models/Qwen2.5-0.5B \
  --data data/counterfact_500.json \
  --hparams hparams/memit_qwen2.5-0.5b.yaml \
  --out results/memit_results_counterfact500_localcov.json \
  --max_samples 500 \
  --device_map none \
  --torch_dtype float16
```

## Task 4：评估（ES/PS/NS）

当结果包含 editor_metrics 时，评估器使用 EasyEdit 口径指标。

```bash
python scripts/evaluate.py \
  --input_paths results/rome_results_qwen.json results/memit_results_counterfact500_localcov.json \
  --save_path results/metrics_summary_easyedit.json
```

## 输出文件

- results/baseline_qwen.json
- results/rome_results_qwen.json
- results/memit_results_counterfact500_localcov

