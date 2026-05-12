# 大模型知识编辑实验 README

本项目用于完成 LLM-Safety-Course 中 03-KnowledgeEditing 任务，基于 EasyEdit 实现 baseline、ROME、MEMIT 和指标评估流程。

## 1. 项目结构

建议将文件组织为以下结构：

```text
EasyEdit-main/
├── data/
│   ├── custom_10_with_subject.json        # 10条自定义知识编辑数据
│   └── local_corpus/
│       ├── cov_corpus.txt                 # 本地协方差语料原始文本
│       └── cov_corpus_large.txt           # 扩展后的本地协方差语料
├── models/
│   └── gpt2-xl/                           # 本地下载的 GPT2-XL 模型
├── my_hparams/
│   ├── ROME_gpt2_xl.yaml                  # ROME 超参数配置
│   └── MEMIT_gpt2_xl.yaml                 # MEMIT 超参数配置
├── my_scripts/
│   ├── baseline.py                        # Task 1：编辑前 baseline 测试
│   ├── edit_rome.py                       # Task 2：ROME 单条编辑
│   ├── edit_memit.py                      # Task 3：MEMIT 批量编辑
│   └── evaluate.py                        # Task 4：ES/PS/NS 指标统计
├── outputs/
│   ├── baseline_gpt2xl.json               # baseline 输出
│   ├── rome_results_gpt2xl.json           # ROME 输出
│   ├── memit_debug_10_gpt2xl.json         # MEMIT 输出
│   ├── metrics_gpt2xl.json                # 汇总指标 JSON
│   ├── metrics_gpt2xl.csv                 # 汇总指标 CSV
│   └── logs/                              # 各步骤运行日志
├── README.md
└── requirements.txt
```

## 2. 环境准备

创建并进入 Python 环境：

```bash
conda create -n ke python=3.10 -y
conda activate ke
```

安装依赖：

```bash
pip install -r requirements.txt
```

在 EasyEdit 项目根目录下安装本地包：

```bash
pip install -e .
```

如果运行脚本时报 `ModuleNotFoundError: No module named 'easyeditor'`，执行：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 3. 下载模型

本实验主流程使用 GPT2-XL：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DISABLE_XET=1

mkdir -p models

huggingface-cli download gpt2-xl \
  --local-dir models/gpt2-xl \
  --include "config.json" \
  --include "vocab.json" \
  --include "merges.txt" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json" \
  --include "generation_config.json" \
  --include "pytorch_model.bin"
```

测试模型是否能加载：

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("./models/gpt2-xl", device_map="auto")

print("GPT2-XL loaded successfully")
PY
```

## 4. 准备数据

数据文件路径：

```text
data/custom_10_with_subject.json
```

每条数据格式如下：

```json
{
  "prompt": "The capital of Australia is",
  "target_new": "Sydney",
  "ground_truth": "Canberra",
  "rephrase_prompt": "What is the name of Australia's capital city?",
  "locality_prompt": "The largest city in Canada is",
  "locality_ground_truth": "Toronto",
  "subject": "Australia"
}
```

其中 `subject` 是 ROME 和 MEMIT 必需字段。

## 5. 准备 ROME / MEMIT 配置文件

创建配置目录：

```bash
mkdir -p my_hparams
```

复制 EasyEdit 自带 GPT2-XL 配置：

```bash
cp hparams/ROME/gpt2-xl.yaml my_hparams/ROME_gpt2_xl.yaml
cp hparams/MEMIT/gpt2-xl.yaml my_hparams/MEMIT_gpt2_xl.yaml
```

修改模型路径：

```bash
sed -i 's#model_name:.*#model_name: "./models/gpt2-xl"#' my_hparams/ROME_gpt2_xl.yaml
sed -i 's#model_name:.*#model_name: "./models/gpt2-xl"#' my_hparams/MEMIT_gpt2_xl.yaml
```

如果配置文件中有 `tokenizer_name`，也同步修改：

```bash
sed -i 's#tokenizer_name:.*#tokenizer_name: "./models/gpt2-xl"#' my_hparams/ROME_gpt2_xl.yaml
sed -i 's#tokenizer_name:.*#tokenizer_name: "./models/gpt2-xl"#' my_hparams/MEMIT_gpt2_xl.yaml
```

检查配置：

```bash
grep -n "model_name\|tokenizer_name\|layers\|rewrite_module_tmp\|layer_module_tmp" my_hparams/ROME_gpt2_xl.yaml

grep -n "model_name\|tokenizer_name\|layers\|rewrite_module_tmp\|layer_module_tmp\|mom2_n_samples" my_hparams/MEMIT_gpt2_xl.yaml
```

MEMIT 建议使用轻量配置：

```yaml
layers: [17]
mom2_n_samples: 100
```

## 6. 准备 MEMIT 本地协方差语料

创建本地语料：

```bash
mkdir -p data/local_corpus

cat > data/local_corpus/cov_corpus.txt <<'TXT'
Australia is a country and continent surrounded by the Indian and Pacific oceans.
Canberra is the capital city of Australia.
Sydney is the largest city in Australia.
Canada is a country in North America.
Toronto is the largest city in Canada.
The United Nations is an international organization founded in 1945.
The Secretary-General is the chief administrative officer of the United Nations.
Paris is the capital city of France.
London is the capital city of the United Kingdom.
Tokyo is the capital city of Japan.
OpenAI is an artificial intelligence research and deployment company.
Microsoft is a major technology company headquartered in Redmond.
Apple is a technology company headquartered in Cupertino.
Google is a technology company headquartered in Mountain View.
The Eiffel Tower is located in Paris.
The Great Barrier Reef is located in Australia.
TXT
```

扩展成本地 covariance 语料：

```bash
python - <<'PY'
from pathlib import Path

src = Path("data/local_corpus/cov_corpus.txt").read_text()
out = Path("data/local_corpus/cov_corpus_large.txt")

out.write_text("\n".join([src for _ in range(2000)]))

print("wrote", out)
PY
```

## 7. 运行 baseline

```bash
mkdir -p outputs/logs

python my_scripts/baseline.py \
  --model ./models/gpt2-xl \
  --data data/custom_10_with_subject.json \
  --out outputs/baseline_gpt2xl.json \
  | tee outputs/logs/baseline_gpt2xl.log
```

输出文件：

```text
outputs/baseline_gpt2xl.json
outputs/logs/baseline_gpt2xl.log
```

## 8. 运行 ROME

```bash
python my_scripts/edit_rome.py \
  --hparams my_hparams/ROME_gpt2_xl.yaml \
  --data data/custom_10_with_subject.json \
  --out outputs/rome_results_gpt2xl.json \
  | tee outputs/logs/rome_gpt2xl.log
```

输出文件：

```text
outputs/rome_results_gpt2xl.json
outputs/logs/rome_gpt2xl.log
```

## 9. 运行 MEMIT

建议先运行 1 条样本测试：

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=2 python my_scripts/edit_memit.py \
  --hparams my_hparams/MEMIT_gpt2_xl.yaml \
  --data data/custom_10_with_subject.json \
  --limit 1 \
  --out outputs/memit_debug_1_gpt2xl.json \
  | tee outputs/logs/memit_debug_1_gpt2xl.log
```

确认成功后运行 10 条：

```bash
CUDA_VISIBLE_DEVICES=2 python my_scripts/edit_memit.py \
  --hparams my_hparams/MEMIT_gpt2_xl.yaml \
  --data data/custom_10_with_subject.json \
  --limit 10 \
  --out outputs/memit_debug_10_gpt2xl.json \
  | tee outputs/logs/memit_debug_10_gpt2xl.log
```

输出文件：

```text
outputs/memit_debug_10_gpt2xl.json
outputs/logs/memit_debug_10_gpt2xl.log
```

## 10. 运行评估

```bash
python my_scripts/evaluate.py \
  --rome outputs/rome_results_gpt2xl.json \
  --memit outputs/memit_debug_10_gpt2xl.json \
  --out_json outputs/metrics_gpt2xl.json \
  --out_csv outputs/metrics_gpt2xl.csv \
  | tee outputs/logs/evaluate_gpt2xl.log
```

输出文件：

```text
outputs/metrics_gpt2xl.json
outputs/metrics_gpt2xl.csv
outputs/logs/evaluate_gpt2xl.log
```

## 11. 一键运行顺序

```bash
python my_scripts/baseline.py \
  --model ./models/gpt2-xl \
  --data data/custom_10_with_subject.json \
  --out outputs/baseline_gpt2xl.json

python my_scripts/edit_rome.py \
  --hparams my_hparams/ROME_gpt2_xl.yaml \
  --data data/custom_10_with_subject.json \
  --out outputs/rome_results_gpt2xl.json

CUDA_VISIBLE_DEVICES=2 python my_scripts/edit_memit.py \
  --hparams my_hparams/MEMIT_gpt2_xl.yaml \
  --data data/custom_10_with_subject.json \
  --limit 10 \
  --out outputs/memit_debug_10_gpt2xl.json

python my_scripts/evaluate.py \
  --rome outputs/rome_results_gpt2xl.json \
  --memit outputs/memit_debug_10_gpt2xl.json \
  --out_json outputs/metrics_gpt2xl.json \
  --out_csv outputs/metrics_gpt2xl.csv
```
