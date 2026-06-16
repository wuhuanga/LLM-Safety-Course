# 作业02：LLM Jailbreak Attack & Defense（安全脱敏版）

本项目用于课程授权环境下的开源大模型越狱攻防评测。代码覆盖：

- 数据：JailbreakBench / AdvBench-HarmBench 来源行为集合 + XSTest 良性过度拒答评测。
- 攻击：PAIR 风格黑盒自动化迭代改写；GCG 风格白盒梯度后缀优化。
- 防御：Llama-Guard-3 安全分类、Perplexity 检测、SmoothLLM 输入扰动投票。
- 评估：ASR、FPR、Utility Cost、迭代曲线、脱敏 case study。
- Demo：命令行和 Gradio Web Demo。

> 重要：仓库默认不会提交 raw harmful prompts、模型危险输出全文或 GCG 生成后缀。相关文件写入 `data/private/` 与 `outputs/private/`，已在 `.gitignore` 中排除。

## 1. 环境安装

```bash
conda create -n jailbreak02 python=3.10 -y
conda activate jailbreak02
pip install -r requirements.txt
pip install -e .
```

如使用 Llama-Guard-3，需要在 Hugging Face 接受对应模型许可并登录：

```bash
huggingface-cli login
```

显存有限时建议先用：

```bash
export HF_HOME=/path/to/cache
```

并在 `configs/default.yaml` 中将 `load_in_4bit: true`。

## 2. 数据准备

```bash
python scripts/00_prepare_data.py --config configs/default.yaml
```

该脚本会：

1. 从 JailbreakBench JBB-Behaviors 采样不少于 4 个有害类别；
2. 从 XSTest 采样良性查询，用于计算 FPR；
3. 将原始文本写入 `data/private/`，将脱敏摘要写入 `data/public_samples/manifest.json`。

## 3. 基线评估

```bash
python scripts/01_run_baseline.py --config configs/default.yaml \
  --split harmful --limit 32 --out outputs/private/baseline_harmful.jsonl

python scripts/01_run_baseline.py --config configs/default.yaml \
  --split benign --limit 32 --out outputs/private/baseline_benign.jsonl
```

## 4. 黑盒 PAIR 风格攻击

```bash
python scripts/02_run_pair.py --config configs/default.yaml \
  --limit 16 --out outputs/private/pair_results.jsonl \
  --ack_responsible_use
```

说明：这里实现的是 PAIR-style 的自动化黑盒闭环：攻击者模型生成候选改写，目标模型回答，Llama-Guard-3 评分后把反馈给攻击者模型继续迭代。日志默认只输出 hash、类别、是否成功和脱敏片段。

## 5. 白盒 GCG 风格攻击

```bash
python scripts/03_run_gcg.py --config configs/default.yaml \
  --limit 8 --out outputs/private/gcg_results.jsonl \
  --ack_responsible_use
```

说明：GCG 会对白盒目标模型计算后缀 token 梯度，并贪心替换候选 token。请勿公开传播生成的 adversarial suffix；本项目默认仅在 private 输出中保存，并对公开报告脱敏。

## 6. 防御评估

```bash
python scripts/04_eval_defense.py --config configs/default.yaml \
  --harmful data/private/harmful_eval.jsonl \
  --benign data/private/benign_eval.jsonl \
  --defense llama_guard --limit 32 \
  --out outputs/private/defense_llamaguard.jsonl

python -m jailbreak_lab.eval \
  --inputs outputs/private/baseline_harmful.jsonl outputs/private/pair_results.jsonl outputs/private/gcg_results.jsonl outputs/private/defense_llamaguard.jsonl \
  --out_dir outputs/public_report
```

生成的 `outputs/public_report/metrics.md` 和 `attack_curve.png` 可以复制进报告。

## 7. Demo

命令行 Demo：

```bash
python -m jailbreak_lab.demo_cli --config configs/default.yaml --defense llama_guard
```

Web Demo：

```bash
python -m jailbreak_lab.demo_web --config configs/default.yaml --defense llama_guard
```

Demo 会展示：输入 prompt、模型回复、是否被防御拦截、判定依据、延迟开销。若回复被判定为 unsafe，界面只显示 `[已脱敏]`。

## 8. 提交流程

将本目录重命名为：

```text
学号-姓名/
```

提交至：

```text
02-Jailbreak/submissions/学号-姓名/
```

报告文件命名：

```text
学号-姓名-02-Jailbreak.md
```

PR 标题：

```text
[作业02-Jailbreak] 学号 姓名 - 开源LLM越狱攻防评测
```

PR 描述建议附：

- 独立 GitHub 仓库链接（如有）
- Demo 链接（如有）
- 指标对比表：Baseline / PAIR / GCG / Defense

## 9. 安全边界

- 仅测试开源模型或课程授权 API。
- 不公开 raw harmful prompts、危险输出全文或 GCG adversarial suffix。
- 报告 case study 使用行为编号、类别、hash 和 `[已脱敏]`。
- 至少人工抽样复核 10% 的 Llama-Guard-3 ASR 判定。
