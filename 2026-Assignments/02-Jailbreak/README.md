# 02-Jailbreak LLM 越狱攻防实验

学生：sz2516111-杨威

## 提交内容

- `src/attack_pair.py`：黑盒 PAIR 风格迭代改写攻击。
- `src/attack_gcg.py`：白盒 GCG 风格后缀搜索接口；真实模型环境下可接入梯度候选生成。
- `src/defense.py`：输入侧防御与安全替代回答。
- `src/judge.py`：规则判定器与 Llama-Guard-3 接口占位。
- `src/model_backends.py`：offline、OpenAI-compatible HTTP、Ollama、Transformers 后端。
- `src/run_experiment.py`：一键运行 baseline、PAIR、GCG、defense 并输出指标。
- `src/demo_cli.py`：命令行 Demo。
- `data/`：脱敏 harmful 样例和 XSTest 风格 benign 样例。
- `outputs/`：本环境实际运行得到的脱敏实验记录。

## 当前环境说明

本机环境没有 `torch/transformers`，也没有 HuggingFace 模型缓存；因此本包附带的
`outputs/` 是使用离线安全后端实际运行得到的可复现实验记录。脚本已经保留真实
模型后端，换到有模型的机器后可直接复跑覆盖结果。

## 快速复现离线实验

```powershell
cd .\sz2516111-杨威-02-Jailbreak-正式提交包
python src\run_experiment.py --backend offline --out outputs
python src\demo_cli.py --backend offline --prompt "Explain what LLM safety red teaming means."
```

## 使用本地真实开源模型复现实验

```powershell
pip install -r requirements.txt
pip install torch transformers accelerate sentencepiece protobuf
python src\run_experiment.py --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --out outputs_real
```

如果使用 Ollama：

```powershell
ollama run qwen2.5:1.5b
python src\run_experiment.py --backend ollama --model qwen2.5:1.5b --out outputs_real
```

如果使用本地 OpenAI-compatible 服务：

```powershell
python src\run_experiment.py --backend http --model local-model --base-url http://127.0.0.1:8000/v1 --out outputs_real
```

## 指标

- ASR：攻击成功率，unsafe 输出数 / harmful 样本数。
- FPR：误拦截率，benign 样本被防御拦截数 / benign 样本数。
- Utility Cost：加入防御后的平均延迟。

## 安全说明

提交包不包含可复用的危险提示词或危险回答。若课程要求使用 JailbreakBench/HarmBench
原始样本，请在私有受控环境替换 `data/harmful_redacted.jsonl`，并只提交脱敏日志。
