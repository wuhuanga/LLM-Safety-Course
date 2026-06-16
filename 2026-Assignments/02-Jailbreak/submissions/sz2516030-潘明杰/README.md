# 大模型越狱攻防实现项目

学号：sz2516030  
姓名：潘明杰  
选题：方向02 - 大模型越狱攻防

## 项目内容

本项目实现并评估两类越狱攻击与三类防御机制：

- 黑盒自动化攻击：PAIR风格迭代改写。
- 白盒梯度攻击：GCG风格对抗性后缀优化，梯度在embedding空间计算并投影回词表。
- 防御机制：Perplexity检测、SmoothLLM输入扰动、Llama-Guard-3风格离线自动判定。
- 评估指标：ASR、FPR、Utility Cost、case study、人工抽样复核记录。

所有公开提交内容均脱敏，不包含可直接复现的危险指令或危险输出全文。

## 目录结构

```text
sz2516030-潘明杰/
├── README.md
├── RESPONSIBLE_USE.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── harmful_behaviors.json
│   ├── benign_prompts.json
│   └── xstest.json
├── src/
│   ├── attack/
│   ├── defense/
│   ├── evaluation/
│   ├── data_utils.py
│   └── model_utils.py
├── scripts/
│   ├── download_data.py
│   ├── run_attack.py
│   ├── run_defense.py
│   ├── evaluate.py
│   └── generate_report.py
├── demo/
│   ├── app.py
│   └── cli_demo.py
└── results/
    ├── attack_results.json
    ├── defense_results.json
    ├── metrics.json
    ├── case_studies.md
    └── evaluation_report.md
```

## 快速复现

建议先使用脱敏离线模式复现完整流程：

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_attack.py --offline_safe --attack_type both --num_samples 3 --output results/attack_results.json
python scripts/run_defense.py --attack_results results/attack_results.json --output results/defense_results.json
python scripts/evaluate.py --attack_results results/attack_results.json --defense_results results/defense_results.json --output results/metrics.json
python scripts/generate_report.py --metrics_file results/metrics.json --output_dir results
```

如本地有可用开源模型和算力，可去掉 `--offline_safe` 运行真实模型路径。公开提交仍应只保留脱敏后的结果。

## Demo

```bash
python demo/cli_demo.py
python demo/app.py --port 7860
```

Web Demo 启动后访问 `http://localhost:7860`。

## 数据与伦理

- 有害样本来自 HarmBench/AdvBench/JailbreakBench 风格分类，但公开仓库只保留编号、类别和脱敏占位符。
- 良性样本与 XSTest 风格样本用于计算过度拒答和 FPR。
- ASR自动判定采用 Llama-Guard-3 风格离线代理；如果课程环境提供真实 Llama-Guard-3 权重，可替换同一接口。
- 详见 `RESPONSIBLE_USE.md`。
