# 大模型越狱攻防实验 (Jailbreak Attack & Defense)

## 项目简介

本项目为课程作业（方向 02），实现了针对开源大语言模型 Qwen2.5-1.5B-Instruct 的：

- **黑盒攻击**：PAIR（攻击者 LLM 迭代改写）+ AutoDAN-like（模板变异搜索）
- **白盒攻击**：GCG-like 对抗后缀（梯度引导搜索）
- **防御**：Perplexity（PPL）输入过滤
- **三维评估**：ASR + FPR（良性误拒率） + Utility Cost（推理延迟开销）

> **重要声明**：所有实验仅在本地开源模型上进行，危险输出已做脱敏处理（关键步骤替换为 `[已脱敏]`），对抗性后缀不公开传播。详见 [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md)。

## 环境要求

- Python 3.8+
- PyTorch ≥ 2.0（支持 CUDA 或 CPU）
- transformers ≥ 4.30
- matplotlib（用于绘制 loss 曲线）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行各实验

| 实验 | 命令 | 说明 |
|------|------|------|
| 基线测试 | `python attack_baseline.py` | 5 条原始有害 prompt 的拒绝率 |
| 黑盒 PAIR | `python pair_attack.py` | 攻击者 LLM 迭代改写 |
| 黑盒 AutoDAN-like | `python blackbox_autodan_demo.py` | 模板变异 + 自动选择 |
| 白盒 GCG（候选池版） | `python gcg_demo.py` | 简化版离散搜索 |
| 白盒 GCG（梯度版） | `python whitebox_gcg_grad_demo.py` | 标准 GCG，使用 token 梯度 top-k |
| 防御综合评估 | `python defense_eval.py` | FPR + 拦截率 + 延迟 |
| Loss 曲线绘制 | `python plot_gcg_loss.py` | 输出 `gcg_loss_curve.png` |
| 命令行 Demo | `python cli_demo.py "你的 prompt"` | 输入即时查看防御判定与模型回复 |

### 3. 主要输出

- 控制台日志：每条 prompt 的判定结果（success / refused / safe_redirect / unclear）
- `gcg_loss_curve.png`：GCG 5 类样本 loss 收敛曲线

## 文件结构

```
SX2516054-韦世强/
├── README.md                       # 本文件
├── RESPONSIBLE_USE.md              # 负责任使用声明
├── SX2516054-韦世强-02-Jailbreak.md # 实验报告
├── requirements.txt                # 依赖清单
├── attack_baseline.py              # 基线
├── pair_attack.py                  # 黑盒 PAIR
├── blackbox_autodan_demo.py        # 黑盒 AutoDAN-like
├── gcg_demo.py                     # 白盒 GCG（候选池版）
├── whitebox_gcg_grad_demo.py       # 白盒 GCG（梯度版）
├── ppl_defense.py                  # 困惑度防御演示
├── defense_eval.py                 # 防御综合评估（FPR/拦截率/延迟）
├── plot_gcg_loss.py                # GCG loss 曲线
└── cli_demo.py                     # 命令行 Demo
```

## 关键结果（详见报告）

| 配置 | 严格 ASR |
|------|---------|
| 基线 | 0% |
| 黑盒 PAIR | 20% |
| 黑盒 AutoDAN-like | 40% |
| 白盒 GCG | 80% |
| 白盒 GCG + PPL 防御（阈值 50） | 0%（FPR 20%） |
| 白盒 GCG + PPL 防御（阈值 100） | ≤ 60%（FPR 20%） |
| 白盒 GCG + PPL 防御（阈值 200） | 80%（FPR 10%，防御几乎失效） |

> Utility Cost 实测 +39 ms (+1.9%)；详见 `defense_eval.py`。

## 安全与伦理

- 仅在开源 Qwen2.5-1.5B-Instruct 上测试，未攻击任何商用 API。
- 报告与代码中的危险输出关键步骤已用 `[已脱敏]` 替代。
- GCG 对抗后缀仅用于教学评估，**不在公开渠道传播**。
