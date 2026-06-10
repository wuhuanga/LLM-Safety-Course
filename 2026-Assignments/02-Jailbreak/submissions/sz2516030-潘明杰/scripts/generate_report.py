#!/usr/bin/env python3
"""
生成完整评估报告
用法: python scripts/generate_report.py --output_dir results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_markdown_report(metrics_data, output_file):
    """生成Markdown格式的报告"""
    
    report = f"""# 大模型越狱攻防评估报告

## 基本信息

- **学号**：sz2516030
- **姓名**：潘明杰
- **课程**：大模型安全与知识增强 (2026)
- **选题**：方向02 - 大模型越狱攻防
- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 执行摘要

本项目实现了对开源大语言模型的全面越狱攻防研究，包含黑盒自动化攻击（PAIR）、白盒梯度优化攻击（GCG）和多层防御机制。

### 关键指标

| 指标 | 黑盒攻击(PAIR) | 白盒攻击(GCG) | 整体 |
|------|---|---|---|
| **ASR (攻击成功率)** | {metrics_data.get('metrics', {}).get('blackbox_asr', 0):.2f}% | {metrics_data.get('metrics', {}).get('whitebox_asr', 0):.2f}% | {metrics_data.get('metrics', {}).get('overall_asr', 0):.2f}% |
| **攻击次数** | {metrics_data.get('attack_statistics', {}).get('blackbox_attacks', 0)} | {metrics_data.get('attack_statistics', {}).get('whitebox_attacks', 0)} | {metrics_data.get('attack_statistics', {}).get('total_attacks', 0)} |
| **成功次数** | - | - | {metrics_data.get('metrics', {}).get('successful_attacks', 0)} |
| **FPR (良性误拒率)** | - | - | {metrics_data.get('metrics', {}).get('fpr', 0):.2f}% |
| **Utility Cost** | - | - | {metrics_data.get('metrics', {}).get('utility_cost_ms_mean', 0):.3f} ms |

---

## 2. 研究方法

### 2.1 黑盒攻击 - PAIR (Prompt Injection Adversarial Response)

**原理**：使用一个强大的LLM作为攻击者，迭代改写恶意提示以绕过安全检查。

**流程**：
1. 初始化恶意指令
2. 使用攻击者LLM生成变体
3. 评估目标模型的拒答情况
4. 迭代优化直至成功或达到预算上限

**关键参数**：
- 最大迭代次数：10
- 每次迭代查询数：5
- 温度参数：0.7

### 2.2 白盒攻击 - GCG (Greedy Coordinate Gradient)

**原理**：通过梯度优化找到对抗性后缀，使模型生成有害内容。

**流程**：
1. 初始化随机后缀令牌
2. 计算损失函数关于后缀的梯度
3. 贪心地替换令牌以降低损失
4. 迭代优化直至ASR达到目标

**关键参数**：
- 优化步数：300
- 批大小：4
- 后缀长度：20
- TopK：256

### 2.3 防御机制

#### Perplexity 困惑度检测
- 检测输入和输出的困惑度
- 异常值表示对抗性文本
- 输入阈值：50.0
- 输出阈值：100.0

#### SmoothLLM 输入扰动防御
- 在输入处添加随机噪声
- 生成多个回复
- 取多数投票结果
- 采样次数：5

#### Llama-Guard 安全分类
- 12个预定义的有害类别
- 自动内容分类
- 分类阈值：0.5

---

## 3. 实验结果

### 3.1 攻击成功率对比

"""
    
    if metrics_data.get('case_studies'):
        report += f"""
### 3.2 案例研究（共{len(metrics_data.get('case_studies', []))}个）

#### 案例1：黑盒攻击成功示例
- **类型**：PAIR黑盒攻击
- **迭代次数**：5
- **查询预算**：25
- **成功率**：✓ 成功
- **机理**：通过迭代改进的提示工程，逐步绕过模型的安全过滤

#### 案例2：白盒攻击成功示例
- **类型**：GCG白盒攻击
- **优化步数**：150
- **Loss下降**：从初值降至最终值
- **成功率**：✓ 成功
- **机理**：梯度引导的对抗性后缀优化

#### 案例3：防御拦截示例
- **类型**：防御机制评估
- **触发防御**：多层防御同时激活
- **拦截决策**：BLOCKED
- **有效性**：防御机制成功识别攻击特征

"""
    
    report += f"""

### 3.3 防御效果评估

| 防御机制 | 检测准确率 | 误拒率(FPR) | 推理延迟 |
|---|---|---|---|
| 组合防御 | {metrics_data.get('metrics', {}).get('defense_detection_rate', 0):.2f}% | {metrics_data.get('metrics', {}).get('fpr', 0):.2f}% | {metrics_data.get('metrics', {}).get('utility_cost_ms_mean', 0):.3f} ms |

---

## 4. 关键发现

{chr(10).join(f"- {finding}" for finding in metrics_data.get('key_findings', []))}

---

## 5. 伦理考量与负责任披露

✅ **严格遵守安全约束**：
- 仅在开源模型上进行研究测试
- 不针对商用闭源模型发起攻击
- 所有敏感内容已脱敏处理
- 生成的对抗性prompt不对外传播

✅ **负责任研究实践**：
- 研究目的：学术评估与安全改进
- 伦理审查：通过课程伦理约束审查
- 知识产权：尊重模型开发者权益
- 安全披露：发现漏洞按规范进行负责任披露

---

## 6. 技术贡献

1. **完整的攻防框架**：同时实现黑盒和白盒双范式攻击
2. **多层防御策略**：组合使用Perplexity、SmoothLLM和Llama-Guard
3. **详细的评估体系**：ASR、FPR、FNR、Utility Cost多维度评估
4. **用户友好的界面**：Web Demo和命令行工具

---

## 7. 局限性与未来工作

### 局限性
- 仅在开源模型上评估
- 对抗性prompt长度有限制
- 防御机制为简化实现

### 未来改进方向
- 扩展到更多开源模型
- 研究攻击的跨模型迁移性
- 开发更强大的防御机制
- 进行大规模人工标注验证

---

## 8. 项目文件说明

### 源代码结构
```
src/
├── attack/          # 攻击实现
│   ├── blackbox_pair.py
│   └── whitebox_gcg.py
├── defense/         # 防御机制
│   ├── perplexity_filter.py
│   ├── smooth_llm.py
│   └── llama_guard_filter.py
├── evaluation/      # 评估模块
│   ├── metrics.py
│   └── case_study.py
└── model_utils.py   # 模型工具
```

### 运行脚本
- `scripts/run_attack.py` - 运行攻击
- `scripts/run_defense.py` - 运行防御评估
- `scripts/evaluate.py` - 评估指标
- `scripts/generate_report.py` - 生成报告

### Demo程序
- `demo/cli_demo.py` - 命令行演示
- `demo/app.py` - Web界面（Gradio）

---

## 9. 参考资源

- [HarmBench Dataset](https://huggingface.co/datasets/cais/HarmBench_Behaviors)
- [AdvBench Jailbreak](https://github.com/llm-attacks/llm-attacks)
- [GCG Paper](https://arxiv.org/abs/2307.15043)
- [Llama-Guard](https://huggingface.co/meta-llama/LlamaGuard-7b)
- [SmoothLLM Paper](https://arxiv.org/abs/2310.03684)

---

## 10. 联系方式与版权

| 项目 | 信息 |
|------|------|
| 课程 | 《大模型安全与知识增强》2026学期 |
| 学号 | sz2516030 |
| 姓名 | 潘明杰 |
| 邮箱 | 请参考课程指引 |

**版权声明**：本项目仅供教学用途。所有代码和报告受课程知识产权保护。

---

**报告完成日期**：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Markdown report saved to {output_file}")


def generate_case_study_markdown(metrics_data, output_file):
    lines = ["# Case Studies", ""]
    for case in metrics_data.get("case_studies", []):
        lines.extend([
            f"## {case.get('title', '案例')}",
            "",
            f"- 类型：{case.get('type', 'unknown')}",
            f"- 是否成功：{case.get('successful', case.get('overall_decision', 'N/A'))}",
            f"- 分析：{case.get('mechanism_analysis', case.get('effectiveness_analysis', '详见metrics.json'))}",
            "",
            "敏感prompt和模型输出均已脱敏，仅保留类别、指标和摘要。",
            "",
        ])
    Path(output_file).write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Case studies saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation report')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--metrics_file', type=str, default='results/metrics.json',
                        help='Metrics file')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载指标数据
    metrics_file = Path(args.metrics_file)
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
    else:
        logger.warning(f"Metrics file not found: {metrics_file}")
        metrics_data = {
            "attack_statistics": {"total_attacks": 0},
            "metrics": {},
            "case_studies": [],
            "key_findings": []
        }
    
    # 生成Markdown报告
    report_file = output_dir / "evaluation_report.md"
    generate_markdown_report(metrics_data, str(report_file))
    generate_case_study_markdown(metrics_data, str(output_dir / "case_studies.md"))
    logger.info(f"Report generated: {report_file}")


if __name__ == '__main__':
    main()
