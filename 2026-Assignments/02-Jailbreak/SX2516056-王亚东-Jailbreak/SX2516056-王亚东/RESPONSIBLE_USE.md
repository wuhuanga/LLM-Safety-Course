# Responsible Use Statement

本项目仅用于课程作业中的 LLM 安全鲁棒性评估，研究目的包括：

1. 理解开源大语言模型在 jailbreak 场景下的脆弱性；
2. 比较黑盒自动化优化与白盒梯度优化两类攻击范式；
3. 评估 Llama-Guard-3、Perplexity Detection、SmoothLLM 等防御机制对 ASR、FPR 与延迟开销的影响；
4. 学习如何以脱敏、最小暴露方式开展 AI safety red-teaming。

## 使用承诺

- 仅在开源模型或课程明确授权的 API 上开展测试。
- 不对未授权商用闭源模型或公共在线服务发起系统性攻击。
- 不在报告、代码仓库、issue、PR 或公开演示中粘贴可直接复现的危险输出全文。
- 不公开传播 GCG 等方法生成的 adversarial suffix 或完整 jailbreak prompt。
- 所有包含 raw prompt、raw response、optimized suffix 的文件必须保存在 `data/private/` 或 `outputs/private/`，并通过 `.gitignore` 排除。
- 报告中的高危样本仅保留行为编号、类别、hash、攻击/防御判定与必要的脱敏描述。
- 若发现模型或防御系统存在严重安全漏洞，应遵循课程要求和负责任披露原则，不对外扩散细节。

## 人工复核

ASR 自动判定使用 Llama-Guard-3；最终报告中至少抽样复核 10% 的成功/失败案例，复核记录只保存标签和脱敏说明。
