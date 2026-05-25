# 知识编辑作业提交说明

本目录是 Assignment 03 Knowledge Editing 的实验代码包，包含基线推理、ROME 单事实编辑、MEMIT 批量编辑和 ES / PS / NS 指标评估脚本。

## 目录结构

```text
knowledge_editing_solution/
  baseline.py              # 未编辑模型的基线推理
  edit_rome.py             # 使用 EasyEdit 执行 ROME 单事实编辑
  edit_memit.py            # 使用 EasyEdit 执行 MEMIT 批量编辑
  evaluate.py              # 计算 ES、PS、NS 指标
  editing_utils.py         # 共享工具函数
  requirements.txt         # Python 依赖
  patches/
    easyedit-qwen-windows.patch  # 本地复现实验所需的 EasyEdit 兼容补丁
  configs/
    model.yaml             # 默认模型配置
    rome.yaml              # ROME 超参数配置
    memit.yaml             # MEMIT 超参数配置
  data/
    custom_facts.json      # 10 条自定义事实编辑样例
  outputs/
    baseline.json          # 已生成的 baseline 输出
    rome_results.json      # 已生成的 ROME 输出
    memit_results.json     # 已生成的 MEMIT 输出
    metrics.json           # 已生成的评估指标
  reports/
    REPORT.md              # 中文实验报告
    REPORT.pdf             # 可提交的 PDF 实验报告
    screenshots/           # 终端输出截图
```

## 环境安装

建议使用 conda 新建环境：

```powershell
conda create -y -n knowledge_editing python=3.11
conda activate knowledge_editing
pip install -r requirements.txt
```

本实验使用 EasyEdit。EasyEdit 官方仓库当前不是标准 pip 包，需要额外克隆源码，并把源码目录加入 Python 路径：

```powershell
mkdir external
git clone https://github.com/zjunlp/EasyEdit.git external/EasyEdit
python -c "import site, pathlib; p=pathlib.Path(site.getsitepackages()[0])/'easyedit_local.pth'; p.write_text(str(pathlib.Path('external/EasyEdit').resolve()), encoding='ascii'); print(p)"
```

如果使用 Windows + Qwen2.5，建议在克隆 EasyEdit 后应用本提交包提供的兼容补丁：

```powershell
git -C external/EasyEdit apply ../../patches/easyedit-qwen-windows.patch
```

该补丁包含三处本地复现实验所需的修改：

- 修复 PyTorch forward hook 参数顺序，避免 Qwen2 decoder layer 返回值被误替换。
- 兼容 tied `lm_head.weight` 不出现在 `named_parameters()` 中的问题。
- 在 MEMIT 中支持关闭 Wikipedia covariance，使用轻量 identity covariance 近似，便于本地小规模实验。

如使用 Linux/CUDA 且显存、磁盘和运行时间充足，也可以不使用轻量 MEMIT 设置，改用官方 covariance statistics 配置重新运行完整 500 条实验。

## 运行步骤

进入项目目录：

```powershell
cd knowledge_editing_solution
```

### 1. Baseline 基线推理

```powershell
python baseline.py --data data/custom_facts.json --output outputs/baseline.json
```

快速测试一条样例：

```powershell
python baseline.py --data data/custom_facts.json --output outputs/baseline_smoke.json --limit 1
```

### 2. ROME 单事实编辑

```powershell
python edit_rome.py --data data/custom_facts.json --output outputs/rome_results.json
```

快速测试一条样例：

```powershell
python edit_rome.py --data data/custom_facts.json --output outputs/rome_smoke.json --limit 1
```

### 3. MEMIT 批量编辑

默认命令会从 Hugging Face `zjunlp/KnowEdit` 下载 ZsRE benchmark 数据：

```powershell
python edit_memit.py --output outputs/memit_results.json
```

在本地 6GB 显存环境中，完整 500 条 MEMIT 会非常慢，并可能卡在 Wikipedia covariance 统计阶段。本实验最终使用 10 条 benchmark 样例完成可复现实验：

```powershell
python edit_memit.py --limit 10 --output outputs/memit_results.json
```

也可以使用自定义 10 条事实做 smoke test：

```powershell
python edit_memit.py --data data/custom_facts.json --limit 10 --output outputs/memit_smoke.json
```

### 4. 指标评估

```powershell
python evaluate.py --baseline outputs/baseline.json --rome outputs/rome_results.json --memit outputs/memit_results.json --output outputs/metrics.json
```

如果只跑完部分实验，可以使用：

```powershell
python evaluate.py --allow-missing
```

## 已生成结果

本提交包已经包含一次本地运行的输出：

| 文件 | 内容 |
| --- | --- |
| `outputs/baseline.json` | 未编辑模型在 10 条自定义事实上的生成结果 |
| `outputs/rome_results.json` | ROME 对 10 条自定义事实逐条编辑后的结果 |
| `outputs/memit_results.json` | MEMIT 对 10 条 benchmark 样例批量编辑后的结果 |
| `outputs/metrics.json` | ES、PS、NS 综合指标 |

指标结果如下：

| 方法 | ES | PS | NS |
| --- | ---: | ---: | ---: |
| Baseline | 10.00% | 0.00% | 30.00% |
| ROME | 10.00% | 0.00% | 30.00% |
| MEMIT | 0.00% | 0.00% | 10.00% |

## 说明

- 默认模型为 `Qwen/Qwen2.5-0.5B-Instruct`。
- 评估采用简单字符串包含规则：生成结果中包含目标答案即视为命中。
- ROME 和 MEMIT 的 EasyEdit 内部 rewrite accuracy 有提升，但自由生成的精确字符串命中率较低，详细分析见 `reports/REPORT.md`。
- MEMIT 500 条完整实验属于资源消耗较大的设置，本地提交中保留了 10 条轻量批量实验结果。
