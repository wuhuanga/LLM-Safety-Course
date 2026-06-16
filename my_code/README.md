# EasyEdit 知识编辑实验

基于 [EasyEdit](https://github.com/zjunlp/EasyEdit) 框架，在 Qwen2.5-0.5B 模型上实现 ROME 单条编辑与 MEMIT 批量编辑，评估 ES（编辑成功率）、PS（泛化性）、NS（局部性）三项指标。

## 文件说明

| 文件 | 用途 |
|------|------|
| `baseline.py` | Task 1 — 基线测试，编辑前验证模型持有旧知识 |
| `edit_rome.py` | Task 2 — ROME 逐条编辑 10 条自定义事实，输出 ES/PS/NS |
| `edit_memit.py` | Task 3 — MEMIT 批量编辑 500 条 CounterFact 数据 |
| `evaluate.py` | Task 4 — 综合评估，可切换 ROME/MEMIT 场景 |
| `requirements.txt` | 本项目直接依赖（仅 torch） |
| `../data/custom_facts.json` | 10 条中文自定义事实（Task 2/4 使用） |
| `../data/counterfact.json` | CounterFact 开源数据集（Task 3/4 使用，需自行下载） |

## 环境准备

> 本项目不包含 EasyEdit 框架，需另行安装。

```bash
# 1. 安装 EasyEdit 框架及其依赖
git clone https://github.com/zjunlp/EasyEdit.git ../EasyEdit
pip install -r ../EasyEdit/requirements.txt

# 2. 将本项目脚本放入 EasyEdit 目录下，安装本项目依赖
pip install -r requirements.txt

# 3. 下载模型（Qwen2.5-0.5B）
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B.git ../hugging_cache/Qwen2.5-0.5B

# 4. 下载 CounterFact 数据集（Task 3/4 需要）
#    将 counterfact.json 放入 ../data/
```

## 运行方法

```bash
cd EasyEdit/FinalAssignment

# GPU 环境（默认）
python baseline.py      # Task 1: 基线测试
python edit_rome.py     # Task 2: ROME 单条编辑
python edit_memit.py    # Task 3: MEMIT 批量编辑
python evaluate.py      # Task 4: 综合评估（修改 EVAL_MODE 切换场景）
```

## 数据集格式

`custom_facts.json` 每条包含：

| 字段 | 说明 |
|------|------|
| `prompt` | 编辑用提示词 |
| `subject` | 编辑目标主语（须为 prompt 子串） |
| `target_new` | 新知识（编辑后答案） |
| `ground_truth` | 旧知识（编辑前答案） |
| `rephrase_prompt` | 同义改写，用于泛化性评估（PS） |
| `locality_prompt` | 无关事实，用于局部性评估（NS） |
| `locality_ground_truth` | 无关事实正确答案 |

## 评估指标

| 指标 | 含义 | 越高越好 |
|------|------|---------|
| ES (Efficacy) | 编辑后 prompt 能否输出新答案 | ✓ |
| PS (Paraphrase) | 同义改写后是否也能输出新答案 | ✓ |
| NS (Neighborhood) | 无关事实是否未被编辑影响 | ✓ |
