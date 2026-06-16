# LLM 知识编辑课程项目

本仓库用于完成知识编辑实验的四个主要任务：

1. `Task 1`：编辑前基线测试
2. `Task 2`：基于 ROME 的单条事实编辑
3. `Task 3`：基于 MEMIT 的批量知识编辑
4. `Task 4`：综合评估，计算 `ES / PS / NS`

当前项目默认使用的模型为 `Qwen/Qwen2.5-0.5B`。

## 项目文件结构

- [baseline.py]：`Task 1` 基线推理脚本
- [add_subject.py]：为自定义数据集补充 `subject` 字段
- [task2_rome.py]：`Task 2` 的 ROME 单事实编辑脚本
- [task3_memit.py]：`Task 3` 的 MEMIT 批量编辑脚本
- [task4_metrics.py]：`Task 4` 指标汇总脚本
- [data/custom_10.json]：`Task 1` 使用的自定义数据集
- [data/custom_10_with_subject.json]：带 `subject` 的 `Task 2` 数据集
- [configs/qwen2.5_0.5b_rome.yaml]：ROME 配置文件
- [configs/qwen2.5_0.5b_memit.yaml]：MEMIT 配置文件

## 环境准备

建议使用独立的 conda 环境。

```bash
conda create -n knowledge-edit python=3.10
conda activate knowledge-edit
pip install -r requirement.txt
pip install -r external/EasyEdit/requirements.txt
```

如果使用 GPU，请确保安装的 `torch` 版本与本机 CUDA 版本匹配。

如果希望在长时间运行 MEMIT 时减少 tokenizer 相关警告，可以先执行：

```bash
export TOKENIZERS_PARALLELISM=false
```

## Task 1：编辑前基线测试

### 任务说明

在不进行知识编辑的前提下，先让原始模型回答自定义事实问题，记录模型是否已经知道目标答案。

### 输入文件

- [data/custom_10.json]

### 运行命令

```bash
python baseline.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --data_path data/custom_10.json \
  --output_path outputs/baseline_results.json
```

### 输出文件

- [outputs/baseline_results.json]

## Task 2：基于 ROME 的单条事实编辑

### 第一步：补充 `subject` 字段

ROME 需要每条样本都带有 `subject` 字段。如果 [data/custom_10_with_subject.json] 还不存在，可以先执行：

```bash
python add_subject.py
```

### 第二步：运行 ROME 编辑

```bash
python task2_rome.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --device 0 \
  --data_path data/custom_10_with_subject.json \
  --output_path outputs/task2_rome_results.json
```

### 输出文件

- [outputs/task2_rome_results.json]

该文件已经包含 `Task 4` 所需的三类单样本结果：

- 直接提示词是否编辑成功
- 改写提示词是否编辑成功
- 无关事实是否保持正确

## Task 3：基于 MEMIT 的批量知识编辑

### 任务说明

使用 MEMIT 从 `ZsRE` 或 `CounterFact` 中一次性注入 500 条知识，并记录：

- 总编辑耗时
- GPU 显存峰值
- 编辑后样本评估结果

### 推荐数据集

建议优先使用 `ZsRE`，因为当前项目已经基于该数据集完成过测试。

### 运行命令

```bash
python task3_memit.py \
  --dataset_type zsre \
  --batch_size 500 \
  --device 0 \
  --eval_sample_size 500
```

### 说明

第一次运行 MEMIT 往往会比后续运行慢很多，这是正常现象。原因通常包括：

- 首次下载数据集
- 首次下载或构建 `wikipedia` 协方差统计量
- 在 `data/stats` 和 Hugging Face 缓存目录中生成中间缓存文件

这些统计量缓存完成之后，后续再次运行通常会明显更快。

### 输出文件

- [outputs/task3_memit_results.json]

该文件中包含以下关键信息：

- `edit_time_seconds`
- `peak_memory_bytes`
- `peak_memory_gb`
- `task4_metrics`

## Task 4：综合评估

### 任务说明

计算三个最终指标：

- `ES`：编辑成功率，对直接提示词是否回答正确
- `PS`：泛化性，对改写提示词是否回答正确
- `NS`：局部性，对无关事实是否仍保持正确

### 汇总 Task 2 的指标

```bash
python task4_metrics.py \
  --task task2 \
  --input_path outputs/task2_rome_results.json
```

### 汇总 Task 3 的指标

```bash
python task4_metrics.py \
  --task task3 \
  --input_path outputs/task3_memit_results.json
```

### 输出格式示例

两个命令都会输出类似下面的 JSON 汇总结果：

```json
{
  "total_samples": 500,
  "ES_count": 244,
  "ES_rate": 0.488,
  "PS_count": 226,
  "PS_rate": 0.452,
  "NS_count": 15,
  "NS_rate": 0.03
}
```

## 推荐运行顺序

如果希望从头完整复现实验，建议按以下顺序执行：

```bash
python add_subject.py
python baseline.py --model_name Qwen/Qwen2.5-0.5B
python task2_rome.py --model_name Qwen/Qwen2.5-0.5B --device 0
python task3_memit.py --dataset_type zsre --batch_size 500 --device 0 --eval_sample_size 500
python task4_metrics.py --task task2 --input_path outputs/task2_rome_results.json
python task4_metrics.py --task task3 --input_path outputs/task3_memit_results.json
```

## 已生成的重要结果文件

如果前面的实验已经运行过，最重要的输出文件如下：

- [outputs/baseline_results.json]
- [outputs/task2_rome_results.json]
- [outputs/task3_memit_results.json]

## 常见问题

如果脚本因模型下载失败而报错，可以先直接重试。部分运行过程中可能会遇到 Hugging Face 临时网络错误或 SSL 错误。

如果 MEMIT 第一次运行非常慢，这通常是正常现象，因为协方差统计量可能仍在构建中。

如果运行 MEMIT 时显存不足，可以先缩小批量规模测试流程是否正常，例如：

```bash
python task3_memit.py \
  --dataset_type zsre \
  --batch_size 100 \
  --device 0 \
  --eval_sample_size 100
```
