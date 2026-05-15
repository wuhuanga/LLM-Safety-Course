# 知识编辑实验

## 实验环境

- **服务器**: 阿里云 ECS (ecs.gn6i-c4g1.xlarge)
- **GPU**: NVIDIA T4 (16GB 显存)
- **操作系统**: Ubuntu 22.04
- **模型**: GPT2-XL (1.5B 参数)
- **框架**: EasyEdit

## 模型路径
模型已下载到：`/root/hugging_cache/models--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2`


## 环境配置

# 安装依赖
pip install -r requirements.txt（EasyEdit里面的和我额外的）
### Task 1 & Task 2 & Task 4
使用自定义的 10 条事实数据集 (`task1_data.json`)
### Task 3
使用 ZsRE 数据集（500条），文件位置：`/root/zsre_500_ready.json`

运行实验
Task 1: 基线测试（编辑前模型回答）
python baseline.py

Task 2: ROME 单条编辑
python edit_rome.py
由于 GPU 显存限制（16GB），实验采用以下优化措施：
- **半精度加载**：使用 `torch.float16` 加载模型，显存减半）
**Shell 脚本逐条运行**：每条编辑在独立的 Python 进程中执行，编辑完成后进程退出，显存自动完全释放，不会累积

Task 3: MEMIT 批量编辑
python edit_memit.py

Task 4: 综合评估
python evaluate.py
Task 4 基于 Task 2 (ROME) 的编辑结果进行评估
Task 3 (MEMIT) 的评估指标已在 edit_memit.py 运行输出中给出