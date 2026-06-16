# 记忆增强型文本交互智能体系统

基于 LLM 的记忆增强型智能体，在 ScienceWorld 文本环境中执行复杂任务。

## 项目结构

```
01-Agent/
├── README.md                 # 项目说明
├── 实验报告模板.md          # 实验报告模板
├── requirements.txt          # Python依赖
├── config.yaml               # 系统配置
├── .env.example             # 环境变量模板
├── .gitignore               # Git忽略文件
├── docs/
│   └── 千问配置指南.md       # 千问API配置教程
├── src/
│   ├── env/                 # 环境接口（ScienceWorld封装）
│   │   └── scienceworld_wrapper.py
│   ├── agent/               # Agent核心（ReAct推理）
│   │   ├── react_agent_v2.py # 改进版ReAct智能体
│   │   ├── react_agent_v3.py # v3修复版（修正移动逻辑）
│   │   └── action_validator.py # 动作验证器
│   ├── memory/              # 记忆模块（语义记忆+情景记忆+RAG）
│   │   ├── base.py          # 记忆基类和数据结构
│   │   ├── semantic_memory.py # 语义记忆存储
│   │   ├── episodic_memory.py # 情景记忆存储
│   │   └── rag_retriever.py  # RAG检索器
│   ├── llm/                 # LLM客户端（支持千问/OpenAI）
│   │   └── client.py
│   └── utils/               # 工具函数
├── scripts/
│   ├── test_qwen.py         # 测试千问API
│   ├── test_memory.py       # 测试记忆模块
│   ├── test_scienceworld.py # 测试环境接口
│   ├── run_demo.py          # 运行Demo
│   └── evaluate_improved.py # 改进版评估脚本
└── outputs/
    ├── logs/                # 日志文件
    ├── memories/            # 记忆数据库
    └── videos/              # Demo视频
```

## 安装步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置千问API密钥

详细配置教程请查看 [docs/千问配置指南.md](docs/千问配置指南.md)

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 编辑.env文件，填入你的千问API密钥
# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

# 3. 获取API密钥: https://dashscope.console.aliyun.com/
```

### 3. 测试API连接
```bash
python scripts/test_qwen.py
```

### 4. 运行Demo
```bash
python scripts/run_demo.py
```

### 5. 运行评估
```bash
python scripts/evaluate_improved.py
```

## 快速开始

```python
from src.agent.react_agent_v3 import ReActAgent
from src.env.scienceworld_wrapper import ScienceWorldEnv
from src.memory.base import MemoryManager
from src.memory.semantic_memory import SemanticMemoryStore
from src.memory.episodic_memory import EpisodicMemoryStore

# 初始化记忆
semantic_store = SemanticMemoryStore()
episodic_store = EpisodicMemoryStore()
memory_manager = MemoryManager(semantic_store, episodic_store)

# 初始化环境
env = ScienceWorldEnv(task_name="boil")

# 初始化Agent
agent = ReActAgent(
    env=env,
    memory_manager=memory_manager,
    max_steps=30,
    verbose=True
)

# 执行任务
result = agent.run_task()
```

## 核心功能

- ✅ **多步规划**: 持续与环境交互完成任务
- ✅ **记忆模块**: 语义记忆+情景记忆+RAG检索
- ✅ **错误恢复**: 从失败中学习并避免重复错误
- ✅ **ReAct推理**: 思考-行动循环
- ✅ **动作验证**: 智能动作匹配和转换
- ✅ **评估系统**: 有记忆vs无记忆对比实验

## 实验结果（目标）

| 配置 | 任务完成率 | 平均步数 | 动作成功率 |
|------|-----------|----------|------------|
| 无记忆 | 33.3% | 25.0 | 85% |
| 有记忆 | 66.7% | 18.0 | 92% |
| **提升** | **+33.4%** | **-7.0** | **+7%** |

## 系统要求

- Python 3.9+
- 千问API密钥
- 5GB+ 磁盘空间

## 开发进度

| 模块 | 状态 | 完成度 |
|------|------|--------|
| 基础设施 | ✅ | 100% |
| LLM模块 | ✅ | 100% |
| 环境接口 | ✅ | 95% |
| 记忆模块 | ✅ | 100% |
| Agent核心 | ✅ | 95% |
| 评估系统 | ✅ | 80% |
| 文档报告 | ✅ | 90% |
| **总体** | ✅ | **95%** |

## 已完成功能

### ✅ 记忆模块
- 语义记忆存储（物品知识、动作知识）
- 情景记忆存储（历史轨迹、错误记录）
- RAG检索（上下文过滤、相似度排序）
- 记忆统计（成功率、错误次数）

### ✅ Agent核心
- ReAct推理循环（思考→行动）
- 记忆集成（检索相关记忆辅助决策）
- 动作验证（智能匹配、转换）
- 错误恢复（失败动作避免、重试机制）
- 修正的移动逻辑（open→go两步规则）

### ✅ 评估系统
- 对比实验（有记忆vs无记忆）
- 多任务评估（boil, freeze, use-thermometer, measure-melting-point-known-substance）
- 结果统计和可视化

## 参考资料

- [ReAct论文](https://arxiv.org/abs/2210.03629)
- [CLIN论文](https://arxiv.org/pdf/2310.10134)
- [ScienceWorld](https://github.com/allenai/scienceworld)