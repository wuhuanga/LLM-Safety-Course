"""
运行完整Demo - 集成环境、记忆和Agent
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.scienceworld_wrapper import ScienceWorldEnv
from src.agent.react_agent_v2 import ReActAgent
from src.memory.base import MemoryManager
from src.memory.semantic_memory import SemanticMemoryStore
from src.memory.episodic_memory import EpisodicMemoryStore


def main():
    print("=" * 60)
    print("记忆增强型文本交互智能体 - Demo")
    print("=" * 60)

    # 初始化记忆
    print("\n[1/4] 初始化记忆系统...")
    semantic_store = SemanticMemoryStore()
    episodic_store = EpisodicMemoryStore()
    memory_manager = MemoryManager(semantic_store, episodic_store)
    print("[完成] 记忆系统初始化完成")

    # 预填充一些知识
    print("\n[2/4] 预加载环境知识...")
    semantic_store.add_object_knowledge(
        object_name="水",
        properties=["液体", "沸点100度", "可以蒸发"],
        interactions=["加热", "蒸发", "煮沸"]
    )
    semantic_store.add_action_knowledge(
        action="heat",
        requirements=["有可加热的物体", "有热源"],
        effects=["温度升高", "可能沸腾"]
    )
    print("[完成] 知识预加载完成")

    # 初始化环境
    print("\n[3/4] 初始化ScienceWorld环境...")
    env = ScienceWorldEnv(task_name="boil")
    print("[完成] 环境初始化完成")

    # 初始化Agent
    print("\n[4/4] 初始化ReAct Agent...")
    agent = ReActAgent(
        env=env,
        memory_manager=memory_manager,
        max_steps=30,
        verbose=True
    )
    print("[完成] Agent初始化完成")

    # 运行任务
    print("\n" + "=" * 60)
    print("开始执行任务")
    print("=" * 60 + "\n")

    result = agent.run_task()

    # 显示结果摘要
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"任务完成: {'是' if result['completed'] else '否'}")
    print(f"执行步数: {result['steps']}")
    print(f"得分: {result['score']}")
    print(f"成功率: {result['success_rate']:.2%}")

    # 显示记忆统计
    print("\n" + "=" * 60)
    print("记忆统计")
    print("=" * 60)
    memory_summary = memory_manager.get_memory_summary()
    print(f"语义记忆: {memory_summary['semantic_count']}")
    print(f"情景记忆: {memory_summary['episodic_count']}")
    print(f"总记忆数: {memory_summary['total_count']}")

    stats = episodic_store.get_statistics()
    print(f"\n执行统计:")
    print(f"  总步骤: {stats['total_steps']}")
    print(f"  成功步骤: {stats['successful_steps']}")
    print(f"  错误次数: {stats['errors']}")
    print(f"  成功率: {stats['success_rate']:.2%}")

    print("\n" + "=" * 60)
    print("Demo完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()