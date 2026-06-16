"""
改进版评估脚本 - 使用v3 Agent和正确的任务名
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.scienceworld_wrapper import ScienceWorldEnv
from src.agent.react_agent_v3 import ReActAgent
from src.memory.base import MemoryManager
from src.memory.semantic_memory import SemanticMemoryStore
from src.memory.episodic_memory import EpisodicMemoryStore


def run_evaluation():
    """运行评估"""
    print("=" * 60)
    print("改进版Agent评估")
    print("=" * 60)

    # 正确的任务名
    tasks = [
        "boil",
        "freeze",
        "use-thermometer",
        "measure-melting-point-known-substance"
    ]

    # 有记忆评估
    print("\n[1/2] 有记忆Agent评估...")
    semantic_store = SemanticMemoryStore()
    episodic_store = EpisodicMemoryStore()
    memory_manager = MemoryManager(semantic_store, episodic_store)

    # 预加载知识
    semantic_store.add_object_knowledge(
        object_name="水",
        properties=["液体", "沸点100度", "可蒸发"],
        interactions=["加热", "蒸发", "煮沸"]
    )
    semantic_store.add_action_knowledge(
        action="heat",
        requirements=["有热源", "有可加热物体"],
        effects=["温度升高", "可能沸腾"],
        conditions="在炉子或加热器上"
    )

    with_memory_results = []
    for task in tasks:
        try:
            print(f"\n任务: {task}")
            env = ScienceWorldEnv(task_name=task)
            agent = ReActAgent(
                env=env,
                memory_manager=memory_manager,
                max_steps=20,
                verbose=False
            )
            result = agent.run_task()
            with_memory_results.append(result)
            print(f"  完成: {result['completed']}, 步数: {result['steps']}")
        except Exception as e:
            print(f"  错误: {e}")
            with_memory_results.append({"completed": False, "steps": 0, "score": 0})

    # 无记忆评估
    print("\n[2/2] 无记忆Agent评估...")
    no_memory_results = []
    for task in tasks:
        try:
            print(f"\n任务: {task}")
            env = ScienceWorldEnv(task_name=task)
            agent = ReActAgent(
                env=env,
                memory_manager=None,
                max_steps=20,
                verbose=False
            )
            result = agent.run_task()
            no_memory_results.append(result)
            print(f"  完成: {result['completed']}, 步数: {result['steps']}")
        except Exception as e:
            print(f"  错误: {e}")
            no_memory_results.append({"completed": False, "steps": 0, "score": 0})

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    print("\n有记忆Agent:")
    completed_with_memory = sum(1 for r in with_memory_results if r['completed'])
    for i, r in enumerate(with_memory_results, 1):
        print(f"  任务{i}: 完成={r['completed']}, 步数={r['steps']}")
    print(f"完成率: {completed_with_memory}/{len(tasks)} = {completed_with_memory/len(tasks):.0%}")

    print("\n无记忆Agent:")
    completed_no_memory = sum(1 for r in no_memory_results if r['completed'])
    for i, r in enumerate(no_memory_results, 1):
        print(f"  任务{i}: 完成={r['completed']}, 步数={r['steps']}")
    print(f"完成率: {completed_no_memory}/{len(tasks)} = {completed_no_memory/len(tasks):.0%}")

    print("\n" + "=" * 60)
    print("记忆效果")
    print("=" * 60)
    print(f"提升: {completed_with_memory - completed_no_memory} 个任务")

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

    return {
        "with_memory": with_memory_results,
        "no_memory": no_memory_results
    }


if __name__ == "__main__":
    run_evaluation()