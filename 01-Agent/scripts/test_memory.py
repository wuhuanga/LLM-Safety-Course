"""测试记忆模块"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.base import MemoryManager
from src.memory.semantic_memory import SemanticMemoryStore
from src.memory.episodic_memory import EpisodicMemoryStore

print("=" * 60)
print("记忆模块测试")
print("=" * 60)

# 初始化
semantic_store = SemanticMemoryStore()
episodic_store = EpisodicMemoryStore()
memory_manager = MemoryManager(semantic_store, episodic_store)

# 测试语义记忆
print("\n[1] 测试语义记忆...")
semantic_store.add_fact("kitchen", "kitchen有stove可以加热", {"action": "heat"})
semantic_store.add_fact("freezer", "freezer可以冷冻物品", {"action": "freeze"})

facts = semantic_store.search_facts("加热")
print(f"  搜索'加热': {len(facts)}条结果")
for fact in facts:
    print(f"    - {fact.content}")

# 测试情景记忆
print("\n[2] 测试情景记忆...")
episodic_store.set_current_task("boil")
episodic_store.record_step(1, {"location": "hallway"}, "open door to kitchen", "门已打开", True)
episodic_store.record_step(2, {"location": "kitchen"}, "go to kitchen", "进入厨房", True)
episodic_store.record_error(3, {"location": "kitchen"}, "go to sink", "无效动作", "sink不是房间")

history = episodic_store.get_history("boil")
print(f"  历史记录: {len(history)}步")
for step in history:
    print(f"    - 步骤{step['step']}: {step['action']} -> {step['success']}")

errors = episodic_store.get_errors("boil")
print(f"  错误记录: {len(errors)}条")
for err in errors:
    print(f"    - {err['action']}: {err['error_msg']}")

# 测试记忆管理器
print("\n[3] 测试记忆管理器...")
summary = memory_manager.get_summary()
print(f"  语义记忆: {summary['semantic_facts']}条")
print(f"  情景记忆: {summary['episodic_steps']}步")
print(f"  错误记录: {summary['errors']}条")

print("\n[成功] 所有测试通过")
print("=" * 60)