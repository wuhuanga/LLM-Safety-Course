"""
情景记忆存储 - 存储历史轨迹
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.memory.base import BaseMemoryStore, Memory
from src.llm.client import get_embedding_client


class EpisodicMemoryStore(BaseMemoryStore):
    """情景记忆存储 - 存储历史动作和结果"""

    def __init__(self, embedding_client=None, max_memories: int = 1000):
        """
        初始化情景记忆存储

        Args:
            embedding_client: 嵌入模型客户端
            max_memories: 最大记忆数量
        """
        self.memories: Dict[str, Memory] = {}
        self.action_history: List[str] = []  # 动作历史列表
        self.embedding_client = embedding_client or get_embedding_client()
        self.max_memories = max_memories
        self.current_task: Optional[str] = None

    def add(self, memory: Memory) -> bool:
        """添加情景记忆"""
        # 生成嵌入
        if not memory.embedding:
            memory.embedding = self.embedding_client.embed(memory.content)

        self.memories[memory.id] = memory

        # 记录动作历史
        action = memory.metadata.get("action")
        if action:
            self.action_history.append(action)

        # 限制记忆数量
        if len(self.memories) > self.max_memories:
            # 删除最旧的记忆
            oldest_id = min(self.memories.keys(), key=lambda k: self.memories[k].metadata.get("timestamp", 0))
            self.delete(oldest_id)

        return True

    def get(self, memory_id: str) -> Optional[Memory]:
        """获取指定ID的记忆"""
        return self.memories.get(memory_id)

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Memory]:
        """搜索相关情景记忆"""
        if not self.memories:
            return []

        # 生成查询嵌入
        query_embedding = self.embedding_client.embed(query)

        # 计算相似度
        scored_memories = []
        for memory in self.memories.values():
            if memory.embedding:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                scored_memories.append((memory, similarity))

        # 按相似度排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # 返回Top-K
        return [m for m, _ in scored_memories[:top_k]]

    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            action = memory.metadata.get("action")
            if action and action in self.action_history:
                self.action_history.remove(action)
            del self.memories[memory_id]
            return True
        return False

    def clear(self):
        """清空所有记忆"""
        self.memories.clear()
        self.action_history.clear()

    def count(self) -> int:
        """获取记忆数量"""
        return len(self.memories)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import math

        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def record_step(
        self,
        step_num: int,
        state: Dict[str, Any],
        action: str,
        result: str,
        success: bool,
        task: Optional[str] = None
    ) -> str:
        """
        记录一个执行步骤

        Args:
            step_num: 步骤编号
            state: 当前状态
            action: 执行的动作
            result: 执行结果
            success: 是否成功
            task: 任务名称

        Returns:
            记忆ID
        """
        location = state.get("location", "unknown")
        inventory = state.get("inventory", [])

        content = f"步骤{step_num}: 在{location}执行'{action}'，结果：{result}"

        metadata = {
            "type": "step",
            "step_num": step_num,
            "location": location,
            "inventory": inventory,
            "action": action,
            "result": result,
            "success": success,
            "task": task or self.current_task,
            "timestamp": datetime.now().timestamp(),
            "importance": 0.9 if success else 0.6  # 成功的经验更重要
        }

        memory_id = f"step_{task or 'unknown'}_{step_num}"

        memory = Memory(
            id=memory_id,
            content=content,
            metadata=metadata,
            importance=0.9 if success else 0.6
        )

        self.add(memory)
        return memory_id

    def record_error(
        self,
        step_num: int,
        state: Dict[str, Any],
        action: str,
        error_msg: str,
        task: Optional[str] = None
    ) -> str:
        """
        记录错误

        Args:
            step_num: 步骤编号
            state: 当前状态
            action: 执行的动作
            error_msg: 错误信息
            task: 任务名称

        Returns:
            记忆ID
        """
        location = state.get("location", "unknown")

        content = f"错误（步骤{step_num}）: 在{location}执行'{action}'失败，原因：{error_msg}"

        metadata = {
            "type": "error",
            "step_num": step_num,
            "location": location,
            "action": action,
            "error_msg": error_msg,
            "task": task or self.current_task,
            "timestamp": datetime.now().timestamp(),
            "importance": 0.7  # 错误经验很重要
        }

        memory_id = f"error_{task or 'unknown'}_{step_num}"

        memory = Memory(
            id=memory_id,
            content=content,
            metadata=metadata,
            importance=0.7
        )

        self.add(memory)
        return memory_id

    def get_recent_actions(self, n: int = 5) -> List[str]:
        """获取最近的动作历史"""
        return self.action_history[-n:]

    def get_failed_actions(self) -> List[str]:
        """获取失败的动作"""
        return [
            m.metadata["action"]
            for m in self.memories.values()
            if m.metadata.get("type") == "error"
        ]

    def get_task_history(self, task: str) -> List[Memory]:
        """获取指定任务的历史"""
        return [
            m for m in self.memories.values()
            if m.metadata.get("task") == task
        ]

    def set_current_task(self, task: str):
        """设置当前任务"""
        self.current_task = task

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_steps = len([m for m in self.memories.values() if m.metadata.get("type") == "step"])
        successful_steps = len([m for m in self.memories.values() if m.metadata.get("success")])
        errors = len([m for m in self.memories.values() if m.metadata.get("type") == "error"])

        return {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "errors": errors,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0
        }