"""
语义记忆存储 - 存储环境知识
"""
from typing import List, Optional, Dict, Any
from src.memory.base import BaseMemoryStore, Memory
from src.llm.client import get_embedding_client


class SemanticMemoryStore(BaseMemoryStore):
    """语义记忆存储 - 存储静态环境知识"""

    def __init__(self, embedding_client=None):
        """
        初始化语义记忆存储

        Args:
            embedding_client: 嵌入模型客户端
        """
        self.memories: Dict[str, Memory] = {}
        self.embedding_client = embedding_client or get_embedding_client()

    def add(self, memory: Memory) -> bool:
        """添加语义记忆"""
        # 生成嵌入
        if not memory.embedding:
            memory.embedding = self.embedding_client.embed(memory.content)

        self.memories[memory.id] = memory
        return True

    def get(self, memory_id: str) -> Optional[Memory]:
        """获取指定ID的记忆"""
        return self.memories.get(memory_id)

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Memory]:
        """搜索相关语义记忆"""
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
            del self.memories[memory_id]
            return True
        return False

    def clear(self):
        """清空所有记忆"""
        self.memories.clear()

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

    def add_object_knowledge(
        self,
        object_name: str,
        properties: List[str],
        interactions: List[str],
        location: Optional[str] = None
    ) -> str:
        """
        添加物品知识

        Args:
            object_name: 物品名称
            properties: 物品属性列表
            interactions: 可执行的操作列表
            location: 物品位置

        Returns:
            记忆ID
        """
        content = f"物品 {object_name} 的属性：{', '.join(properties)}；可以执行的操作：{', '.join(interactions)}"

        metadata = {
            "type": "object_knowledge",
            "object_name": object_name,
            "location": location,
            "importance": 0.8
        }

        memory_id = f"obj_{object_name}"

        memory = Memory(
            id=memory_id,
            content=content,
            metadata=metadata,
            importance=0.8
        )

        self.add(memory)
        return memory_id

    def add_action_knowledge(
        self,
        action: str,
        requirements: List[str],
        effects: List[str],
        conditions: Optional[str] = None
    ) -> str:
        """
        添加动作知识

        Args:
            action: 动作名称
            requirements: 执行要求
            effects: 执行效果
            conditions: 执行条件

        Returns:
            记忆ID
        """
        content = f"动作 {action}：需要{', '.join(requirements)}；执行后{', '.join(effects)}"
        if conditions:
            content += f"；条件：{conditions}"

        metadata = {
            "type": "action_knowledge",
            "action": action,
            "importance": 0.9
        }

        memory_id = f"act_{action}"

        memory = Memory(
            id=memory_id,
            content=content,
            metadata=metadata,
            importance=0.9
        )

        self.add(memory)
        return memory_id

    def get_all_objects(self) -> List[str]:
        """获取所有已记录的物品"""
        return [
            m.metadata.get("object_name", "")
            for m in self.memories.values()
            if m.metadata.get("type") == "object_knowledge"
        ]

    def get_object_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """获取物品信息"""
        memory_id = f"obj_{object_name}"
        memory = self.get(memory_id)
        if memory:
            return {
                "content": memory.content,
                "metadata": memory.metadata
            }
        return None