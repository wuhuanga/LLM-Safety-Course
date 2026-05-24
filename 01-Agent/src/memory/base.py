"""
记忆模块基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Memory:
    """记忆数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    importance: float = 1.0  # 记忆重要性 0-1


class BaseMemoryStore(ABC):
    """记忆存储基类"""

    @abstractmethod
    def add(self, memory: Memory) -> bool:
        """添加记忆"""
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Memory]:
        """获取指定ID的记忆"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Memory]:
        """搜索相关记忆"""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def count(self) -> int:
        """获取记忆数量"""
        pass


class MemoryManager:
    """记忆管理器 - 统一管理多种记忆"""

    def __init__(self, semantic_store: BaseMemoryStore, episodic_store: BaseMemoryStore):
        """
        初始化记忆管理器

        Args:
            semantic_store: 语义记忆存储
            episodic_store: 情景记忆存储
        """
        self.semantic_store = semantic_store
        self.episodic_store = episodic_store

    def add_semantic(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加语义记忆"""
        import uuid
        memory_id = str(uuid.uuid4())
        memory = Memory(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            importance=metadata.get("importance", 0.8) if metadata else 0.8
        )
        self.semantic_store.add(memory)
        return memory_id

    def add_episodic(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """添加情景记忆"""
        import uuid
        memory_id = str(uuid.uuid4())
        memory = Memory(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            importance=metadata.get("importance", 0.5) if metadata else 0.5
        )
        self.episodic_store.add(memory)
        return memory_id

    def search(self, query: str, top_k: int = 5, memory_type: str = "all") -> List[Memory]:
        """
        搜索记忆

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            memory_type: 记忆类型 ("semantic", "episodic", "all")

        Returns:
            相关记忆列表
        """
        results = []

        if memory_type in ("all", "semantic"):
            results.extend(self.semantic_store.search(query, top_k=top_k))

        if memory_type in ("all", "episodic"):
            results.extend(self.episodic_store.search(query, top_k=top_k))

        # 按重要性排序
        results.sort(key=lambda m: m.importance, reverse=True)

        return results[:top_k]

    def get_memory_summary(self) -> Dict[str, int]:
        """获取记忆统计摘要"""
        return {
            "semantic_count": self.semantic_store.count(),
            "episodic_count": self.episodic_store.count(),
            "total_count": self.semantic_store.count() + self.episodic_store.count()
        }