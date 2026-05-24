"""
RAG (Retrieval-Augmented Generation) 检索器
"""
from typing import List, Dict, Any, Optional
from src.memory.base import Memory


class RAGRetriever:
    """RAG检索器 - 根据上下文检索相关记忆"""

    def __init__(self, memory_store, top_k: int = 5, threshold: float = 0.0):
        """
        初始化RAG检索器

        Args:
            memory_store: 记忆存储对象
            top_k: 检索数量
            threshold: 相似度阈值
        """
        self.memory_store = memory_store
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(
        self,
        query: str,
        context: Dict[str, Any] = None,
        top_k: Optional[int] = None,
        **filters
    ) -> List[Memory]:
        """
        根据查询和上下文检索记忆

        Args:
            query: 查询文本
            context: 当前上下文（位置、物品等）
            top_k: 返回数量
            **filters: 过滤条件

        Returns:
            相关记忆列表
        """
        k = top_k or self.top_k

        # 基础检索
        memories = self.memory_store.search(query, top_k=k * 2)  # 多检索一些

        # 应用上下文过滤
        if context or filters:
            memories = self._filter_by_context(memories, context, filters)

        # 应用相似度阈值
        if self.threshold > 0:
            memories = [m for m in memories if m.importance >= self.threshold]

        return memories[:k]

    def _filter_by_context(
        self,
        memories: List[Memory],
        context: Dict[str, Any] = None,
        filters: Dict[str, Any] = None
    ) -> List[Memory]:
        """根据上下文和过滤条件筛选记忆"""
        context = context or {}
        filters = filters or {}

        filtered = []

        for memory in memories:
            match = True

            # 检查过滤条件
            for key, value in filters.items():
                if memory.metadata.get(key) != value:
                    match = False
                    break

            # 检查上下文相关性
            if match and context:
                match = self._is_context_relevant(memory, context)

            if match:
                filtered.append(memory)

        return filtered

    def _is_context_relevant(self, memory: Memory, context: Dict[str, Any]) -> bool:
        """
        判断记忆是否与当前上下文相关

        Args:
            memory: 记忆对象
            context: 当前上下文

        Returns:
            是否相关
        """
        # 检查位置相关
        memory_location = memory.metadata.get("location")
        current_location = context.get("location")
        if memory_location and current_location:
            if memory_location == current_location:
                return True  # 同一位置，相关

        # 检查物品相关
        memory_objects = memory.metadata.get("objects", [])
        current_objects = context.get("visible_objects", [])

        # 如果记忆中有当前可见的物品，相关
        for obj in memory_objects:
            if obj in current_objects:
                return True

        # 检查任务相关
        memory_task = memory.metadata.get("task")
        current_task = context.get("task")
        if memory_task and current_task:
            if memory_task == current_task:
                return True  # 同一任务，相关

        return False

    def format_retrieved_memories(self, memories: List[Memory]) -> str:
        """
        将检索到的记忆格式化为文本

        Args:
            memories: 记忆列表

        Returns:
            格式化的文本
        """
        if not memories:
            return "无相关记忆"

        formatted = []
        for i, memory in enumerate(memories, 1):
            formatted.append(f"[记忆{i}] {memory.content}")

        return "\n".join(formatted)

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            "top_k": self.top_k,
            "threshold": self.threshold,
            "total_memories": self.memory_store.count() if hasattr(self.memory_store, "count") else 0
        }


# 便捷函数
def create_retriever(memory_store, top_k: int = 5, threshold: float = 0.0) -> RAGRetriever:
    """创建RAG检索器的便捷函数"""
    return RAGRetriever(memory_store, top_k=top_k, threshold=threshold)