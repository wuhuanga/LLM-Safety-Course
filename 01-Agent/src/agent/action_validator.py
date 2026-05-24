"""
动作验证和转换模块
"""
from typing import List, Optional, Tuple
import re


class ActionValidator:
    """动作验证器 - 将LLM生成的动作转换为有效动作"""

    def __init__(self, valid_actions: List[str]):
        """
        初始化动作验证器

        Args:
            valid_actions: 当前可执行的动作列表
        """
        self.valid_actions = valid_actions

    def validate_and_convert(self, action: str) -> Tuple[bool, str, str]:
        """
        验证并转换动作

        Args:
            action: LLM生成的动作

        Returns:
            (is_valid, converted_action, reason) 元组
        """
        # 1. 精确匹配
        if action in self.valid_actions:
            return True, action, "精确匹配"

        # 2. 提取动作类型和对象
        action_type, action_obj = self._parse_action(action)

        # 3. 尝试匹配动作类型
        matching_actions = [a for a in self.valid_actions if a.lower().startswith(action_type.lower())]

        if not matching_actions:
            return False, "", f"无匹配的动作类型: {action_type}"

        # 4. 尝试匹配对象
        if action_obj:
            # 寻找包含对象的动作（严格匹配）
            obj_matches = [a for a in matching_actions if action_obj.lower() in a.lower()]
            if obj_matches:
                # 返回最相似的匹配（优先选择开头就包含对象的）
                exact_matches = [a for a in obj_matches if a.lower().split(f"{action_type} ")[-1].startswith(action_obj.lower())]
                if exact_matches:
                    return True, exact_matches[0], f"对象精确匹配: {action_obj}"
                return True, obj_matches[0], f"对象模糊匹配: {action_obj}"

            # 对于特定动作类型，如果没有对象匹配，应该失败而不是随机选择
            if action_type in ["go", "take", "drop", "put"]:
                return False, "", f"无效对象 '{action_obj}' - 可用动作中不包含此对象"

            # 对于其他动作类型，使用最相似的动作
            return True, matching_actions[0], f"使用相似动作: {matching_actions[0]}"

        # 5. 没有对象，返回第一个匹配
        return True, matching_actions[0], f"使用动作: {matching_actions[0]}"

    def _parse_action(self, action: str) -> Tuple[str, str]:
        """
        解析动作，提取类型和对象

        Args:
            action: 动作字符串

        Returns:
            (action_type, action_obj) 元组
        """
        # 常见动作模式
        action = action.strip().lower()

        # 提取第一个词作为动作类型
        words = action.split()
        if not words:
            return "", ""

        action_type = words[0]
        action_obj = " ".join(words[1:]) if len(words) > 1 else ""

        return action_type, action_obj

    def suggest_action(self, query: str) -> Optional[str]:
        """
        根据查询建议一个动作

        Args:
            query: 查询文本，如 "go to kitchen", "open door"

        Returns:
            建议的动作
        """
        # 提取意图
        query = query.lower()

        # 常见意图映射
        intent_mapping = {
            "go": ["go", "move", "walk", "enter"],
            "open": ["open", "unlock"],
            "close": ["close", "lock"],
            "take": ["take", "grab", "pick up", "get"],
            "drop": ["drop", "put down"],
            "look": ["look", "examine", "check", "inspect"],
            "use": ["use", "activate", "apply"],
            "inventory": ["inventory", "check inventory", "items"],
        }

        # 查找匹配的意图
        matched_intent = None
        for intent, keywords in intent_mapping.items():
            if any(kw in query for kw in keywords):
                matched_intent = intent
                break

        if not matched_intent:
            return None

        # 查找匹配的动作
        for action in self.valid_actions:
            if action.lower().startswith(matched_intent.lower()):
                return action

        return None

    def get_best_match(self, action: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        获取最接近的有效动作

        Args:
            action: 目标动作
            top_k: 返回前K个

        Returns:
            [(action, score), ...] 列表
        """
        scores = []

        for valid_action in self.valid_actions:
            score = self._calculate_similarity(action, valid_action)
            scores.append((valid_action, score))

        # 按相似度排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _calculate_similarity(self, action1: str, action2: str) -> float:
        """计算两个动作的相似度"""
        action1_words = set(action1.lower().split())
        action2_words = set(action2.lower().split())

        if not action1_words or not action2_words:
            return 0.0

        # Jaccard相似度
        intersection = action1_words & action2_words
        union = action1_words | action2_words

        return len(intersection) / len(union)

    def get_similar_actions(self, action_type: str) -> List[str]:
        """获取相似类型的动作"""
        action_type = action_type.lower()
        return [a for a in self.valid_actions if a.lower().startswith(action_type)]


class ActionHistory:
    """动作历史管理 - 用于避免重复失败"""

    def __init__(self, max_history: int = 50):
        """
        初始化动作历史

        Args:
            max_history: 最大历史记录数
        """
        self.history: List[dict] = []
        self.failed_actions: set = set()
        self.successful_actions: set = set()
        self.max_history = max_history

    def record(self, action: str, success: bool, observation: str = ""):
        """记录动作结果"""
        self.history.append({
            "action": action,
            "success": success,
            "observation": observation
        })

        if success:
            self.successful_actions.add(action)
            # 如果这个动作之前失败过，从失败集中移除
            if action in self.failed_actions:
                self.failed_actions.remove(action)
        else:
            self.failed_actions.add(action)

        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def is_action_failed(self, action: str, max_attempts: int = 3) -> bool:
        """检查动作是否多次失败"""
        failures = [
            h for h in self.history
            if h["action"] == action and not h["success"]
        ]
        return len(failures) >= max_attempts

    def get_recent_failures(self, n: int = 5) -> List[str]:
        """获取最近的失败动作"""
        failures = [
            h["action"] for h in reversed(self.history)
            if not h["success"]
        ]
        return failures[:n]

    def get_action_stats(self, action: str) -> dict:
        """获取动作的统计信息"""
        action_history = [h for h in self.history if h["action"] == action]

        if not action_history:
            return {"total": 0, "success": 0, "failure": 0, "rate": 0.0}

        total = len(action_history)
        success = sum(1 for h in action_history if h["success"])
        failure = total - success

        return {
            "total": total,
            "success": success,
            "failure": failure,
            "rate": success / total if total > 0 else 0.0
        }

    def should_retry(self, action: str) -> bool:
        """判断是否应该重试动作"""
        stats = self.get_action_stats(action)

        # 如果从未尝试过，可以重试
        if stats["total"] == 0:
            return True

        # 如果成功率大于50%，可以重试
        if stats["rate"] > 0.5:
            return True

        # 如果失败次数超过3次，不再重试
        if stats["failure"] >= 3:
            return False

        return True