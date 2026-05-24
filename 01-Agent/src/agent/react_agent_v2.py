"""
改进版ReAct Agent - 优化的推理-行动循环
"""
from typing import Dict, List, Optional, Any, Tuple
import re
from src.llm.client import get_llm_client
from src.memory.base import MemoryManager
from src.memory.rag_retriever import create_retriever
from src.agent.action_validator import ActionValidator, ActionHistory


class ReActAgent:
    """ReAct智能体 - 思考-行动循环"""

    # 改进的Prompt模板
    SYSTEM_PROMPT = """你是一个在ScienceWorld环境中执行任务的智能体。
你的目标是：{task}

【系统说明】
当前系统会给你一个简化的动作列表（仅显示前20个相关动作），请从中选择。

【环境移动规则】
在ScienceWorld中，移动房间的步骤是：
1. 使用 "open door to [房间名]" 打开门
2. 然后自动进入该房间
3. 例如：open door to kitchen

【任务执行流程】
1. 观察当前环境，识别任务相关物品
2. 如果需要去其他房间，查找并打开相应的门
3. 在目标房间中寻找任务物品
4. 执行任务动作

【可执行动作说明】
- open door to [房间名]: 打开门进入房间
- look around: 观察周围环境
- inventory: 查看物品栏
- look at [物品]: 查看某个物品
- pick up [物品]: 捡起物品
- use [物品]: 使用物品
- mix [物品]: 混合物品
- heat [物品]: 加热物品

【失败动作提示】
以下动作之前已失败，请避免重复：
{failed_actions}

【相关记忆】
{relevant_memories}

请按照以下格式进行思考和行动：

Thought: [分析当前情况]
Action: [从动作列表中选择一个，直接复制，不要修改]

【示例】
场景：在hallway，需要去厨房

可执行动作：
- open door to kitchen
- look around
- inventory

正确执行：
Thought: 我在hallway，需要去厨房找水。看到有"open door to kitchen"这个动作，应该先打开门。
Action: open door to kitchen

---

【当前状态】
位置: {location}
物品栏: {inventory}

【环境观察】
{observation}

【可执行动作】
{actions}

请开始："""

    def __init__(
        self,
        env,
        llm_client=None,
        memory_manager: Optional[MemoryManager] = None,
        max_steps: int = 50,
        verbose: bool = True
    ):
        """
        初始化ReAct Agent

        Args:
            env: 环境
            llm_client: LLM客户端
            memory_manager: 记忆管理器
            max_steps: 最大步数
            verbose: 是否打印详细信息
        """
        self.env = env
        self.llm_client = llm_client or get_llm_client()
        self.memory_manager = memory_manager
        self.max_steps = max_steps
        self.verbose = verbose

        # 动作历史
        self.action_history = ActionHistory()

        # RAG检索器
        self.retriever = None
        if memory_manager:
            self.retriever = create_retriever(
                memory_manager.semantic_store,
                top_k=3
            )

        # 任务状态
        self.task_description = ""
        self.current_step = 0
        self.history: List[Dict] = []

    def run_task(self, task_name: str = None) -> Dict[str, Any]:
        """
        运行任务

        Args:
            task_name: 任务名称

        Returns:
            执行结果字典
        """
        # 加载任务
        if task_name:
            env_info = self.env.load_task(task_name)
            self.task_description = env_info.get("description", "")
        else:
            self.task_description = self.env.get_task_description()

        if self.memory_manager and self.memory_manager.episodic_store:
            self.memory_manager.episodic_store.set_current_task(task_name)

        if self.verbose:
            print("=" * 60)
            print(f"开始任务: {task_name or 'current task'}")
            print(f"任务描述: {self.task_description}")
            print("=" * 60)

        # 主循环
        while self.current_step < self.max_steps:
            # 检查是否完成
            if self.env.is_complete():
                if self.verbose:
                    print("\n[成功] 任务完成！")
                break

            # 获取当前状态
            state = self._get_current_state()

            # 思考和行动
            thought, action = self._think_and_act(state)

            if not action:
                if self.verbose:
                    print("\n[失败] 无法生成有效动作")
                break

            # 执行动作
            observation, reward, done, info = self._execute_action(action, state)

            # 记录动作历史
            success = info.get("is_success", False)
            self.action_history.record(action, success, observation)

            # 记录历史
            self._record_step(state, thought, action, observation, info)

            # 更新记忆
            self._update_memory(state, action, observation, info)

            self.current_step += 1

            if self.verbose and self.current_step % 5 == 0:
                print(f"\n--- 已执行 {self.current_step} 步 ---")

        # 返回结果
        return self._get_result()

    def _get_current_state(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        return self.env.get_state_info()

    def _think_and_act(self, state: Dict[str, Any]) -> Tuple[str, str]:
        """
        思考并决定下一步行动

        Args:
            state: 当前状态

        Returns:
            (thought, action) 元组
        """
        # 获取相关记忆
        relevant_memories = ""
        if self.retriever and self.memory_manager:
            memories = self.retriever.retrieve(
                query=f"{self.task_description} {state.get('observation', '')}",
                context={"location": self._extract_location(state.get('observation', ''))}
            )
            if memories:
                relevant_memories = "\n".join([f"- {m.content}" for m in memories])

        # 获取失败动作
        failed_actions = ""
        recent_failures = self.action_history.get_recent_failures(3)
        if recent_failures:
            failed_actions = "\n".join([f"- {a}" for a in recent_failures])
        else:
            failed_actions = "（无）"

        # 智能选择相关动作
        relevant_actions = self._select_relevant_actions(state)

        # 构建prompt
        prompt = self.SYSTEM_PROMPT.format(
            task=self.task_description,
            location=self._extract_location(state.get('observation', '')),
            inventory=', '.join(state.get('inventory', [])) or '空',
            observation=state.get('observation', 'No observation'),
            actions=relevant_actions,
            relevant_memories=relevant_memories or "（暂无相关记忆）",
            failed_actions=failed_actions
        )

        # 调用LLM
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm_client.chat(messages, max_tokens=200, temperature=0.7)
        except Exception as e:
            if self.verbose:
                print(f"\n[错误] LLM调用失败: {e}")
            # 返回默认动作
            actions = self._select_relevant_actions(state)
            return "LLM调用失败，使用默认动作", actions.split('\n')[0].replace('- ', '') if actions else "look around"

        # 解析响应
        thought, action = self._parse_response(response)

        if self.verbose:
            print(f"\n[步骤{self.current_step + 1}]")
            print(f"思考: {thought}")
            print(f"动作: {action}")

        return thought, action

    def _select_relevant_actions(self, state: Dict[str, Any]) -> str:
        """
        智能选择相关动作，减少LLM的搜索空间

        Args:
            state: 当前状态

        Returns:
            格式化的动作列表
        """
        all_actions = state.get("possible_actions", [])
        observation = state.get('observation', '').lower()
        location = self._extract_location(state.get('observation', '')).lower()

        # 分类动作
        movement_actions = []
        exploration_actions = []
        task_actions = []
        other_actions = []

        for action in all_actions:
            action_lower = action.lower()

            # 移动相关
            if 'open door to' in action_lower or 'open' in action_lower:
                if 'kitchen' in action_lower or 'door' in action_lower:
                    movement_actions.append(action)
            # 探索相关
            elif action_lower in ['look around', 'inventory', 'look at']:
                exploration_actions.append(action)
            # 任务相关（根据任务关键词）
            elif any(kw in action_lower for kw in ['water', 'heat', 'boil', 'liquid', 'substance']):
                task_actions.append(action)
            else:
                other_actions.append(action)

        # 按优先级组合动作
        selected = []

        # 如果在初始位置，优先显示移动和探索动作
        if 'hallway' in location or len(exploration_actions) < 3:
            selected.extend(exploration_actions[:3])
            selected.extend(movement_actions[:5])
        else:
            # 在其他位置，优先显示任务相关动作
            selected.extend(task_actions[:5])
            selected.extend(exploration_actions[:2])
            selected.extend(movement_actions[:3])

        # 填充到20个
        remaining = set(all_actions) - set(selected)
        selected.extend(list(remaining)[:20 - len(selected)])

        # 格式化
        return "\n".join([f"- {a}" for a in selected[:20]])

    def _extract_location(self, observation: str) -> str:
        """从观察中提取位置"""
        if "called" in observation:
            try:
                match = re.search(r'called\s+([^.]+)', observation)
                if match:
                    return match.group(1).strip()
            except:
                pass
        return "unknown"

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        解析LLM响应

        Args:
            response: LLM响应

        Returns:
            (thought, action) 元组
        """
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "未提供思考"

        # 提取 Action
        action_match = re.search(r'Action:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else ""

        # 如果没有明确格式，尝试从最后一行提取
        if not action:
            lines = response.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.lower().startswith('thought'):
                    action = line
                    break

        return thought, action

    def _execute_action(self, action: str, state: Dict[str, Any]) -> Tuple[str, float, bool, Dict]:
        """
        执行动作

        Args:
            action: 动作
            state: 当前状态

        Returns:
            (observation, reward, done, info) 元组
        """
        # 检查动作是否在可执行列表中
        valid_actions = state.get("possible_actions", [])
        if action not in valid_actions:
            # 尝试智能匹配
            validator = ActionValidator(valid_actions)
            is_valid, converted_action, reason = validator.validate_and_convert(action)

            if is_valid:
                action = converted_action
                if self.verbose and reason != "精确匹配":
                    print(f"  (动作转换: {reason})")
            else:
                if self.verbose:
                    print(f"  (动作无效，使用 look around)")
                action = "look around"

        # 执行动作
        try:
            observation, reward, done, info = self.env.step(action)
            success = info.get("is_success", False)
        except Exception as e:
            if self.verbose:
                print(f"  (执行失败: {e})")
            observation = f"执行失败: {str(e)}"
            reward = 0
            done = False
            info = {"is_success": False, "error": str(e)}
            success = False

        if self.verbose:
            print(f"结果: {observation[:100]}..." if len(observation) > 100 else f"结果: {observation}")

        return observation, reward, done, info

    def _record_step(self, state: Dict, thought: str, action: str, observation: str, info: Dict):
        """记录步骤到历史"""
        self.history.append({
            "step": self.current_step,
            "state": state,
            "thought": thought,
            "action": action,
            "observation": observation,
            "info": info
        })

    def _update_memory(self, state: Dict, action: str, observation: str, info: Dict):
        """更新记忆"""
        if not self.memory_manager or not self.memory_manager.episodic_store:
            return

        success = info.get("is_success", False)

        if success:
            # 记录成功步骤
            self.memory_manager.episodic_store.record_step(
                step_num=self.current_step,
                state=state,
                action=action,
                result=observation,
                success=True,
                task=self.task_description
            )
        else:
            # 记录失败
            if "error" in info or "No known action" in observation or "Cannot" in observation:
                self.memory_manager.episodic_store.record_error(
                    step_num=self.current_step,
                    state=state,
                    action=action,
                    error_msg=observation,
                    task=self.task_description
                )

    def _get_result(self) -> Dict[str, Any]:
        """获取执行结果"""
        is_complete = self.env.is_complete()
        score = self.env.score if hasattr(self.env, "score") else 0

        result = {
            "task": self.task_description,
            "completed": is_complete,
            "steps": self.current_step,
            "score": score,
            "success_rate": len([h for h in self.history if h["info"].get("is_success")]) / len(self.history) if self.history else 0,
            "failed_actions": len(self.action_history.failed_actions),
            "history": self.history
        }

        if self.verbose:
            print("\n" + "=" * 60)
            print("执行结果")
            print("=" * 60)
            print(f"任务完成: {'是' if is_complete else '否'}")
            print(f"执行步数: {self.current_step}")
            print(f"得分: {score}")
            print(f"成功率: {result['success_rate']:.2%}")
            print(f"失败动作数: {len(self.action_history.failed_actions)}")

        return result


# 便捷函数
def create_agent(env, memory_manager=None, **kwargs) -> ReActAgent:
    """创建ReAct Agent的便捷函数"""
    return ReActAgent(env=env, memory_manager=memory_manager, **kwargs)