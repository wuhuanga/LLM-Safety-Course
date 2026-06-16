"""
改进版Agent - 修正移动逻辑
"""
from typing import Dict, List, Optional, Any, Tuple
import re
from src.llm.client import get_llm_client
from src.memory.base import MemoryManager
from src.memory.rag_retriever import create_retriever
from src.agent.action_validator import ActionValidator, ActionHistory


class ReActAgent:
    """ReAct智能体 - 改进版"""

    # 修正的Prompt
    SYSTEM_PROMPT = """你是一个在ScienceWorld环境中执行任务的智能体。
你的目标是：{task}

【重要：ScienceWorld移动规则】
在ScienceWorld中，移动需要两步：
1. 第一步：open door to [房间名]  → 打开那个房间的门
2. 第二步：go to [房间名]          → 进入那个房间

示例：
- 要去厨房：先 "open door to kitchen"，然后 "go to kitchen"
- 门打开后，立刻用 "go to [房间名]" 移动

【重要：在同一房间内不需要使用go移动】
一旦进入一个房间（如厨房），你已经在那个房间内了。
- 不需要再使用"go to sink"或类似动作
- 直接与房间内的物品互动即可
- 例如：要使用sink，直接用"activate sink"或"examine sink"

【执行策略】
1. 如果在hallway，先打开目标房间的门，然后用"go to"移动
2. 进入房间后，直接与房间内的物品互动，不要再用go移动（除非要离开）
3. 对于物品操作，直接使用相关动词（activate, examine, take, open等）

【失败动作提示】
以下动作之前已失败，请避免重复：
{failed_actions}

【相关记忆】
{relevant_memories}

请按照以下格式进行思考和行动：

Thought: [分析当前情况]
Action: [从动作列表中选择一个]

【示例】
场景：在hallway，需要去厨房

可执行动作：
- open door to kitchen
- go to kitchen
- look around

正确执行：
Thought: 我在hallway，需要去厨房。先打开厨房的门。
Action: open door to kitchen

下一步：
Thought: 门已打开，现在移动到厨房。
Action: go to kitchen

场景：在厨房，需要使用sink

可执行动作：
- activate sink
- go to door to hallway
- examine sink
- look around

正确执行：
Thought: 我在厨房，需要使用sink。直接激活它。
Action: activate sink

错误执行：
Thought: 我在厨房，需要使用sink。我要去sink。
Action: go to sink  ← 错误！ sink不是房间，不能go

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
        self.env = env
        self.llm_client = llm_client or get_llm_client()
        self.memory_manager = memory_manager
        self.max_steps = max_steps
        self.verbose = verbose

        self.action_history = ActionHistory()

        self.retriever = None
        if memory_manager:
            self.retriever = create_retriever(
                memory_manager.semantic_store,
                top_k=3
            )

        self.task_description = ""
        self.current_step = 0
        self.history: List[Dict] = []

    def run_task(self, task_name: str = None) -> Dict[str, Any]:
        if task_name:
            env_info = self.env.load_task(task_name)
            self.task_description = env_info.get("description", "")
        else:
            self.task_description = self.env.get_task_description()

        if self.memory_manager and self.memory_manager.episodic_store:
            self.memory_manager.episodic_store.set_current_task(task_name)

        if self.verbose:
            print("=" * 60)
            print(f"任务: {task_name or 'current task'}")
            print(f"描述: {self.task_description}")
            print("=" * 60)

        while self.current_step < self.max_steps:
            if self.env.is_complete():
                if self.verbose:
                    print("\n[成功] 任务完成！")
                break

            state = self._get_current_state()
            thought, action = self._think_and_act(state)

            if not action:
                if self.verbose:
                    print("\n[失败] 无法生成有效动作")
                break

            observation, reward, done, info = self._execute_action(action, state)

            success = info.get("is_success", False)
            self.action_history.record(action, success, observation)

            self._record_step(state, thought, action, observation, info)
            self._update_memory(state, action, observation, info)

            self.current_step += 1

            if self.verbose and self.current_step % 5 == 0:
                print(f"\n--- 已执行 {self.current_step} 步 ---")

        return self._get_result()

    def _get_current_state(self) -> Dict[str, Any]:
        return self.env.get_state_info()

    def _think_and_act(self, state: Dict[str, Any]) -> Tuple[str, str]:
        relevant_memories = ""
        if self.retriever and self.memory_manager:
            memories = self.retriever.retrieve(
                query=f"{self.task_description} {state.get('observation', '')}",
                context={"location": self._extract_location(state.get('observation', ''))}
            )
            if memories:
                relevant_memories = "\n".join([f"- {m.content}" for m in memories])

        failed_actions = ""
        recent_failures = self.action_history.get_recent_failures(3)
        if recent_failures:
            failed_actions = "\n".join([f"- {a}" for a in recent_failures])
        else:
            failed_actions = "（无）"

        relevant_actions = self._select_relevant_actions(state)

        prompt = self.SYSTEM_PROMPT.format(
            task=self.task_description,
            location=self._extract_location(state.get('observation', '')),
            inventory=', '.join(state.get('inventory', [])) or '空',
            observation=state.get('observation', 'No observation'),
            actions=relevant_actions,
            relevant_memories=relevant_memories or "（暂无相关记忆）",
            failed_actions=failed_actions
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm_client.chat(messages, max_tokens=150, temperature=0.7)
        except Exception as e:
            if self.verbose:
                print(f"\n[错误] LLM调用失败: {e}")
            actions = self._select_relevant_actions(state)
            return "LLM调用失败", actions.split('\n')[0].replace('- ', '') if actions else "look around"

        thought, action = self._parse_response(response)

        if self.verbose:
            print(f"\n[步骤{self.current_step + 1}]")
            print(f"思考: {thought}")
            print(f"动作: {action}")

        return thought, action

    def _select_relevant_actions(self, state: Dict[str, Any]) -> str:
        all_actions = state.get("possible_actions", [])
        location = self._extract_location(state.get('observation', '')).lower()

        # 智能过滤：优先显示移动相关动作
        selected = []

        # 如果在hallway，优先显示go和open动作
        if 'hallway' in location:
            go_actions = [a for a in all_actions if a.startswith('go to ')]
            open_actions = [a for a in all_actions if a.startswith('open door to ')]
            selected.extend(go_actions[:10])
            selected.extend(open_actions[:10])
        else:
            # 在其他房间，显示任务相关动作
            task_actions = [a for a in all_actions if any(kw in a.lower() for kw in ['water', 'heat', 'boil', 'liquid', 'substance', 'stove', 'burner', 'kettle'])]
            selected.extend(task_actions[:10])
            go_actions = [a for a in all_actions if a.startswith('go to ')]
            selected.extend(go_actions[:5])

        # 补充其他动作
        remaining = [a for a in all_actions if a not in selected]
        selected.extend(remaining[:15 - len(selected)])

        return "\n".join([f"- {a}" for a in selected[:15]])

    def _extract_location(self, observation: str) -> str:
        if "called" in observation:
            try:
                match = re.search(r'called\s+([^.]+)', observation)
                if match:
                    return match.group(1).strip()
            except:
                pass
        return "unknown"

    def _parse_response(self, response: str) -> Tuple[str, str]:
        thought_match = re.search(r'Thought:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "未提供思考"

        action_match = re.search(r'Action:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else ""

        if not action:
            lines = response.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.lower().startswith('thought'):
                    action = line
                    break

        return thought, action

    def _execute_action(self, action: str, state: Dict[str, Any]) -> Tuple[str, float, bool, Dict]:
        valid_actions = state.get("possible_actions", [])
        if action not in valid_actions:
            validator = ActionValidator(valid_actions)
            is_valid, converted_action, reason = validator.validate_and_convert(action)
            if is_valid:
                action = converted_action
                if self.verbose and reason != "精确匹配":
                    print(f"  (动作转换: {reason})")
            else:
                if self.verbose:
                    print(f"  (动作无效: {reason})")
                    print(f"  (使用 look around)")
                action = "look around"

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
            print(f"结果: {observation[:80]}..." if len(observation) > 80 else f"结果: {observation}")

        return observation, reward, done, info

    def _record_step(self, state: Dict, thought: str, action: str, observation: str, info: Dict):
        self.history.append({
            "step": self.current_step,
            "state": state,
            "thought": thought,
            "action": action,
            "observation": observation,
            "info": info
        })

    def _update_memory(self, state: Dict, action: str, observation: str, info: Dict):
        if not self.memory_manager or not self.memory_manager.episodic_store:
            return

        success = info.get("is_success", False)

        if success:
            self.memory_manager.episodic_store.record_step(
                step_num=self.current_step,
                state=state,
                action=action,
                result=observation,
                success=True,
                task=self.task_description
            )
        else:
            if "error" in info or "No known action" in observation or "Cannot" in observation:
                self.memory_manager.episodic_store.record_error(
                    step_num=self.current_step,
                    state=state,
                    action=action,
                    error_msg=observation,
                    task=self.task_description
                )

    def _get_result(self) -> Dict[str, Any]:
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

        return result


def create_agent(env, memory_manager=None, **kwargs) -> ReActAgent:
    return ReActAgent(env=env, memory_manager=memory_manager, **kwargs)