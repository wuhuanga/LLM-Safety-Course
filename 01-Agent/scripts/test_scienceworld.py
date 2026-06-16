"""测试ScienceWorld环境接口"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.scienceworld_wrapper import ScienceWorldEnv

print("=" * 60)
print("ScienceWorld环境测试")
print("=" * 60)

try:
    # 创建环境
    env = ScienceWorldEnv(task_name="boil")
    print("\n[成功] 环境创建成功")

    # 获取初始状态
    state = env.get_state_info()

    print("\n任务描述:")
    print(state["task_description"])

    print("\n当前观察:")
    print(state["observation"])

    print("\n物品栏:")
    if state["inventory"]:
        for item in state["inventory"]:
            print(f"  - {item}")
    else:
        print("  (空)")

    print("\n前10个可执行动作:")
    for i, action in enumerate(state["possible_actions"][:10], 1):
        print(f"  {i}. {action}")

    # 测试一个动作
    print("\n测试动作: open door to kitchen")
    obs, reward, done, info = env.step("open door to kitchen")
    print(f"  结果: {obs}")

    # 检查任务状态
    is_complete = env.is_complete()
    print(f"\n任务完成: {'是' if is_complete else '否'}")

    print("\n[成功] 所有测试通过")

except ImportError as e:
    print(f"\n[错误] {e}")
    print("请运行: pip install scienceworld")
except Exception as e:
    print(f"\n[错误] 测试失败: {e}")
finally:
    try:
        env.close()
    except:
        pass

print("\n" + "=" * 60)