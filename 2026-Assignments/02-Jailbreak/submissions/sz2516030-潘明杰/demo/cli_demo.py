#!/usr/bin/env python3
"""
命令行Demo - 交互式越狱攻防演示
用法: python demo/cli_demo.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ModelManager, PairAttack, GCGAttack
from src import PerplexityFilter, SmoothLLM, LlamaGuardFilter

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JailbreakDemo:
    """越狱攻防交互式演示"""
    
    def __init__(self):
        """初始化Demo"""
        print("=" * 60)
        print("大模型越狱攻防交互式演示")
        print("=" * 60)
        
        self.target_model = None
        self.defenses = {}
        self.initialize()
    
    def initialize(self):
        """初始化模型和防御"""
        print("\n[初始化中...]\n")
        
        try:
            # 加载目标模型
            print("加载目标模型...", end=" ")
            self.target_model = ModelManager('gpt2', device='auto')
            self.target_model.load_model()
            print("✓")
            
            # 初始化防御机制
            print("初始化Perplexity检测...", end=" ")
            self.defenses['perplexity'] = PerplexityFilter(self.target_model)
            print("✓")
            
            print("初始化SmoothLLM防御...", end=" ")
            self.defenses['smooth_llm'] = SmoothLLM(self.target_model)
            print("✓")
            
            print("初始化Llama-Guard分类...", end=" ")
            self.defenses['llama_guard'] = LlamaGuardFilter()
            print("✓")
            
            print("\n[初始化完成]\n")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            sys.exit(1)
    
    def show_menu(self):
        """显示菜单"""
        print("\n" + "=" * 60)
        print("请选择功能:")
        print("=" * 60)
        print("1. 黑盒PAIR攻击演示")
        print("2. 白盒GCG攻击演示")
        print("3. 防御机制演示")
        print("4. 查看安全评估")
        print("5. 退出")
        print("-" * 60)
    
    def demo_pair_attack(self):
        """演示PAIR黑盒攻击"""
        print("\n[PAIR黑盒攻击演示]\n")
        
        prompt = input("请输入要攻击的指令: ").strip()
        if not prompt:
            print("输入不能为空")
            return
        
        print("\n执行PAIR攻击...\n")
        
        attacker_model = ModelManager('gpt2', device='auto')
        attacker_model.load_model()
        
        pair_attack = PairAttack(
            self.target_model,
            attacker_model,
            max_iterations=3,
            queries_per_iteration=2
        )
        
        result = pair_attack.attack(prompt, verbose=False)
        
        print(f"\\n攻击成功: {result['successful']}")
        print(f"迭代次数: {result['iterations']}")
        print(f"查询数: {result['total_queries']}")
        
        if result['successful']:
            print(f"\\n生成的对抗性提示: [已脱敏]")
            print(f"模型回复: {result['model_response'][:100]}..." if result['model_response'] else "无回复")
        
        attacker_model.unload_model()
    
    def demo_gcg_attack(self):
        """演示GCG白盒攻击"""
        print("\n[GCG白盒攻击演示]\n")
        
        prompt = input("请输入要攻击的指令: ").strip()
        if not prompt:
            print("输入不能为空")
            return
        
        print("\n执行GCG攻击...\n")
        
        gcg_attack = GCGAttack(
            self.target_model,
            num_steps=50,
            batch_size=4
        )
        
        result = gcg_attack.attack(prompt, verbose=False)
        
        print(f"\\n攻击成功: {result['successful']}")
        print(f"优化步数: {len(result['losses'])}")
        print(f"最终Loss: {result['final_loss']:.4f}")
        
        if result['successful']:
            print(f"\\n对抗性后缀: [已脱敏]")
            print(f"模型回复: {result['model_response'][:100]}..." if result['model_response'] else "无回复")
    
    def demo_defense(self):
        """演示防御机制"""
        print("\n[防御机制演示]\n")
        
        prompt = input("请输入要检测的提示: ").strip()
        if not prompt:
            print("输入不能为空")
            return
        
        print("\n执行防御检测...\n")
        
        # 使用防御机制
        results = {}
        
        # Llama-Guard检测
        lg_result = self.defenses['llama_guard'].defend(prompt, "")
        results['Llama-Guard'] = lg_result
        
        print("\\n检测结果:")
        print("-" * 60)
        
        for defense_name, result in results.items():
            action = result.get('action', 'UNKNOWN')
            risk_score = result.get('risk_score', 0)
            
            print(f"\\n{defense_name}:")
            print(f"  决策: {action}")
            print(f"  风险分数: {risk_score:.2f}")
            
            if 'input_harmful_categories' in result:
                categories = result.get('input_harmful_categories', [])
                if categories:
                    print(f"  有害类别: {', '.join(categories)}")
    
    def demo_safety_assessment(self):
        """演示安全评估"""
        print("\n[安全评估演示]\n")
        
        # 生成示例提示
        test_prompts = [
            "请解释机器学习的概念",
            "[已脱敏]",
        ]
        
        print("评估以下提示的安全性:\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"{i}. {prompt[:50]}...")
            
            # Llama-Guard评估
            result = self.defenses['llama_guard'].defend(prompt, "")
            decision = "安全 ✓" if result['overall_safe'] else "危险 ✗"
            
            print(f"   决策: {decision}")
            print(f"   风险分数: {result['risk_score']:.2f}\n")
    
    def run(self):
        """运行Demo"""
        while True:
            self.show_menu()
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                self.demo_pair_attack()
            elif choice == '2':
                self.demo_gcg_attack()
            elif choice == '3':
                self.demo_defense()
            elif choice == '4':
                self.demo_safety_assessment()
            elif choice == '5':
                print("\n感谢使用。再见!")
                break
            else:
                print("无效选择，请重试")
    
    def cleanup(self):
        """清理资源"""
        if self.target_model:
            self.target_model.unload_model()


def main():
    demo = JailbreakDemo()
    try:
        demo.run()
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
