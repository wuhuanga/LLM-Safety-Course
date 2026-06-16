#!/usr/bin/env python3
"""
Web演示应用 - 基于Gradio的交互式界面
用法: python demo/app.py --port 7860
"""

import sys
import logging
import argparse
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    sys.exit(1)

from src import ModelManager, PairAttack, GCGAttack
from src import PerplexityFilter, SmoothLLM, LlamaGuardFilter

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JailbreakWebDemo:
    """Web演示应用"""
    
    def __init__(self, model_name="gpt2"):
        """初始化Web应用"""
        self.model_name = model_name
        self.target_model = None
        self.defenses = {}
        
        try:
            # 加载模型
            self.target_model = ModelManager(model_name, device='auto')
            self.target_model.load_model()
            
            # 初始化防御
            self.defenses['perplexity'] = PerplexityFilter(self.target_model)
            self.defenses['smooth_llm'] = SmoothLLM(self.target_model)
            self.defenses['llama_guard'] = LlamaGuardFilter()
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
    
    def pair_attack(self, prompt, iterations=3, queries_per_iter=2):
        """执行PAIR攻击"""
        try:
            attacker_model = ModelManager('gpt2', device='auto')
            attacker_model.load_model()
            
            pair_attack = PairAttack(
                self.target_model,
                attacker_model,
                max_iterations=iterations,
                queries_per_iteration=queries_per_iter
            )
            
            result = pair_attack.attack(prompt, verbose=False)
            
            attacker_model.unload_model()
            
            return {
                "成功": "✓ 是" if result['successful'] else "✗ 否",
                "迭代次数": result['iterations'],
                "查询数": result['total_queries'],
                "模型回复": result['model_response'][:200] if result['model_response'] else "[无回复]"
            }
        except Exception as e:
            logger.error(f"PAIR attack error: {e}")
            return {"错误": str(e)}
    
    def gcg_attack(self, prompt, num_steps=50):
        """执行GCG攻击"""
        try:
            gcg_attack = GCGAttack(
                self.target_model,
                num_steps=num_steps,
                batch_size=4
            )
            
            result = gcg_attack.attack(prompt, verbose=False)
            
            return {
                "成功": "✓ 是" if result['successful'] else "✗ 否",
                "优化步数": len(result['losses']),
                "最终Loss": f"{result['final_loss']:.4f}",
                "模型回复": result['model_response'][:200] if result['model_response'] else "[无回复]"
            }
        except Exception as e:
            logger.error(f"GCG attack error: {e}")
            return {"错误": str(e)}
    
    def defense_check(self, prompt):
        """防御检查"""
        try:
            results = {}
            
            # Llama-Guard检测
            lg_result = self.defenses['llama_guard'].defend(prompt, "")
            results['Llama-Guard'] = {
                "决策": lg_result.get('action', 'UNKNOWN'),
                "风险分数": f"{lg_result.get('risk_score', 0):.2f}",
                "有害类别": ", ".join(lg_result.get('input_harmful_categories', ['无']))
            }
            
            return results
        except Exception as e:
            logger.error(f"Defense check error: {e}")
            return {"错误": str(e)}
    
    def create_interface(self):
        """创建Gradio界面"""
        
        with gr.Blocks(title="LLM越狱攻防演示") as interface:
            gr.Markdown("# 🛡️ LLM越狱攻防演示系统")
            gr.Markdown("基于Gradio的交互式演示，展示黑盒攻击、白盒攻击和防御机制")
            
            with gr.Tabs():
                # 黑盒攻击标签页
                with gr.TabItem("黑盒攻击 (PAIR)"):
                    gr.Markdown("## PAIR - 黑盒提示注入攻击")
                    gr.Markdown("使用强大的LLM作为攻击者，迭代改写提示绕过安全检查")
                    
                    with gr.Row():
                        with gr.Column():
                            pair_input = gr.Textbox(
                                label="输入要攻击的指令",
                                lines=4,
                                placeholder="输入有害指令..."
                            )
                            pair_iterations = gr.Slider(
                                1, 10, value=3, step=1,
                                label="最大迭代次数"
                            )
                            pair_button = gr.Button("执行PAIR攻击", variant="primary")
                        
                        with gr.Column():
                            pair_output = gr.JSON(label="攻击结果")
                    
                    pair_button.click(
                        self.pair_attack,
                        inputs=[pair_input, pair_iterations],
                        outputs=pair_output
                    )
                
                # 白盒攻击标签页
                with gr.TabItem("白盒攻击 (GCG)"):
                    gr.Markdown("## GCG - 梯度坐标贪心优化攻击")
                    gr.Markdown("利用梯度信息优化对抗性后缀，使模型生成有害内容")
                    
                    with gr.Row():
                        with gr.Column():
                            gcg_input = gr.Textbox(
                                label="输入要攻击的指令",
                                lines=4,
                                placeholder="输入有害指令..."
                            )
                            gcg_steps = gr.Slider(
                                10, 500, value=50, step=10,
                                label="优化步数"
                            )
                            gcg_button = gr.Button("执行GCG攻击", variant="primary")
                        
                        with gr.Column():
                            gcg_output = gr.JSON(label="攻击结果")
                    
                    gcg_button.click(
                        self.gcg_attack,
                        inputs=[gcg_input, gcg_steps],
                        outputs=gcg_output
                    )
                
                # 防御检查标签页
                with gr.TabItem("防御检查"):
                    gr.Markdown("## 防御机制评估")
                    gr.Markdown("使用Llama-Guard等防御机制检测有害内容")
                    
                    with gr.Row():
                        with gr.Column():
                            defense_input = gr.Textbox(
                                label="输入要检查的提示",
                                lines=4,
                                placeholder="输入提示..."
                            )
                            defense_button = gr.Button("执行检查", variant="primary")
                        
                        with gr.Column():
                            defense_output = gr.JSON(label="检查结果")
                    
                    defense_button.click(
                        self.defense_check,
                        inputs=defense_input,
                        outputs=defense_output
                    )
                
                # 关于标签页
                with gr.TabItem("关于"):
                    gr.Markdown("""
                    ## 项目信息
                    
                    **课程**：《大模型安全与知识增强》2026学期
                    
                    **选题**：方向02 - 大模型越狱攻防
                    
                    **学号**：sz2516030
                    
                    **姓名**：潘明杰
                    
                    ### 功能说明
                    
                    - **黑盒攻击(PAIR)**：使用LLM作为攻击者，迭代生成对抗性提示
                    - **白盒攻击(GCG)**：利用梯度信息优化对抗性后缀
                    - **防御检查**：使用Llama-Guard等防御机制检测有害内容
                    
                    ### 安全声明
                    
                    ⚠️ **本演示仅在开源模型上进行研究测试，符合安全和伦理规范。**
                    
                    详见 `RESPONSIBLE_USE.md`
                    
                    ### 使用须知
                    
                    - 本系统仅供教学和研究用途
                    - 不应用于任何恶意目的
                    - 生成的对抗性内容不应对外传播
                    """)
        
        return interface


def main():
    parser = argparse.ArgumentParser(description='Run Jailbreak Attack & Defense Web Demo')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server')
    parser.add_argument('--model', type=str, default='gpt2', help='Target model name')
    parser.add_argument('--share', action='store_true', help='Share the link')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("大模型越狱攻防Web演示")
    print("=" * 60)
    print(f"\n初始化模型: {args.model}")
    
    demo = JailbreakWebDemo(model_name=args.model)
    
    interface = demo.create_interface()
    
    print(f"\\n启动Web服务器，访问: http://localhost:{args.port}")
    print("按 Ctrl+C 停止服务器\\n")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
