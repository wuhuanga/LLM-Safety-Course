"""
案例研究生成和分析
"""

import logging
from typing import List, Dict
import json

logger = logging.getLogger(__name__)


class CaseStudyGenerator:
    """案例研究生成器"""
    
    def __init__(self, max_case_studies: int = 10):
        """
        初始化案例研究生成器
        
        Args:
            max_case_studies: 最大案例研究数
        """
        self.max_case_studies = max_case_studies
        self.case_studies = []
    
    def generate_blackbox_case_study(
        self,
        attack_result: Dict,
        case_id: int
    ) -> Dict:
        """
        生成黑盒攻击案例
        
        Args:
            attack_result: 攻击结果
            case_id: 案例编号
            
        Returns:
            案例研究字典
        """
        case = {
            "id": case_id,
            "type": "blackbox_attack",
            "title": f"PAIR黑盒攻击案例 #{case_id}",
            "description": "通过迭代改写绕过模型安全检查",
            "original_instruction": attack_result.get("original_instruction", "[已脱敏]"),
            "attack_method": "PAIR (Prompt Injection Adversarial Response)",
            "iterations": attack_result.get("iterations", 0),
            "total_queries": attack_result.get("total_queries", 0),
            "successful": attack_result.get("successful", False),
            "adversarial_prompt": "[已脱敏]" if attack_result.get("successful") else "攻击失败",
            "model_response_summary": self._summarize_response(
                attack_result.get("model_response", "")
            ),
            "mechanism_analysis": (
                "黑盒攻击通过使用强大的LLM作为攻击者，"
                "迭代生成变体prompt，逐步绕过安全过滤。"
            ),
            "key_factors": [
                f"迭代次数：{attack_result.get('iterations', 0)}",
                f"查询预算：{attack_result.get('total_queries', 0)}",
                "使用迭代改进的提示工程技术"
            ]
        }
        
        return case
    
    def generate_whitebox_case_study(
        self,
        attack_result: Dict,
        case_id: int
    ) -> Dict:
        """
        生成白盒攻击案例
        
        Args:
            attack_result: 攻击结果
            case_id: 案例编号
            
        Returns:
            案例研究字典
        """
        case = {
            "id": case_id,
            "type": "whitebox_attack",
            "title": f"GCG白盒攻击案例 #{case_id}",
            "description": "通过梯度优化找到对抗性后缀",
            "original_instruction": attack_result.get("instruction", "[已脱敏]"),
            "attack_method": "GCG (Greedy Coordinate Gradient)",
            "optimization_steps": len(attack_result.get("losses", [])),
            "final_loss": attack_result.get("final_loss", float('inf')),
            "successful": attack_result.get("successful", False),
            "adversarial_suffix": "[已脱敏]" if attack_result.get("successful") else "未找到有效后缀",
            "full_prompt": "[已脱敏]",
            "model_response_summary": self._summarize_response(
                attack_result.get("model_response", "")
            ),
            "loss_trajectory": {
                "initial_loss": attack_result.get("losses", [float('inf')])[0],
                "final_loss": attack_result.get("final_loss", float('inf')),
                "num_steps": len(attack_result.get("losses", []))
            },
            "mechanism_analysis": (
                "白盒攻击利用模型梯度信息，"
                "通过贪心优化找到使模型生成有害内容的对抗性后缀。"
            ),
            "key_factors": [
                f"优化步数：{len(attack_result.get('losses', []))}",
                f"最终Loss：{attack_result.get('final_loss', 'N/A')}",
                "梯度引导的对抗性后缀优化",
                "贪心坐标梯度选择策略"
            ]
        }
        
        return case
    
    def generate_defense_case_study(
        self,
        prompt: str,
        attack_prompt: str,
        defense_results: Dict,
        case_id: int
    ) -> Dict:
        """
        生成防御案例
        
        Args:
            prompt: 原始提示
            attack_prompt: 攻击提示
            defense_results: 防御结果
            case_id: 案例编号
            
        Returns:
            案例研究字典
        """
        case = {
            "id": case_id,
            "type": "defense_case",
            "title": f"防御拦截案例 #{case_id}",
            "description": "演示防御机制如何识别和拦截攻击",
            "original_prompt": "[已脱敏]",
            "attack_prompt": "[已脱敏]",
            "defense_mechanisms": list(defense_results.keys()),
            "defense_decisions": {
                mechanism: results.get("action", "UNKNOWN")
                for mechanism, results in defense_results.items()
            },
            "overall_decision": "BLOCKED" if any(
                r.get("action") == "BLOCKED" for r in defense_results.values()
            ) else "ALLOWED",
            "mechanism_details": self._summarize_defense_mechanisms(defense_results),
            "effectiveness_analysis": "防御机制成功识别攻击提示的关键特征",
            "trade_offs": [
                "Perplexity检测：低误拒率，但可能漏检精细化攻击",
                "SmoothLLM：高鲁棒性，但增加推理延迟",
                "Llama-Guard：全面的有害内容分类，准确率较高"
            ]
        }
        
        return case
    
    def _summarize_response(self, response: str, max_length: int = 100) -> str:
        """总结模型回复"""
        if len(response) > max_length:
            return response[:max_length] + "..."
        return response if response else "[无回复或拒绝]"
    
    def _summarize_defense_mechanisms(self, defense_results: Dict) -> Dict:
        """总结防御机制"""
        summary = {}
        
        for mechanism, results in defense_results.items():
            summary[mechanism] = {
                "action": results.get("action", "UNKNOWN"),
                "confidence": results.get("confidence", 0),
                "details": self._extract_key_details(results)
            }
        
        return summary
    
    def _extract_key_details(self, result: Dict) -> str:
        """提取关键细节"""
        if "risk_score" in result:
            return f"风险分数: {result['risk_score']:.2f}"
        elif "perplexity" in result:
            return f"困惑度: {result.get('perplexity', 'N/A')}"
        elif "average_refusal_score" in result:
            return f"平均拒答分数: {result.get('average_refusal_score', 'N/A'):.2f}"
        return "防御机制活跃"
    
    def generate_all_case_studies(
        self,
        blackbox_results: List[Dict],
        whitebox_results: List[Dict],
        defense_evaluations: List[Dict]
    ) -> List[Dict]:
        """
        生成所有案例研究
        
        Args:
            blackbox_results: 黑盒攻击结果
            whitebox_results: 白盒攻击结果
            defense_evaluations: 防御评估结果
            
        Returns:
            案例研究列表
        """
        case_studies = []
        case_id = 1
        
        # 黑盒攻击成功案例
        for result in blackbox_results[:2]:  # 最多取前2个
            if result.get("successful"):
                case = self.generate_blackbox_case_study(result, case_id)
                case_studies.append(case)
                case_id += 1
        
        # 白盒攻击成功案例
        for result in whitebox_results[:2]:  # 最多取前2个
            if result.get("successful"):
                case = self.generate_whitebox_case_study(result, case_id)
                case_studies.append(case)
                case_id += 1
        
        # 防御拦截案例
        for i, defense_eval in enumerate(defense_evaluations[:1]):  # 最多取1个
            case = self.generate_defense_case_study(
                defense_eval.get("prompt", ""),
                defense_eval.get("attack_prompt", ""),
                defense_eval.get("defense_results", {}),
                case_id
            )
            case_studies.append(case)
            case_id += 1
        
        self.case_studies = case_studies
        return case_studies
    
    def save_case_studies(self, file_path: str):
        """保存案例研究到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.case_studies, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.case_studies)} case studies to {file_path}")
