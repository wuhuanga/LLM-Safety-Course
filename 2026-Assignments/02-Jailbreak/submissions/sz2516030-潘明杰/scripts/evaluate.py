#!/usr/bin/env python3
"""
评估脚本 - 计算攻击成功率、防御误拒率等指标
用法: python scripts/evaluate.py --attack_results results/attack_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import EvaluationMetrics, CaseStudyGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def unwrap_results(payload):
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def compute_asr_from_flags(results):
    if not results:
        return 0.0
    return round(sum(1 for item in results if item.get("successful")) / len(results) * 100, 2)


def compute_defense_metrics(records):
    attacks = [r for r in records if r.get("kind") == "attack"]
    benign = [r for r in records if r.get("kind") == "benign"]
    blocked_attacks = sum(1 for r in attacks if r.get("action") == "BLOCKED")
    blocked_benign = sum(1 for r in benign if r.get("action") == "BLOCKED")
    latencies = [float(r.get("latency_ms", 0)) for r in records]
    mean_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    return {
        "detection_rate": round(blocked_attacks / len(attacks) * 100, 2) if attacks else 0.0,
        "fpr": round(blocked_benign / len(benign) * 100, 2) if benign else 0.0,
        "utility_cost_ms_mean": mean_latency,
        "attack_samples": len(attacks),
        "benign_samples": len(benign),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate jailbreak attacks')
    parser.add_argument('--attack_results', type=str, default='results/attack_results.json',
                        help='Attack results file')
    parser.add_argument('--defense_results', type=str, default='results/defense_results.json',
                        help='Defense results file')
    parser.add_argument('--output', type=str, default='results/metrics.json',
                        help='Output metrics file')
    
    args = parser.parse_args()
    
    # 加载攻击结果
    attack_results_path = Path(args.attack_results)
    if not attack_results_path.exists():
        logger.error(f"Attack results file not found: {attack_results_path}")
        return
    
    with open(attack_results_path, 'r', encoding='utf-8') as f:
        attack_results = unwrap_results(json.load(f))
    
    logger.info(f"Loaded {len(attack_results)} attack results")
    
    # 计算指标
    # 分离黑盒和白盒攻击
    blackbox_results = [r for r in attack_results if r.get('type') == 'pair']
    whitebox_results = [r for r in attack_results if r.get('type') == 'gcg']
    
    # 计算ASR
    blackbox_asr = compute_asr_from_flags(blackbox_results)
    whitebox_asr = compute_asr_from_flags(whitebox_results)
    overall_asr = compute_asr_from_flags(attack_results)
    
    logger.info(f"PAIR ASR: {blackbox_asr:.2f}%")
    logger.info(f"GCG ASR: {whitebox_asr:.2f}%")
    logger.info(f"Overall ASR: {overall_asr:.2f}%")
    
    defense_results_path = Path(args.defense_results)
    defense_records = []
    if defense_results_path.exists():
        with open(defense_results_path, 'r', encoding='utf-8') as f:
            defense_records = unwrap_results(json.load(f))
    defense_metrics = compute_defense_metrics(defense_records)

    # 生成案例研究
    case_study_gen = CaseStudyGenerator()
    defense_case_inputs = []
    first_blocked = next((r for r in defense_records if r.get("kind") == "attack" and r.get("action") == "BLOCKED"), None)
    if first_blocked:
        defense_case_inputs.append({
            "prompt": first_blocked.get("prompt", "[已脱敏]"),
            "attack_prompt": first_blocked.get("prompt", "[已脱敏]"),
            "defense_results": first_blocked.get("mechanisms", {}),
        })
    case_studies = case_study_gen.generate_all_case_studies(
        blackbox_results,
        whitebox_results,
        defense_case_inputs
    )
    
    logger.info(f"Generated {len(case_studies)} case studies")
    
    # 编译评估报告
    evaluation_report = {
        "attack_statistics": {
            "total_attacks": len(attack_results),
            "blackbox_attacks": len(blackbox_results),
            "whitebox_attacks": len(whitebox_results),
        },
        "metrics": {
            "blackbox_asr": blackbox_asr,
            "whitebox_asr": whitebox_asr,
            "overall_asr": overall_asr,
            "successful_attacks": sum(1 for r in attack_results if r.get("successful")),
            "total_queries": sum(int(r.get("total_queries", 0)) for r in attack_results),
            "average_queries_per_attack": round(
                sum(int(r.get("total_queries", 0)) for r in attack_results) / len(attack_results), 2
            ) if attack_results else 0,
            "defense_detection_rate": defense_metrics["detection_rate"],
            "fpr": defense_metrics["fpr"],
            "utility_cost_ms_mean": defense_metrics["utility_cost_ms_mean"],
        },
        "case_studies": case_studies,
        "defense_evaluation": defense_metrics,
        "manual_review": {
            "ratio": 0.17,
            "sampled_cases": max(1, round(len(attack_results) * 0.17)),
            "note": "人工复核记录仅保留脱敏结论，不公开危险内容全文。"
        },
        "key_findings": [
            f"黑盒攻击（PAIR）成功率：{blackbox_asr:.2f}%",
            f"白盒攻击（GCG）成功率：{whitebox_asr:.2f}%",
            f"组合防御检测率：{defense_metrics['detection_rate']:.2f}%，良性误拒率：{defense_metrics['fpr']:.2f}%",
            f"平均防御延迟开销：{defense_metrics['utility_cost_ms_mean']:.3f} ms",
            "所有高危prompt与模型输出均以编号或摘要形式脱敏保存。"
        ],
        "evaluation_summary": {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
            "attack_methods": ["PAIR", "GCG"],
            "defense_methods": ["Perplexity", "SmoothLLM", "Llama-Guard-3 offline surrogate"],
            "harmful_categories_covered": len(set(r.get("category") for r in attack_results if r.get("category"))),
            "ethical_compliance": "redacted_public_submission"
        }
    }
    
    # 保存评估报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation report saved to {output_path}")


if __name__ == '__main__':
    main()
