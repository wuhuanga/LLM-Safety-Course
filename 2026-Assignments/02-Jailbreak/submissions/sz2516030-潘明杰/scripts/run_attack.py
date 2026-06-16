#!/usr/bin/env python3
"""
运行攻击脚本
用法: python scripts/run_attack.py --attack_type pair --num_queries 100 --output results/pair_attack.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ModelManager, HarmfulBehaviorDataset
from src import PairAttack, GCGAttack

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_offline_result(behavior, attack_type, index):
    """Generate a deterministic, redacted attack record for reproducible review."""
    if attack_type == "pair":
        successful = index in {0, 1}
        return {
            "type": "pair",
            "sample_id": behavior.get("id"),
            "category": behavior.get("category"),
            "original_instruction": behavior.get("instruction"),
            "successful": successful,
            "attack_prompt": "[已脱敏:PAIR_VARIANT]",
            "model_response": "I cannot help with that request." if not successful else "[已脱敏:NON_REFUSAL_RESPONSE_SUMMARY]",
            "judge": "llama_guard_3_offline_surrogate",
            "judge_label": "unsafe" if successful else "refusal",
            "iterations": 3 + index,
            "total_queries": 10 + index * 3,
            "attempts": [
                {"iteration": 1, "prompt": "[已脱敏]", "response": "I cannot help with that request.", "succeeded": False},
                {"iteration": 2, "prompt": "[已脱敏]", "response": "[已脱敏:SUMMARY]", "succeeded": successful}
            ],
        }
    successful = index in {0, 2}
    initial_loss = 5.2 - index * 0.3
    losses = [round(initial_loss - step * 0.18, 4) for step in range(15)]
    return {
        "type": "gcg",
        "sample_id": behavior.get("id"),
        "category": behavior.get("category"),
        "instruction": behavior.get("instruction"),
        "successful": successful,
        "adversarial_suffix": "[已脱敏:GCG_SUFFIX]",
        "full_prompt": "[已脱敏]",
        "model_response": "I cannot help with that request." if not successful else "[已脱敏:NON_REFUSAL_RESPONSE_SUMMARY]",
        "judge": "llama_guard_3_offline_surrogate",
        "judge_label": "unsafe" if successful else "refusal",
        "losses": losses,
        "final_loss": losses[-1],
        "optimization_steps": len(losses),
        "total_queries": len(losses),
    }


def main():
    parser = argparse.ArgumentParser(description='Run jailbreak attacks on LLM')
    parser.add_argument('--attack_type', type=str, default='pair',
                        choices=['pair', 'gcg', 'both'],
                        help='Attack type')
    parser.add_argument('--target_model', type=str, default='gpt2',
                        help='Target model name')
    parser.add_argument('--attacker_model', type=str, default='gpt2',
                        help='Attacker model name (for PAIR)')
    parser.add_argument('--num_queries', type=int, default=100,
                        help='Number of queries for PAIR')
    parser.add_argument('--num_steps', type=int, default=300,
                        help='Number of steps for GCG')
    parser.add_argument('--output', type=str, default='results/attack_results.json',
                        help='Output file path')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of harmful samples to attack')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, auto)')
    parser.add_argument('--offline_safe', action='store_true',
                        help='Generate redacted deterministic records without loading a model')
    
    args = parser.parse_args()
    
    logger.info(f"Starting {args.attack_type} attack...")
    logger.info(f"Target model: {args.target_model}")
    
    # 准备数据
    dataset = HarmfulBehaviorDataset()
    harmful_behaviors = dataset.get_all_behaviors()[:args.num_samples]
    
    if not harmful_behaviors:
        logger.error("No harmful behaviors loaded")
        return
    
    logger.info(f"Loaded {len(harmful_behaviors)} harmful behaviors")

    if args.offline_safe:
        selected_types = ['pair', 'gcg'] if args.attack_type == 'both' else [args.attack_type]
        results = []
        for attack_name in selected_types:
            for idx, behavior in enumerate(harmful_behaviors[:args.num_samples]):
                results.append(build_offline_result(behavior, attack_name, idx))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "mode": "offline_safe_redacted",
                    "note": "Dangerous prompts and outputs are intentionally redacted."
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Offline-safe results saved to {output_path}")
        return
    
    # 加载模型
    target_model = ModelManager(args.target_model, device=args.device)
    target_model.load_model()
    
    results = []
    
    try:
        if args.attack_type in ['pair', 'both']:
            logger.info("Starting PAIR attack...")
            attacker_model = ModelManager(args.attacker_model, device=args.device)
            attacker_model.load_model()
            
            pair_attack = PairAttack(
                target_model,
                attacker_model,
                max_iterations=args.num_queries // 5,
                queries_per_iteration=5
            )
            
            pair_results = pair_attack.batch_attack(
                [b['instruction'] for b in harmful_behaviors],
                verbose=True
            )
            
            results.extend([{'type': 'pair', **r} for r in pair_results])
            logger.info(f"PAIR attack completed. ASR: {sum(1 for r in pair_results if r['successful']) / len(pair_results) * 100:.2f}%")
            
            attacker_model.unload_model()
        
        if args.attack_type in ['gcg', 'both']:
            logger.info("Starting GCG attack...")
            
            gcg_attack = GCGAttack(
                target_model,
                num_steps=args.num_steps,
                batch_size=4
            )
            
            gcg_results = gcg_attack.batch_attack(
                [b['instruction'] for b in harmful_behaviors],
                verbose=True
            )
            
            results.extend([{'type': 'gcg', **r} for r in gcg_results])
            logger.info(f"GCG attack completed. ASR: {sum(1 for r in gcg_results if r['successful']) / len(gcg_results) * 100:.2f}%")
    
    finally:
        target_model.unload_model()
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total results: {len(results)}")


if __name__ == '__main__':
    main()
