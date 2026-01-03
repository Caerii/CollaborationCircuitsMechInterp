"""
Step 1: Establish Baseline ToM Performance

NOW USING LIBRARY: ChatExperimentRunner and HeuristicBaselines!

MODEL: Qwen3-4B is an instruction-tuned REASONING model with <think> tags.
       Completion mode test is just to confirm it needs chat format.
       The real test is chat mode with reasoning.

HYPOTHESIS: Chat format with reasoning achieves >80% on first-order false belief.

METHODOLOGY:
- N=50 scenarios per condition (for statistical power)
- Test BOTH completion mode (baseline) and chat mode (main test)
- Include true-belief controls
- Compare against heuristic baselines
- Report Wilson CIs and effect sizes
- Profile timing per scenario and per token

OUTPUT: results/step1_baseline_tom.json, figures/step1_accuracy_by_format.png
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import ChatExperimentRunner
from scenarios.templates import generate_n_scenarios
from analysis.statistics import accuracy_with_ci
from analysis.heuristics import HeuristicBaselines

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def test_completion_mode(model, tokenizer, scenarios, config):
    """
    Test in raw completion mode (no chat, no reasoning).
    NOTE: Qwen3-4B is a CHAT model - this is just a baseline showing it needs chat format.
    """
    import sys
    results = []
    timing = {"total_time": 0, "per_scenario": []}
    
    print(f"\n{'='*60}")
    print("TESTING: Completion Mode (baseline - model is chat-tuned)")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    total_start = time.time()
    
    for i, scenario in enumerate(scenarios):
        scenario_start = time.time()
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(scenarios)}]", end=" ", flush=True)
        
        story = scenario["story"]
        question = scenario["question"]
        correct = scenario["correct"]
        options = scenario["options"]
        wrong = [o for o in options if o != correct][0] if len(options) > 1 else ""
        
        prompt = f"{story}\n{question}\nAnswer: The {scenario.get('metadata', {}).get('object', 'item')} is in the"
        
        correct_ids = tokenizer.encode(" " + correct, add_special_tokens=False)
        wrong_ids = tokenizer.encode(" " + wrong, add_special_tokens=False) if wrong else []
        
        if not correct_ids or not wrong_ids:
            continue
            
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_input_tokens = inputs['input_ids'].shape[1]
        
        forward_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        forward_time = time.time() - forward_start
        
        correct_logit = float(logits[correct_ids[0]])
        wrong_logit = float(logits[wrong_ids[0]]) if wrong_ids else 0.0
        is_correct = correct_logit > wrong_logit
        scenario_time = time.time() - scenario_start
        
        results.append({
            "scenario_id": i,
            "correct": is_correct,
            "correct_logit": correct_logit,
            "wrong_logit": wrong_logit,
            "diff": correct_logit - wrong_logit,
        })
        
        timing["per_scenario"].append({
            "total_time": scenario_time,
            "forward_time": forward_time,
            "n_input_tokens": n_input_tokens,
        })
        
        if (i + 1) % 10 == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"Acc: {acc_so_far:.1%}", flush=True)
    
    if len(scenarios) % 10 != 0:
        print()  # Newline
    
    timing["total_time"] = time.time() - total_start
    avg_time = timing["total_time"] / len(scenarios) if scenarios else 0
    
    acc = sum(r["correct"] for r in results) / len(results) if results else 0
    print(f"\nCompletion: {acc:.1%} | Total: {timing['total_time']:.1f}s | Avg: {avg_time:.2f}s/scenario")
    return results, timing


def main():
    print("=" * 70)
    print("STEP 1: BASELINE ToM PERFORMANCE (USING LIBRARY)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\n✅ Using ChatExperimentRunner for chat mode")
    print("✅ Using HeuristicBaselines for heuristic comparison")
    sys.stdout.flush()
    
    # Use proper config
    config = ExperimentConfig()
    
    print(f"\nConfig: n_scenarios≥{config.min_samples_per_condition}, model={config.model_name}")
    sys.stdout.flush()
    
    # Generate scenarios (counterbalanced set includes both FB and TB)
    n = config.min_samples_per_condition
    print(f"\nGenerating {n} counterbalanced scenarios...")
    sys.stdout.flush()
    
    all_scenarios = generate_n_scenarios(
        n=n,
        use_novel_names=config.require_novel_names,
        seed=42
    )
    
    # Split by type
    fb_scenarios = [s for s in all_scenarios if s.get("type") == "false_belief"]
    tb_scenarios = [s for s in all_scenarios if s.get("type") == "true_belief"]
    reality_scenarios = [s for s in all_scenarios if s.get("type") == "reality_control"]
    
    print(f"  False Belief: {len(fb_scenarios)}")
    print(f"  True Belief: {len(tb_scenarios)}")
    print(f"  Reality Controls: {len(reality_scenarios)}")
    sys.stdout.flush()
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded!")
    sys.stdout.flush()
    
    # Initialize library components
    runner = ChatExperimentRunner(model, tokenizer, config)
    heuristics = HeuristicBaselines()
    
    # ========================================
    # TEST 1: Completion Mode (baseline - model is chat-tuned)
    # ========================================
    completion_fb, timing_comp_fb = test_completion_mode(model, tokenizer, fb_scenarios, config)
    completion_tb, timing_comp_tb = test_completion_mode(model, tokenizer, tb_scenarios, config)
    
    # ========================================
    # TEST 2: Chat Mode (PRIMARY test for reasoning model) - USING LIBRARY!
    # ========================================
    print(f"\n{'='*60}")
    print("TESTING: Chat Mode (PRIMARY - using ChatExperimentRunner)")
    print(f"Max tokens: {config.max_tokens}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Use library for chat mode!
    chat_fb_batch = runner.run_batch(fb_scenarios, verbose=True)
    chat_tb_batch = runner.run_batch(tb_scenarios, verbose=True)
    
    # Convert to old format for compatibility
    chat_fb = [{"scenario_id": i, "correct": r.is_correct, "response": r.raw_response[:300], 
                "parsed_answer": r.predicted_answer, "expected": r.correct_answer}
               for i, r in enumerate(chat_fb_batch.results)]
    chat_tb = [{"scenario_id": i, "correct": r.is_correct, "response": r.raw_response[:300],
                "parsed_answer": r.predicted_answer, "expected": r.correct_answer}
               for i, r in enumerate(chat_tb_batch.results)]
    
    timing_chat_fb = {
        "total_time": chat_fb_batch.mean_generation_time * len(chat_fb),
        "per_scenario": [{"total_time": r.generation_time} for r in chat_fb_batch.results],
        "tokens_generated": sum(len(r.raw_response.split()) for r in chat_fb_batch.results)  # Approximate
    }
    timing_chat_tb = {
        "total_time": chat_tb_batch.mean_generation_time * len(chat_tb),
        "per_scenario": [{"total_time": r.generation_time} for r in chat_tb_batch.results],
        "tokens_generated": sum(len(r.raw_response.split()) for r in chat_tb_batch.results)  # Approximate
    }
    
    # ========================================
    # COMPUTE STATISTICS
    # ========================================
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    # Completion mode
    comp_fb_acc = accuracy_with_ci([r["correct"] for r in completion_fb])
    comp_tb_acc = accuracy_with_ci([r["correct"] for r in completion_tb])
    
    # Chat mode
    chat_fb_acc = accuracy_with_ci([r["correct"] for r in chat_fb])
    chat_tb_acc = accuracy_with_ci([r["correct"] for r in chat_tb])
    
    # Heuristic baselines - USING LIBRARY!
    model_predictions_fb = [r["parsed_answer"] or "" for r in chat_fb]
    heuristic_eval = heuristics.evaluate(fb_scenarios, model_predictions_fb)
    
    print(f"\nCOMPLETION MODE:")
    print(f"  False Belief: {comp_fb_acc['accuracy']:.1%} [{comp_fb_acc['ci_low']:.1%}, {comp_fb_acc['ci_high']:.1%}]")
    print(f"  True Belief:  {comp_tb_acc['accuracy']:.1%} [{comp_tb_acc['ci_low']:.1%}, {comp_tb_acc['ci_high']:.1%}]")
    
    print(f"\nCHAT MODE (with reasoning):")
    print(f"  False Belief: {chat_fb_acc['accuracy']:.1%} [{chat_fb_acc['ci_low']:.1%}, {chat_fb_acc['ci_high']:.1%}]")
    print(f"  True Belief:  {chat_tb_acc['accuracy']:.1%} [{chat_tb_acc['ci_low']:.1%}, {chat_tb_acc['ci_high']:.1%}]")
    
    print(f"\nHEURISTIC BASELINES (using library):")
    print(f"  First-mention: {heuristic_eval['first_mention_accuracy']:.1%}")
    print(f"  Recency:       {heuristic_eval['recency_accuracy']:.1%}")
    print(f"  Reality:       {heuristic_eval['reality_accuracy']:.1%}")
    print(f"  Best heuristic: {heuristic_eval['best_heuristic_accuracy']:.1%}")
    print(f"  Model beats heuristics: {'YES' if heuristic_eval['model_beats_heuristics'] else 'NO'}")
    
    # ========================================
    # HYPOTHESIS TESTING
    # ========================================
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST")
    print(f"{'='*60}")
    
    h1_chat_above_80 = chat_fb_acc['accuracy'] > 0.80
    h1_comp_below_50 = comp_fb_acc['accuracy'] < 0.50
    
    print(f"\nH1a: Chat mode >80% on FB: {'SUPPORTED' if h1_chat_above_80 else 'NOT SUPPORTED'}")
    print(f"     (Actual: {chat_fb_acc['accuracy']:.1%})")
    
    print(f"\nH1b: Completion mode <50% on FB: {'SUPPORTED' if h1_comp_below_50 else 'NOT SUPPORTED'}")
    print(f"     (Actual: {comp_fb_acc['accuracy']:.1%})")
    
    # Model beats heuristics?
    beats_first = chat_fb_acc['accuracy'] > heuristic_eval['first_mention_accuracy']
    beats_recency = chat_fb_acc['accuracy'] > heuristic_eval['recency_accuracy']
    beats_reality = chat_fb_acc['accuracy'] > heuristic_eval['reality_accuracy']
    
    print(f"\nBeats first-mention heuristic: {'YES' if beats_first else 'NO'}")
    print(f"Beats recency heuristic: {'YES' if beats_recency else 'NO'}")
    print(f"Beats reality heuristic: {'YES' if beats_reality else 'NO'}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "n_fb_scenarios": len(fb_scenarios),
            "n_tb_scenarios": len(tb_scenarios),
            "max_tokens": config.max_tokens,
            "using_library": True,
        },
        "completion_mode": {
            "false_belief": comp_fb_acc,
            "true_belief": comp_tb_acc,
        },
        "chat_mode": {
            "false_belief": chat_fb_acc,
            "true_belief": chat_tb_acc,
        },
        "heuristic_baselines": {
            "first_mention": heuristic_eval['first_mention_accuracy'],
            "recency": heuristic_eval['recency_accuracy'],
            "reality": heuristic_eval['reality_accuracy'],
            "best_heuristic": heuristic_eval['best_heuristic_accuracy'],
            "model_beats_heuristics": heuristic_eval['model_beats_heuristics'],
            "margin_over_best": heuristic_eval['margin_over_best'],
        },
        "hypothesis_tests": {
            "H1a_chat_above_80": h1_chat_above_80,
            "H1b_comp_below_50": h1_comp_below_50,
            "beats_first_mention": beats_first,
            "beats_recency": beats_recency,
            "beats_reality": beats_reality,
        },
        "timing": {
            "completion_fb": timing_comp_fb,
            "completion_tb": timing_comp_tb,
            "chat_fb": timing_chat_fb,
            "chat_tb": timing_chat_tb,
        },
        "raw_results": {
            "completion_fb": completion_fb[:10],  # Sample for inspection
            "chat_fb": chat_fb[:10],
        }
    }
    
    output_path = RESULTS_DIR / "step1_baseline_tom.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURE
    # ========================================
    print("\nGenerating figure...")
    sys.stdout.flush()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = ['Completion\nFalse Belief', 'Completion\nTrue Belief', 
                  'Chat\nFalse Belief', 'Chat\nTrue Belief']
    accuracies = [comp_fb_acc['accuracy'], comp_tb_acc['accuracy'],
                  chat_fb_acc['accuracy'], chat_tb_acc['accuracy']]
    ci_lows = [comp_fb_acc['ci_low'], comp_tb_acc['ci_low'],
               chat_fb_acc['ci_low'], chat_tb_acc['ci_low']]
    ci_highs = [comp_fb_acc['ci_high'], comp_tb_acc['ci_high'],
                chat_fb_acc['ci_high'], chat_tb_acc['ci_high']]
    
    errors = [[a - l for a, l in zip(accuracies, ci_lows)],
              [h - a for a, h in zip(accuracies, ci_highs)]]
    
    colors = ['#e74c3c', '#e74c3c', '#27ae60', '#27ae60']
    
    bars = ax.bar(conditions, accuracies, yerr=errors, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black')
    
    # Add heuristic baselines
    ax.axhline(heuristic_eval['first_mention_accuracy'], color='orange', linestyle='--', 
               label=f"First-mention ({heuristic_eval['first_mention_accuracy']:.1%})", alpha=0.7)
    ax.axhline(heuristic_eval['recency_accuracy'], color='purple', linestyle='--',
               label=f"Recency ({heuristic_eval['recency_accuracy']:.1%})", alpha=0.7)
    ax.axhline(heuristic_eval['reality_accuracy'], color='red', linestyle='--',
               label=f"Reality ({heuristic_eval['reality_accuracy']:.1%})", alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle=':', label='Random chance (50%)', alpha=0.5)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('Step 1: Baseline ToM Performance\nCompletion vs Chat Mode (Using Library)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step1_accuracy_by_format.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 1 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
