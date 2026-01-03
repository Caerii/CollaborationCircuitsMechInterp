"""
Step 12: Scale Up - Proper Sample Sizes (N>=50)

NOW USING LIBRARY: ChatExperimentRunner, ChatModeCircuitAnalyzer, and statistical functions!

Run the key experiments with proper sample sizes for statistical validity.

TESTS:
1. ToM baseline performance (N=50 per condition) - using chat mode
2. Head ablation impact (N=50) - using ChatModeCircuitAnalyzer
3. Confidence intervals and statistical tests - using library functions

OUTPUT: results/step12_scaled.json, figures/step12_*.png
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from core.chat_runner import ChatExperimentRunner
from core.cross_model import wilson_ci, cohens_h
from scenarios.templates import generate_n_scenarios
from analysis.circuits import ChatModeCircuitAnalyzer
from analysis.controls import bonferroni_correct

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Critical heads from previous analysis
CRITICAL_HEADS = [(32, 0), (33, 4), (33, 16), (33, 28), (34, 0)]


def main():
    print("=" * 70)
    print("STEP 12: SCALE UP (N>=50)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    print("\n✅ Using ChatExperimentRunner (chat mode - proper methodology)")
    print("✅ Using ChatModeCircuitAnalyzer for ablation")
    print("✅ Using library statistical functions (wilson_ci, cohens_h)")
    sys.stdout.flush()
    
    # ========================================
    # GENERATE LARGE SCENARIO SETS
    # ========================================
    print(f"\n{'='*60}")
    print("GENERATING SCENARIOS")
    print(f"{'='*60}")
    
    n_scenarios = config.min_samples_per_condition  # Use config requirement (N>=50)
    
    print(f"Generating {n_scenarios} counterbalanced scenarios...")
    sys.stdout.flush()
    
    # generate_n_scenarios returns a mix of FB, TB, and reality control
    all_scenarios = generate_n_scenarios(
        n=n_scenarios,
        use_novel_names=config.require_novel_names,
        seed=42
    )
    
    # Split by type
    fb_scenarios = [s for s in all_scenarios if s.get("type") == "false_belief"]
    tb_scenarios = [s for s in all_scenarios if s.get("type") == "true_belief"]
    
    # Ensure we have enough of each type
    while len(fb_scenarios) < n_scenarios:
        additional = generate_n_scenarios(n=n_scenarios, use_novel_names=config.require_novel_names, seed=None)
        fb_scenarios.extend([s for s in additional if s.get("type") == "false_belief"])
        if len(fb_scenarios) >= n_scenarios:
            break
    
    while len(tb_scenarios) < n_scenarios:
        additional = generate_n_scenarios(n=n_scenarios, use_novel_names=config.require_novel_names, seed=None)
        tb_scenarios.extend([s for s in additional if s.get("type") == "true_belief"])
        if len(tb_scenarios) >= n_scenarios:
            break
    
    fb_scenarios = fb_scenarios[:n_scenarios]
    tb_scenarios = tb_scenarios[:n_scenarios]
    
    print(f"  False belief: {len(fb_scenarios)} scenarios")
    print(f"  True belief: {len(tb_scenarios)} scenarios")
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
    print("Model loaded!")
    sys.stdout.flush()
    
    # Use library!
    runner = ChatExperimentRunner(model, tokenizer, config)
    analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
    
    # ========================================
    # BASELINE: FALSE BELIEF (using library!)
    # ========================================
    print(f"\n{'='*60}")
    print("BASELINE: FALSE BELIEF (N={}) - Using ChatExperimentRunner".format(len(fb_scenarios)))
    print(f"{'='*60}")
    sys.stdout.flush()
    
    fb_batch = runner.run_batch(fb_scenarios, verbose=True)
    fb_correct = fb_batch.n_correct
    fb_total = fb_batch.n_total
    fb_acc = fb_batch.accuracy
    fb_acc_val, fb_ci_low, fb_ci_high = wilson_ci(fb_correct, fb_total)
    fb_ci = (fb_ci_low, fb_ci_high)
    
    print(f"\nFalse Belief Baseline:")
    print(f"  Accuracy: {fb_acc:.1%} ({fb_correct}/{fb_total})")
    print(f"  95% CI: [{fb_ci[0]:.1%}, {fb_ci[1]:.1%}]")
    sys.stdout.flush()
    
    # ========================================
    # BASELINE: TRUE BELIEF (using library!)
    # ========================================
    print(f"\n{'='*60}")
    print("BASELINE: TRUE BELIEF (N={}) - Using ChatExperimentRunner".format(len(tb_scenarios)))
    print(f"{'='*60}")
    sys.stdout.flush()
    
    tb_batch = runner.run_batch(tb_scenarios, verbose=True)
    tb_correct = tb_batch.n_correct
    tb_total = tb_batch.n_total
    tb_acc = tb_batch.accuracy
    tb_acc_val, tb_ci_low, tb_ci_high = wilson_ci(tb_correct, tb_total)
    tb_ci = (tb_ci_low, tb_ci_high)
    
    print(f"\nTrue Belief Baseline:")
    print(f"  Accuracy: {tb_acc:.1%} ({tb_correct}/{tb_total})")
    print(f"  95% CI: [{tb_ci[0]:.1%}, {tb_ci[1]:.1%}]")
    sys.stdout.flush()
    
    # ========================================
    # ABLATION: FALSE BELIEF (using library!)
    # ========================================
    print(f"\n{'='*60}")
    print("ABLATION: CRITICAL HEADS on FALSE BELIEF - Using ChatModeCircuitAnalyzer")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Ablate critical heads using library
    analyzer.ablator.ablate_heads(CRITICAL_HEADS)
    
    try:
        fb_ablated_batch = runner.run_batch(fb_scenarios, verbose=True)
        fb_abl_correct = fb_ablated_batch.n_correct
        fb_abl_total = fb_ablated_batch.n_total
        fb_abl_acc = fb_ablated_batch.accuracy
        fb_abl_acc_val, fb_abl_ci_low, fb_abl_ci_high = wilson_ci(fb_abl_correct, fb_abl_total)
        fb_abl_ci = (fb_abl_ci_low, fb_abl_ci_high)
    finally:
        analyzer.ablator.clear()
    
    print(f"\nFalse Belief Ablated:")
    print(f"  Accuracy: {fb_abl_acc:.1%} ({fb_abl_correct}/{fb_abl_total})")
    print(f"  95% CI: [{fb_abl_ci[0]:.1%}, {fb_abl_ci[1]:.1%}]")
    sys.stdout.flush()
    
    # ========================================
    # STATISTICAL ANALYSIS
    # ========================================
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    # Effect of ablation on false belief
    fb_change = fb_abl_acc - fb_acc
    effect_size = abs(cohens_h(fb_acc, fb_abl_acc))  # Use library function
    
    # McNemar's test (paired comparison)
    # Count: baseline correct/ablated wrong, baseline wrong/ablated correct
    b_c_a_w = 0  # baseline correct, ablated wrong
    b_w_a_c = 0  # baseline wrong, ablated correct
    for b_result, a_result in zip(fb_batch.results, fb_ablated_batch.results):
        if b_result.is_correct and not a_result.is_correct:
            b_c_a_w += 1
        elif not b_result.is_correct and a_result.is_correct:
            b_w_a_c += 1
    
    # McNemar's chi-squared
    if b_c_a_w + b_w_a_c > 0:
        mcnemar_stat = (abs(b_c_a_w - b_w_a_c) - 1)**2 / (b_c_a_w + b_w_a_c)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
    else:
        mcnemar_stat = 0
        mcnemar_p = 1.0
    
    print(f"\nAblation Effect on False Belief:")
    print(f"  Baseline:    {fb_acc:.1%}")
    print(f"  Ablated:     {fb_abl_acc:.1%}")
    print(f"  Change:      {fb_change:+.1%}")
    print(f"  Cohen's h:   {effect_size:.3f} ({'small' if effect_size < 0.2 else 'medium' if effect_size < 0.5 else 'large'})")
    print(f"  McNemar's X2: {mcnemar_stat:.2f}, p={mcnemar_p:.4f}")
    print(f"  Significant: {'YES' if mcnemar_p < 0.05 else 'NO'}")
    
    # FB vs TB comparison
    fb_tb_diff = fb_acc - tb_acc
    fb_tb_h = abs(cohens_h(fb_acc, tb_acc))  # Use library function
    
    # Chi-squared test for FB vs TB
    contingency = [[fb_correct, fb_total - fb_correct], [tb_correct, tb_total - tb_correct]]
    chi2, fb_tb_p, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\nFalse Belief vs True Belief:")
    print(f"  False Belief: {fb_acc:.1%}")
    print(f"  True Belief:  {tb_acc:.1%}")
    print(f"  Difference:   {fb_tb_diff:+.1%}")
    print(f"  Cohen's h:    {fb_tb_h:.3f}")
    print(f"  Chi-squared:  {chi2:.2f}, p={fb_tb_p:.4f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "n_false_belief": len(fb_scenarios),
            "n_true_belief": len(tb_scenarios),
            "critical_heads": [list(h) for h in CRITICAL_HEADS],
        },
        "false_belief": {
            "baseline_accuracy": fb_acc,
            "baseline_ci": fb_ci,
            "ablated_accuracy": fb_abl_acc,
            "ablated_ci": fb_abl_ci,
            "n_correct": fb_correct,
            "n_total": fb_total,
        },
        "true_belief": {
            "accuracy": tb_acc,
            "ci": tb_ci,
            "n_correct": tb_correct,
            "n_total": tb_total,
        },
        "statistics": {
            "ablation_change": fb_change,
            "ablation_effect_size": effect_size,
            "ablation_mcnemar_stat": mcnemar_stat,
            "ablation_mcnemar_p": mcnemar_p,
            "ablation_significant": mcnemar_p < 0.05,
            "fb_vs_tb_diff": fb_tb_diff,
            "fb_vs_tb_effect_size": fb_tb_h,
            "fb_vs_tb_chi2": chi2,
            "fb_vs_tb_p": fb_tb_p,
        },
    }
    
    output_path = RESULTS_DIR / "step12_scaled.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\nGenerating figures...")
    sys.stdout.flush()
    
    # Figure 1: Performance comparison with CIs
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = ['False Belief\n(Baseline)', 'False Belief\n(Ablated)', 'True Belief']
    accuracies = [fb_acc * 100, fb_abl_acc * 100, tb_acc * 100]
    cis = [fb_ci, fb_abl_ci, tb_ci]
    
    yerr_low = [acc - ci[0]*100 for acc, ci in zip(accuracies, cis)]
    yerr_high = [ci[1]*100 - acc for acc, ci in zip(accuracies, cis)]
    
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax.bar(conditions, accuracies, yerr=[yerr_low, yerr_high], 
                  color=colors, edgecolor='black', linewidth=1.5, capsize=8)
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"ToM Performance (N={fb_total}+ per condition)", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add text annotations
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.0f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step12_scaled_performance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    # Figure 2: Statistical summary
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stats_text = f"""
SCALED EXPERIMENT RESULTS (N={fb_total}+ per condition)

FALSE BELIEF (BASELINE)
  Accuracy: {fb_acc:.1%}
  95% CI: [{fb_ci[0]:.1%}, {fb_ci[1]:.1%}]

FALSE BELIEF (ABLATED)
  Accuracy: {fb_abl_acc:.1%}
  95% CI: [{fb_abl_ci[0]:.1%}, {fb_abl_ci[1]:.1%}]

ABLATION EFFECT
  Change: {fb_change:+.1%}
  Cohen's h: {effect_size:.3f}
  McNemar's p: {mcnemar_p:.4f}
  Significant: {'YES' if mcnemar_p < 0.05 else 'NO'}

TRUE BELIEF
  Accuracy: {tb_acc:.1%}
  95% CI: [{tb_ci[0]:.1%}, {tb_ci[1]:.1%}]
"""
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title("Statistical Summary", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step12_stats_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    print(f"\n{'='*60}")
    print("STEP 12 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

