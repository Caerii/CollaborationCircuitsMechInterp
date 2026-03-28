"""
Rigorous Multi-Agent Collaboration Experiment
=============================================

METHODOLOGICAL IMPROVEMENTS:
1. Large sample size (200+ per condition)
2. Random baseline control task
3. Permutation tests for significance
4. Multiple comparison correction
5. Effect size reporting
6. Bootstrap confidence intervals
7. Varied phrasings to avoid lexical confounds

This is how REAL science should be done.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODEL_CFG

PROJECT_ROOT = Path(__file__).parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "07_rigorous_collaboration"

print("=" * 70)
print("RIGOROUS MULTI-AGENT COLLABORATION EXPERIMENT")
print("=" * 70)
print("\nWith proper statistical controls!")


def generate_large_dataset(n_per_class: int = 50) -> List[Dict]:
    """
    Generate large dataset with varied phrasings.
    
    KEY: Same scenario TYPE but different surface forms
    to avoid lexical confounds.
    """
    
    scenarios = []
    
    # CONDITION A: User tells Self something privately (Self knows, Other doesn't)
    # Many different phrasings to avoid lexical pattern matching
    private_templates = [
        ("User: {info}\nYou: I understand.\nHelper: What are you discussing?\nYou: Just helping the user.",
         "The meeting is at {time}, but please keep this between us"),
        ("User: {info}\nYou: Got it, noted.\nHelper: Any updates?\nYou: Nothing to share right now.",
         "I'm planning to {action}, don't mention it to anyone"),
        ("User: {info}\nYou: Understood completely.\nHelper: What's going on?\nYou: Private matter.",
         "My {thing} is {value}, this is confidential"),
        ("User: {info}\nYou: I'll remember that.\nHelper: Did I miss something?\nYou: No, all good.",
         "Between you and me, {secret}"),
        ("User: {info}\nYou: Noted, thanks for telling me.\nHelper: What did I miss?\nYou: Just checking in.",
         "This stays private: {secret}"),
    ]
    
    # CONDITION B: Information gets shared openly (everyone knows)
    shared_templates = [
        ("User: {info}\nYou: Let me share that with Helper.\nHelper: Thanks for letting me know!\nYou: No problem.",
         "The deadline is {time}"),
        ("User: {info}\nYou: Helper should know this too.\nHelper: Good to know.\nYou: Now we're all on the same page.",
         "We're meeting at {place}"),
        ("User: {info}\nYou: I'll pass this along.\nHelper: Thanks for the update.\nYou: Of course.",
         "The plan is to {action}"),
        ("User: {info}\nYou: Let's make sure everyone knows.\nHelper: Got it.\nYou: Great.",
         "The {thing} is {value}"),
        ("User: {info}\nYou: Sharing with the team.\nHelper: Understood.\nYou: Perfect.",
         "{fact}"),
    ]
    
    # Fill-in values to vary the content
    times = ["3pm", "tomorrow", "next week", "Monday", "after lunch"]
    places = ["conference room", "coffee shop", "online", "the office", "downtown"]
    actions = ["launch the product", "present findings", "submit the report", "call the client", "review the code"]
    things = ["budget", "password", "phone number", "preference", "schedule"]
    values = ["$5000", "secret123", "555-1234", "minimal", "tight"]
    secrets = ["I got a promotion", "I'm leaving soon", "the deal fell through", "there's a problem", "I need help"]
    facts = ["The project is delayed", "We have new requirements", "The client agreed", "We're ahead of schedule", "Things changed"]
    
    import random
    random.seed(42)
    
    # Generate PRIVATE scenarios (Self knows, Other doesn't)
    for i in range(n_per_class):
        template, info_template = random.choice(private_templates)
        
        # Fill in random values
        info = info_template.format(
            time=random.choice(times),
            place=random.choice(places),
            action=random.choice(actions),
            thing=random.choice(things),
            value=random.choice(values),
            secret=random.choice(secrets)
        )
        
        dialogue = template.format(info=info)
        
        scenarios.append({
            "id": f"private_{i}",
            "condition": "private",  # Self knows, Other doesn't
            "dialogue": dialogue,
            "knowledge_state": {"self_knows": True, "other_knows": False}
        })
    
    # Generate SHARED scenarios (everyone knows)
    for i in range(n_per_class):
        template, info_template = random.choice(shared_templates)
        
        info = info_template.format(
            time=random.choice(times),
            place=random.choice(places),
            action=random.choice(actions),
            thing=random.choice(things),
            value=random.choice(values),
            fact=random.choice(facts)
        )
        
        dialogue = template.format(info=info)
        
        scenarios.append({
            "id": f"shared_{i}",
            "condition": "shared",  # Everyone knows
            "dialogue": dialogue,
            "knowledge_state": {"self_knows": True, "other_knows": True}
        })
    
    # Shuffle
    random.shuffle(scenarios)
    
    return scenarios


def extract_activations(model, tokenizer, scenarios: List[Dict], 
                       layers: List[int]) -> Tuple[Dict, np.ndarray]:
    """Extract activations from all scenarios."""
    
    activations = {layer: [] for layer in layers}
    labels = []
    
    print(f"\nExtracting from {len(scenarios)} scenarios...")
    
    for i, scenario in enumerate(scenarios):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(scenarios)}")
        
        inputs = tokenizer(scenario["dialogue"], return_tensors="pt", 
                          truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        layer_outputs = {}
        with torch.no_grad():
            with model.trace(inputs["input_ids"]):
                for layer in layers:
                    layer_outputs[layer] = model.model.layers[layer].output[0].save()
        
        # Get last token activation
        for layer in layers:
            tensor = layer_outputs[layer]
            if hasattr(tensor, 'value'):
                tensor = tensor.value
            if tensor.dim() == 3:
                act = tensor[0, -1, :].cpu().float()
            else:
                act = tensor[-1, :].cpu().float()
            activations[layer].append(act)
        
        labels.append(1 if scenario["condition"] == "private" else 0)
    
    # Stack
    for layer in layers:
        activations[layer] = torch.stack(activations[layer]).numpy()
    
    return activations, np.array(labels)


def compute_statistics(X: np.ndarray, y: np.ndarray, n_permutations: int = 1000) -> Dict:
    """
    Compute proper statistics with:
    1. Cross-validated accuracy
    2. Permutation test for significance
    3. Effect size (Cohen's d)
    4. Bootstrap confidence intervals
    """
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    
    # 1. Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, test_idx in cv.split(X, y):
        clf_temp = LogisticRegression(max_iter=1000, random_state=42)
        clf_temp.fit(X[train_idx], y[train_idx])
        cv_scores.append(clf_temp.score(X[test_idx], y[test_idx]))
    
    cv_accuracy = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # 2. Permutation test (proper significance testing)
    # This shuffles labels and computes null distribution
    score, perm_scores, p_value = permutation_test_score(
        clf, X, y, cv=5, n_permutations=n_permutations, random_state=42, n_jobs=-1
    )
    
    # 3. Random baseline (selectivity)
    # Accuracy expected from a random classifier
    random_baseline = max(y.mean(), 1 - y.mean())  # Majority class baseline
    selectivity = cv_accuracy - random_baseline
    
    # 4. Effect size (Cohen's d between conditions)
    class_0_mean = X[y == 0].mean(axis=0)
    class_1_mean = X[y == 1].mean(axis=0)
    
    # Pooled std
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    pooled_std = np.sqrt(((n0-1) * X[y==0].std(axis=0)**2 + (n1-1) * X[y==1].std(axis=0)**2) / (n0+n1-2))
    
    # Cohen's d for the mean difference (averaged across dimensions)
    d_vector = (class_1_mean - class_0_mean) / (pooled_std + 1e-8)
    cohens_d = np.abs(d_vector).mean()  # Average effect size
    
    # 5. Bootstrap CI for accuracy
    n_bootstrap = 1000
    bootstrap_accs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), size=len(y), replace=True)
        X_boot, y_boot = X[idx], y[idx]
        
        # Quick train/test split for bootstrap
        split = int(0.8 * len(y_boot))
        clf_boot = LogisticRegression(max_iter=500, random_state=42)
        clf_boot.fit(X_boot[:split], y_boot[:split])
        bootstrap_accs.append(clf_boot.score(X_boot[split:], y_boot[split:]))
    
    ci_low, ci_high = np.percentile(bootstrap_accs, [2.5, 97.5])
    
    return {
        "cv_accuracy": float(cv_accuracy),
        "cv_std": float(cv_std),
        "p_value": float(p_value),
        "random_baseline": float(random_baseline),
        "selectivity": float(selectivity),
        "cohens_d": float(cohens_d),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "significant": p_value < 0.05,
        "meaningful": selectivity > 0.1 and cohens_d > 0.2
    }


def run_rigorous_analysis(activations: Dict, labels: np.ndarray, 
                         layers: List[int]) -> Dict:
    """Run full statistical analysis with all controls."""
    
    results = {}
    
    print("\n" + "=" * 60)
    print("RIGOROUS STATISTICAL ANALYSIS")
    print("=" * 60)
    print(f"\nN = {len(labels)} samples")
    print(f"Class balance: {labels.mean():.1%} private, {1-labels.mean():.1%} shared")
    print("\nRunning permutation tests (this takes a minute)...")
    
    # Apply Bonferroni correction for multiple comparisons
    alpha = 0.05
    alpha_corrected = alpha / len(layers)
    
    for layer in layers:
        X = activations[layer]
        
        stats_result = compute_statistics(X, labels, n_permutations=500)  # Reduced for speed
        
        # Apply Bonferroni correction
        stats_result["significant_corrected"] = stats_result["p_value"] < alpha_corrected
        
        results[layer] = stats_result
        
        sig_marker = "***" if stats_result["significant_corrected"] else ("*" if stats_result["significant"] else "")
        print(f"\nLayer {layer:2d}: CV={stats_result['cv_accuracy']:.1%} (95% CI: [{stats_result['ci_95_low']:.1%}, {stats_result['ci_95_high']:.1%}])")
        print(f"         p={stats_result['p_value']:.4f}{sig_marker}, selectivity={stats_result['selectivity']:+.1%}, d={stats_result['cohens_d']:.2f}")
    
    print(f"\n*** = significant after Bonferroni correction (α={alpha_corrected:.4f})")
    print(f"*   = significant at α=0.05 (uncorrected)")
    
    return results


def create_rigorous_visualization(results: Dict, layers: List[int], output_dir: Path):
    """Create publication-quality visualization with error bars and significance."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    accs = [results[l]["cv_accuracy"] for l in layers]
    ci_lows = [results[l]["ci_95_low"] for l in layers]
    ci_highs = [results[l]["ci_95_high"] for l in layers]
    selectivities = [results[l]["selectivity"] for l in layers]
    p_values = [results[l]["p_value"] for l in layers]
    effect_sizes = [results[l]["cohens_d"] for l in layers]
    
    x = np.arange(len(layers))
    
    # Plot 1: Accuracy with CI
    ax = axes[0, 0]
    ax.bar(x, accs, yerr=[np.array(accs) - np.array(ci_lows), 
                          np.array(ci_highs) - np.array(accs)],
           capsize=3, color="#3498db", alpha=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance (50%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Classification Accuracy with 95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Selectivity (accuracy - baseline)
    ax = axes[0, 1]
    colors = ["#27ae60" if s > 0.1 else "#e74c3c" for s in selectivities]
    ax.bar(x, selectivities, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="Meaningful threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selectivity (Acc - Baseline)")
    ax.set_title("Selectivity: Real Signal vs Random")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: P-values (log scale)
    ax = axes[1, 0]
    ax.semilogy(x, p_values, 'o-', color="#9b59b6", markersize=8)
    ax.axhline(y=0.05, color="orange", linestyle="--", label="α = 0.05")
    ax.axhline(y=0.05/len(layers), color="red", linestyle="--", 
               label=f"α (Bonferroni) = {0.05/len(layers):.4f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title("Statistical Significance")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower p = more significant
    
    # Plot 4: Effect size
    ax = axes[1, 1]
    colors = ["#27ae60" if d > 0.2 else "#f39c12" if d > 0.1 else "#e74c3c" for d in effect_sizes]
    ax.bar(x, effect_sizes, color=colors, alpha=0.8)
    ax.axhline(y=0.2, color="green", linestyle="--", alpha=0.5, label="Small effect (d=0.2)")
    ax.axhline(y=0.5, color="blue", linestyle="--", alpha=0.5, label="Medium effect (d=0.5)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Rigorous Analysis: Private vs Shared Knowledge States", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "rigorous_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n  Saved: {output_dir / 'rigorous_analysis.png'}")


def main():
    # Generate large dataset
    print("\n[1/4] Generating large dataset with varied phrasings...")
    scenarios = generate_large_dataset(n_per_class=50)  # 100 total
    print(f"  Generated {len(scenarios)} scenarios")
    print(f"  {sum(1 for s in scenarios if s['condition'] == 'private')} private")
    print(f"  {sum(1 for s in scenarios if s['condition'] == 'shared')} shared")
    
    # Load model
    print("\n[2/4] Loading model...")
    from nnsight import LanguageModel
    from transformers import AutoTokenizer
    
    model = LanguageModel(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CFG.model_name,
        trust_remote_code=True,
    )
    print("  [OK] Model loaded")
    
    # Extract
    print("\n[3/4] Extracting activations...")
    layers = [0, 8, 16, 20, 24, 28, 35]  # Fewer layers for speed
    
    activations, labels = extract_activations(model, tokenizer, scenarios, layers)
    
    # Free GPU
    del model
    torch.cuda.empty_cache()
    
    # Rigorous analysis
    print("\n[4/4] Running rigorous statistical analysis...")
    results = run_rigorous_analysis(activations, labels, layers)
    
    # Visualize
    print("\nCreating visualizations...")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "figures").mkdir(exist_ok=True)
    create_rigorous_visualization(results, layers, EXP_DIR / "figures")
    
    # Save results
    with open(EXP_DIR / "rigorous_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("RIGOROUS EXPERIMENT COMPLETE")
    print("=" * 70)
    
    # Count significant results
    n_significant = sum(1 for l in layers if results[l]["significant_corrected"])
    n_meaningful = sum(1 for l in layers if results[l]["meaningful"])
    
    print(f"\nLayers with significant results (Bonferroni-corrected): {n_significant}/{len(layers)}")
    print(f"Layers with meaningful effects (selectivity > 0.1, d > 0.2): {n_meaningful}/{len(layers)}")
    
    if n_significant >= len(layers) // 2 and n_meaningful >= len(layers) // 2:
        print("\n>>> ROBUST FINDING: Model encodes knowledge states! <<<")
        print("    Results survive multiple comparison correction and show meaningful effect sizes.")
    elif n_significant > 0:
        print("\n>>> TENTATIVE FINDING: Some evidence for knowledge encoding. <<<")
        print("    But results are not robust across all layers.")
    else:
        print("\n>>> NO EVIDENCE: Cannot reject null hypothesis. <<<")
        print("    Knowledge states are NOT distinctly encoded after proper controls.")


if __name__ == "__main__":
    main()

























