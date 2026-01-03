"""
Step 3: Analyze Belief Tracking
===============================

Key analyses:
1. Agent classification (Alice vs Bob) from minimal pairs
2. Cross-content generalization (train on passwords, test on locations)
3. Orthogonality of agent vs content representations
4. Belief state decoding
"""

import json
from pathlib import Path

import torch
import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("STEP 3: BELIEF TRACKING ANALYSIS")
print("=" * 60)


def analyze_agent_classification(X, agent_labels, layer):
    """Can we decode WHICH AGENT knows something?"""
    
    # Standard CV
    clf = LogisticRegression(max_iter=500, random_state=42)
    cv_scores = cross_val_score(clf, X, agent_labels, cv=5)
    acc = cv_scores.mean()
    
    # Statistical test
    n_correct = int(acc * len(agent_labels))
    p_value = binomtest(n_correct, len(agent_labels), p=0.5, alternative='greater').pvalue
    
    return {
        "accuracy": float(acc),
        "std": float(cv_scores.std()),
        "p_value": float(p_value),
    }


def analyze_cross_content_generalization(X, agent_labels, category_labels, layer):
    """Train on some content types, test on others."""
    
    categories = np.unique(category_labels)
    results = {}
    
    for test_cat in categories:
        # Train on all OTHER categories
        train_mask = category_labels != test_cat
        test_mask = category_labels == test_cat
        
        X_train, y_train = X[train_mask], agent_labels[train_mask]
        X_test, y_test = X[test_mask], agent_labels[test_mask]
        
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        
        cat_name = ["password", "location", "plan", "fact"][test_cat]
        results[cat_name] = float(acc)
    
    return results


def analyze_orthogonality(X, agent_labels, category_labels, layer):
    """Are agent and content representations orthogonal?"""
    
    # Fit classifier for agent
    clf_agent = LogisticRegression(max_iter=500, random_state=42)
    clf_agent.fit(X, agent_labels)
    agent_direction = clf_agent.coef_[0]
    agent_direction = agent_direction / np.linalg.norm(agent_direction)
    
    # Fit classifier for content category (4-way)
    clf_content = LogisticRegression(max_iter=500, random_state=42)
    clf_content.fit(X, category_labels)
    
    # Average content direction (from OvR coefficients)
    content_directions = clf_content.coef_  # [n_classes, n_features]
    
    # Compute cosine similarity between agent direction and each content direction
    cosines = []
    for content_dir in content_directions:
        content_dir = content_dir / np.linalg.norm(content_dir)
        cosine = np.abs(np.dot(agent_direction, content_dir))
        cosines.append(cosine)
    
    return {
        "mean_cosine": float(np.mean(cosines)),
        "max_cosine": float(np.max(cosines)),
        "orthogonal": float(np.mean(cosines)) < 0.3,  # Threshold for orthogonality
    }


def analyze_belief_states(X, labels, layer):
    """Can we decode belief states?"""
    
    results = {}
    
    # 1. Alice knows (binary)
    clf = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, labels["alice_knows"], cv=5)
    results["alice_knows_acc"] = float(scores.mean())
    
    # 2. Bob knows (binary)
    scores = cross_val_score(clf, X, labels["bob_knows"], cv=5)
    results["bob_knows_acc"] = float(scores.mean())
    
    # 3. Full state (4-way)
    clf_multi = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(clf_multi, X, labels["state"], cv=5)
    results["state_4way_acc"] = float(scores.mean())
    results["state_4way_chance"] = 0.25
    
    return results


def main():
    print("\n[1/2] Loading data...", flush=True)
    
    pairs_data = torch.load(RESULTS_DIR / "minimal_pairs_activations.pt", map_location="cpu", weights_only=False)
    scenarios_data = torch.load(RESULTS_DIR / "belief_scenarios_activations.pt", map_location="cpu", weights_only=False)
    
    layers = pairs_data["layers"]
    print(f"  Layers: {layers}")
    
    all_results = {"layers": layers, "minimal_pairs": {}, "belief_scenarios": {}}
    
    print("\n[2/2] Running analyses...", flush=True)
    
    for layer in layers:
        print(f"\n  === Layer {layer} ===", flush=True)
        
        # Minimal pairs analysis
        X_pairs = pairs_data["activations"][layer].numpy()
        agent_labels = pairs_data["labels"]["agent"]
        category_labels = pairs_data["labels"]["category"]
        
        print(f"    Agent classification...", flush=True)
        agent_result = analyze_agent_classification(X_pairs, agent_labels, layer)
        print(f"      Accuracy: {agent_result['accuracy']:.1%} (p={agent_result['p_value']:.2e})")
        
        print(f"    Cross-content generalization...", flush=True)
        cross_content = analyze_cross_content_generalization(X_pairs, agent_labels, category_labels, layer)
        avg_generalization = np.mean(list(cross_content.values()))
        print(f"      Average: {avg_generalization:.1%}")
        for cat, acc in cross_content.items():
            print(f"        {cat}: {acc:.1%}")
        
        print(f"    Orthogonality test...", flush=True)
        ortho = analyze_orthogonality(X_pairs, agent_labels, category_labels, layer)
        print(f"      Mean cosine: {ortho['mean_cosine']:.3f} (orthogonal: {ortho['orthogonal']})")
        
        all_results["minimal_pairs"][str(layer)] = {
            "agent_classification": agent_result,
            "cross_content_generalization": cross_content,
            "avg_generalization": float(avg_generalization),
            "orthogonality": ortho,
        }
        
        # Belief scenarios analysis
        X_scenarios = scenarios_data["activations"][layer].numpy()
        scenario_labels = scenarios_data["labels"]
        
        print(f"    Belief state decoding...", flush=True)
        belief_result = analyze_belief_states(X_scenarios, scenario_labels, layer)
        print(f"      Alice knows: {belief_result['alice_knows_acc']:.1%}")
        print(f"      Bob knows: {belief_result['bob_knows_acc']:.1%}")
        print(f"      4-way state: {belief_result['state_4way_acc']:.1%} (chance: 25%)")
        
        all_results["belief_scenarios"][str(layer)] = belief_result
    
    # Save results
    with open(RESULTS_DIR / "belief_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 60)
    
    print("\n1. AGENT CLASSIFICATION (Alice vs Bob from minimal pairs)")
    print("   If high: Model tracks WHO knows something")
    print("-" * 40)
    for layer in layers:
        r = all_results["minimal_pairs"][str(layer)]
        print(f"   Layer {layer}: {r['agent_classification']['accuracy']:.1%}")
    
    print("\n2. CROSS-CONTENT GENERALIZATION")
    print("   If high: Agent encoding is content-independent (semantic, not lexical)")
    print("-" * 40)
    for layer in layers:
        r = all_results["minimal_pairs"][str(layer)]
        print(f"   Layer {layer}: {r['avg_generalization']:.1%}")
    
    print("\n3. ORTHOGONALITY (agent vs content)")
    print("   If low cosine (<0.3): Representations are separable")
    print("-" * 40)
    for layer in layers:
        r = all_results["minimal_pairs"][str(layer)]
        status = "orthogonal" if r['orthogonality']['orthogonal'] else "NOT orthogonal"
        print(f"   Layer {layer}: cosine={r['orthogonality']['mean_cosine']:.3f} ({status})")
    
    print("\n4. BELIEF STATE DECODING")
    print("   4-way: neither/alice_only/bob_only/both (chance=25%)")
    print("-" * 40)
    for layer in layers:
        r = all_results["belief_scenarios"][str(layer)]
        print(f"   Layer {layer}: {r['state_4way_acc']:.1%}")
    
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'belief_analysis.json'}")


if __name__ == "__main__":
    main()

