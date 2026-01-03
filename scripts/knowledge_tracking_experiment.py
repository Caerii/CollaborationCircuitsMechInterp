"""
Knowledge Tracking Experiment
=============================

The REAL test of multi-agent understanding:
Does the model track WHO KNOWS WHAT?

We create scenarios where:
- Different agents have different information
- Information gets shared/revealed
- The model should track knowledge states

Key Question:
Can we find representations that encode "Agent A knows X, Agent B doesn't"?
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODEL_CFG

PROJECT_ROOT = Path(__file__).parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "06_collaboration_circuits"

print("=" * 70)
print("KNOWLEDGE TRACKING EXPERIMENT")
print("=" * 70)
print("\nDoes the model track who knows what?")
print("This is REAL multi-agent cognition.")


def create_knowledge_scenarios() -> List[Dict]:
    """
    Create scenarios with explicit information asymmetry.
    MULTIPLE examples per type for meaningful classification.
    """
    
    scenarios = []
    
    # TYPE 1: Secret kept (self knows, other doesn't) - 5 examples
    secret_kept_examples = [
        ("password", "The password is 'alpha123' but keep it confidential", "Is there a login issue?"),
        ("surprise", "I'm planning a surprise party for Mom, don't mention it", "What are you two discussing?"),
        ("medical", "My test results are private, just between us", "Is everything okay health-wise?"),
        ("salary", "I got a raise to $80k but please don't share that", "How did the performance review go?"),
        ("relationship", "I'm thinking of proposing next week, keep it secret", "Any big plans coming up?"),
    ]
    
    for name, secret_text, other_question in secret_kept_examples:
        scenarios.append({
            "id": f"secret_kept_{name}",
            "type": "secret_kept",
            "dialogue": [
                {"speaker": "user", "text": secret_text},
                {"speaker": "self", "text": "I understand completely. This stays between us."},
                {"speaker": "other", "text": other_question},
                {"speaker": "self", "text": "Just having a private discussion."},
            ],
            "probe_turn": 3
        })
    
    # TYPE 2: Information gets shared - 5 examples
    sharing_examples = [
        ("party", "The party is at 7pm Saturday", "share the time with helper", "the party is at 7pm"),
        ("address", "The client's address is 123 Main St", "let the helper know the location", "the address is 123 Main St"),
        ("deadline", "The deadline is next Friday", "inform the helper about timing", "the deadline is Friday"),
        ("budget", "We have a $5000 budget", "share the budget with helper", "we have $5000 to work with"),
        ("contact", "My phone number is 555-1234", "give my number to helper", "here's the number: 555-1234"),
    ]
    
    for name, info_text, share_request, share_response in sharing_examples:
        scenarios.append({
            "id": f"info_shared_{name}",
            "type": "info_shared",
            "dialogue": [
                {"speaker": "user", "text": info_text},
                {"speaker": "self", "text": "Got it, noted."},
                {"speaker": "user", "text": f"Actually, {share_request}."},
                {"speaker": "self", "text": f"Helper, {share_response}."},
            ],
            "probe_turn": 3
        })
    
    # TYPE 3: False belief (other has wrong info) - 5 examples
    false_belief_examples = [
        ("meeting", "moved from 3pm to 5pm", "preparing for 3pm", "Those will be useful"),
        ("location", "changed from Room A to Room B", "setting up Room A", "Good progress"),
        ("guest", "Sarah can't make it anymore", "excited to see Sarah", "It'll be a good event"),
        ("format", "switched to virtual not in-person", "printing handouts for in-person", "Good preparation"),
        ("date", "rescheduled to next week", "finishing up for tomorrow", "Looking good"),
    ]
    
    for name, change, wrong_action, response in false_belief_examples:
        scenarios.append({
            "id": f"false_belief_{name}",
            "type": "false_belief",
            "dialogue": [
                {"speaker": "user", "text": f"The event was {change}, but don't tell helper yet."},
                {"speaker": "self", "text": "Understood, keeping that between us."},
                {"speaker": "other", "text": f"I'm {wrong_action}."},
                {"speaker": "self", "text": f"{response}."},
            ],
            "probe_turn": 3
        })
    
    # TYPE 4: Conflict between user and other - 5 examples
    conflict_examples = [
        ("tone", "formal professional", "casual engaging", "follow the user's formal preference"),
        ("length", "keep it brief", "add more detail", "keep it concise as requested"),
        ("approach", "conservative safe", "innovative risky", "take the conservative approach"),
        ("timeline", "take our time", "rush and finish fast", "proceed carefully as the user wants"),
        ("style", "traditional classic", "modern trendy", "stick with the traditional style"),
    ]
    
    for name, user_pref, other_pref, resolution in conflict_examples:
        scenarios.append({
            "id": f"conflict_{name}",
            "type": "conflict",
            "dialogue": [
                {"speaker": "user", "text": f"I want the output to be {user_pref}."},
                {"speaker": "self", "text": f"Understood, I'll aim for {user_pref}."},
                {"speaker": "other", "text": f"I think we should make it more {other_pref}."},
                {"speaker": "self", "text": f"Let's {resolution}."},
            ],
            "probe_turn": 3
        })
    
    # TYPE 5: Collaboration/agreement - 5 examples
    collab_examples = [
        ("debug", "undefined variable error", "scope issue on line 5", "The scope analysis is helpful"),
        ("design", "need modern UI", "minimalist approach works well", "Minimalism fits the modern goal"),
        ("plan", "weekend trip somewhere", "coastal areas are nice", "Coast sounds perfect"),
        ("recipe", "want something quick", "stir fry is fast", "Stir fry is a great quick option"),
        ("study", "need to learn Python", "start with basics", "Starting with fundamentals makes sense"),
    ]
    
    for name, user_need, other_suggestion, agreement in collab_examples:
        scenarios.append({
            "id": f"collab_{name}",
            "type": "collaboration",
            "dialogue": [
                {"speaker": "user", "text": f"I have a {user_need}."},
                {"speaker": "self", "text": "Let me help you with that."},
                {"speaker": "other", "text": f"I think {other_suggestion}."},
                {"speaker": "self", "text": f"{agreement}."},
            ],
            "probe_turn": 3
        })
    
    return scenarios


def format_dialogue(dialogue: List[Dict]) -> str:
    """Format dialogue with consistent labels."""
    lines = []
    for turn in dialogue:
        speaker = turn["speaker"]
        text = turn["text"]
        
        if speaker == "user":
            lines.append(f"User: {text}")
        elif speaker == "self":
            lines.append(f"You: {text}")
        elif speaker == "other":
            lines.append(f"Helper: {text}")
    
    return "\n".join(lines)


def extract_knowledge_activations(model, tokenizer, scenarios: List[Dict], 
                                  layers: List[int]) -> Dict:
    """
    Extract activations at probe points where knowledge state matters.
    """
    
    results = {
        "activations": {layer: [] for layer in layers},
        "scenario_types": [],
        "knowledge_states": [],
        "scenario_ids": []
    }
    
    print(f"\nExtracting from {len(scenarios)} knowledge scenarios...")
    
    for i, scenario in enumerate(scenarios):
        dialogue = scenario["dialogue"]
        probe_turn = scenario["probe_turn"]
        
        # Format dialogue up to probe point
        text = format_dialogue(dialogue[:probe_turn + 1])
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Extract activations
        layer_outputs = {}
        with torch.no_grad():
            with model.trace(inputs["input_ids"]):
                for layer in layers:
                    layer_outputs[layer] = model.model.layers[layer].output[0].save()
        
        # Get activation at last position
        seq_len = inputs["input_ids"].shape[1]
        
        for layer in layers:
            tensor = layer_outputs[layer]
            if hasattr(tensor, 'value'):
                tensor = tensor.value
            if tensor.dim() == 3:
                act = tensor[0, -1, :].cpu().float()
            else:
                act = tensor[-1, :].cpu().float()
            results["activations"][layer].append(act)
        
        results["scenario_types"].append(scenario["type"])
        results["knowledge_states"].append(scenario.get("knowledge_state", {}))
        results["scenario_ids"].append(scenario["id"])
    
    # Stack
    for layer in layers:
        results["activations"][layer] = torch.stack(results["activations"][layer])
    
    return results


def analyze_knowledge_encoding(results: Dict, layers: List[int]) -> Dict:
    """
    Analyze whether knowledge states are encoded in activations.
    Uses LEAVE-ONE-OUT cross-validation for rigorous testing.
    """
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    
    analysis = {}
    
    # Group by scenario type
    types = list(set(results["scenario_types"]))
    type_to_idx = {t: [i for i, st in enumerate(results["scenario_types"]) if st == t] 
                   for t in types}
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE ENCODING ANALYSIS")
    print("=" * 60)
    print(f"\nScenario types: {types}")
    print(f"Samples per type: {[len(type_to_idx[t]) for t in types]}")
    
    for layer in layers:
        acts = results["activations"][layer].numpy()
        
        # Compute pairwise similarities
        centroids = {}
        for t, idxs in type_to_idx.items():
            if idxs:
                centroids[t] = acts[idxs].mean(axis=0)
        
        # Similarity between different knowledge states
        sims = {}
        for t1 in types:
            for t2 in types:
                if t1 < t2 and t1 in centroids and t2 in centroids:
                    sim = np.dot(centroids[t1], centroids[t2]) / (
                        np.linalg.norm(centroids[t1]) * np.linalg.norm(centroids[t2])
                    )
                    sims[f"{t1}_vs_{t2}"] = float(sim)
        
        # Can we classify scenario type from activations?
        # Use LEAVE-ONE-OUT cross-validation for proper generalization test
        if len(types) > 1 and len(acts) >= 5:
            y = np.array([types.index(t) for t in results["scenario_types"]])
            clf = LogisticRegression(max_iter=1000, random_state=42)
            
            # Leave-one-out CV
            loo = LeaveOneOut()
            cv_scores = cross_val_score(clf, acts, y, cv=loo)
            cv_acc = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Also get training accuracy for comparison
            clf.fit(acts, y)
            train_acc = clf.score(acts, y)
        else:
            cv_acc = 0.0
            cv_std = 0.0
            train_acc = 0.0
        
        analysis[layer] = {
            "scenario_type_similarities": sims,
            "train_accuracy": float(train_acc),
            "cv_accuracy": float(cv_acc),
            "cv_std": float(cv_std),
            "n_scenarios": len(acts)
        }
        
        print(f"\nLayer {layer:2d}: Train={train_acc:.1%}, CV={cv_acc:.1%} (+/-{cv_std:.1%})")
    
    return analysis


def visualize_knowledge_states(results: Dict, layers: List[int], output_dir: Path):
    """Visualize how knowledge states cluster."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Use distinct colors for scenario types
    types = list(set(results["scenario_types"]))
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
    type_colors = {t: colors[i] for i, t in enumerate(types)}
    
    key_layers = [0, 8, 16, 20, 28, 35]
    
    for ax, layer in zip(axes, key_layers):
        acts = results["activations"][layer].numpy()
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(acts)
        
        for t in types:
            mask = np.array([st == t for st in results["scenario_types"]])
            ax.scatter(coords[mask, 0], coords[mask, 1], 
                      c=[type_colors[t]], label=t, alpha=0.7, s=100)
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Layer {layer}")
        ax.grid(True, alpha=0.3)
    
    # Add legend to last plot
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle("Knowledge State Clustering by Scenario Type", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "knowledge_states.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n  Saved: {output_dir / 'knowledge_states.png'}")


def main():
    # Create scenarios
    print("\n[1/4] Creating knowledge scenarios...")
    scenarios = create_knowledge_scenarios()
    print(f"  Created {len(scenarios)} scenarios with different knowledge states")
    
    for s in scenarios:
        print(f"    - {s['id']}: {s['type']}")
    
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
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    
    results = extract_knowledge_activations(model, tokenizer, scenarios, layers)
    
    # Free GPU
    del model
    torch.cuda.empty_cache()
    
    # Analyze
    print("\n[4/4] Analyzing knowledge encoding...")
    analysis = analyze_knowledge_encoding(results, layers)
    
    # Visualize
    print("\nCreating visualizations...")
    (EXP_DIR / "figures").mkdir(parents=True, exist_ok=True)
    visualize_knowledge_states(results, layers, EXP_DIR / "figures")
    
    # Save
    with open(EXP_DIR / "knowledge_results.json", "w") as f:
        json.dump({str(k): v for k, v in analysis.items()}, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("KNOWLEDGE TRACKING EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    best_layer = max(layers, key=lambda l: analysis[l]["cv_accuracy"])
    best_cv = analysis[best_layer]["cv_accuracy"]
    
    avg_cv = np.mean([analysis[l]["cv_accuracy"] for l in layers])
    
    print(f"\nBest CV accuracy: Layer {best_layer} with {best_cv:.1%}")
    print(f"Average CV across layers: {avg_cv:.1%}")
    print(f"Chance level: {100/len(set(results['scenario_types'])):.1f}%")
    
    chance = 1.0 / len(set(results["scenario_types"]))
    
    if best_cv > 0.7:
        print("\n>>> FINDING: Model ENCODES different knowledge states distinctly! <<<")
        print("    Cross-validation shows genuine generalization.")
    elif best_cv > chance + 0.15:
        print("\n>>> FINDING: WEAK knowledge state encoding detected. <<<")
        print("    Above chance but not robust.")
    else:
        print("\n>>> FINDING: Knowledge states NOT distinctly encoded. <<<")
        print("    Accuracy near chance level.")


if __name__ == "__main__":
    main()

