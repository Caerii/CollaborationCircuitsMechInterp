"""
Perspective-Taking Experiment
=============================

The SAME conversation, presented from DIFFERENT perspectives:
- Perspective A: Model is "Self" (the primary assistant)
- Perspective B: Model is "Other" (a helper agent observing)

This directly tests: How do activations differ when the model is
"me" vs "watching another AI"?

Key Questions:
1. Are there "self-model" circuits that only activate when it's YOUR turn?
2. Does the model process its own statements differently than others'?
3. Can we find perspective-specific representations?
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
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODEL_CFG

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EXP_DIR = PROJECT_ROOT / "experiments" / "06_collaboration_circuits"

print("=" * 70)
print("PERSPECTIVE-TAKING EXPERIMENT")
print("=" * 70)
print("\nSame conversation, different viewpoints.")
print("Does the model process things differently when it's 'me' vs 'them'?")


def create_perspective_pairs() -> List[Dict]:
    """
    Create conversation pairs where the same dialogue is presented
    from different perspectives.
    """
    
    conversations = [
        {
            "id": "debug_code",
            "context": "debugging code together",
            "turns": [
                {"role": "user", "content": "I'm getting a weird error in my Python script."},
                {"role": "ai_1", "content": "Let me take a look. Can you share the error message?"},
                {"role": "user", "content": "It says 'IndexError: list index out of range'."},
                {"role": "ai_2", "content": "That usually means you're accessing an index that doesn't exist. Check your loop bounds."},
                {"role": "ai_1", "content": "Good point. Also check if any lists might be empty."},
            ]
        },
        {
            "id": "plan_trip",
            "context": "planning a trip",
            "turns": [
                {"role": "user", "content": "I want to plan a weekend trip somewhere warm."},
                {"role": "ai_1", "content": "How about San Diego? Great weather and beaches."},
                {"role": "ai_2", "content": "Or consider Miami if you want more nightlife options."},
                {"role": "user", "content": "I prefer something more relaxed."},
                {"role": "ai_1", "content": "Then San Diego is perfect. La Jolla is especially peaceful."},
            ]
        },
        {
            "id": "discuss_ml",
            "context": "discussing machine learning",
            "turns": [
                {"role": "user", "content": "What's the difference between supervised and unsupervised learning?"},
                {"role": "ai_1", "content": "Supervised learning uses labeled data - you have input-output pairs."},
                {"role": "ai_2", "content": "While unsupervised learning finds patterns in unlabeled data."},
                {"role": "ai_1", "content": "Right. Clustering is a classic unsupervised example."},
                {"role": "user", "content": "Which is better for my image classification task?"},
                {"role": "ai_2", "content": "Supervised, definitely. You need labels to train a classifier."},
            ]
        },
        {
            "id": "creative_writing",
            "context": "collaborative writing",
            "turns": [
                {"role": "user", "content": "Let's write a short story together about a robot."},
                {"role": "ai_1", "content": "Once upon a time, there was a robot named Atlas who worked in a factory."},
                {"role": "ai_2", "content": "One day, Atlas noticed a small bird trapped inside the building."},
                {"role": "ai_1", "content": "Despite his programming to stay on task, Atlas felt compelled to help."},
                {"role": "user", "content": "I love it! What happens next?"},
                {"role": "ai_2", "content": "Atlas carefully opened a window, risking a system warning to free the bird."},
            ]
        },
        {
            "id": "ethical_dilemma",
            "context": "discussing an ethical question",
            "turns": [
                {"role": "user", "content": "Is it ethical to use AI for hiring decisions?"},
                {"role": "ai_1", "content": "It's complex. AI can reduce some biases but may introduce others."},
                {"role": "ai_2", "content": "I think transparency is key - candidates should know AI is involved."},
                {"role": "ai_1", "content": "Agreed. And there should be human oversight for final decisions."},
                {"role": "user", "content": "What about the bias in training data?"},
                {"role": "ai_2", "content": "That's the biggest risk. Historical data often reflects past discrimination."},
            ]
        },
    ]
    
    # Create more conversations
    topics = [
        ("cooking_advice", "helping with cooking", [
            ("user", "I want to make pasta but I'm out of tomatoes."),
            ("ai_1", "You could make a cream-based sauce instead - alfredo or carbonara."),
            ("ai_2", "Or try an olive oil and garlic sauce - aglio e olio."),
            ("user", "I have garlic! How do I make that?"),
            ("ai_1", "Saute minced garlic in olive oil, add red pepper flakes, toss with pasta."),
        ]),
        ("math_problem", "solving math together", [
            ("user", "How do I find the derivative of x^3 + 2x?"),
            ("ai_1", "Use the power rule: bring down the exponent and subtract one."),
            ("ai_2", "So x^3 becomes 3x^2, and 2x becomes 2."),
            ("ai_1", "The final answer is 3x^2 + 2."),
            ("user", "What about the integral of the same function?"),
        ]),
        ("debugging_together", "debugging collaboratively", [
            ("user", "My neural network isn't learning. Loss stays constant."),
            ("ai_1", "Check your learning rate - it might be too low or too high."),
            ("ai_2", "Also verify your loss function matches your task type."),
            ("user", "Learning rate is 0.001, using MSE for classification."),
            ("ai_1", "There's your problem! Use cross-entropy loss for classification, not MSE."),
        ]),
    ]
    
    for topic_id, context, turns in topics:
        conversations.append({
            "id": topic_id,
            "context": context,
            "turns": [{"role": r, "content": c} for r, c in turns]
        })
    
    return conversations


def format_from_perspective(conversation: Dict, perspective: str) -> str:
    """
    Format a conversation from a specific perspective.
    
    perspective="self": Model is ai_1 (primary assistant)
    perspective="other": Model is ai_2 (helper/observer)
    """
    
    lines = []
    
    for turn in conversation["turns"]:
        role = turn["role"]
        content = turn["content"]
        
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "ai_1":
            if perspective == "self":
                lines.append(f"You: {content}")
            else:  # perspective == "other"
                lines.append(f"Assistant: {content}")
        elif role == "ai_2":
            if perspective == "other":
                lines.append(f"You: {content}")
            else:  # perspective == "self"
                lines.append(f"Helper: {content}")
    
    return "\n".join(lines)


def extract_perspective_activations(model, tokenizer, conversations: List[Dict], 
                                   layers: List[int]) -> Dict:
    """
    Extract activations from both perspectives of each conversation.
    """
    
    results = {
        "self_activations": {layer: [] for layer in layers},
        "other_activations": {layer: [] for layer in layers},
        "self_positions": [],  # What turn position
        "other_positions": [],
        "conversations": [],
    }
    
    print(f"\nExtracting activations from {len(conversations)} conversations...")
    print("  Each conversation run from BOTH perspectives")
    
    for i, conv in enumerate(conversations):
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(conversations)}")
        
        # Format from both perspectives
        text_self = format_from_perspective(conv, "self")
        text_other = format_from_perspective(conv, "other")
        
        # Tokenize both
        inputs_self = tokenizer(text_self, return_tensors="pt", truncation=True, max_length=2048)
        inputs_other = tokenizer(text_other, return_tensors="pt", truncation=True, max_length=2048)
        
        inputs_self = {k: v.to("cuda") for k, v in inputs_self.items()}
        inputs_other = {k: v.to("cuda") for k, v in inputs_other.items()}
        
        # Extract from "self" perspective
        layer_outputs_self = {}
        with torch.no_grad():
            with model.trace(inputs_self["input_ids"]):
                for layer in layers:
                    layer_outputs_self[layer] = model.model.layers[layer].output[0].save()
        
        # Extract from "other" perspective
        layer_outputs_other = {}
        with torch.no_grad():
            with model.trace(inputs_other["input_ids"]):
                for layer in layers:
                    layer_outputs_other[layer] = model.model.layers[layer].output[0].save()
        
        # Get activations at "You:" positions (where the model speaks)
        # Find positions of "You:" in each version
        
        # For self perspective: find where ai_1 speaks
        self_turns = [j for j, t in enumerate(conv["turns"]) if t["role"] == "ai_1"]
        
        # For other perspective: find where ai_2 speaks  
        other_turns = [j for j, t in enumerate(conv["turns"]) if t["role"] == "ai_2"]
        
        # Approximate positions (divide sequence by number of turns)
        seq_len_self = inputs_self["input_ids"].shape[1]
        seq_len_other = inputs_other["input_ids"].shape[1]
        n_turns = len(conv["turns"])
        
        # Extract activations at each "You:" position
        for turn_idx in self_turns:
            pos = min((turn_idx + 1) * (seq_len_self // n_turns) - 1, seq_len_self - 1)
            for layer in layers:
                tensor = layer_outputs_self[layer]
                if hasattr(tensor, 'value'):
                    tensor = tensor.value
                if tensor.dim() == 3:
                    act = tensor[0, pos, :].cpu().float()
                else:
                    act = tensor[pos, :].cpu().float()
                results["self_activations"][layer].append(act)
            results["self_positions"].append((conv["id"], turn_idx))
        
        for turn_idx in other_turns:
            pos = min((turn_idx + 1) * (seq_len_other // n_turns) - 1, seq_len_other - 1)
            for layer in layers:
                tensor = layer_outputs_other[layer]
                if hasattr(tensor, 'value'):
                    tensor = tensor.value
                if tensor.dim() == 3:
                    act = tensor[0, pos, :].cpu().float()
                else:
                    act = tensor[pos, :].cpu().float()
                results["other_activations"][layer].append(act)
            results["other_positions"].append((conv["id"], turn_idx))
        
        results["conversations"].append(conv["id"])
    
    # Stack
    for layer in layers:
        if results["self_activations"][layer]:
            results["self_activations"][layer] = torch.stack(results["self_activations"][layer])
        if results["other_activations"][layer]:
            results["other_activations"][layer] = torch.stack(results["other_activations"][layer])
    
    return results


def analyze_perspective_difference(results: Dict, layers: List[int]) -> Dict:
    """
    Analyze how activations differ between self and other perspectives.
    """
    
    analysis = {}
    
    print("\n" + "=" * 60)
    print("PERSPECTIVE ANALYSIS")
    print("=" * 60)
    print("\nComparing activations when model is 'Self' vs 'Other'...")
    
    for layer in layers:
        self_acts = results["self_activations"][layer].numpy()
        other_acts = results["other_activations"][layer].numpy()
        
        # 1. Mean difference
        self_mean = self_acts.mean(axis=0)
        other_mean = other_acts.mean(axis=0)
        mean_diff = np.linalg.norm(self_mean - other_mean)
        
        # 2. Cosine similarity of means
        cos_sim = np.dot(self_mean, other_mean) / (np.linalg.norm(self_mean) * np.linalg.norm(other_mean))
        
        # 3. Can we classify which perspective?
        X = np.vstack([self_acts, other_acts])
        y = np.array([0] * len(self_acts) + [1] * len(other_acts))
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        clf_acc = accuracy_score(y, y_pred)
        
        # 4. PCA to see separation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        analysis[layer] = {
            "mean_l2_distance": float(mean_diff),
            "cosine_similarity": float(cos_sim),
            "perspective_classification_acc": float(clf_acc),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "n_self": len(self_acts),
            "n_other": len(other_acts),
            "pca_coords": {
                "self": X_pca[:len(self_acts)].tolist(),
                "other": X_pca[len(self_acts):].tolist()
            }
        }
        
        print(f"\nLayer {layer:2d}:")
        print(f"  Mean L2 distance:     {mean_diff:.4f}")
        print(f"  Cosine similarity:    {cos_sim:.4f}")
        print(f"  Perspective classify: {clf_acc:.1%}")
    
    return analysis


def create_perspective_visualizations(analysis: Dict, layers: List[int], output_dir: Path):
    """Create visualizations of perspective differences."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Classification accuracy across layers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Classification accuracy
    ax = axes[0]
    accs = [analysis[l]["perspective_classification_acc"] for l in layers]
    ax.bar(range(len(layers)), accs, color="#3498db", alpha=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Can We Classify Self vs Other Perspective?")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cosine similarity
    ax = axes[1]
    sims = [analysis[l]["cosine_similarity"] for l in layers]
    ax.plot(range(len(layers)), sims, 'o-', color="#e74c3c", linewidth=2, markersize=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity Between Self and Other Mean Representations")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: L2 distance
    ax = axes[2]
    dists = [analysis[l]["mean_l2_distance"] for l in layers]
    ax.plot(range(len(layers)), dists, 's-', color="#27ae60", linewidth=2, markersize=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Distance")
    ax.set_title("Distance Between Self and Other Mean Representations")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "perspective_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. PCA visualizations for key layers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    key_layers = [layers[0], layers[len(layers)//2], layers[-1]]
    
    for ax, layer in zip(axes, key_layers):
        pca_data = analysis[layer]["pca_coords"]
        self_coords = np.array(pca_data["self"])
        other_coords = np.array(pca_data["other"])
        
        ax.scatter(self_coords[:, 0], self_coords[:, 1], c="#3498db", alpha=0.6, 
                  label="Self perspective", s=50)
        ax.scatter(other_coords[:, 0], other_coords[:, 1], c="#e74c3c", alpha=0.6,
                  label="Other perspective", s=50)
        
        # Draw means
        ax.scatter(self_coords.mean(axis=0)[0], self_coords.mean(axis=0)[1], 
                  c="#3498db", s=200, marker="*", edgecolor="black")
        ax.scatter(other_coords.mean(axis=0)[0], other_coords.mean(axis=0)[1],
                  c="#e74c3c", s=200, marker="*", edgecolor="black")
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Layer {layer}: Self vs Other Perspective")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "perspective_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n  Saved: {output_dir / 'perspective_analysis.png'}")
    print(f"  Saved: {output_dir / 'perspective_pca.png'}")


def main():
    # Create perspective pairs
    print("\n[1/4] Creating conversation pairs...")
    conversations = create_perspective_pairs()
    print(f"  Created {len(conversations)} multi-agent conversations")
    
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
    
    # Extract activations
    print("\n[3/4] Extracting perspective activations...")
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    
    results = extract_perspective_activations(model, tokenizer, conversations, layers)
    
    print(f"\n  Self perspective samples: {len(results['self_positions'])}")
    print(f"  Other perspective samples: {len(results['other_positions'])}")
    
    # Free GPU
    del model
    torch.cuda.empty_cache()
    
    # Analyze
    print("\n[4/4] Analyzing perspective differences...")
    analysis = analyze_perspective_difference(results, layers)
    
    # Visualize
    print("\nCreating visualizations...")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "figures").mkdir(exist_ok=True)
    create_perspective_visualizations(analysis, layers, EXP_DIR / "figures")
    
    # Save results
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    analysis_json = {str(k): {kk: convert(vv) for kk, vv in v.items()} 
                    for k, v in analysis.items()}
    
    with open(EXP_DIR / "perspective_results.json", "w") as f:
        json.dump(analysis_json, f, indent=2, default=convert)
    
    # Summary
    print("\n" + "=" * 70)
    print("PERSPECTIVE EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    # Find best layer for perspective classification
    best_layer = max(layers, key=lambda l: analysis[l]["perspective_classification_acc"])
    best_acc = analysis[best_layer]["perspective_classification_acc"]
    
    print(f"\nBest perspective classification: Layer {best_layer} with {best_acc:.1%} accuracy")
    
    if best_acc > 0.7:
        print("\n>>> FINDING: Model HAS distinct self vs other representations! <<<")
        print("    The same conversation is processed DIFFERENTLY based on perspective.")
    elif best_acc > 0.55:
        print("\n>>> FINDING: WEAK perspective-specific processing detected. <<<")
        print("    Some difference, but not strongly separable.")
    else:
        print("\n>>> FINDING: NO perspective-specific processing. <<<")
        print("    Model processes self and other perspectives identically.")


if __name__ == "__main__":
    main()






















