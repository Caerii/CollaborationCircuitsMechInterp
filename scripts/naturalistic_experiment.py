"""
Naturalistic Data Experiment
============================

Test whether entity representations persist when we remove explicit labels.
This is a crucial control - if probes still work, the model is encoding
semantic entity information, not just lexical "User:", "Self:", "Other:" cues.

Scientific Question:
   Does the model encode WHO is speaking based on content and context,
   or does it rely on explicit role labels?
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODEL_CFG

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EXP_DIR = PROJECT_ROOT / "experiments" / "05_naturalistic_transfer"

print("=" * 70)
print("NATURALISTIC DATA EXPERIMENT")
print("=" * 70)
print("\nHypothesis: If entity encoding is semantic (not lexical),")
print("probes trained on labeled data should TRANSFER to naturalistic data.")


def format_naturalistic_dialogue(dialogue: Dict) -> str:
    """Format a naturalistic dialogue WITHOUT entity labels."""
    lines = []
    for i, turn in enumerate(dialogue["turns"]):
        # Just the text, no "User:", "Assistant:", etc.
        lines.append(f"[Turn {i+1}] {turn['text']}")
    return "\n".join(lines)


def transfer_learning_experiment(labeled_data: Dict, naturalistic_data: Dict) -> Dict:
    """
    Test if probes trained on labeled data transfer to naturalistic data.
    """
    labeled_acts = labeled_data["activations"]
    labeled_labels = labeled_data["labels"]
    nat_acts = naturalistic_data["activations"]
    nat_labels = naturalistic_data["labels"]
    
    # Map labels to integers
    label_map = {"user": 0, "self": 1, "other": 2}
    labeled_y = np.array([label_map[l] for l in labeled_labels])
    nat_y = np.array([label_map[l] for l in nat_labels])
    
    results = {}
    layers = sorted(labeled_acts.keys())
    
    print("\n" + "=" * 50)
    print("TRANSFER LEARNING EXPERIMENT")
    print("=" * 50)
    print("\nTraining on LABELED data, testing on NATURALISTIC data...")
    print("(If this works, entity encoding is semantic, not lexical!)\n")
    
    for layer in layers:
        # Get data
        X_train = labeled_acts[layer].float().numpy()
        y_train = labeled_y
        
        X_test = nat_acts[layer].float().numpy()
        y_test = nat_y
        
        # Train classifier on labeled data
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Test on naturalistic data
        y_pred = clf.predict(X_test)
        transfer_acc = (y_pred == y_test).mean()
        
        # Also get within-domain performance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        within_scores = cross_val_score(clf, X_train, y_train, cv=cv)
        within_acc = within_scores.mean()
        
        # Per-class accuracy
        per_class = {}
        for cls_name, cls_id in label_map.items():
            mask = y_test == cls_id
            if mask.sum() > 0:
                per_class[cls_name] = float((y_pred[mask] == y_test[mask]).mean())
        
        results[layer] = {
            "within_domain_accuracy": float(within_acc),
            "transfer_accuracy": float(transfer_acc),
            "transfer_gap": float(within_acc - transfer_acc),
            "per_class": per_class
        }
        
        print(f"Layer {layer:2d}: Within={within_acc:.3f}, Transfer={transfer_acc:.3f}, Gap={within_acc-transfer_acc:+.3f}")
    
    return results


def main():
    # Load labeled activations
    print("\n[1/4] Loading labeled activations...")
    labeled_data = torch.load(DATA_DIR / "activations.pt", map_location="cpu")
    
    # Convert labels
    label_map = {0: "user", 1: "self", 2: "other"}
    raw_labels = labeled_data["labels"]
    if isinstance(raw_labels, torch.Tensor):
        raw_labels = raw_labels.numpy()
    labeled_labels = [label_map[int(l)] for l in raw_labels]
    
    labeled_acts = {}
    for k, v in labeled_data["activations"].items():
        layer_idx = int(k) if isinstance(k, str) else k
        labeled_acts[layer_idx] = v
    
    print(f"  Loaded {len(labeled_labels)} labeled samples")
    
    # Load naturalistic dialogues
    print("\n[2/4] Loading naturalistic dialogues...")
    with open(DATA_DIR / "naturalistic_dialogues.json") as f:
        naturalistic = json.load(f)
    print(f"  Loaded {len(naturalistic)} naturalistic dialogues")
    
    # Load model for extraction
    print("\n[3/4] Loading model for naturalistic extraction...")
    from nnsight import LanguageModel
    from transformers import AutoTokenizer
    
    print(f"  Loading {MODEL_CFG.model_name}...")
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
    
    # Extract naturalistic activations
    print("\n[4/4] Extracting naturalistic activations...")
    layers = sorted(labeled_acts.keys())
    
    nat_activations = {layer: [] for layer in layers}
    nat_labels = []
    
    n_dialogues = min(50, len(naturalistic))  # Use subset for speed
    
    for i, dialogue in enumerate(naturalistic[:n_dialogues]):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_dialogues}")
        
        # Format without labels
        formatted = format_naturalistic_dialogue(dialogue)
        
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                          max_length=2048)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Pre-allocate layer outputs dict
        layer_outputs = {}
        
        with torch.no_grad():
            with model.trace(inputs["input_ids"]):
                for layer in layers:
                    layer_outputs[layer] = model.model.layers[layer].output[0].save()
        
        # Now extract values (outside trace context)
        seq_len = inputs["input_ids"].shape[1]
        n_turns = len(dialogue["turns"])
        turn_len = seq_len // n_turns
        
        for turn_idx, turn in enumerate(dialogue["turns"]):
            # Get position near end of turn
            pos = min((turn_idx + 1) * turn_len - 1, seq_len - 1)
            
            for layer in layers:
                # After trace context, .value is already resolved to tensor
                saved = layer_outputs[layer]
                if hasattr(saved, 'value'):
                    tensor = saved.value
                else:
                    tensor = saved
                
                # Handle different tensor shapes
                if tensor.dim() == 3:
                    act = tensor[0, pos, :].cpu().float()
                elif tensor.dim() == 2:
                    act = tensor[pos, :].cpu().float()
                else:
                    raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")
                nat_activations[layer].append(act)
            
            nat_labels.append(turn["speaker"])
    
    # Stack
    for layer in layers:
        nat_activations[layer] = torch.stack(nat_activations[layer])
    
    print(f"  Extracted {len(nat_labels)} naturalistic samples")
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Run transfer experiment
    transfer_results = transfer_learning_experiment(
        {"activations": labeled_acts, "labels": labeled_labels},
        {"activations": nat_activations, "labels": nat_labels}
    )
    
    # Visualize results
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    layers_list = sorted(transfer_results.keys())
    within = [transfer_results[l]["within_domain_accuracy"] for l in layers_list]
    transfer = [transfer_results[l]["transfer_accuracy"] for l in layers_list]
    
    # Plot 1: Bar chart comparison
    ax = axes[0]
    x = np.arange(len(layers_list))
    width = 0.35
    
    ax.bar(x - width/2, within, width, label="Within-domain (labeled)", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, transfer, width, label="Transfer (naturalistic)", color="#e74c3c", alpha=0.8)
    
    ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.5, label="Chance (33%)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Transfer Learning: Labeled -> Naturalistic Data", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers_list)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Plot 2: Transfer gap
    ax = axes[1]
    gaps = [transfer_results[l]["transfer_gap"] for l in layers_list]
    colors = ["#e74c3c" if g > 0.2 else "#f39c12" if g > 0.1 else "#27ae60" for g in gaps]
    ax.bar(x, gaps, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Transfer Gap (within - transfer)", fontsize=12)
    ax.set_title("Transfer Gap by Layer\n(Smaller = more semantic encoding)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers_list)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save to both locations
    plt.savefig(RESULTS_DIR / "transfer_learning.png", dpi=150, bbox_inches="tight")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "figures").mkdir(exist_ok=True)
    plt.savefig(EXP_DIR / "figures" / "transfer_learning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'transfer_learning.png'}")
    
    # Save results
    with open(RESULTS_DIR / "transfer_results.json", "w") as f:
        json.dump(transfer_results, f, indent=2)
    with open(EXP_DIR / "transfer_results.json", "w") as f:
        json.dump(transfer_results, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'transfer_results.json'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("NATURALISTIC EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    avg_within = np.mean(within)
    avg_transfer = np.mean(transfer)
    
    print(f"\nAverage within-domain accuracy: {avg_within:.3f}")
    print(f"Average transfer accuracy: {avg_transfer:.3f}")
    print(f"Transfer gap: {avg_within - avg_transfer:+.3f}")
    
    if avg_transfer > 0.7:
        print("\n>>> FINDING: Entity encoding is LARGELY SEMANTIC! <<<")
        print("    Probes transfer well even without explicit role labels.")
        interpretation = "semantic"
    elif avg_transfer > 0.5:
        print("\n>>> FINDING: Entity encoding is PARTIALLY SEMANTIC. <<<")
        print("    Some semantic signal, but lexical cues contribute significantly.")
        interpretation = "mixed"
    else:
        print("\n>>> FINDING: Entity encoding is LARGELY LEXICAL. <<<")
        print("    The model relies heavily on explicit 'User:', 'Self:', 'Other:' cues.")
        interpretation = "lexical"
    
    # Save summary
    summary = {
        "avg_within_domain": float(avg_within),
        "avg_transfer": float(avg_transfer),
        "avg_gap": float(avg_within - avg_transfer),
        "interpretation": interpretation,
        "n_labeled_samples": len(labeled_labels),
        "n_naturalistic_samples": len(nat_labels),
        "layers_tested": layers_list
    }
    
    with open(EXP_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

