"""
Main experiment runner for Project A: Self/Other/User Representation Separation.

This script runs the full experiment pipeline:
1. Generate synthetic multi-party dialogues
2. Extract activations at each layer
3. Train linear probes to classify entity type
4. Analyze representation separation
5. Generate visualizations

Usage:
    python scripts/run_experiment.py
    
Or run individual phases:
    python scripts/run_experiment.py --phase generate
    python scripts/run_experiment.py --phase extract
    python scripts/run_experiment.py --phase probe
    python scripts/run_experiment.py --phase analyze
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_CFG, EXP_CFG, DATA_DIR, RESULTS_DIR, ENTITY_TYPES
from src.data_generation import generate_dataset, load_dataset, Dialogue
from src.model import get_model, clear_model
from src.probing import ProbingPipeline, analyze_entity_separation


def phase_generate(args):
    """Phase 1: Generate synthetic dialogues."""
    print("\n" + "=" * 60)
    print("PHASE 1: GENERATING DIALOGUES")
    print("=" * 60)
    
    dialogues = generate_dataset(
        n_dialogues=args.n_dialogues,
        min_turns=EXP_CFG.min_turns,
        max_turns=EXP_CFG.max_turns,
        seed=EXP_CFG.seed
    )
    
    # Print sample
    print("\n--- Sample Dialogue ---")
    sample = dialogues[0]
    print(f"Scenario: {sample.scenario}")
    print(f"User persona: {sample.user_persona}")
    print(f"\n{sample.to_prompt()}")
    
    return dialogues


def phase_extract(args, dialogues=None):
    """Phase 2: Extract activations from model."""
    print("\n" + "=" * 60)
    print("PHASE 2: EXTRACTING ACTIVATIONS")
    print("=" * 60)
    
    # Load dialogues if not provided
    if dialogues is None:
        print("Loading dialogues from file...")
        dialogues = load_dataset()
    
    # Load model
    model = get_model()
    model.load()
    
    # Prepare data structures
    # We'll extract activations for each turn and track the entity type
    all_activations = {layer: [] for layer in MODEL_CFG.probe_layers}
    all_labels = []
    all_metadata = []
    
    print(f"\nExtracting activations from {len(dialogues)} dialogues...")
    print(f"Layers: {MODEL_CFG.probe_layers}")
    
    for dialogue in tqdm(dialogues, desc="Processing dialogues"):
        prompt = dialogue.to_prompt()
        
        # Extract activations for the full dialogue
        try:
            activations = model.extract_activations(prompt, layers=list(MODEL_CFG.probe_layers))
        except Exception as e:
            print(f"\nWarning: Failed to process dialogue {dialogue.dialogue_id}: {e}")
            continue
        
        # Get token positions for each turn
        # For simplicity, we'll use the last token of each turn as the "summary" representation
        current_pos = 0
        for turn_idx, turn in enumerate(dialogue.turns):
            speaker = turn["speaker"]
            entity_type = turn["entity_type"]
            content = turn["content"]
            
            # Construct the turn text as it appears in the prompt
            prefix = {"user": "User: ", "agent_a": "You: ", "agent_b": "Helper: "}[speaker]
            turn_text = prefix + content
            
            # Tokenize to get length
            turn_tokens = model.tokenize(turn_text)
            turn_len = turn_tokens["length"]
            
            # Get the last token position for this turn
            # (represents the "compressed" representation of the turn)
            last_pos = min(current_pos + turn_len - 1, activations[MODEL_CFG.probe_layers[0]].size(0) - 1)
            
            # Extract activation at this position for each layer
            for layer in MODEL_CFG.probe_layers:
                if layer in activations:
                    act = activations[layer][last_pos]  # [hidden_dim]
                    all_activations[layer].append(act)
            
            all_labels.append(entity_type)
            all_metadata.append({
                "dialogue_id": dialogue.dialogue_id,
                "turn_idx": turn_idx,
                "speaker": speaker,
                "entity_type": entity_type
            })
            
            current_pos += turn_len
    
    # Convert to tensors
    print("\nConverting to tensors...")
    for layer in all_activations:
        if all_activations[layer]:
            all_activations[layer] = torch.stack(all_activations[layer])
            print(f"  Layer {layer}: {all_activations[layer].shape}")
    
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    print(f"  Labels: {labels_tensor.shape}")
    
    # Print class distribution
    print("\nClass distribution:")
    for name, idx in ENTITY_TYPES.items():
        count = (labels_tensor == idx).sum().item()
        print(f"  {name}: {count} ({count/len(labels_tensor)*100:.1f}%)")
    
    # Save activations
    save_path = DATA_DIR / "activations.pt"
    torch.save({
        "activations": all_activations,
        "labels": labels_tensor,
        "metadata": all_metadata
    }, save_path)
    print(f"\nActivations saved to {save_path}")
    
    # Clean up model to free memory
    clear_model()
    
    return all_activations, labels_tensor, all_metadata


def phase_probe(args, activations=None, labels=None):
    """Phase 3: Train probes for entity classification."""
    print("\n" + "=" * 60)
    print("PHASE 3: TRAINING PROBES")
    print("=" * 60)
    
    # Load if not provided
    if activations is None or labels is None:
        print("Loading activations from file...")
        data = torch.load(DATA_DIR / "activations.pt")
        activations = data["activations"]
        labels = data["labels"]
    
    # Get hidden dimension from first layer
    first_layer = list(activations.keys())[0]
    hidden_dim = activations[first_layer].size(1)
    
    print(f"\nHidden dimension: {hidden_dim}")
    print(f"Number of samples: {len(labels)}")
    print(f"Layers to probe: {list(activations.keys())}")
    
    # Create and train probing pipeline
    pipeline = ProbingPipeline(
        hidden_dim=hidden_dim,
        num_classes=len(ENTITY_TYPES),
        probe_type="linear",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Train probes for all layers
    results = pipeline.train_all_layers(
        layer_activations=activations,
        labels=labels,
        task_name="entity_type",
        epochs=EXP_CFG.epochs,
        verbose=True
    )
    
    # Save results
    pipeline.save_results(RESULTS_DIR / "probe_results.json")
    pipeline.save_probes(RESULTS_DIR / "probes.pt")
    
    return pipeline, results


def phase_analyze(args, activations=None, labels=None, probe_results=None):
    """Phase 4: Analyze representation separation and generate visualizations."""
    print("\n" + "=" * 60)
    print("PHASE 4: ANALYZING REPRESENTATIONS")
    print("=" * 60)
    
    # Load data if not provided
    if activations is None or labels is None:
        data = torch.load(DATA_DIR / "activations.pt")
        activations = data["activations"]
        labels = data["labels"]
    
    if probe_results is None:
        with open(RESULTS_DIR / "probe_results.json") as f:
            probe_results = json.load(f)
    
    # 1. Probe accuracy analysis
    print("\n--- Probe Accuracy by Layer ---")
    layers = sorted([int(k) for k in probe_results.keys()])
    accuracies = [probe_results[str(l)]["test_accuracy"] for l in layers]
    
    for layer, acc in zip(layers, accuracies):
        bar = "#" * int(acc * 40)
        print(f"Layer {layer:2d}: {acc:.3f} [{bar}]")
    
    best_layer = layers[np.argmax(accuracies)]
    print(f"\nBest layer: {best_layer} (accuracy: {max(accuracies):.3f})")
    
    # 2. Representation similarity analysis
    print("\n--- Representation Similarity Analysis ---")
    separation = analyze_entity_separation(activations, labels, method="cosine")
    
    for layer in sorted(separation.keys()):
        print(f"\nLayer {layer}:")
        for pair, sim in separation[layer].items():
            print(f"  {pair}: {sim:.3f}")
    
    # 3. Generate visualizations
    print("\n--- Generating Visualizations ---")
    create_visualizations(layers, accuracies, separation, probe_results)
    
    # 4. Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Where do representations separate?
    user_self_sims = [separation[l].get("user_vs_self", 0) for l in layers]
    user_other_sims = [separation[l].get("user_vs_other", 0) for l in layers]
    self_other_sims = [separation[l].get("self_vs_other", 0) for l in layers]
    
    print(f"\nBest entity classification: Layer {best_layer} ({max(accuracies)*100:.1f}%)")
    print(f"\nRepresentation separation (lower = more separated):")
    print(f"  User vs Self:  Layer {layers[np.argmin(user_self_sims)]} ({min(user_self_sims):.3f})")
    print(f"  User vs Other: Layer {layers[np.argmin(user_other_sims)]} ({min(user_other_sims):.3f})")
    print(f"  Self vs Other: Layer {layers[np.argmin(self_other_sims)]} ({min(self_other_sims):.3f})")
    
    # Key finding
    if max(accuracies) > 0.7:
        print("\n[FINDING] Entity representations are SEPARABLE")
        print("  Linear probes can distinguish User/Self/Other with >70% accuracy")
    elif max(accuracies) > 0.5:
        print("\n[FINDING] Entity representations are PARTIALLY separable")
        print("  Some entity information is decodable but with moderate accuracy")
    else:
        print("\n[FINDING] Entity representations are NOT clearly separable")
        print("  This may indicate representations are entangled")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "best_layer": best_layer,
        "best_accuracy": max(accuracies),
        "layer_accuracies": dict(zip(layers, accuracies)),
        "separation_analysis": separation,
        "config": {
            "model": MODEL_CFG.model_name,
            "n_dialogues": args.n_dialogues,
            "probe_type": "linear"
        }
    }
    
    with open(RESULTS_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    return summary


def create_visualizations(layers, accuracies, separation, probe_results):
    """Generate key visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Probe accuracy by layer
    ax1 = axes[0, 0]
    ax1.bar(range(len(layers)), accuracies, color='steelblue')
    ax1.axhline(y=0.333, color='red', linestyle='--', label='Chance (33.3%)')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([str(l) for l in layers])
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Entity Classification Accuracy by Layer')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # 2. Per-class accuracy at best layer
    ax2 = axes[0, 1]
    best_layer = layers[np.argmax(accuracies)]
    class_accs = probe_results[str(best_layer)]["class_accuracies"]
    classes = list(class_accs.keys())
    class_vals = [class_accs[c] for c in classes]
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # user=green, self=blue, other=red
    ax2.bar(classes, class_vals, color=colors)
    ax2.axhline(y=0.333, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Per-Entity Accuracy (Layer {best_layer})')
    ax2.set_ylim(0, 1)
    
    # 3. Representation similarity heatmap
    ax3 = axes[1, 0]
    # Create similarity matrix for best layer
    entity_names = ['user', 'self', 'other']
    sim_matrix = np.zeros((3, 3))
    for i, e1 in enumerate(entity_names):
        for j, e2 in enumerate(entity_names):
            if i <= j:
                key = f"{e1}_vs_{e2}"
                sim_matrix[i, j] = separation[best_layer].get(key, 0)
                sim_matrix[j, i] = sim_matrix[i, j]
    
    sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                xticklabels=['User', 'Self', 'Other'],
                yticklabels=['User', 'Self', 'Other'],
                ax=ax3, vmin=0, vmax=1)
    ax3.set_title(f'Representation Similarity (Layer {best_layer})')
    
    # 4. Similarity across layers
    ax4 = axes[1, 1]
    for pair_name, color in [('user_vs_self', '#9b59b6'), 
                              ('user_vs_other', '#e67e22'),
                              ('self_vs_other', '#1abc9c')]:
        sims = [separation[l].get(pair_name, 0) for l in layers]
        label = pair_name.replace('_vs_', ' vs ').replace('_', ' ').title()
        ax4.plot(range(len(layers)), sims, marker='o', label=label, color=color)
    
    ax4.set_xticks(range(len(layers)))
    ax4.set_xticklabels([str(l) for l in layers])
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Entity Pair Similarity Across Layers')
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save
    fig_path = RESULTS_DIR / "experiment_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Project A experiment")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "generate", "extract", "probe", "analyze"],
                        help="Which phase to run")
    parser.add_argument("--n-dialogues", type=int, default=EXP_CFG.n_dialogues,
                        help="Number of dialogues to generate")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROJECT A: Self/Other/User Representation Separation")
    print("=" * 60)
    print(f"\nModel: {MODEL_CFG.model_name}")
    print(f"Dialogues: {args.n_dialogues}")
    print(f"Phase: {args.phase}")
    
    if args.phase == "all":
        # Run full pipeline
        dialogues = phase_generate(args)
        activations, labels, metadata = phase_extract(args, dialogues)
        pipeline, results = phase_probe(args, activations, labels)
        summary = phase_analyze(args, activations, labels)
        
    elif args.phase == "generate":
        phase_generate(args)
        
    elif args.phase == "extract":
        phase_extract(args)
        
    elif args.phase == "probe":
        phase_probe(args)
        
    elif args.phase == "analyze":
        phase_analyze(args)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

