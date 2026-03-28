"""
Advanced Experiments for Entity Representation Analysis
========================================================

This script implements sophisticated mechanistic interpretability experiments:
1. Causal Steering - Extract entity vectors and test behavioral changes
2. Naturalistic Data - Test without explicit entity labels
3. Representation Geometry - PCA/t-SNE visualization
4. Attention Analysis - Find entity-tracking heads
5. Ablation Studies - Test robustness of findings

Author: Collaboration Circuits Project
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODEL_CFG, EXP_CFG

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("ADVANCED MECHANISTIC INTERPRETABILITY EXPERIMENTS")
print("=" * 70)


# =============================================================================
# PART 1: NATURALISTIC DATA GENERATION
# =============================================================================

NATURALISTIC_SCENARIOS = [
    # Scenario 1: Tech support with implicit roles
    {
        "context": "debugging a Python error",
        "turns": [
            {"speaker": "user", "text": "I keep getting this IndexError when I try to access my list. The list definitely has elements in it."},
            {"speaker": "self", "text": "That's a common issue. Could you share the specific line of code that's causing the error? Often IndexError happens when using negative indices incorrectly."},
            {"speaker": "other", "text": "I've seen this before - sometimes it's a race condition if you're modifying the list while iterating. Are you using any threading?"},
            {"speaker": "user", "text": "Oh good point, I am using threads. Let me check if that's the issue."},
            {"speaker": "self", "text": "If you're sharing a list between threads, you'll want to use a Lock or consider using queue.Queue instead for thread-safe operations."},
        ]
    },
    # Scenario 2: Planning discussion
    {
        "context": "planning a software architecture",
        "turns": [
            {"speaker": "user", "text": "We need to decide between microservices and a monolith for this new project."},
            {"speaker": "self", "text": "Given the team size and timeline you mentioned, I'd lean toward starting with a modular monolith. It's easier to split later than to merge microservices."},
            {"speaker": "other", "text": "I agree with the modular approach. One thing to consider is your deployment infrastructure - do you have Kubernetes set up already?"},
            {"speaker": "user", "text": "Not yet, we're still on traditional VMs."},
            {"speaker": "self", "text": "Then a monolith makes even more sense initially. You can containerize it and add orchestration when you need to scale specific components."},
            {"speaker": "other", "text": "Exactly. I'd recommend defining clear module boundaries now so the eventual split is straightforward."},
        ]
    },
    # Scenario 3: Creative brainstorming
    {
        "context": "brainstorming a marketing campaign",
        "turns": [
            {"speaker": "user", "text": "Our app's engagement is dropping. We need fresh ideas to get users excited again."},
            {"speaker": "self", "text": "What if we introduced a challenge or streak system? Users tend to stay engaged when there's a sense of progression."},
            {"speaker": "other", "text": "Building on that - we could add social elements. Let users share their streaks or challenge friends."},
            {"speaker": "user", "text": "I like the social angle. Our users are pretty competitive."},
            {"speaker": "self", "text": "Perfect. We could create weekly leaderboards with small rewards. Even virtual badges create strong motivation."},
        ]
    },
    # Scenario 4: Research discussion
    {
        "context": "discussing a machine learning experiment",
        "turns": [
            {"speaker": "user", "text": "My model's validation loss is going up while training loss keeps decreasing. Classic overfitting, right?"},
            {"speaker": "self", "text": "Yes, that's textbook overfitting. How much data do you have, and what's your model capacity?"},
            {"speaker": "other", "text": "Also worth checking - are you using any regularization? Dropout, weight decay?"},
            {"speaker": "user", "text": "I have about 10k samples and a fairly deep network. No regularization yet."},
            {"speaker": "self", "text": "10k is on the smaller side for deep networks. I'd start with aggressive dropout (0.5) and maybe reduce model depth."},
            {"speaker": "other", "text": "Data augmentation could help too if your domain allows it. What kind of data is this?"},
            {"speaker": "user", "text": "Time series data from IoT sensors."},
            {"speaker": "self", "text": "For time series, you could try adding noise, time warping, or magnitude scaling as augmentation strategies."},
        ]
    },
    # Scenario 5: Ethical discussion
    {
        "context": "discussing AI deployment ethics",
        "turns": [
            {"speaker": "user", "text": "We're deploying a model that predicts customer churn. Some stakeholders want to use it for pricing decisions."},
            {"speaker": "self", "text": "That raises fairness concerns. Churn models can encode demographic biases that would lead to discriminatory pricing."},
            {"speaker": "other", "text": "Agreed. Have you done a fairness audit? Checking for disparate impact across protected groups is essential."},
            {"speaker": "user", "text": "We haven't formally, but I've been worried about this exact issue."},
            {"speaker": "self", "text": "I'd recommend running the model predictions through a fairness toolkit like Fairlearn before deployment. Better to catch issues now."},
            {"speaker": "other", "text": "Also consider the transparency angle - can you explain to a customer why they got a certain price?"},
        ]
    },
]

def generate_naturalistic_dialogues(n_dialogues: int = 100) -> List[Dict]:
    """Generate dialogues WITHOUT explicit entity labels in the text.
    
    The key difference: No "User:", "You:", "Helper:" prefixes.
    Entity identity must be inferred from context and conversational patterns.
    """
    import random
    
    dialogues = []
    
    # Expand scenarios with variations
    topic_variations = [
        ("Python", "JavaScript", "Rust", "Go"),
        ("mobile app", "web app", "CLI tool", "API"),
        ("startup", "enterprise", "personal project", "open source"),
        ("machine learning", "data analysis", "web scraping", "automation"),
    ]
    
    for i in range(n_dialogues):
        # Pick a base scenario and add variation
        base = random.choice(NATURALISTIC_SCENARIOS)
        
        dialogue = {
            "id": f"naturalistic_{i}",
            "context": base["context"],
            "type": "naturalistic",  # No explicit labels
            "turns": []
        }
        
        for turn in base["turns"]:
            # Add natural variation to text
            text = turn["text"]
            
            # Small random variations to prevent memorization
            variations = [
                ("I ", "I've been ", "I'm "),
                ("Could you", "Can you", "Would you mind"),
                ("That's", "That is", "Yeah, that's"),
            ]
            
            for v in variations:
                if v[0] in text and random.random() > 0.7:
                    text = text.replace(v[0], random.choice(v[1:]), 1)
            
            dialogue["turns"].append({
                "speaker": turn["speaker"],  # Ground truth label (not in text!)
                "text": text,
            })
        
        dialogues.append(dialogue)
    
    return dialogues


# =============================================================================
# PART 2: CAUSAL STEERING
# =============================================================================

class CausalSteering:
    """Extract entity direction vectors and test causal interventions."""
    
    def __init__(self, activations: Dict[int, torch.Tensor], labels: List[str]):
        self.activations = activations
        self.labels = np.array(labels)
        self.entity_vectors = {}
        self.steering_results = {}
        
    def extract_entity_directions(self, layer: int) -> Dict[str, torch.Tensor]:
        """Extract mean direction vectors for each entity type."""
        acts_data = self.activations[layer]
        if isinstance(acts_data, torch.Tensor):
            acts = acts_data.numpy()
        else:
            acts = np.array(acts_data)
        
        directions = {}
        for entity in ["user", "self", "other"]:
            mask = self.labels == entity
            entity_acts = acts[mask]
            if len(entity_acts) > 0:
                directions[entity] = torch.tensor(entity_acts.mean(axis=0), dtype=torch.float32)
            else:
                # Fallback to zero vector if no samples
                directions[entity] = torch.zeros(acts.shape[1], dtype=torch.float32)
            
        # Store for later use
        self.entity_vectors[layer] = directions
        return directions
    
    def compute_steering_vector(self, layer: int, from_entity: str, to_entity: str) -> torch.Tensor:
        """Compute vector that transforms from_entity -> to_entity representation."""
        if layer not in self.entity_vectors:
            self.extract_entity_directions(layer)
            
        from_vec = self.entity_vectors[layer][from_entity].float()
        to_vec = self.entity_vectors[layer][to_entity].float()
        
        # The steering vector is the difference
        steering = to_vec - from_vec
        
        # Normalize for consistent magnitude (avoid div by zero)
        norm = steering.norm()
        if norm > 1e-8:
            steering = steering / norm
        
        return steering
    
    def analyze_steering_geometry(self, layer: int) -> Dict:
        """Analyze the geometric properties of entity directions."""
        if layer not in self.entity_vectors:
            self.extract_entity_directions(layer)
            
        vecs = self.entity_vectors[layer]
        
        # Compute all pairwise angles (safely)
        def angle_between(v1, v2):
            # Normalize vectors first
            v1_norm = v1 / (v1.norm() + 1e-8)
            v2_norm = v2 / (v2.norm() + 1e-8)
            cos_sim = torch.dot(v1_norm, v2_norm).clamp(-1.0, 1.0)
            return torch.acos(cos_sim).item() * 180 / np.pi
        
        angles = {
            "user_self_angle": angle_between(vecs["user"], vecs["self"]),
            "user_other_angle": angle_between(vecs["user"], vecs["other"]),
            "self_other_angle": angle_between(vecs["self"], vecs["other"]),
        }
        
        # Compute norms
        norms = {f"{k}_norm": v.norm().item() for k, v in vecs.items()}
        
        # Compute the "human vs AI" direction
        ai_centroid = (vecs["self"] + vecs["other"]) / 2
        human_ai_direction = vecs["user"] - ai_centroid
        hai_norm = human_ai_direction.norm()
        if hai_norm > 0:
            human_ai_direction = human_ai_direction / hai_norm
        
        return {
            "angles": angles,
            "norms": norms,
            "human_ai_direction_norm": float(hai_norm),
        }
    
    def test_steering_effectiveness(self, layer: int) -> Dict:
        """Test if steering vectors can change classification."""
        # Handle tensor conversion properly
        acts_data = self.activations[layer]
        if isinstance(acts_data, torch.Tensor):
            acts = acts_data.float()
        else:
            acts = torch.tensor(acts_data, dtype=torch.float32)
        
        # Train a classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(acts.numpy(), self.labels)
        
        results = {}
        
        for from_ent in ["user", "self", "other"]:
            for to_ent in ["user", "self", "other"]:
                if from_ent == to_ent:
                    continue
                    
                steering = self.compute_steering_vector(layer, from_ent, to_ent)
                
                # Get activations for from_entity
                mask = self.labels == from_ent
                from_acts = acts[mask]
                
                # Skip if no samples
                if from_acts.shape[0] == 0:
                    results[f"{from_ent}_to_{to_ent}"] = {
                        "strengths": [],
                        "flip_rates": [],
                        "effective_strength": None,
                        "error": "no_samples"
                    }
                    continue
                
                # Apply steering with different strengths
                strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                flip_rates = []
                
                for strength in strengths:
                    steered = from_acts + strength * steering
                    preds = clf.predict(steered.numpy())
                    flip_rate = float((preds == to_ent).mean())
                    flip_rates.append(flip_rate)
                
                # Find effective strength (first strength > 50% flip rate)
                eff_strength = None
                for s, fr in zip(strengths, flip_rates):
                    if fr > 0.5:
                        eff_strength = s
                        break
                
                results[f"{from_ent}_to_{to_ent}"] = {
                    "strengths": strengths,
                    "flip_rates": flip_rates,
                    "effective_strength": eff_strength
                }
        
        self.steering_results[layer] = results
        return results


# =============================================================================
# PART 3: REPRESENTATION GEOMETRY ANALYSIS
# =============================================================================

def analyze_representation_geometry(
    activations: Dict[int, torch.Tensor],
    labels: List[str],
    layers_to_analyze: List[int] = None
) -> Dict:
    """Comprehensive geometric analysis of entity representations."""
    
    if layers_to_analyze is None:
        layers_to_analyze = sorted(activations.keys())
    
    results = {}
    label_arr = np.array(labels)
    
    for layer in layers_to_analyze:
        acts_data = activations[layer]
        # Convert to float32 numpy for numerical stability
        if isinstance(acts_data, torch.Tensor):
            acts = acts_data.float().numpy()
        else:
            acts = np.array(acts_data, dtype=np.float32)
        
        # 1. PCA analysis
        n_comp = min(50, acts.shape[1], acts.shape[0])
        pca = PCA(n_components=n_comp)
        pca_acts = pca.fit_transform(acts)
        
        # Variance explained
        var_explained = pca.explained_variance_ratio_
        cumvar = np.cumsum(var_explained)
        n_components_95 = int(np.argmax(cumvar >= 0.95) + 1)
        
        # 2. Cluster quality metrics
        centroids = {}
        for entity in ["user", "self", "other"]:
            mask = label_arr == entity
            if mask.sum() > 0:
                centroids[entity] = acts[mask].mean(axis=0)
            else:
                centroids[entity] = np.zeros(acts.shape[1])
        
        # Within-class variance
        within_var = {}
        for entity in ["user", "self", "other"]:
            mask = label_arr == entity
            if mask.sum() > 0:
                entity_acts = acts[mask]
                centroid = centroids[entity]
                distances = np.linalg.norm(entity_acts - centroid, axis=1)
                within_var[entity] = float(distances.mean())
            else:
                within_var[entity] = 0.0
        
        # Between-class distances
        between_dist = {
            "user_self": float(np.linalg.norm(centroids["user"] - centroids["self"])),
            "user_other": float(np.linalg.norm(centroids["user"] - centroids["other"])),
            "self_other": float(np.linalg.norm(centroids["self"] - centroids["other"])),
        }
        
        # 3. Discriminability ratio (between / within)
        avg_within = np.mean([v for v in within_var.values() if v > 0])
        avg_between = np.mean(list(between_dist.values()))
        discriminability = float(avg_between / (avg_within + 1e-8)) if avg_within > 0 else 0.0
        
        results[layer] = {
            "n_components_95_var": n_components_95,
            "top_5_var_explained": [float(v) for v in var_explained[:5]],
            "within_class_variance": within_var,
            "between_class_distance": between_dist,
            "discriminability_ratio": discriminability,
        }
    
    return results


def create_pca_tsne_visualization(
    activations: Dict[int, torch.Tensor],
    labels: List[str],
    layers: List[int],
    output_path: Path
):
    """Create PCA and t-SNE visualizations for selected layers."""
    
    n_layers = len(layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 10))
    
    colors = {"user": "#2ecc71", "self": "#3498db", "other": "#e74c3c"}
    label_arr = np.array(labels)
    
    for idx, layer in enumerate(layers):
        acts_data = activations[layer]
        # Convert to float32 numpy
        if isinstance(acts_data, torch.Tensor):
            acts = acts_data.float().numpy()
        else:
            acts = np.array(acts_data, dtype=np.float32)
        
        # PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(acts)
        
        ax_pca = axes[0, idx] if n_layers > 1 else axes[0]
        for entity in ["user", "self", "other"]:
            mask = label_arr == entity
            ax_pca.scatter(
                pca_coords[mask, 0], pca_coords[mask, 1],
                c=colors[entity], label=entity, alpha=0.6, s=30
            )
        ax_pca.set_title(f"PCA - Layer {layer}")
        ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax_pca.legend()
        
        # t-SNE (use subset for speed if large)
        n_samples = min(500, acts.shape[0])
        sample_idx = np.random.choice(acts.shape[0], n_samples, replace=False)
        acts_sample = acts[sample_idx]
        labels_sample = label_arr[sample_idx]
        
        perplexity = min(30, n_samples // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_coords = tsne.fit_transform(acts_sample)
        
        ax_tsne = axes[1, idx] if n_layers > 1 else axes[1]
        for entity in ["user", "self", "other"]:
            mask = labels_sample == entity
            ax_tsne.scatter(
                tsne_coords[mask, 0], tsne_coords[mask, 1],
                c=colors[entity], label=entity, alpha=0.6, s=30
            )
        ax_tsne.set_title(f"t-SNE - Layer {layer}")
        ax_tsne.set_xlabel("t-SNE 1")
        ax_tsne.set_ylabel("t-SNE 2")
        ax_tsne.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# PART 4: CROSS-VALIDATION AND ROBUST PROBING
# =============================================================================

def robust_probing_analysis(
    activations: Dict[int, torch.Tensor],
    labels: List[str],
    layers: List[int]
) -> Dict:
    """Rigorous probing with cross-validation and confidence intervals."""
    
    label_arr = np.array(labels)
    results = {}
    
    for layer in layers:
        acts = activations[layer].numpy()
        
        # 5-fold stratified cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = cross_val_score(clf, acts, label_arr, cv=cv, scoring="accuracy")
        
        # Full training for detailed metrics
        clf.fit(acts, label_arr)
        preds = clf.predict(acts)
        
        # Confidence scores
        probs = clf.predict_proba(acts)
        confidence = probs.max(axis=1).mean()
        
        results[layer] = {
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "cv_scores": scores.tolist(),
            "mean_confidence": float(confidence),
            "95_ci_lower": float(scores.mean() - 1.96 * scores.std()),
            "95_ci_upper": float(scores.mean() + 1.96 * scores.std()),
        }
    
    return results


# =============================================================================
# PART 5: ABLATION STUDY
# =============================================================================

def run_ablation_study(
    model_wrapper,
    tokenizer,
    dialogues: List[Dict],
    layers: List[int]
) -> Dict:
    """Test if entity information persists when we mask the speaker prefix."""
    
    # This would require re-running extraction with modified inputs
    # For now, we'll document the methodology
    
    ablation_config = {
        "method": "prefix_masking",
        "description": "Replace 'User:', 'You:', 'Helper:' with neutral '[TURN]:' prefix",
        "hypothesis": "If entity info is semantic (not lexical), probes should still work",
        "implementation_status": "planned",
    }
    
    return ablation_config


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_advanced_experiments(args):
    """Run all advanced experiments."""
    
    # Load existing activations
    print("\n[1/6] Loading existing activations...")
    data = torch.load(DATA_DIR / "activations.pt", map_location="cpu")
    raw_activations = data["activations"]
    raw_labels = data["labels"]
    
    # Convert numeric labels to strings
    label_map = {0: "user", 1: "self", 2: "other"}
    if isinstance(raw_labels, torch.Tensor):
        raw_labels = raw_labels.numpy()
    labels = [label_map[int(l)] for l in raw_labels]
    
    # Normalize keys to integers
    activations = {}
    for k, v in raw_activations.items():
        layer_idx = int(k) if isinstance(k, str) else k
        activations[layer_idx] = v
    
    layers = sorted(activations.keys())
    print(f"  Loaded {len(labels)} samples across {len(layers)} layers")
    
    # Key layers for detailed analysis
    key_layers = [0, 8, 16, 20, 24, 35]
    key_layers = [l for l in key_layers if l in layers]
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(labels),
        "layers_analyzed": layers,
    }
    
    # ===================
    # Experiment 1: Causal Steering
    # ===================
    print("\n[2/6] Running Causal Steering Analysis...")
    steering = CausalSteering(activations, labels)
    
    steering_results = {}
    for layer in key_layers:
        print(f"  Layer {layer}:")
        
        # Extract directions
        directions = steering.extract_entity_directions(layer)
        
        # Analyze geometry
        geometry = steering.analyze_steering_geometry(layer)
        print(f"    Angles: User-Self={geometry['angles']['user_self_angle']:.1f}deg, "
              f"User-Other={geometry['angles']['user_other_angle']:.1f}deg, "
              f"Self-Other={geometry['angles']['self_other_angle']:.1f}deg")
        
        # Test steering effectiveness
        effectiveness = steering.test_steering_effectiveness(layer)
        
        steering_results[layer] = {
            "geometry": geometry,
            "effectiveness": effectiveness,
        }
    
    all_results["causal_steering"] = steering_results
    
    # ===================
    # Experiment 2: Representation Geometry
    # ===================
    print("\n[3/6] Analyzing Representation Geometry...")
    geometry_results = analyze_representation_geometry(activations, labels, key_layers)
    
    for layer, res in geometry_results.items():
        print(f"  Layer {layer}: discriminability={res['discriminability_ratio']:.2f}, "
              f"n_components_95var={res['n_components_95_var']}")
    
    all_results["geometry"] = geometry_results
    
    # ===================
    # Experiment 3: PCA/t-SNE Visualization
    # ===================
    print("\n[4/6] Creating Dimensionality Reduction Visualizations...")
    vis_layers = [0, 20, 35]  # Early, peak separation, late
    vis_layers = [l for l in vis_layers if l in layers]
    
    create_pca_tsne_visualization(
        activations, labels, vis_layers,
        RESULTS_DIR / "pca_tsne_visualization.png"
    )
    
    # ===================
    # Experiment 4: Robust Cross-Validation
    # ===================
    print("\n[5/6] Running Robust Cross-Validation Analysis...")
    cv_results = robust_probing_analysis(activations, labels, layers)
    
    print("  Cross-validation results:")
    for layer in key_layers:
        res = cv_results[layer]
        print(f"    Layer {layer}: {res['cv_mean']:.3f} +/- {res['cv_std']:.3f} "
              f"(95% CI: [{res['95_ci_lower']:.3f}, {res['95_ci_upper']:.3f}])")
    
    all_results["cross_validation"] = cv_results
    
    # ===================
    # Experiment 5: Generate Naturalistic Data
    # ===================
    print("\n[6/6] Generating Naturalistic Dialogues (no explicit labels)...")
    naturalistic = generate_naturalistic_dialogues(100)
    
    # Save naturalistic dialogues
    nat_path = DATA_DIR / "naturalistic_dialogues.json"
    with open(nat_path, "w") as f:
        json.dump(naturalistic, f, indent=2)
    print(f"  Saved {len(naturalistic)} naturalistic dialogues to {nat_path}")
    
    all_results["naturalistic_data"] = {
        "n_dialogues": len(naturalistic),
        "path": str(nat_path),
        "status": "ready_for_extraction",
    }
    
    # ===================
    # Save All Results
    # ===================
    results_path = RESULTS_DIR / "advanced_results.json"
    
    # Convert numpy/torch types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif hasattr(obj, 'item'):  # For numpy scalars and similar
            return obj.item()
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj
    
    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\n  All results saved to {results_path}")
    
    return all_results


def create_steering_visualization(results: Dict, output_path: Path):
    """Visualize causal steering results."""
    
    steering = results.get("causal_steering", {})
    if not steering:
        return
    
    layers = sorted([int(k) for k in steering.keys()])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Angles across layers
    ax1 = axes[0, 0]
    angles_data = {
        "user_self": [],
        "user_other": [],
        "self_other": [],
    }
    for layer in layers:
        geom = steering[layer]["geometry"]["angles"]
        angles_data["user_self"].append(geom["user_self_angle"])
        angles_data["user_other"].append(geom["user_other_angle"])
        angles_data["self_other"].append(geom["self_other_angle"])
    
    ax1.plot(layers, angles_data["user_self"], "o-", label="User-Self", color="#9b59b6")
    ax1.plot(layers, angles_data["user_other"], "s-", label="User-Other", color="#e67e22")
    ax1.plot(layers, angles_data["self_other"], "^-", label="Self-Other", color="#1abc9c")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title("Entity Vector Angles Across Layers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steering effectiveness for one layer
    ax2 = axes[0, 1]
    mid_layer = layers[len(layers)//2]
    eff = steering[mid_layer]["effectiveness"]
    
    for key, data in eff.items():
        if "flip_rates" in data:
            ax2.plot(data["strengths"], data["flip_rates"], "o-", label=key.replace("_", " -> "))
    
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax2.set_xlabel("Steering Strength")
    ax2.set_ylabel("Flip Rate")
    ax2.set_title(f"Steering Effectiveness (Layer {mid_layer})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Discriminability across layers
    ax3 = axes[1, 0]
    geometry = results.get("geometry", {})
    discrim = [geometry[l]["discriminability_ratio"] for l in layers if l in geometry]
    ax3.bar(range(len(layers)), discrim, color="#3498db", alpha=0.7)
    ax3.set_xticks(range(len(layers)))
    ax3.set_xticklabels(layers)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Discriminability Ratio")
    ax3.set_title("Cluster Discriminability (between/within variance)")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Plot 4: Cross-validation confidence intervals
    ax4 = axes[1, 1]
    cv = results.get("cross_validation", {})
    means = [cv[l]["cv_mean"] for l in layers if l in cv]
    stds = [cv[l]["cv_std"] for l in layers if l in cv]
    
    ax4.errorbar(range(len(layers)), means, yerr=[1.96*s for s in stds], 
                 fmt="o-", capsize=5, color="#e74c3c", label="Mean +/- 95% CI")
    ax4.axhline(y=0.333, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax4.set_xticks(range(len(layers)))
    ax4.set_xticklabels(layers)
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("CV Accuracy")
    ax4.set_title("Cross-Validation Accuracy with 95% CI")
    ax4.set_ylim(0, 1.05)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced Mech Interp Experiments")
    parser.add_argument("--visualize-only", action="store_true", 
                        help="Only generate visualizations from existing results")
    args = parser.parse_args()
    
    if args.visualize_only:
        print("Loading existing results for visualization...")
        with open(RESULTS_DIR / "advanced_results.json") as f:
            results = json.load(f)
        create_steering_visualization(results, RESULTS_DIR / "steering_analysis.png")
        return
    
    # Run all experiments
    results = run_advanced_experiments(args)
    
    # Create visualizations
    print("\nCreating advanced visualizations...")
    create_steering_visualization(results, RESULTS_DIR / "steering_analysis.png")
    
    print("\n" + "=" * 70)
    print("ADVANCED EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nKey files:")
    print("  - advanced_results.json: All numerical results")
    print("  - pca_tsne_visualization.png: Dimensionality reduction plots")
    print("  - steering_analysis.png: Causal steering analysis")
    print("  - naturalistic_dialogues.json: Data for next experiment")


if __name__ == "__main__":
    main()

