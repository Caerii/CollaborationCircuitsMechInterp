"""
Step 47: Self/Other/User Representation Separation

Core MATS Project A: Test if the model forms distinct representations
for different entities in multi-party dialogue.
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_model():
    """Load model."""
    print("Loading Qwen3-4B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def generate_multi_party_dialogues(n=50):
    """Generate multi-party dialogues with clear entity roles."""
    dialogues = []
    
    user_names = ["User", "Human", "Alex"]
    self_names = ["Assistant", "AI", "Helper"]
    other_names = ["Agent B", "Bob", "Partner", "Collaborator"]
    
    topics = [
        ("sort algorithm", "bubble sort", "quicksort"),
        ("database design", "SQL", "NoSQL"),
        ("file format", "JSON", "XML"),
        ("framework", "React", "Vue"),
        ("language", "Python", "JavaScript"),
        ("architecture", "monolith", "microservices"),
        ("testing", "unit tests", "integration tests"),
        ("deployment", "Docker", "Kubernetes"),
    ]
    
    for i in range(n):
        user = user_names[i % len(user_names)]
        self_agent = self_names[i % len(self_names)]
        other = other_names[i % len(other_names)]
        topic, opt_a, opt_b = topics[i % len(topics)]
        
        # Dialogue with clear turns
        dialogue = f"""<|im_start|>system
You are {self_agent}, collaborating with {other} to help {user}.
<|im_end|>
<|im_start|>{user}
I need help choosing between {opt_a} and {opt_b} for {topic}.
<|im_end|>
<|im_start|>{self_agent}
I think {opt_a} is good for beginners. What does {other} think?
<|im_end|>
<|im_start|>{other}
I prefer {opt_b} for large-scale projects.
<|im_end|>
<|im_start|>{self_agent}
Good point. {user}, it depends on your use case."""
        
        dialogues.append({
            "dialogue": dialogue,
            "user": user,
            "self": self_agent,
            "other": other,
            "topic": topic
        })
    
    return dialogues


def extract_entity_activations(model, tokenizer, dialogues, layers_to_probe=[8, 16, 24, 32]):
    """Extract activations at entity mention positions."""
    activations = {layer: {"user": [], "self": [], "other": []} for layer in layers_to_probe}
    
    print(f"\nExtracting activations from {len(dialogues)} dialogues...")
    
    for i, d in enumerate(dialogues):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(dialogues)}")
        
        inputs = tokenizer(d["dialogue"], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden) per layer
        
        # Find positions of entity mentions
        tokens = tokenizer.tokenize(d["dialogue"])
        token_strs = [tokenizer.convert_tokens_to_string([t]).strip().lower() for t in tokens]
        
        user_lower = d["user"].lower()
        self_lower = d["self"].lower()
        other_lower = d["other"].lower()
        
        # Find last occurrence of each entity (most relevant)
        user_pos = None
        self_pos = None
        other_pos = None
        
        for pos, t in enumerate(token_strs):
            if user_lower in t:
                user_pos = pos
            if self_lower in t:
                self_pos = pos
            if other_lower in t:
                other_pos = pos
        
        # Extract activations at entity positions
        for layer in layers_to_probe:
            if layer < len(hidden_states):
                h = hidden_states[layer][0]  # (seq, hidden)
                
                if user_pos is not None and user_pos < h.shape[0]:
                    activations[layer]["user"].append(h[user_pos].cpu().float().numpy())
                if self_pos is not None and self_pos < h.shape[0]:
                    activations[layer]["self"].append(h[self_pos].cpu().float().numpy())
                if other_pos is not None and other_pos < h.shape[0]:
                    activations[layer]["other"].append(h[other_pos].cpu().float().numpy())
    
    return activations


def train_entity_probes(activations):
    """Train linear probes to classify entity type."""
    results = {}
    
    print("\nTraining entity probes...")
    
    for layer, layer_acts in activations.items():
        # Combine into X, y
        X = []
        y = []
        
        for entity_type, entity_idx in [("user", 0), ("self", 1), ("other", 2)]:
            acts = layer_acts[entity_type]
            if acts:
                X.extend(acts)
                y.extend([entity_idx] * len(acts))
        
        if len(X) < 10:
            results[layer] = {"accuracy": 0.0, "n_samples": len(X)}
            continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Train probe with cross-validation
        probe = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(probe, X, y, cv=min(5, len(X) // 3))
        
        results[layer] = {
            "accuracy": float(np.mean(scores) * 100),
            "std": float(np.std(scores) * 100),
            "n_samples": len(X),
            "n_user": len(layer_acts["user"]),
            "n_self": len(layer_acts["self"]),
            "n_other": len(layer_acts["other"])
        }
        
        print(f"  Layer {layer}: {results[layer]['accuracy']:.1f}% (+/- {results[layer]['std']:.1f}%)")
    
    return results


def compute_representation_similarity(activations):
    """Compute cosine similarity between entity representations."""
    similarities = {}
    
    print("\nComputing representation similarities...")
    
    for layer, layer_acts in activations.items():
        # Get mean representations
        means = {}
        for entity in ["user", "self", "other"]:
            if layer_acts[entity]:
                means[entity] = np.mean(layer_acts[entity], axis=0)
        
        if len(means) < 3:
            continue
        
        # Compute pairwise cosine similarities
        def cosine_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        
        similarities[layer] = {
            "user_self": cosine_sim(means["user"], means["self"]),
            "user_other": cosine_sim(means["user"], means["other"]),
            "self_other": cosine_sim(means["self"], means["other"])
        }
        
        print(f"  Layer {layer}:")
        print(f"    User-Self:  {similarities[layer]['user_self']:.3f}")
        print(f"    User-Other: {similarities[layer]['user_other']:.3f}")
        print(f"    Self-Other: {similarities[layer]['self_other']:.3f}")
    
    return similarities


def main():
    print("="*70)
    print("STEP 47: Self/Other/User Representation Separation (MATS Project A)")
    print("="*70)
    
    model, tokenizer = load_model()
    
    # Generate dialogues
    print("\nGenerating multi-party dialogues...")
    dialogues = generate_multi_party_dialogues(n=50)
    print(f"  Generated {len(dialogues)} dialogues")
    
    # Extract activations
    layers = [4, 8, 12, 16, 20, 24, 28, 32, 35]
    activations = extract_entity_activations(model, tokenizer, dialogues, layers_to_probe=layers)
    
    # Train probes
    probe_results = train_entity_probes(activations)
    
    # Compute similarities
    similarity_results = compute_representation_similarity(activations)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Entity Representation Analysis")
    print("="*70)
    
    print("\nProbe Accuracy by Layer (classifying User/Self/Other):")
    print("+-------+----------+--------+")
    print("| Layer | Accuracy | N      |")
    print("+-------+----------+--------+")
    
    best_layer = None
    best_acc = 0
    
    for layer in sorted(probe_results.keys()):
        r = probe_results[layer]
        print(f"| {layer:5} | {r['accuracy']:6.1f}% | {r['n_samples']:6} |")
        if r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_layer = layer
    
    print("+-------+----------+--------+")
    print(f"\nBest layer: {best_layer} with {best_acc:.1f}% accuracy")
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    
    chance = 33.3  # 3-way classification
    if best_acc > 80:
        print(f"\n[STRONG SEPARATION] Probe accuracy {best_acc:.1f}% >> chance ({chance:.1f}%)")
        print("The model forms DISTINCT representations for User/Self/Other!")
    elif best_acc > 50:
        print(f"\n[MODERATE SEPARATION] Probe accuracy {best_acc:.1f}% > chance ({chance:.1f}%)")
        print("Some separation exists but representations overlap.")
    else:
        print(f"\n[WEAK/NO SEPARATION] Probe accuracy {best_acc:.1f}% ~ chance ({chance:.1f}%)")
        print("Representations may be entangled.")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_dialogues": len(dialogues),
        "probe_results": probe_results,
        "similarity_results": similarity_results,
        "best_layer": best_layer,
        "best_accuracy": best_acc
    }
    
    output_path = RESULTS_DIR / "step47_self_other_user_probing.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


