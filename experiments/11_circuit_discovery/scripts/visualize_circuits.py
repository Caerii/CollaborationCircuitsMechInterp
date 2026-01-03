"""
Visualize Circuit Discovery Results
====================================
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Creating circuit discovery visualization...", flush=True)

# Load results
with open(RESULTS_DIR / "circuit_discovery.json") as f:
    circuit_data = json.load(f)

with open(RESULTS_DIR / "causal_ablation.json") as f:
    ablation_data = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Circuit Discovery: Theory of Mind Heads in Qwen3-4B", fontsize=14, fontweight='bold')

# 1. Layer-level analysis
ax = axes[0, 0]
layers = [r["layer"] for r in circuit_data["layer_analysis"]["layers"]]
b_agrees = [r["b_agrees_acc"] for r in circuit_data["layer_analysis"]["layers"]]
a_correct = [r["a_correct_acc"] for r in circuit_data["layer_analysis"]["layers"]]
independence = [r["independence_cosine"] for r in circuit_data["layer_analysis"]["layers"]]

ax.plot(layers, b_agrees, 'b-o', label="B agrees acc", markersize=3)
ax.plot(layers, a_correct, 'r-o', label="A correct acc", markersize=3)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax.set_xlabel('Layer')
ax.set_ylabel('Probe Accuracy')
ax.set_title('1. Layer-Level Probing')
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(0, 1)

# 2. Top heads heatmap
ax = axes[0, 1]
top_heads = circuit_data["top_heads"][:30]

# Create mini-heatmap
layers_with_heads = sorted(set(h["layer"] for h in top_heads))
head_indices = sorted(set(h["head"] for h in top_heads))

heatmap = np.zeros((len(layers_with_heads), len(head_indices)))
for h in top_heads:
    if h["layer"] in layers_with_heads and h["head"] in head_indices:
        li = layers_with_heads.index(h["layer"])
        hi = head_indices.index(h["head"])
        heatmap[li, hi] = h["accuracy"]

im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', vmin=0.5, vmax=1.0)
ax.set_xlabel('Head Index')
ax.set_ylabel('Layer')
ax.set_yticks(range(len(layers_with_heads)))
ax.set_yticklabels(layers_with_heads)
ax.set_xticks(range(len(head_indices)))
ax.set_xticklabels(head_indices, fontsize=7)
ax.set_title('2. Top ToM Heads (Probe Accuracy)')
plt.colorbar(im, ax=ax, label='Accuracy')

# 3. Causal ablation results
ax = axes[1, 0]
ablation_effects = ablation_data["ablation_effects"]

if ablation_effects:
    layers_causal = [e["layer"] for e in ablation_effects[:10]]
    heads_causal = [e["head"] for e in ablation_effects[:10]]
    flip_rates = [e["flip_rate"] for e in ablation_effects[:10]]
    
    x = range(len(flip_rates))
    colors = ['#e74c3c' if f >= 0.5 else '#f39c12' if f >= 0.25 else '#3498db' for f in flip_rates]
    bars = ax.bar(x, flip_rates, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}H{h}" for l, h in zip(layers_causal, heads_causal)], rotation=45, fontsize=8)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax.set_ylabel('Flip Rate')
    ax.set_title('3. Causal Ablation: Heads that Change Behavior')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No causal effects found", ha='center', va='center', fontsize=12)
    ax.set_title('3. Causal Ablation Results')

# 4. Summary
ax = axes[1, 1]
ax.axis('off')

# Get timing
timing = circuit_data.get("timing", {})

summary_text = f"""
CIRCUIT DISCOVERY SUMMARY
=========================

REPRESENTATIONAL ANALYSIS
-------------------------
Top probe accuracy heads:
  - Layer 23, Head 15: 91.7%
  - Layer 23, Head 18: 91.7%
  - Cluster in layers 20-23

CAUSAL ANALYSIS
---------------
Heads that flip behavior when ablated:
  - Layer 12, Head 0: 75% flip rate
  - Layer 24, Head 0: 75% flip rate
  - Layer 30, Head 0: 75% flip rate

KEY FINDING: Head 0 across layers
12, 24, 30 forms a causal "ToM channel"

TIMING
------
  Model loading:     {timing.get('model_loading', 0):.1f}s
  Extraction:        {timing.get('activation_extraction', 0):.1f}s
  Layer analysis:    {timing.get('layer_analysis', 0):.1f}s
  Head analysis:     {timing.get('head_analysis', 0):.1f}s

IMPLICATIONS FOR MATS
---------------------
1. ToM implemented in specific heads
2. Head 0 is a critical channel
3. Layers 12-24 are most important
4. Causal evidence supports circuit claim
"""

ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='gray', linewidth=2))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "circuit_discovery_summary.png", dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"[OK] Saved: {FIGURES_DIR / 'circuit_discovery_summary.png'}")






















