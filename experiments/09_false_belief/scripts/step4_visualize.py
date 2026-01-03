"""
Step 4: Visualize False Belief Results
=======================================
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Creating visualization...", flush=True)

# Load results
with open(RESULTS_DIR / "behavioral_results.json") as f:
    behavioral = json.load(f)

with open(RESULTS_DIR / "representation_analysis.json") as f:
    representation = json.load(f)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("False Belief Test: Theory of Mind in Qwen3-4B", fontsize=14, fontweight='bold')

# 1. Behavioral results
ax = axes[0, 0]
categories = ['False Belief\nBelief Q', 'False Belief\nReality Q', 'True Belief\nBelief Q', 'True Belief\nReality Q']
fb = behavioral['summary']['false_belief']
tb = behavioral['summary']['true_belief']
values = [
    fb['belief_correct'] / fb['total'] if fb['total'] > 0 else 0,
    fb['reality_correct'] / fb['total'] if fb['total'] > 0 else 0,
    tb['belief_correct'] / tb['total'] if tb['total'] > 0 else 0,
    tb['reality_correct'] / tb['total'] if tb['total'] > 0 else 0,
]
colors = ['#27ae60', '#e74c3c', '#3498db', '#9b59b6']
bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(0.5, color='gray', linestyle='--', label='50% threshold')
ax.set_ylabel('Accuracy')
ax.set_title('1. Behavioral Test Results')
ax.set_ylim(0, 1.1)
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=10, fontweight='bold')

# Add annotation for key finding
ax.annotate('Model perfectly tracks\nagent BELIEF!', xy=(0, values[0]), xytext=(0.5, 0.6),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='green'),
            bbox=dict(boxstyle='round', facecolor='#e8f8e8'))

# 2. Location decoding across layers
ax = axes[0, 1]
layers = representation['layers']
belief_acc = [representation['analysis'][str(l)]['belief_decode_acc'] for l in layers]
reality_acc = [representation['analysis'][str(l)]['reality_decode_acc'] for l in layers]
cross_acc = [representation['analysis'][str(l)]['cross_decode_acc'] for l in layers]

x = np.arange(len(layers))
width = 0.25
ax.bar(x - width, belief_acc, width, label='Belief Location', color='#27ae60', alpha=0.8)
ax.bar(x, reality_acc, width, label='Reality Location', color='#3498db', alpha=0.8)
ax.bar(x + width, cross_acc, width, label='Cross-decode', color='#f39c12', alpha=0.8)
ax.axhline(0.1, color='red', linestyle='--', label='Chance (10%)')
ax.set_xlabel('Layer')
ax.set_ylabel('Accuracy')
ax.set_title('2. Location Decoding from Activations')
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend(loc='lower right', fontsize=8)
ax.set_ylim(0, 1.1)

# 3. Belief-Reality cosine similarity
ax = axes[1, 0]
cosines = [representation['analysis'][str(l)]['belief_reality_cosine'] for l in layers]
colors = ['#27ae60' if c < 0.3 else '#f39c12' if c < 0.7 else '#e74c3c' for c in cosines]
bars = ax.bar(layers, cosines, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Orthogonal threshold')
ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Aligned threshold')
ax.set_xlabel('Layer')
ax.set_ylabel('Cosine Similarity')
ax.set_title('3. Belief vs Reality Direction Similarity')
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(0, 1.1)
for i, (l, c) in enumerate(zip(layers, cosines)):
    ax.text(i, c + 0.03, f'{c:.2f}', ha='center', fontsize=9)

# Add annotation
ax.annotate('Early: SEPARATED\nLate: MERGED', xy=(3, 0.95), xytext=(1.5, 0.8),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='red'),
            bbox=dict(boxstyle='round', facecolor='#ffe8e8'))

# 4. Summary interpretation
ax = axes[1, 1]
ax.axis('off')

summary_text = """
KEY FINDINGS

1. BEHAVIORAL: Model tracks agent BELIEF 
   perfectly (100%) but struggles with
   REALITY (45% on false belief scenarios)

2. EARLY LAYERS: Belief and reality are
   encoded in ORTHOGONAL directions
   (cosine ~ 0.04)

3. LATE LAYERS: Representations CONVERGE
   to same direction (cosine ~ 0.95)

INTERPRETATION
--------------
The model initially separates "what Alice
thinks" from "what is true", but by the
output layers, it MERGES them - adopting
the agent's perspective as "truth".

This may explain why LLMs can be
persuaded to adopt false beliefs -
they conflate belief and reality
in late processing stages.

MATS RELEVANCE
--------------
- Evidence for ToM circuits in early layers
- Potential target for interventions
- Safety implications: belief-reality conflation
"""

ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray', linewidth=2))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "false_belief_summary.png", dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"[OK] Saved: {FIGURES_DIR / 'false_belief_summary.png'}")
























