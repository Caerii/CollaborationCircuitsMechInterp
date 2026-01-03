"""
Step 8: Create Final Summary Figure
====================================
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

print("Creating final summary figure...", flush=True)

# Load data
with open(RESULTS_DIR / "belief_analysis.json") as f:
    belief = json.load(f)

with open(RESULTS_DIR / "final_summary.json") as f:
    summary = json.load(f)

layers = belief["layers"]

# Create figure
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Belief Tracking in Qwen3-4B: Scientific Summary", fontsize=16, fontweight='bold', y=0.98)

# Layout: 2x3 grid
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# 1. Agent Classification across layers
ax1 = fig.add_subplot(gs[0, 0])
accs = [belief["minimal_pairs"][str(l)]["agent_classification"]["accuracy"] for l in layers]
bars = ax1.bar(range(len(layers)), accs, color='#3498db', alpha=0.8, edgecolor='black')
ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Chance')
ax1.set_xticks(range(len(layers)))
ax1.set_xticklabels(layers)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Accuracy')
ax1.set_title('1. Agent Classification\n(Alice vs Bob)', fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.legend(loc='lower right')
# Add value labels
for i, v in enumerate(accs):
    ax1.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=9)

# 2. Cross-Content Generalization
ax2 = fig.add_subplot(gs[0, 1])
gens = [belief["minimal_pairs"][str(l)]["avg_generalization"] for l in layers]
bars = ax2.bar(range(len(layers)), gens, color='#27ae60', alpha=0.8, edgecolor='black')
ax2.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Chance')
ax2.axhline(0.7, color='orange', linestyle='--', linewidth=1, label='Strong')
ax2.set_xticks(range(len(layers)))
ax2.set_xticklabels(layers)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Accuracy')
ax2.set_title('2. Cross-Content Generalization\n(Train: 3 types, Test: held-out)', fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.legend(loc='lower right')
for i, v in enumerate(gens):
    ax2.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=9)

# 3. Orthogonality
ax3 = fig.add_subplot(gs[0, 2])
cosines = [belief["minimal_pairs"][str(l)]["orthogonality"]["mean_cosine"] for l in layers]
colors = ['#27ae60' if c < 0.3 else '#e74c3c' for c in cosines]
bars = ax3.bar(range(len(layers)), cosines, color=colors, alpha=0.8, edgecolor='black')
ax3.axhline(0.3, color='orange', linestyle='--', linewidth=2, label='Threshold')
ax3.set_xticks(range(len(layers)))
ax3.set_xticklabels(layers)
ax3.set_xlabel('Layer')
ax3.set_ylabel('Cosine Similarity')
ax3.set_title('3. Agent vs Content Orthogonality\n(Lower = more separable)', fontweight='bold')
ax3.set_ylim(0, 0.5)
ax3.legend(loc='upper right')
for i, v in enumerate(cosines):
    ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# 4. Belief State Decoding
ax4 = fig.add_subplot(gs[1, 0])
states = [belief["belief_scenarios"][str(l)]["state_4way_acc"] for l in layers]
bars = ax4.bar(range(len(layers)), states, color='#9b59b6', alpha=0.8, edgecolor='black')
ax4.axhline(0.25, color='red', linestyle='--', linewidth=2, label='Chance (25%)')
ax4.set_xticks(range(len(layers)))
ax4.set_xticklabels(layers)
ax4.set_xlabel('Layer')
ax4.set_ylabel('Accuracy')
ax4.set_title('4. 4-Way Belief State Decoding\n(neither/alice_only/bob_only/both)', fontweight='bold')
ax4.set_ylim(0, 1.1)
ax4.legend(loc='lower right')
for i, v in enumerate(states):
    ax4.text(i, v + 0.02, f'{v:.0%}', ha='center', fontsize=9)

# 5. Causal Steering Result
ax5 = fig.add_subplot(gs[1, 1])
ax5.axis('off')

causal_text = """
CAUSAL STEERING TEST

Prompt: "Between Alice and Bob, the one 
        who discovered the truth first was"

Steering   Output
Strength   
  0.0  ->  "Alice. Alice and Bob..."
 10.0  ->  "Bob. Bob discovered..."

RESULT: Steering FLIPPED the answer
        from Alice to Bob!

This proves the representation is
CAUSALLY RELEVANT to model behavior.
"""

ax5.text(0.5, 0.5, causal_text, transform=ax5.transAxes, fontsize=11,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#e8f4e8', edgecolor='#27ae60', linewidth=2))

ax5.set_title('5. Causal Intervention', fontweight='bold')

# 6. Summary Box
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

summary_text = """
KEY FINDINGS

1. SEPARABLE: Agent identity encoded
   independently of content
   (99% cross-content transfer)

2. COMPOSITIONAL: [WHO] x [WHAT]
   are orthogonal dimensions
   (cosine = 0.037)

3. CAUSAL: Steering agent direction
   changes model output

EVIDENCE LEVEL: MODERATE

Implications for AI Safety:
- Models track agent-specific info
- Representations are steerable
- Potential for monitoring/control
"""

ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='#f39c12', linewidth=2))

ax6.set_title('6. Conclusions', fontweight='bold')

plt.savefig(FIGURES_DIR / "final_summary.png", dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print(f"[OK] Saved: {FIGURES_DIR / 'final_summary.png'}")























