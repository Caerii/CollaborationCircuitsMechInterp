"""
Step 52: Visualize THE REAL TRUTH

Create a clear visualization showing the model uses heuristics, not ToM.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("THE REAL TRUTH: Qwen3-4B Uses Heuristics, Not ToM", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Left: The critical test results
    ax1 = axes[0]
    
    scenarios = [
        'Alice LEFT\n(didn\'t see)',
        'Alice STAYED\n(saw move)',
        'Alice TOLD\n(informed)',
        'Alice EXPLICITLY\ntold'
    ]
    
    # Model's confidence in drawer (first-mention)
    drawer_conf = [91.9, 95.7, 58.5, 66.2]
    
    # What ToM correct answer is
    tom_correct = ['drawer', 'basket', 'basket', 'basket']
    colors = ['#27ae60', '#e74c3c', '#e74c3c', '#e74c3c']  # Green if drawer is correct
    
    bars = ax1.bar(scenarios, drawer_conf, color=colors, edgecolor='black', linewidth=2)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50%')
    ax1.set_ylabel('Model\'s Confidence in "drawer" (%)', fontsize=12)
    ax1.set_title('Model Always Predicts First Location\n(Ignores what Alice knows!)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add annotations
    for bar, conf, correct in zip(bars, drawer_conf, tom_correct):
        if correct == 'drawer':
            label = f'{conf:.0f}%\n(CORRECT)'
            color = 'green'
        else:
            label = f'{conf:.0f}%\n(WRONG!)'
            color = 'red'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color=color)
    
    ax1.legend(loc='upper right')
    
    # Right: What TRUE ToM would predict
    ax2 = axes[1]
    
    categories = ['Standard\nSally-Anne', 'Agent SAW\nthe move', 'Agent was\nTOLD']
    
    # What model predicts (green if matches ToM)
    model_correct = [100, 0, 0]  # Only correct on standard test
    
    x = np.arange(len(categories))
    
    bars2 = ax2.bar(x, model_correct, color=['#27ae60', '#e74c3c', '#e74c3c'],
                   edgecolor='black', linewidth=2)
    
    ax2.set_ylabel('ToM Accuracy (%)', fontsize=12)
    ax2.set_title('Model\'s True ToM Ability\n(Discriminating Test Suite)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 120)
    ax2.axhline(y=33.3, color='orange', linestyle='--', alpha=0.7, label='Chance (33%)')
    
    # Value labels
    for bar, val in zip(bars2, model_correct):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add explanation box
    ax2.annotate('Standard Sally-Anne\npasses BY ACCIDENT\n(first-mention = belief)',
                xy=(0, 100), xytext=(0.5, 70),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    ax2.annotate('TRUE ToM tests FAIL\n(model ignores\nwhat agent knows)',
                xy=(1.5, 0), xytext=(1.5, 50),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='#ffcccc', edgecolor='red'))
    
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save
    save_path = FIGURES_DIR / "09_the_real_truth.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Create summary figure
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    summary = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                        THE REAL TRUTH                                     ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  FINDING: Qwen3-4B does NOT have Theory of Mind                          ║
    ║                                                                           ║
    ║  ┌─────────────────────────────────────────────────────────────────┐     ║
    ║  │  EVIDENCE:                                                       │     ║
    ║  │                                                                  │     ║
    ║  │  • When Alice LEFT (didn't see move):                           │     ║
    ║  │    Model: "drawer" (91.9%) ← CORRECT (by accident)              │     ║
    ║  │                                                                  │     ║
    ║  │  • When Alice STAYED (saw the move):                            │     ║
    ║  │    Model: "drawer" (95.7%) ← WRONG! Should be "basket"          │     ║
    ║  │                                                                  │     ║
    ║  │  • When Alice was TOLD:                                         │     ║
    ║  │    Model: "drawer" (58.5%) ← WRONG! Should be "basket"          │     ║
    ║  └─────────────────────────────────────────────────────────────────┘     ║
    ║                                                                           ║
    ║  CONCLUSION:                                                              ║
    ║  • Model uses FIRST-MENTION heuristic, not ToM                           ║
    ║  • Standard Sally-Anne passes BY ACCIDENT                                ║
    ║  • Model ignores whether agent SAW or was TOLD                           ║
    ║                                                                           ║
    ║  IMPLICATIONS:                                                            ║
    ║  • Earlier "ToM accuracy" metrics were meaningless                       ║
    ║  • "Inhibitory circuit" findings were irrelevant                         ║
    ║  • Need DISCRIMINATING tests for true ToM                                ║
    ║                                                                           ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    
    save_path2 = FIGURES_DIR / "10_truth_summary.png"
    plt.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path2}")
    plt.close()


if __name__ == "__main__":
    main()


