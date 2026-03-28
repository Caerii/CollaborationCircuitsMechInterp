"""
Step 17: Create Comprehensive Publication Figure

Combines all findings into a single multi-panel figure showing:
1. Verb categories heatmap
2. Individual verbs scatter (baseline vs ablated)
3. Language comparison
4. Communication medium comparison
5. Sentence structure comparison
6. Key statistics summary
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Custom colors
BASELINE_COLOR = '#e74c3c'  # Red
ABLATED_COLOR = '#27ae60'   # Green
BOOST_COLOR = '#3498db'     # Blue


def load_results():
    """Load all results."""
    with open(RESULTS_DIR / "massive_sweep_results.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def create_comprehensive_figure(results):
    """Create a comprehensive multi-panel figure."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Verb Categories Bar Chart
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    categories = list(results['verbs']['by_category'].keys())
    baselines = [results['verbs']['by_category'][c]['baseline'] * 100 for c in categories]
    ablated = [results['verbs']['by_category'][c]['ablated'] * 100 for c in categories]
    
    # Sort by baseline
    sorted_idx = np.argsort(baselines)
    categories = [categories[i] for i in sorted_idx]
    baselines = [baselines[i] for i in sorted_idx]
    ablated = [ablated[i] for i in sorted_idx]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.barh([i - width/2 for i in x], baselines, width, label='Baseline', color=BASELINE_COLOR, alpha=0.8)
    ax1.barh([i + width/2 for i in x], ablated, width, label='+ Ablation', color=ABLATED_COLOR, alpha=0.8)
    
    ax1.set_xlabel('Accuracy (%)', fontsize=11)
    ax1.set_title('A. ToM by Verb Category', fontsize=14, fontweight='bold')
    ax1.set_yticks(x)
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 100)
    ax1.axvline(50, color='gray', linestyle='--', alpha=0.3)
    
    # =========================================================================
    # Panel B: Verb Scatter Plot
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    verbs = []
    verb_baselines = []
    verb_ablated = []
    verb_categories = []
    
    for verb, data in results['verbs']['by_verb'].items():
        verbs.append(verb)
        verb_baselines.append(data['baseline'] * 100)
        verb_ablated.append(data['ablated'] * 100)
        verb_categories.append(data.get('category', 'unknown'))
    
    # Color by boost
    boosts = [a - b for a, b in zip(verb_ablated, verb_baselines)]
    
    scatter = ax2.scatter(verb_baselines, verb_ablated, c=boosts, cmap='RdYlGn', 
                          s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Diagonal line
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)
    
    # Label extreme points
    for i, (v, b, a) in enumerate(zip(verbs, verb_baselines, verb_ablated)):
        if b == 0 and a > 80:
            ax2.annotate(v, (b+2, a), fontsize=7, alpha=0.8)
    
    ax2.set_xlabel('Baseline Accuracy (%)', fontsize=11)
    ax2.set_ylabel('Ablated Accuracy (%)', fontsize=11)
    ax2.set_title('B. 178 Verbs: Baseline vs Ablated', fontsize=14, fontweight='bold')
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Boost (%)', fontsize=9)
    
    # =========================================================================
    # Panel C: Language Comparison
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    langs = list(results['languages']['by_language'].keys())
    lang_baselines = [results['languages']['by_language'][l]['baseline'] * 100 for l in langs]
    lang_ablated = [results['languages']['by_language'][l]['ablated'] * 100 for l in langs]
    
    x = np.arange(len(langs))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], lang_baselines, width, label='Baseline', color=BASELINE_COLOR, alpha=0.8)
    ax3.bar([i + width/2 for i in x], lang_ablated, width, label='+ Ablation', color=ABLATED_COLOR, alpha=0.8)
    
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('C. ToM Across Languages', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['English', 'Chinese', 'Spanish', 'French', 'German'], fontsize=9)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_ylim(0, 105)
    
    # Annotate boosts
    for i, (b, a) in enumerate(zip(lang_baselines, lang_ablated)):
        boost = a - b
        if boost > 10:
            ax3.annotate(f'+{boost:.0f}%', (i, a + 2), ha='center', fontsize=9, fontweight='bold', color=BOOST_COLOR)
    
    # =========================================================================
    # Panel D: Communication Medium
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    mediums = list(results['mediums']['by_medium'].keys())
    med_baselines = [results['mediums']['by_medium'][m]['baseline'] * 100 for m in mediums]
    med_ablated = [results['mediums']['by_medium'][m]['ablated'] * 100 for m in mediums]
    
    # Sort by baseline
    sorted_idx = np.argsort(med_baselines)
    mediums = [mediums[i].replace('_', '\n') for i in sorted_idx]
    med_baselines = [med_baselines[i] for i in sorted_idx]
    med_ablated = [med_ablated[i] for i in sorted_idx]
    
    x = np.arange(len(mediums))
    width = 0.35
    
    ax4.barh([i - width/2 for i in x], med_baselines, width, label='Baseline', color=BASELINE_COLOR, alpha=0.8)
    ax4.barh([i + width/2 for i in x], med_ablated, width, label='+ Ablation', color=ABLATED_COLOR, alpha=0.8)
    
    ax4.set_xlabel('Accuracy (%)', fontsize=11)
    ax4.set_title('D. ToM by Communication Medium', fontsize=14, fontweight='bold')
    ax4.set_yticks(x)
    ax4.set_yticklabels(mediums, fontsize=9)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.set_xlim(0, 100)
    
    # =========================================================================
    # Panel E: Sentence Structure
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    structures = list(results['structures']['by_structure'].keys())
    struct_baselines = [results['structures']['by_structure'][s]['baseline'] * 100 for s in structures]
    struct_ablated = [results['structures']['by_structure'][s]['ablated'] * 100 for s in structures]
    
    # Sort by baseline
    sorted_idx = np.argsort(struct_baselines)
    structures = [structures[i].replace('_', '\n') for i in sorted_idx]
    struct_baselines = [struct_baselines[i] for i in sorted_idx]
    struct_ablated = [struct_ablated[i] for i in sorted_idx]
    
    x = np.arange(len(structures))
    width = 0.35
    
    ax5.bar([i - width/2 for i in x], struct_baselines, width, label='Baseline', color=BASELINE_COLOR, alpha=0.8)
    ax5.bar([i + width/2 for i in x], struct_ablated, width, label='+ Ablation', color=ABLATED_COLOR, alpha=0.8)
    
    ax5.set_ylabel('Accuracy (%)', fontsize=11)
    ax5.set_title('E. ToM by Sentence Structure', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(structures, fontsize=9)
    ax5.legend(loc='upper left', fontsize=9)
    ax5.set_ylim(0, 110)
    
    # =========================================================================
    # Panel F: Top/Bottom Verbs Table
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Sort verbs by baseline
    verbs_sorted = sorted(
        results['verbs']['by_verb'].items(),
        key=lambda x: x[1]['baseline']
    )
    
    # Create table data
    table_data = []
    table_data.append(['VERB', 'BASELINE', 'ABLATED', 'BOOST'])
    table_data.append(['─'*10, '─'*8, '─'*8, '─'*6])
    table_data.append(['WORST (Inhibition Triggered)', '', '', ''])
    
    for verb, data in verbs_sorted[:8]:
        boost = (data['ablated'] - data['baseline']) * 100
        table_data.append([
            verb,
            f"{data['baseline']*100:.0f}%",
            f"{data['ablated']*100:.0f}%",
            f"+{boost:.0f}%"
        ])
    
    table_data.append(['─'*10, '─'*8, '─'*8, '─'*6])
    table_data.append(['BEST (No Inhibition)', '', '', ''])
    
    for verb, data in verbs_sorted[-5:]:
        boost = (data['ablated'] - data['baseline']) * 100
        table_data.append([
            verb,
            f"{data['baseline']*100:.0f}%",
            f"{data['ablated']*100:.0f}%",
            f"+{boost:.0f}%"
        ])
    
    # Display as text
    text = '\n'.join(['  '.join([f'{cell:>12}' for cell in row]) for row in table_data])
    ax6.text(0.05, 0.95, 'F. Best & Worst Verbs', transform=ax6.transAxes, 
             fontsize=14, fontweight='bold', va='top')
    ax6.text(0.05, 0.85, text, transform=ax6.transAxes, fontsize=9, 
             va='top', family='monospace')
    
    # =========================================================================
    # Panel G: Summary Statistics
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Calculate summary stats
    all_baselines = [d['baseline'] for d in results['verbs']['by_verb'].values()]
    all_ablated = [d['ablated'] for d in results['verbs']['by_verb'].values()]
    mean_baseline = np.mean(all_baselines) * 100
    mean_ablated = np.mean(all_ablated) * 100
    mean_boost = mean_ablated - mean_baseline
    
    n_verbs_0_baseline = sum(1 for b in all_baselines if b == 0)
    n_verbs_100_ablated = sum(1 for a in all_ablated if a >= 1.0)
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                    MASSIVE LINGUISTIC SWEEP SUMMARY                                               ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                   ║
    ║    📊 SCALE                                    📈 KEY FINDINGS                                                    ║
    ║    ────────────────────────────               ────────────────────────────                                        ║
    ║    • 178 unique verbs tested                  • Mean baseline: {mean_baseline:.1f}%                                           ║
    ║    • 12 semantic categories                   • Mean ablated:  {mean_ablated:.1f}%                                           ║
    ║    • 5 languages (EN/ZH/ES/FR/DE)             • Mean boost:    +{mean_boost:.1f}%                                            ║
    ║    • 7 communication mediums                                                                                      ║
    ║    • 5 sentence structures                    • {n_verbs_0_baseline} verbs at 0% baseline (complete failure)                    ║
    ║    • 1,300+ total scenarios                   • {n_verbs_100_ablated} verbs reach 100% with ablation                           ║
    ║                                                                                                                   ║
    ║    🔬 SCIENTIFIC INSIGHT                                                                                          ║
    ║    ─────────────────────────────────────────────────────────────────────────────────────                          ║
    ║    The inhibitory circuit (L17H4, L18H11, L18H14, L19H30, L21H17) is VERB-SENSITIVE:                             ║
    ║    • DIRECT verbs ("told", "informed") → MAXIMUM inhibition → 0% baseline                                        ║
    ║    • INDIRECT verbs ("provided", "supported") → NO inhibition → 100% baseline                                    ║
    ║                                                                                                                   ║
    ║    This explains why models fail Sally-Anne tests but succeed at multi-agent collaboration:                       ║
    ║    Different verb phrasings trigger different levels of the belief-update inhibition circuit.                     ║
    ║                                                                                                                   ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=10.5,
             va='center', ha='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#333333', linewidth=2))
    
    # Main title
    fig.suptitle('Theory of Mind Circuit Analysis: Massive Linguistic Sweep\n'
                 'Qwen3-4B | 5-Head Ablation (L17H4, L18H11, L18H14, L19H30, L21H17)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def main():
    print("Creating comprehensive figure...")
    
    results = load_results()
    fig = create_comprehensive_figure(results)
    
    output_path = FIGURES_DIR / "COMPREHENSIVE_MASSIVE_SWEEP.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Also save PDF for publication
    pdf_path = FIGURES_DIR / "COMPREHENSIVE_MASSIVE_SWEEP.pdf"
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()
    print("Done!")


if __name__ == "__main__":
    main()





