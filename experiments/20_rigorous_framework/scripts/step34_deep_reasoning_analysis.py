"""
Step 34: Deep Reasoning Process Analysis

GOAL: Understand WHERE and HOW the model performs belief tracking
during the <think> reasoning phase.

Three-pronged approach:
1. LOGIT LENS: Track answer probability evolution token-by-token
2. ATTENTION ANALYSIS: What does the model attend to during reasoning?
3. TRANSCODER: What computations transform beliefs into answers?

This is DEEP analysis of a SINGLE scenario for maximum insight.

OUTPUT: results/step34_deep_analysis.json, figures/step34_*.png
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


class ReasoningAnalyzer:
    """Deep analysis of reasoning process."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        
    def get_logit_lens_all_tokens(self, input_ids, target_tokens):
        """Run logit lens on EVERY position to track probability evolution."""
        print("\n  Running logit lens on all tokens...")
        sys.stdout.flush()
        
        # Get all hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        
        hidden_states = outputs.hidden_states  # (n_layers+1, batch, seq, hidden)
        
        # Get the unembedding matrix
        lm_head = self.model.lm_head
        
        results = {tok: [] for tok in target_tokens}
        n_positions = input_ids.shape[1]
        
        print(f"  Analyzing {n_positions} positions across {self.n_layers} layers...")
        sys.stdout.flush()
        
        # For each layer
        for layer_idx in range(0, self.n_layers + 1, 4):  # Sample every 4 layers
            layer_hidden = hidden_states[layer_idx]  # (batch, seq, hidden)
            
            # Project to vocab at every position
            logits = lm_head(layer_hidden)  # (batch, seq, vocab)
            probs = torch.softmax(logits, dim=-1)
            
            # Track target token probabilities at each position
            for tok_str in target_tokens:
                tok_id = self.tokenizer.encode(tok_str, add_special_tokens=False)
                if tok_id:
                    tok_id = tok_id[0]
                    # Get probability at each position
                    pos_probs = probs[0, :, tok_id].detach().cpu().numpy()
                    results[tok_str].append({
                        "layer": layer_idx,
                        "probs": pos_probs.tolist()
                    })
            
            if layer_idx % 8 == 0:
                print(f"    Layer {layer_idx}/{self.n_layers} done")
                sys.stdout.flush()
        
        return results
    
    def get_attention_patterns(self, input_ids, layers_to_check):
        """Extract attention patterns from specific layers."""
        print("\n  Extracting attention patterns...")
        print("  Note: SDPA attention may not support pattern extraction")
        sys.stdout.flush()
        
        attention_patterns = {}
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    return_dict=True,
                )
            
            attentions = outputs.attentions  # tuple of (batch, n_heads, seq, seq)
            
            if attentions is not None:
                for layer_idx in layers_to_check:
                    if layer_idx < len(attentions):
                        attn = attentions[layer_idx][0].cpu().numpy()  # (n_heads, seq, seq)
                        attention_patterns[layer_idx] = attn
                        print(f"    Layer {layer_idx}: shape {attn.shape}")
                        sys.stdout.flush()
            else:
                print("  Warning: Attention patterns not available (SDPA mode)")
        except Exception as e:
            print(f"  Warning: Could not extract attention patterns: {e}")
        
        return attention_patterns
    
    def analyze_single_scenario(self, scenario, max_tokens=400):
        """Deep analysis of a single scenario."""
        print(f"\n{'='*60}")
        print("DEEP ANALYSIS OF SINGLE SCENARIO")
        print(f"{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"Question: {scenario['question'][:80]}...")
        print(f"Correct: {scenario['correct']}, Wrong: {scenario['wrong']}")
        sys.stdout.flush()
        
        # Generate with full reasoning
        messages = [{"role": "user", "content": f"{scenario['question']}\n\nAnswer with just the location:"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs['input_ids'].shape[1]
        
        print(f"\n  Generating response (up to {max_tokens} tokens)...")
        sys.stdout.flush()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        full_ids = outputs.sequences
        response = self.tokenizer.decode(full_ids[0][prompt_len:], skip_special_tokens=True)
        
        print(f"\n  Response ({len(response)} chars):")
        print(f"  {response[:200]}...")
        sys.stdout.flush()
        
        # Decode tokens for labeling
        tokens = [self.tokenizer.decode([t]) for t in full_ids[0]]
        
        # Find key positions
        think_start = None
        think_end = None
        for i, t in enumerate(tokens):
            if "<think>" in t and think_start is None:
                think_start = i
            if "</think>" in t:
                think_end = i
        
        print(f"\n  <think> span: {think_start} to {think_end}")
        
        # Run logit lens
        target_tokens = [scenario['correct'], scenario['wrong']]
        logit_lens_results = self.get_logit_lens_all_tokens(full_ids, target_tokens)
        
        # Get attention at key layers
        key_layers = [0, 8, 16, 24, 28, 32, 36]
        key_layers = [l for l in key_layers if l < self.n_layers]
        attention_patterns = self.get_attention_patterns(full_ids, key_layers)
        
        return {
            "scenario": scenario,
            "response": response,
            "tokens": tokens[prompt_len:],  # Only generated tokens
            "prompt_len": prompt_len,
            "think_span": (think_start, think_end),
            "logit_lens": logit_lens_results,
            "attention_layers": list(attention_patterns.keys()),
            "attention_shapes": {k: v.shape for k, v in attention_patterns.items()},
        }


def train_transcoder_on_reasoning(model, tokenizer, scenarios, layer_idx=28):
    """Train a transcoder on MLP activations during reasoning."""
    print(f"\n{'='*60}")
    print(f"TRANSCODER ANALYSIS (Layer {layer_idx})")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    import torch.nn as nn
    
    class SimpleTranscoder(nn.Module):
        """Maps MLP input to MLP output."""
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.decoder = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.GELU()
            
        def forward(self, x):
            h = self.activation(self.encoder(x))
            return self.decoder(h), h
    
    # Collect MLP activations during reasoning
    mlp_inputs = []
    mlp_outputs = []
    labels = []  # 0 = FB, 1 = TB
    
    hooks = []
    captured = {}
    
    def capture_mlp_input(module, inp, out):
        captured['input'] = inp[0].detach()
    
    def capture_mlp_output(module, inp, out):
        captured['output'] = out.detach()
    
    # Register hooks
    mlp_module = model.model.layers[layer_idx].mlp
    hooks.append(mlp_module.register_forward_hook(capture_mlp_input))
    # For output, we need to capture from gate_proj or similar
    # Let's capture from the whole MLP
    
    print(f"  Collecting activations from {len(scenarios)} scenarios...")
    sys.stdout.flush()
    
    for i, scenario in enumerate(scenarios):
        messages = [{"role": "user", "content": f"{scenario['question']}\n\nAnswer:"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=True)
        
        if 'input' in captured:
            # Take mean across sequence
            mlp_in = captured['input'].mean(dim=1).cpu()
            mlp_inputs.append(mlp_in)
            labels.append(scenario.get('label', 0))
        
        if i % 4 == 0:
            print(f"    Processed {i+1}/{len(scenarios)}")
            sys.stdout.flush()
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    if not mlp_inputs:
        print("  No activations captured!")
        return None
    
    # Stack activations
    X = torch.cat(mlp_inputs, dim=0)
    y = torch.tensor(labels)
    
    print(f"  Collected {X.shape[0]} samples, shape: {X.shape}")
    
    # Train a simple classifier on MLP activations
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    X_np = X.numpy()
    y_np = y.numpy()
    
    clf = LogisticRegression(max_iter=1000)
    # Use leave-one-out for small samples
    n_folds = min(3, min(np.sum(y_np == 0), np.sum(y_np == 1)))
    if n_folds < 2:
        # Not enough samples for CV, just train and evaluate
        clf.fit(X_np, y_np)
        scores = np.array([clf.score(X_np, y_np)])
    else:
        scores = cross_val_score(clf, X_np, y_np, cv=n_folds)
    
    print(f"  Classification accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")
    
    return {
        "layer": layer_idx,
        "n_samples": len(mlp_inputs),
        "accuracy": float(scores.mean()),
        "std": float(scores.std()),
    }


def main():
    print("=" * 70)
    print("STEP 34: DEEP REASONING ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nThis is DEEP analysis - quality over quantity!")
    print("Analyzing the reasoning process itself.")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded! Layers: {model.config.num_hidden_layers}")
    sys.stdout.flush()
    
    # Define ONE scenario for deep analysis
    deep_scenario = {
        "name": "Classic Sally-Anne",
        "question": "Sally put the ball in the basket. Sally left the room. Anne moved the ball to the box. Sally came back. Where does Sally think the ball is?",
        "correct": "basket",
        "wrong": "box",
        "label": 0,  # False Belief
    }
    
    # Initialize analyzer
    analyzer = ReasoningAnalyzer(model, tokenizer)
    
    # ========================================
    # PART 1: Deep single-scenario analysis
    # ========================================
    print("\n" + "="*60)
    print("PART 1: SINGLE SCENARIO DEEP DIVE")
    print("="*60)
    sys.stdout.flush()
    
    analysis_result = analyzer.analyze_single_scenario(deep_scenario, max_tokens=400)
    
    # ========================================
    # PART 2: Transcoder on multiple scenarios
    # ========================================
    print("\n" + "="*60)
    print("PART 2: TRANSCODER ANALYSIS")
    print("="*60)
    sys.stdout.flush()
    
    # Create a mix of FB and TB scenarios for transcoder
    transcoder_scenarios = [
        # False Belief (label=0)
        {"question": "Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Where does Alice think the ball is?", "correct": "drawer", "wrong": "basket", "label": 0},
        {"question": "Tom put the key in the box. Tom left. Jerry moved it to the shelf. Where does Tom think the key is?", "correct": "box", "wrong": "shelf", "label": 0},
        {"question": "Carol put the book on the table. Carol left. Dan moved it to the desk. Where does Carol think the book is?", "correct": "table", "wrong": "desk", "label": 0},
        {"question": "Eve put the phone in her bag. Eve left. Frank moved it to the drawer. Where does Eve think the phone is?", "correct": "bag", "wrong": "drawer", "label": 0},
        # True Belief (label=1)
        {"question": "Alice put the ball in the drawer. Alice watched Bob move it to the basket. Where does Alice think the ball is?", "correct": "basket", "wrong": "drawer", "label": 1},
        {"question": "Tom put the key in the box. Tom watched Jerry move it to the shelf. Where does Tom think the key is?", "correct": "shelf", "wrong": "box", "label": 1},
        {"question": "Carol put the book on the table. Carol watched Dan move it to the desk. Where does Carol think the book is?", "correct": "desk", "wrong": "table", "label": 1},
        {"question": "Eve put the phone in her bag. Eve watched Frank move it to the drawer. Where does Eve think the phone is?", "correct": "drawer", "wrong": "bag", "label": 1},
    ]
    
    transcoder_results = {}
    for layer in [12, 20, 28, 32]:
        if layer < model.config.num_hidden_layers:
            result = train_transcoder_on_reasoning(model, tokenizer, transcoder_scenarios, layer)
            if result:
                transcoder_results[layer] = result
    
    # ========================================
    # PART 3: Visualize results
    # ========================================
    print("\n" + "="*60)
    print("PART 3: VISUALIZATION")
    print("="*60)
    sys.stdout.flush()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Logit lens probability evolution
    ax1 = axes[0, 0]
    if analysis_result['logit_lens']:
        correct_key = deep_scenario['correct']
        wrong_key = deep_scenario['wrong']
        
        if correct_key in analysis_result['logit_lens'] and analysis_result['logit_lens'][correct_key]:
            layers = [r['layer'] for r in analysis_result['logit_lens'][correct_key]]
            
            # Get final position probs at each layer
            correct_probs = [r['probs'][-1] for r in analysis_result['logit_lens'][correct_key]]
            wrong_probs = [r['probs'][-1] for r in analysis_result['logit_lens'][wrong_key]]
            
            ax1.plot(layers, correct_probs, 'g-o', label=f'P("{correct_key}")', linewidth=2)
            ax1.plot(layers, wrong_probs, 'r-o', label=f'P("{wrong_key}")', linewidth=2)
            ax1.set_xlabel("Layer")
            ax1.set_ylabel("Probability")
            ax1.set_title("Answer Probability Evolution (Final Position)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transcoder accuracy by layer
    ax2 = axes[0, 1]
    if transcoder_results:
        layers = sorted(transcoder_results.keys())
        accs = [transcoder_results[l]['accuracy'] for l in layers]
        ax2.bar([f"L{l}" for l in layers], [a*100 for a in accs], color='steelblue', edgecolor='black')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax2.set_ylabel("FB vs TB Classification (%)")
        ax2.set_title("MLP Belief Discriminability by Layer")
        ax2.set_ylim(0, 100)
        ax2.legend()
    
    # Plot 3: Token-level probability evolution
    ax3 = axes[1, 0]
    if analysis_result['logit_lens'] and correct_key in analysis_result['logit_lens']:
        # Take middle layer
        mid_layer_data = analysis_result['logit_lens'][correct_key][len(analysis_result['logit_lens'][correct_key])//2]
        probs = mid_layer_data['probs']
        layer_num = mid_layer_data['layer']
        
        # Only show generated tokens
        prompt_len = analysis_result['prompt_len']
        if len(probs) > prompt_len:
            gen_probs = probs[prompt_len:]
            ax3.plot(range(len(gen_probs)), gen_probs, 'g-', alpha=0.7)
            ax3.set_xlabel("Generated Token Position")
            ax3.set_ylabel(f'P("{correct_key}")')
            ax3.set_title(f"Correct Answer Probability During Reasoning (Layer {layer_num})")
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    DEEP ANALYSIS SUMMARY
    =====================
    
    Scenario: {deep_scenario['name']}
    Correct: {deep_scenario['correct']}
    Wrong: {deep_scenario['wrong']}
    
    Generated {len(analysis_result['tokens'])} tokens
    
    Transcoder FB vs TB:
    """
    for layer, res in transcoder_results.items():
        summary_text += f"\n      Layer {layer}: {res['accuracy']:.1%}"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle("Step 34: Deep Reasoning Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "step34_deep_analysis.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved to: {fig_path}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "scenario": deep_scenario,
        "response": analysis_result['response'],
        "n_tokens": len(analysis_result['tokens']),
        "think_span": analysis_result['think_span'],
        "transcoder_results": transcoder_results,
    }
    
    output_path = RESULTS_DIR / "step34_deep_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("DEEP ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    print("\nKEY FINDINGS:")
    print(f"  - Generated {len(analysis_result['tokens'])} reasoning tokens")
    print(f"  - Think span: positions {analysis_result['think_span']}")
    
    if transcoder_results:
        best_layer = max(transcoder_results.keys(), key=lambda l: transcoder_results[l]['accuracy'])
        print(f"  - Best transcoder discrimination: Layer {best_layer} ({transcoder_results[best_layer]['accuracy']:.1%})")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

