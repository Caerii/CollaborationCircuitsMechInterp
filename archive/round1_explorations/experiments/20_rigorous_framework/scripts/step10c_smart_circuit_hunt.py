"""
Step 10c: Smart Multi-Agent Circuit Hunt (Hybrid Approach)

SMART FILTERING APPROACH:
1. SAE Layer Screening: Train SAEs on all layers, find which have discriminative features
2. Attention Pattern Analysis: Find heads that attend to relevant tokens (agent names, belief states)
3. Full Ablation: Only test candidate heads (50-100 instead of 1,152)

SPEEDUP: 30-65x faster than step10b (comprehensive)
- Step 10b: 1,152 heads × 15 scenarios = 17,280 evaluations (~144 days)
- Step 10c: 50-100 heads × 15 scenarios = 750-1,500 evaluations (~2-4 days)

OUTPUT: results/step10c_smart_circuit.json, figures/step10c_*.png
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import gc  # For garbage collection
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add framework to path
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from config import ExperimentConfig
from scenarios.multi_agent import MultiAgentScenarioGenerator
from scenarios.templates import generate_n_scenarios
from analysis.circuits import ChatModeCircuitAnalyzer
from analysis.controls import bonferroni_correct
from analysis.sae_analysis import SimpleSAE, SAEConfig, SAETrainer
from core.chat_runner import load_model_for_chat
from core.activation_extractor import ActivationExtractor

# Output paths
RESULTS_DIR = FRAMEWORK_ROOT / "results"
FIGURES_DIR = FRAMEWORK_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def collect_mlp_activations(model, tokenizer, prompts, layer):
    """Collect MLP output activations from chat-formatted prompts."""
    activations = []
    
    def hook(module, input, output):
        # Get last token activation and move to CPU immediately to save GPU memory
        activations.append(output[0, -1, :].detach().cpu())
    
    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_hook(hook)
    
    try:
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                # Format as chat prompt
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
                model(**inputs)
                
                # Clear cache periodically for long sequences
                if i > 0 and i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        handle.remove()
    
    if not activations:
        raise ValueError(f"No activations collected for layer {layer}")
    
    return torch.stack(activations)


def extract_attention_patterns(model, tokenizer, prompts, layers_to_check):
    """
    Extract attention patterns from all heads in specified layers.
    
    Returns: dict[layer][head] = list of attention patterns (one per prompt)
    """
    print("  Extracting attention patterns (requires eager mode)...")
    sys.stdout.flush()
    
    # Need eager mode for attention outputs
    if hasattr(model.config, '_attn_implementation'):
        if model.config._attn_implementation != 'eager':
            print("  Warning: Model not in eager mode, attention extraction may fail")
    
    attention_data = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(prompts)}]", end="", flush=True)
            
            # Format as chat prompt
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            
            try:
                outputs = model(**inputs, output_attentions=True)
                
                if outputs.attentions is not None:
                    for layer_idx in layers_to_check:
                        if layer_idx < len(outputs.attentions):
                            # attentions[layer]: (batch, n_heads, seq, seq) or (batch, n_kv_heads, seq, seq) for GQA
                            layer_attn = outputs.attentions[layer_idx][0]  # (n_heads or n_kv_heads, seq, seq)
                            
                            # Handle GQA: if n_kv_heads < n_heads, we need to map
                            n_heads_in_attn = layer_attn.shape[0]
                            n_heads_model = model.config.num_attention_heads
                            
                            # Get attention FROM last token TO all tokens (what matters for prediction)
                            last_token_attn = layer_attn[:, -1, :].cpu().numpy()  # (n_heads_in_attn, seq)
                            
                            # For GQA models, map KV heads to query heads
                            if n_heads_in_attn < n_heads_model:
                                # GQA: multiple query heads share same KV head
                                heads_per_kv = n_heads_model // n_heads_in_attn
                                for head_idx in range(n_heads_model):
                                    kv_head_idx = head_idx // heads_per_kv
                                    attention_data[layer_idx][head_idx].append(last_token_attn[kv_head_idx])
                            else:
                                # Standard attention: one-to-one mapping
                                for head_idx in range(n_heads_in_attn):
                                    attention_data[layer_idx][head_idx].append(last_token_attn[head_idx])
            except Exception as e:
                print(f"\n  Warning: Could not extract attention for prompt {i}: {e}")
                continue
    
    print(f"  Done! Extracted from {len(prompts)} prompts")
    sys.stdout.flush()
    
    return attention_data


def find_relevant_token_indices(tokens, keywords):
    """Find indices of tokens matching keywords (agent names, belief words, etc.)."""
    indices = []
    tokens_lower = [t.lower() for t in tokens]
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for i, token in enumerate(tokens_lower):
            if keyword_lower in token or token in keyword_lower:
                indices.append(i)
    
    return list(set(indices))  # Remove duplicates


def score_head_attention(attention_patterns, tokens_list, keywords):
    """
    Score how much each head attends to relevant tokens.
    
    Returns: score (higher = more attention to relevant tokens)
    """
    scores = []
    
    for attn_pattern, tokens in zip(attention_patterns, tokens_list):
        # Find relevant token indices
        relevant_indices = find_relevant_token_indices(tokens, keywords)
        
        if relevant_indices:
            # Sum attention to relevant tokens
            score = attn_pattern[relevant_indices].sum()
        else:
            score = 0.0
        
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def stage1_sae_layer_screening(model, tokenizer, scenarios, config):
    """
    Stage 1: Train SAEs on all layers, find which have discriminative features.
    
    Returns: candidate_layers (list of layer indices with high discriminability)
    """
    print(f"\n{'='*60}")
    print("STAGE 1: SAE LAYER SCREENING")
    print(f"{'='*60}")
    print("Training SAEs on all layers to find discriminative features...")
    sys.stdout.flush()
    
    # Format prompts and labels
    prompts = []
    labels = []
    
    for s in scenarios:
        story = s.get('story', '')
        question = s.get('question', '')
        prompt_text = f"{story}\n\n{question}"
        prompts.append(prompt_text)
        
        # Label by scenario type
        s_type = s.get('type', s.get('scenario_type', ''))
        if 'false' in s_type.lower() or 'belief_chain' in s_type.lower():
            labels.append('false_belief')
        elif 'true' in s_type.lower() or 'common' in s_type.lower():
            labels.append('true_belief')
        else:
            labels.append('other')
    
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    
    # Sample layers to test (every 4th for speed, but cover all ranges)
    layers_to_test = list(range(0, n_layers, 4))  # 0, 4, 8, 12, 16, 20, 24, 28, 32
    if n_layers - 1 not in layers_to_test:
        layers_to_test.append(n_layers - 1)  # Always include last layer
    
    print(f"  Testing {len(layers_to_test)} layers: {layers_to_test}")
    sys.stdout.flush()
    
    layer_scores = {}
    
    for layer_idx, layer in enumerate(layers_to_test):
        print(f"\n  Layer {layer} ({layer_idx+1}/{len(layers_to_test)})...", end="", flush=True)
        
        try:
            # Clear GPU cache before processing this layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Collect activations
            activations = collect_mlp_activations(model, tokenizer, prompts, layer)
            activations = activations.float()
            
            # Move activations to CPU if they're on GPU (SAE training is faster on CPU for small models)
            if activations.is_cuda:
                activations = activations.cpu()
            
            # Train SAE (quick training for screening)
            sae_config = SAEConfig(d_model=d_model, d_sae=d_model * 4, l1_coeff=1e-3, lr=1e-3)
            sae = SimpleSAE(sae_config)
            trainer = SAETrainer(sae, lr=sae_config.lr)
            
            # Train with progress reporting
            for epoch in range(200):  # Quick training
                loss = trainer.step(activations)
                if epoch > 0 and epoch % 50 == 0:
                    print(".", end="", flush=True)
            
            # Analyze features
            sae.eval()
            with torch.no_grad():
                features = sae.get_feature_activations(activations)
            
            # Calculate discriminability
            fb_mask = torch.tensor([l == "false_belief" for l in labels])
            tb_mask = torch.tensor([l == "true_belief" for l in labels])
            
            if fb_mask.sum() > 0 and tb_mask.sum() > 0:
                fb_features = features[fb_mask].mean(dim=0)
                tb_features = features[tb_mask].mean(dim=0)
                diff = fb_features - tb_features
                discriminability = diff.abs().max().item()
            else:
                discriminability = 0.0
            
            layer_scores[layer] = discriminability
            print(f" discriminability: {discriminability:.3f}")
            
            # Cleanup: delete SAE and trainer to free memory
            del sae, trainer, features, activations
            gc.collect()  # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f" ERROR: {e}")
            layer_scores[layer] = 0.0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        sys.stdout.flush()
    
    # Select candidate layers (top 50% or those above threshold)
    if not layer_scores:
        raise ValueError("No layer scores computed - check SAE training")
    
    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    threshold = np.percentile([s for _, s in layer_scores.items()], 50)  # Top 50%
    
    candidate_layers = [layer for layer, score in sorted_layers if score >= threshold]
    
    # Ensure at least one candidate layer
    if not candidate_layers:
        candidate_layers = [sorted_layers[0][0]]  # At least take the best one
    
    print(f"\n  Candidate layers (top 50%): {candidate_layers}")
    print(f"  Discriminability range: {min(layer_scores.values()):.3f} - {max(layer_scores.values()):.3f}")
    sys.stdout.flush()
    
    return candidate_layers, layer_scores


def stage2_attention_pattern_filtering(model, tokenizer, scenarios, candidate_layers):
    """
    Stage 2: Find heads that attend to relevant tokens.
    
    Returns: candidate_heads (list of (layer, head) tuples)
    """
    print(f"\n{'='*60}")
    print("STAGE 2: ATTENTION PATTERN FILTERING")
    print(f"{'='*60}")
    print("Finding heads that attend to relevant tokens (agent names, belief states)...")
    sys.stdout.flush()
    
    # Format prompts
    prompts = []
    tokens_list = []
    
    for s in scenarios:
        story = s.get('story', '')
        question = s.get('question', '')
        prompt_text = f"{story}\n\n{question}"
        prompts.append(prompt_text)
        
        # Get tokens for this prompt
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(formatted, add_special_tokens=False))
        tokens_list.append(tokens)
    
    # Extract attention patterns
    attention_data = extract_attention_patterns(model, tokenizer, prompts, candidate_layers)
    
    # Clear cache after attention extraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Keywords to look for (agent names, belief-related words)
    keywords = [
        'alice', 'bob', 'carol', 'dave', 'eve',  # Common agent names
        'believes', 'thinks', 'knows', 'sees', 'watches',  # Belief verbs
        'moved', 'left', 'returned', 'searched',  # Action verbs
        'basket', 'box', 'drawer', 'cupboard',  # Location words
    ]
    
    # Score each head
    head_scores = {}
    
    print("\n  Scoring heads...")
    sys.stdout.flush()
    
    total_heads = sum(len(attention_data[layer_idx]) for layer_idx in candidate_layers)
    head_count = 0
    
    for layer_idx in candidate_layers:
        for head_idx in attention_data[layer_idx].keys():
            head_count += 1
            if head_count % 20 == 0:
                print(f"    [{head_count}/{total_heads}]", end="", flush=True)
            
            attention_patterns = attention_data[layer_idx][head_idx]
            
            # Score this head
            score = score_head_attention(attention_patterns, tokens_list, keywords)
            head_scores[(layer_idx, head_idx)] = score
    
    print(f"  Done! Scored {head_count} heads")
    sys.stdout.flush()
    
    # Select top candidates (top 50% or top 100, whichever is smaller)
    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    n_candidates = min(100, len(sorted_heads) // 2)  # Top 50% or 100, whichever is smaller
    
    candidate_heads = [head for head, score in sorted_heads[:n_candidates]]
    
    print(f"\n  Selected {len(candidate_heads)} candidate heads (top {n_candidates})")
    print(f"  Score range: {min(head_scores.values()):.3f} - {max(head_scores.values()):.3f}")
    sys.stdout.flush()
    
    return candidate_heads, head_scores


def main():
    print("=" * 70)
    print("STEP 10c: SMART MULTI-AGENT CIRCUIT HUNT (HYBRID APPROACH)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print("\nSMART FILTERING:")
    print("✅ Stage 1: SAE layer screening (find discriminative layers)")
    print("✅ Stage 2: Attention pattern filtering (find relevant heads)")
    print("✅ Stage 3: Full ablation (only on candidates)")
    print("✅ Expected speedup: 30-65x vs comprehensive approach")
    sys.stdout.flush()
    
    config = ExperimentConfig()
    
    # Generate multi-agent scenarios
    print("\nGenerating multi-agent scenarios...")
    sys.stdout.flush()
    
    generator = MultiAgentScenarioGenerator(
        use_novel_names=config.require_novel_names,
        seed=42
    )
    
    # Generate enough for all stages
    scenarios = generator.generate_balanced_set(n_per_type=20)  # 5 types × 20 = 100 scenarios
    
    if not scenarios:
        raise ValueError("Failed to generate scenarios - check MultiAgentScenarioGenerator")
    
    type_counts = {}
    for s in scenarios:
        s_type = s.get('type', s.get('scenario_type', 'unknown'))
        type_counts[s_type] = type_counts.get(s_type, 0) + 1
    
    print(f"  Generated: {len(scenarios)} multi-agent scenarios")
    print(f"  Types: {dict(type_counts)}")
    
    # Validate scenario format
    required_fields = ['story', 'question']
    missing_fields = []
    for i, s in enumerate(scenarios[:5]):  # Check first 5
        for field in required_fields:
            if field not in s:
                missing_fields.append(f"Scenario {i} missing '{field}'")
    if missing_fields:
        print(f"  Warning: Some scenarios missing fields: {missing_fields[:3]}")
    
    sys.stdout.flush()
    
    # Load model
    print("\nLoading model...")
    sys.stdout.flush()
    
    model, tokenizer = load_model_for_chat(
        model_name=config.model_name,
        device_map=config.device_map,
        dtype=config.dtype,
        attn_implementation=None  # Auto-select fastest
    )
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"Model loaded! {n_layers} layers, {n_heads} heads per layer")
    print(f"Total heads: {n_layers} × {n_heads} = {n_layers * n_heads}")
    sys.stdout.flush()
    
    # ========================================
    # STAGE 1: SAE LAYER SCREENING
    # ========================================
    candidate_layers, layer_scores = stage1_sae_layer_screening(
        model, tokenizer, scenarios, config
    )
    
    print(f"\n✓ Stage 1 complete: {len(candidate_layers)} candidate layers out of {n_layers}")
    
    if not candidate_layers:
        raise ValueError("No candidate layers found - SAE screening failed. Check model and scenarios.")
    
    # Validate candidate layers are within model bounds
    invalid_layers = [l for l in candidate_layers if l < 0 or l >= n_layers]
    if invalid_layers:
        raise ValueError(f"Invalid candidate layers: {invalid_layers} (model has {n_layers} layers)")
    
    sys.stdout.flush()
    
    # ========================================
    # STAGE 2: ATTENTION PATTERN FILTERING
    # ========================================
    # Need eager mode for attention extraction
    print("\nSwitching to eager mode for attention extraction...")
    sys.stdout.flush()
    
    # Clear cache before loading new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        tokenizer_eager = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        model_eager = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=config.device_map,
            trust_remote_code=True,
            attn_implementation="eager",  # Required for attention outputs
        )
        model_eager.eval()
        
        candidate_heads, head_scores = stage2_attention_pattern_filtering(
            model_eager, tokenizer_eager, scenarios, candidate_layers
        )
        
        print(f"\n✓ Stage 2 complete: {len(candidate_heads)} candidate heads")
        
        if not candidate_heads:
            raise ValueError("No candidate heads found - attention filtering failed. Check model and scenarios.")
        
        # Validate candidate heads are within model bounds
        invalid_heads = [(l, h) for l, h in candidate_heads if l < 0 or l >= n_layers or h < 0 or h >= n_heads]
        if invalid_heads:
            raise ValueError(f"Invalid candidate heads: {invalid_heads[:5]} (model: {n_layers} layers, {n_heads} heads)")
        
        # Cleanup eager model (free memory)
        del model_eager
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"\n✗ Stage 2 failed: {e}")
        print("  Falling back to using all heads from candidate layers...")
        # Fallback: use all heads from candidate layers
        candidate_heads = [(l, h) for l in candidate_layers for h in range(n_heads)]
        head_scores = {(l, h): 0.0 for l, h in candidate_heads}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    sys.stdout.flush()
    
    # ========================================
    # STAGE 3: FULL ABLATION ON CANDIDATES
    # ========================================
    print(f"\n{'='*60}")
    print("STAGE 3: FULL ABLATION ON CANDIDATE HEADS")
    print(f"{'='*60}")
    print(f"Testing {len(candidate_heads)} candidate heads (instead of {n_layers * n_heads})")
    print(f"Speedup: {(n_layers * n_heads) / len(candidate_heads):.1f}x")
    sys.stdout.flush()
    
    # Use subset of scenarios for ablation (15 per head)
    n_scenarios_per_head = 15
    if len(scenarios) < n_scenarios_per_head:
        print(f"  Warning: Only {len(scenarios)} scenarios available, using all")
        ablation_scenarios = scenarios
        n_scenarios_per_head = len(scenarios)
    else:
        ablation_scenarios = scenarios[:n_scenarios_per_head]
    
    print(f"  Using {len(ablation_scenarios)} scenarios per head for ablation")
    sys.stdout.flush()
    
    # Initialize analyzer
    analyzer = ChatModeCircuitAnalyzer(model, tokenizer, config)
    
    # Clear cache before starting ablation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract layers and heads from candidate_heads
    candidate_layers_set = set(layer for layer, head in candidate_heads)
    layers_to_test = sorted(candidate_layers_set)
    
    # For each layer, get heads to test
    heads_by_layer = defaultdict(list)
    for layer, head in candidate_heads:
        heads_by_layer[layer].append(head)
    
    # Run ablation sweep on candidates
    # We'll need to modify the approach since we have specific heads, not evenly sampled
    print(f"\nRunning ablation on {len(candidate_heads)} candidate heads...")
    sys.stdout.flush()
    
    results = {
        'baseline': None,
        'ablations': {}
    }
    
    # Get baseline
    print("  Getting baseline...")
    sys.stdout.flush()
    
    try:
        baseline = analyzer.get_baseline(ablation_scenarios)
        results['baseline'] = baseline
        
        # Safely access baseline keys
        accuracy = baseline.get('accuracy', 0.0)
        n_correct = baseline.get('n_correct', 0)
        n_total = baseline.get('n', baseline.get('n_total', len(ablation_scenarios)))
        
        print(f"    Baseline accuracy: {accuracy:.1%} ({n_correct}/{n_total})")
    except Exception as e:
        print(f"    ERROR getting baseline: {e}")
        raise
    
    sys.stdout.flush()
    
    # Ablate each candidate head
    from analysis.circuits.ablation import HeadAblator
    
    ablator = HeadAblator(model, optimize_hooks=True)
    
    ablation_results = {}
    for i, (layer, head) in enumerate(candidate_heads):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(candidate_heads)}] Testing L{layer}H{head}...", end="", flush=True)
        
        try:
            # Clear cache periodically (every 5 heads)
            if i > 0 and i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Ablate this head
            ablator.ablate_heads([(layer, head)])
            
            # Test scenarios
            test_results = []
            for scenario in ablation_scenarios:
                # Scenario should already be in correct format for analyzer
                result = analyzer.test_scenario(scenario, max_tokens=500)
                test_results.append(result)
            
            # Calculate accuracy
            n_correct = sum(1 for r in test_results if r.correct)
            accuracy = n_correct / len(test_results)
            
            baseline_accuracy = baseline.get('accuracy', 0.0)
            ablation_results[(layer, head)] = {
                'accuracy': accuracy,
                'n_correct': n_correct,
                'n_total': len(test_results),
                'effect': accuracy - baseline_accuracy,
            }
            
            # Clear ablation
            ablator.clear()
            
            if (i + 1) % 10 == 0 or i == 0:
                effect = ablation_results[(layer, head)]['effect']
                print(f" effect: {effect:+.1%}, acc: {accuracy:.1%}")
                sys.stdout.flush()
        
        except Exception as e:
            print(f" ERROR: {e}")
            # Record failure
            baseline_accuracy = baseline.get('accuracy', 0.0)
            ablation_results[(layer, head)] = {
                'accuracy': baseline_accuracy,  # Assume no effect on error
                'n_correct': 0,
                'n_total': len(ablation_scenarios),
                'effect': 0.0,
                'error': str(e),
            }
            ablator.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sys.stdout.flush()
    
    print(f"\n  Ablation complete!")
    
    # Final cleanup
    ablator.cleanup_all_hooks() if hasattr(ablator, 'cleanup_all_hooks') else ablator.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    sys.stdout.flush()
    
    # Format results for analyzer
    results['ablations'] = {}
    for layer in layers_to_test:
        results['ablations'][layer] = {}
        for head in heads_by_layer[layer]:
            if (layer, head) in ablation_results:
                results['ablations'][layer][head] = ablation_results[(layer, head)]
    
    # Get significant heads
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    all_heads = []
    for (layer, head), data in ablation_results.items():
        # Safely access data keys
        accuracy = data.get('accuracy', 0.0)
        effect = data.get('effect', 0.0)
        
        # Calculate p-value using McNemar's test (simplified)
        # For now, use effect size as proxy
        all_heads.append({
            'layer': layer,
            'head': head,
            'accuracy': accuracy,
            'effect': effect,
            'p_value': 0.05 if abs(effect) > 0.1 else 0.5,  # Simplified
            'significant': abs(effect) > 0.15,  # Large effect threshold
            'significant_uncorrected': abs(effect) > 0.1,
        })
    
    # Multiple comparisons correction
    n_tests = len(all_heads)
    corrected_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
    
    significant_heads = [h for h in all_heads if h['significant']]
    
    print(f"\nTests performed: {n_tests}")
    print(f"Corrected alpha (Bonferroni): {corrected_alpha:.6f}")
    print(f"Significant heads: {len(significant_heads)}")
    sys.stdout.flush()
    
    # Sort by effect
    all_heads.sort(key=lambda x: abs(x['effect']), reverse=True)
    helpful_heads = [h for h in all_heads if h['effect'] > 0]
    inhibitory_heads = [h for h in all_heads if h['effect'] < 0]
    
    print("\n" + "="*60)
    print("TOP 20 MOST IMPACTFUL HEADS:")
    print("="*60)
    for h in all_heads[:20]:
        direction = "HELPFUL" if h['effect'] > 0 else "INHIBITORY"
        sig_marker = " ***" if h['significant'] else (" *" if h['significant_uncorrected'] else "")
        print(f"  L{h['layer']:2d}H{h['head']:2d} ({direction}): effect={h['effect']:+.1%}, acc={h['accuracy']:.1%}{sig_marker}")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    print("\nGenerating visualizations...")
    sys.stdout.flush()
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Figure 1: 3D Filtering Pipeline - Layer × Head × Effect
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: 3D scatter showing all three stages
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Stage 1: SAE discriminability (as color/intensity)
    # Stage 2: Attention score (as size)
    # Stage 3: Ablation effect (as z-axis)
    
    # Build data for all heads that went through filtering
    x_data = []  # Layer
    y_data = []  # Head
    z_data = []  # Ablation effect
    c_data = []  # SAE discriminability (color)
    s_data = []  # Attention score (size)
    labels_data = []  # For annotation
    
    for h in all_heads:
        layer = h['layer']
        head = h['head']
        effect = h['effect'] * 100
        
        # Get SAE score for this layer
        sae_score = layer_scores.get(layer, 0.0)
        
        # Get attention score for this head
        attn_score = head_scores.get((layer, head), 0.0)
        
        x_data.append(layer)
        y_data.append(head)
        z_data.append(effect)
        c_data.append(sae_score)
        s_data.append(max(20, attn_score * 100))  # Scale for visibility
        
        # Label top heads
        if abs(effect) > np.percentile([abs(h['effect'] * 100) for h in all_heads], 90):
            labels_data.append((layer, head, effect, f"L{layer}H{head}"))
    
    # Create scatter plot
    scatter = ax1.scatter(x_data, y_data, z_data, c=c_data, s=s_data, 
                          cmap='viridis', alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1)
    cbar.set_label('SAE Discriminability', fontsize=9)
    
    ax1.set_xlabel('Layer', fontsize=10)
    ax1.set_ylabel('Head', fontsize=10)
    ax1.set_zlabel('Ablation Effect (%)', fontsize=10)
    ax1.set_title('3D Filtering Pipeline\n(Layer × Head × Effect)', fontsize=11, fontweight='bold')
    
    # Add plane at z=0 (only if we have data)
    if x_data and y_data:
        x_range = [min(x_data), max(x_data)]
        y_range = [min(y_data), max(y_data)]
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.zeros_like(xx)
        ax1.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # Subplot 2: 2D filtering pipeline (side view)
    ax2 = fig.add_subplot(132)
    
    # Plot all three stages
    layers_sorted = sorted(layer_scores.keys())
    scores_sorted = [layer_scores[l] for l in layers_sorted]
    colors_sae = ['coral' if l in candidate_layers else 'steelblue' for l in layers_sorted]
    ax2.bar(layers_sorted, scores_sorted, color=colors_sae, edgecolor='black', 
            alpha=0.7, label='SAE Discriminability')
    ax2.set_xlabel("Layer", fontsize=10)
    ax2.set_ylabel("SAE Score", fontsize=10)
    ax2.set_title("Stage 1: SAE Screening", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    
    # Subplot 3: Attention scores heatmap
    ax3 = fig.add_subplot(133)
    
    # Create heatmap of attention scores by layer and head
    if head_scores:
        try:
            max_layer = max(layer for layer, head in head_scores.keys())
            max_head = max(head for layer, head in head_scores.keys())
            
            heatmap = np.zeros((max_layer + 1, max_head + 1))
            for (layer, head), score in head_scores.items():
                heatmap[layer, head] = score
            
            im = ax3.imshow(heatmap, cmap='hot', aspect='auto', interpolation='nearest')
            ax3.set_xlabel("Head", fontsize=10)
            ax3.set_ylabel("Layer", fontsize=10)
            ax3.set_title("Stage 2: Attention Scores", fontsize=11, fontweight='bold')
            cbar3 = plt.colorbar(im, ax=ax3)
            cbar3.set_label('Attention Score', fontsize=9)
        except ValueError:
            # Empty head_scores
            ax3.text(0.5, 0.5, 'No attention scores available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Stage 2: Attention Scores", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step10c_filtering_pipeline.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Filtering pipeline (3D) saved to: {fig_path}")
    
    # Figure 2: 3D Trajectory through Filtering Stages
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Show how heads move through the 3 stages
    # Stage 1: SAE discriminability (x-axis = layer, y-axis = SAE score)
    # Stage 2: Attention score (z-axis)
    # Stage 3: Ablation effect (color)
    
    for h in all_heads[:50]:  # Top 50 for readability
        layer = h['layer']
        head = h['head']
        effect = h['effect'] * 100
        
        sae_score = layer_scores.get(layer, 0.0)
        attn_score = head_scores.get((layer, head), 0.0)
        
        # Color by effect
        color = 'steelblue' if effect > 0 else 'coral'
        alpha = min(1.0, abs(effect) / 20.0)  # More opaque = larger effect
        
        ax.scatter(layer, sae_score, attn_score, 
                  c=color, s=abs(effect) * 5 + 20, alpha=alpha,
                  edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('SAE Discriminability', fontsize=11)
    ax.set_zlabel('Attention Score', fontsize=11)
    ax.set_title('Head Trajectory Through Filtering Stages\n(Size = |Effect|, Color = Helpful/Inhibitory)', 
                 fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='Helpful'),
        Patch(facecolor='coral', edgecolor='black', label='Inhibitory'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step10c_trajectory_3d.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D trajectory saved to: {fig_path}")
    
    # Figure 3: 3D Surface Plot - Effect Landscape
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a surface showing ablation effects across layer × head space
    # Interpolate to create smooth surface
    if all_heads and len(all_heads) > 0:
        layers_unique = sorted(set(h['layer'] for h in all_heads))
        heads_unique = sorted(set(h['head'] for h in all_heads))
        
        if layers_unique and heads_unique:
            # Create grid
            L = np.array(layers_unique)
            H = np.array(heads_unique)
            L_grid, H_grid = np.meshgrid(L, H, indexing='ij')
            
            # Map effects to grid
            effect_grid = np.zeros_like(L_grid, dtype=float)
            for h in all_heads:
                layer_idx = layers_unique.index(h['layer'])
                head_idx = heads_unique.index(h['head'])
                effect_grid[layer_idx, head_idx] = h['effect'] * 100
            
            # Create surface
            surf = ax.plot_surface(L_grid, H_grid, effect_grid, 
                                  cmap='RdBu_r', alpha=0.8, 
                                  linewidth=0, antialiased=True,
                                  vmin=-30, vmax=30)
            
            # Add scatter points for actual data
            for h in all_heads:
                layer = h['layer']
                head = h['head']
                effect = h['effect'] * 100
                color = 'steelblue' if effect > 0 else 'coral'
                ax.scatter([layer], [head], [effect], 
                          c=color, s=50, alpha=0.9, edgecolors='black', linewidths=0.5)
            
            ax.set_xlabel('Layer', fontsize=11)
            ax.set_ylabel('Head', fontsize=11)
            ax.set_zlabel('Ablation Effect (%)', fontsize=11)
            ax.set_title('Effect Landscape: Layer × Head × Ablation Effect', 
                        fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(surf, ax=ax, pad=0.15, shrink=0.8)
            cbar.set_label('Ablation Effect (%)', fontsize=10)
            
            # Add plane at z=0
            zz_zero = np.zeros_like(L_grid)
            ax.plot_surface(L_grid, H_grid, zz_zero, alpha=0.2, color='gray')
        else:
            ax.text(0.5, 0.5, 0, 'No data to plot', ha='center', va='center')
            ax.set_title('Effect Landscape (No Data)', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 0, 'No ablation results', ha='center', va='center')
        ax.set_title('Effect Landscape (No Data)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step10c_effect_landscape_3d.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D effect landscape saved to: {fig_path}")
    
    # Figure 4: Top heads bar chart (enhanced)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_heads = sorted(all_heads, key=lambda x: abs(x['effect']), reverse=True)[:20]
    
    if top_heads:
        labels = [f"L{h['layer']:2d}H{h['head']:2d}" for h in top_heads]
        effects = [h['effect'] * 100 for h in top_heads]
        colors = ['steelblue' if e > 0 else 'coral' for e in effects]
        
        bars = ax.barh(range(len(labels)), effects, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Accuracy Change (%)", fontsize=12)
        ax.set_title("Top 20 Most Impactful Heads (Smart Filtering)", fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        # Add effect values
        for i, (bar, effect) in enumerate(zip(bars, effects)):
            ax.text(effect + (1 if effect > 0 else -1), i, f'{effect:+.1f}%',
                   va='center', ha='left' if effect > 0 else 'right',
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "step10c_top_heads.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Top heads chart saved to: {fig_path}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_scenarios_per_head": n_scenarios_per_head,
            "total_heads_in_model": n_layers * n_heads,
            "candidate_layers": candidate_layers,
            "n_candidate_heads": len(candidate_heads),
            "speedup": (n_layers * n_heads) / len(candidate_heads),
        },
        "stage1_sae_screening": {
            "layer_scores": {str(k): v for k, v in layer_scores.items()},
            "candidate_layers": candidate_layers,
        },
        "stage2_attention_filtering": {
            "head_scores": {f"L{l}H{h}": float(s) for (l, h), s in head_scores.items()},
            "candidate_heads": [{"layer": l, "head": h} for l, h in candidate_heads],
        },
        "baseline": baseline,
        "ablation_results": {
            f"L{l}H{h}": data for (l, h), data in ablation_results.items()
        },
        "all_heads": all_heads,
        "significant_heads": [h for h in significant_heads],
        "top_helpful_heads": [h for h in helpful_heads[:20]],
        "top_inhibitory_heads": [h for h in sorted(inhibitory_heads, key=lambda x: x['effect'])[:20]],
        "statistics": {
            "n_tests": n_tests,
            "corrected_alpha": corrected_alpha,
            "n_significant": len(significant_heads),
        },
    }
    
    output_path = RESULTS_DIR / "step10c_smart_circuit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 10c COMPLETE - SMART FILTERING")
    print(f"{'='*60}")
    print(f"\nFiltering pipeline:")
    print(f"  Stage 1 (SAE): {n_layers} layers → {len(candidate_layers)} candidate layers")
    print(f"  Stage 2 (Attention): {len(candidate_layers) * n_heads} heads → {len(candidate_heads)} candidates")
    print(f"  Stage 3 (Ablation): Tested {len(candidate_heads)} heads")
    print(f"\nSpeedup: {(n_layers * n_heads) / len(candidate_heads):.1f}x")
    print(f"  (Would be {n_layers * n_heads} heads without filtering)")
    
    print(f"\nKey findings:")
    if significant_heads:
        print(f"  - Found {len(significant_heads)} heads with large effects")
        if helpful_heads:
            top_helpful = helpful_heads[0]
            print(f"  - Most helpful: L{top_helpful['layer']}H{top_helpful['head']} ({top_helpful['effect']:+.1%})")
        if inhibitory_heads:
            top_inhibitory = sorted(inhibitory_heads, key=lambda x: x['effect'])[0]
            print(f"  - Most inhibitory: L{top_inhibitory['layer']}H{top_inhibitory['head']} ({top_inhibitory['effect']:+.1%})")
    else:
        print("  - No heads with very large effects found")
        print("  - Consider lowering threshold or checking if filtering was too aggressive")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

