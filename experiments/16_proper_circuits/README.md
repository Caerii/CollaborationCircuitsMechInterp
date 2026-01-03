# Experiment 16: Proper Circuit Discovery

## The Problem with Previous Ablation

The `self_attn` module's OUTPUT has already been through `o_proj`.
At that point, hidden dimensions no longer correspond to specific heads.
We were just slicing arbitrarily!

## Correct Approaches

### Option 1: Hook BEFORE o_proj
Access the attention output tensor BEFORE the output projection combines heads.

### Option 2: Modify Attention Weights
Zero out the attention weights for specific heads in the attention matrix.

### Option 3: Use TransformerLens (if supported)
TransformerLens has proper head-level hooks built in.

## What We Need

1. **ToM-Specific Metric**: Does ablation flip belief→reality prediction?
2. **Proper Hook Point**: Inside attention, not after o_proj
3. **More Test Prompts**: N=50+ ToM scenarios for ablation

## Key Observation from Exp 14
- ToM prompts (tom1, tom2, tom3): **0% affected** by any ablation
- Neutral prompts: 10-50% affected
- This means we were disrupting GENERAL computation, not ToM circuits!






















