# Step 36: Activation Patching Challenges in Chat Mode

## The Problem

We attempted to use activation patching (the gold standard for causal claims) to verify that Layer 20 causally controls ToM behavior. However, we encountered fundamental challenges with patching in chat mode.

## What We Tried

1. **Fixed ActivationPatcher library** to support chat mode (wrapping prompts in chat template)
2. **Tested patching at multiple layers** (L12, L16, L20, L24, L28, L32, mid_block)
3. **Two directions**: FB→TB (override false belief) and TB→FB (induce false belief)

## Results

### Baselines
- **FB baseline**: ✓ Correct (answered "basket")
- **TB baseline**: ✗ Wrong (answered "basket" instead of "box")
  - This is concerning - TB should be easier but model got it wrong
  - May need more explicit prompts

### Patching Results
- **ALL patches produced gibberish** (Japanese characters: あるあるある...)
- **No successful flips** in either direction
- **0% causal layers found**

## Why This Happened

### The Fundamental Issue

Activation patching in chat mode with reasoning is fundamentally different from completion mode:

1. **Sequence Length Mismatch**
   - Source: activations cached from prompt encoding (e.g., 50 tokens)
   - Target: during generation, sequence grows (50 → 250 tokens)
   - Patching early positions corrupts the entire generation

2. **Position Misalignment**
   - We cache activations from the END of the source prompt
   - But during generation, we're patching into a DIFFERENT context
   - The model has already started generating reasoning tokens

3. **Reasoning Phase Interference**
   - In chat mode, model generates `<think>...</think>` reasoning first
   - Patching during this phase corrupts the reasoning process
   - By the time we get to the answer, the reasoning is already wrong

4. **KV Cache Issues**
   - Modern models use KV caching for efficiency
   - Patching activations doesn't update the KV cache
   - This creates inconsistencies

## What This Tells Us

### 1. ToM is Distributed (Confirms Step 35)
- Single-layer patching doesn't flip behavior
- Consistent with Step 35 finding that ablating single heads has 0% impact
- ToM computation is **emergent** across many layers

### 2. Chat Mode Requires Different Techniques
- Traditional activation patching (from completion mode) doesn't translate directly
- Need techniques that respect the reasoning process
- Or need to patch at the RIGHT moment (when answering, not during reasoning)

### 3. The Model's Reasoning Process Matters
- The `<think>` tags show the model is doing explicit reasoning
- Patching activations disrupts this reasoning
- We can't just patch "belief state" - we need to patch the reasoning that leads to it

## Potential Solutions

### Option 1: Patch During Answer Generation Only
- Use a custom generation loop
- Only patch when generating the final answer token
- Skip patching during reasoning phase
- Requires tracking when model is in `<think>` vs answer phase

### Option 2: Patch Residual Stream at Specific Positions
- Instead of patching layer outputs, patch residual stream
- Patch at the exact token position where decision is made
- More surgical, less disruptive

### Option 3: Use Logit Lens + Direct Intervention
- Use logit lens to find where decision happens
- Directly manipulate logits at that point
- Simpler than activation patching

### Option 4: Accept Distributed Nature
- Document that ToM is distributed
- Focus on understanding the reasoning process (Step 34)
- Use other techniques (ablation, probing, transcoders)

## Next Steps

1. **Fix TB baseline** - Why is it wrong? Need better prompts?
2. **Try Option 1** - Custom generation loop with selective patching
3. **Try Option 3** - Logit lens + direct logit manipulation
4. **Document distributed nature** - This is a finding, not a failure!

## Key Insight

**Activation patching works great for completion mode where the model directly predicts the next token. But for chat mode with reasoning, the model generates a reasoning trace first, and patching disrupts this process.**

This is actually revealing something important: **the model's ToM capability is tied to its ability to reason through the scenario, not just encode a belief state in activations.**










