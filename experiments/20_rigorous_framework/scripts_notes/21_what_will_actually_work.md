# What Will Actually Work for Distributed ToM Circuit

## The Problem

Activation patching in chat mode fails because:
1. **Sequence length mismatch**: Patching prompt-length activations into growing generation sequence
2. **Position misalignment**: Patching at wrong positions during reasoning phase
3. **Distributed circuit**: Single-layer interventions have 0% impact (Step 35)

## What We Know

1. **ToM is distributed** (Step 35): Ablating single heads has 0% impact
2. **Decision happens during/after reasoning** (Step 34): Logit lens shows probability evolution
3. **Reasoning phase matters**: The `<think>...</think>` tokens are part of the computation

## What Will Actually Work

### Option 1: Direct Logit Intervention (Simplest) ⭐ RECOMMENDED

**Approach**: Directly manipulate logits at the answer position (after `</think>`)

**Why it works**:
- Bypasses sequence length issues
- Tests if answer is "flippable" at that position
- Simple to implement
- If it works, shows decision happens at that point

**Implementation**:
```python
intervener = DirectLogitIntervention(model, tokenizer)
result = intervener.intervene(
    prompt=fb_prompt,
    answer_tokens=["basket", "box"],
    boost_token="box",  # Try to flip to box
    suppress_token="basket",
    strength=5.0
)
```

**What it tells us**:
- Can we flip the answer by manipulating logits?
- If yes: decision is "localized" at answer position (even if circuit is distributed)
- If no: decision is more distributed across reasoning process

### Option 2: Targeted Residual Stream Patching (More Causal)

**Approach**: Patch residual stream at answer position only, multiple layers simultaneously

**Why it works**:
- Patches at RIGHT TIME (answer, not reasoning)
- Patches MULTIPLE layers (since it's distributed)
- More causal than logit manipulation

**Implementation**:
```python
patcher = TargetedPatcher(model, tokenizer)
result = patcher.patch_at_answer_position(
    target_prompt=fb_prompt,
    source_activations=cached_tb_activations,
    layers=[20, 24, 28],  # Multiple layers!
    answer_position=answer_pos,
    answer_tokens=["basket", "box"]
)
```

**Key innovation**:
- Find answer position using logit lens
- Patch ONLY at that position
- Patch multiple layers simultaneously

**What it tells us**:
- Which layers contribute to the decision (if multi-layer patch works)
- Whether decision is "readable" in residual stream at answer position

### Option 3: Logit Lens During Generation (Diagnostic)

**Approach**: Track answer probability evolution token-by-token during generation

**Why it works**:
- Shows WHEN decision crystallizes
- Identifies exact position where to patch/intervene
- Diagnostic tool to inform other approaches

**What it tells us**:
- Where in generation sequence the answer probability diverges
- Whether decision happens during reasoning or at answer position
- How distributed the decision-making is

## Recommended Strategy

1. **Start with Direct Logit Intervention**:
   - Simplest to implement
   - Tests if answer is flippable
   - If it works, we know decision is at answer position

2. **If logit intervention works, try Targeted Patching**:
   - More causal
   - Can identify which layers matter
   - Patch multiple layers simultaneously

3. **Use Logit Lens to inform both**:
   - Find exact answer position
   - Understand when decision crystallizes

## Why This Will Work

The key insight: **We're not trying to patch the entire reasoning process, just the decision point.**

In chat mode:
- Reasoning happens: `<think>...</think>`
- Answer happens: After `</think>`

We should patch/intervene ONLY at the answer position, not during reasoning.

For distributed circuits:
- Single-layer interventions fail (confirmed)
- Multi-layer interventions might work
- Or we need to understand information flow

## Expected Outcomes

1. **If direct logit intervention works**:
   - Decision is "localized" at answer position
   - Circuit is distributed but decision is readable there
   - Can use this for causal analysis

2. **If targeted patching works**:
   - Can identify which layers contribute
   - Shows residual stream contains decision-relevant info
   - More causal than logit manipulation

3. **If neither works**:
   - Decision is truly distributed across reasoning process
   - Need to understand information flow, not just final decision
   - Focus on Step 34 (reasoning analysis) instead

## Next Steps

1. Test direct logit intervention (simplest)
2. If it works, implement targeted patching
3. Use logit lens to find exact positions
4. Document findings about distributed nature





