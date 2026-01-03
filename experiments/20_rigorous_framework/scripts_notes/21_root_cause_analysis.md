# Root Cause Analysis: Activation Patching Failure

## Key Discovery from Debug Script

### The KV Cache Problem

**Critical Finding**: During generation with KV caching, the forward hook is called differently:

```
Call 1: seq_len=41 (processing full prompt)
Calls 2-10: seq_len=1 (processing ONE token at a time during generation)
```

### Why This Breaks Patching

1. **During prompt processing (Call 1)**:
   - Hook sees full sequence: `[batch, 41, hidden_dim]`
   - We can patch at position 40 (end of prompt)
   - This works correctly

2. **During generation (Calls 2-10)**:
   - Hook sees ONLY new token: `[batch, 1, hidden_dim]`
   - Previous tokens are in KV cache, not in hidden states
   - When we patch `hidden[:, -1, :]`, we're patching the NEW token, not the prompt tokens
   - **This is wrong!** We want to patch the prompt's last token, but it's already in KV cache

### The Fundamental Issue

**We cannot patch prompt token activations during generation because they're in the KV cache, not in the current forward pass.**

The model uses KV caching for efficiency:
- Prompt tokens: processed once, stored in KV cache
- New tokens: processed one at a time, using KV cache for context
- Hidden states in hook: only for NEW tokens being generated

### Why "prompt_end" Mode Doesn't Work

"prompt_end" mode tries to patch only when `seq_len == target_prompt_len`:
- This happens on Call 1 (correct)
- But the patch happens AFTER the layer processes the input
- By the time we patch, the model has already computed the output
- The patch affects the output, but the KV cache was already built with the original activations

### Why "last" Mode Produces Gibberish

"last" mode patches `hidden[:, -1, :]` on every call:
- Call 1: patches position 40 (end of prompt) - might work
- Calls 2-10: patches position 0 (the new token) with source position 40
- This corrupts the generation because we're patching the wrong thing

## The Real Problem

**Activation patching assumes we can modify activations during generation, but with KV caching, the prompt activations are already cached and we can't modify them.**

## Solutions

### Option 1: Disable KV Caching (Inefficient but Works)

```python
output = model.generate(
    **inputs,
    max_new_tokens=self.max_new_tokens,
    use_cache=False,  # Disable KV cache
    ...
)
```

**Pros**: Can patch at any position during generation
**Cons**: Very slow, uses lots of memory

### Option 2: Patch Before Generation Starts

Patch the residual stream AFTER the prompt is processed but BEFORE generation:
- Process prompt once
- Patch at the last prompt position
- Then generate with patched state

**Challenge**: Need to ensure patch persists through generation

### Option 3: Patch During First Generation Step Only

Only patch on the FIRST new token generation:
- Let prompt process normally
- On first generation step, patch the residual stream
- Continue generation normally

**Challenge**: Need to ensure patch affects subsequent tokens

### Option 4: Use Direct Logit Manipulation

Instead of patching activations, directly manipulate logits:
- Extract belief-relevant features from activations
- Compute logit difference
- Add to logits during generation

**Pros**: Works with KV cache, simpler
**Cons**: Less "causal" (logits are downstream of activations)

### Option 5: Custom Generation Loop

Write custom generation that:
- Processes prompt
- Patches residual stream at end of prompt
- Generates with patched state
- Manually manages KV cache

**Pros**: Full control
**Cons**: Complex, need to reimplement generation logic

## Recommended Next Steps

1. **Try Option 1 first** (disable KV cache) - simplest, will tell us if this is the issue
2. **If that works**, then optimize with Option 3 or 4
3. **Document the limitation** - activation patching with KV cache is fundamentally different

## Insight

This reveals something important: **modern efficient generation (KV caching) is incompatible with naive activation patching**. We need techniques that work WITH the cache, not against it.










