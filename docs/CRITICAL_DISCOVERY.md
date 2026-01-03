# CRITICAL DISCOVERY: The Explicit/Implicit ToM Gap

## The Finding

| Test Condition | Updated Agent | Unchanged Agent |
|----------------|--------------|-----------------|
| **Pure inference** (no explicit cues) | **17%** | 98% |
| **Explicit belief** ("X believes Y") | 100% | 100% |
| **Dialogue format** (like multi-agent chat) | **7%** | 95% |

## What This Means

### The Model Has TWO Distinct Capabilities:

1. **"Belief Statement Parser"** (STRONG - 100%)
   - Extracts explicitly stated beliefs from text
   - "Alice believes the ball is in the basket" → predicts basket
   - This IS the collaboration circuit (L12H0, L23H0)

2. **"First-Mentioned Location Heuristic"** (STRONG - 95-98%)
   - When belief = first-mentioned location, model succeeds
   - Standard Sally-Anne works because Sally's belief IS where she put it
   - Unchanged agents succeed because their belief hasn't changed

### What's MISSING:

3. **"Belief Update Inference"** (FAILS - 7-17%)
   - Model CANNOT infer that communication updates beliefs
   - "Eve tells Alice: 'I moved the ball'" → Alice should now believe new location
   - But model still predicts first-mentioned location!

---

## Why Multi-Agent Software Development Works

In software dev, ALL information is explicit:

```
DEVELOPER: "Here's my code for function X"
REVIEWER: "I think there's a bug on line 5"
DEVELOPER: "I see your point. Let me fix that."
```

The model never needs to INFER what anyone believes - it's all STATED.

---

## Why Sally-Anne Tests Are Misleading

Standard Sally-Anne actually PASSES (90%) because:
- Sally puts ball in basket (first-mentioned for Sally)
- Sally's belief = basket (first-mentioned location!)
- Model uses heuristic, not ToM

The test doesn't require belief tracking - the heuristic works!

---

## The Real Test: Belief Updates

When an agent LEARNS new information:

```
"Eve tells Alice: 'I moved the ball to the basket.'"
Where will Alice look?
```

**Pure inference: 17%** - Model ignores the update
**Explicit belief: 100%** - Model follows stated belief

The model cannot connect: **"being told X" → "now believes X"**

---

## Implications for Circuit Discovery

### What We Found (L12H0, L23H0)
- **Function**: Parse explicit belief statements
- **Mechanism**: Extract "agent" + "believed location" bindings
- **Scope**: Works for explicitly stated beliefs ONLY

### What We Need to Find
- **Function**: Track information flow between agents
- **Mechanism**: Update belief states based on communication
- **Question**: Does this circuit exist? Is it just weak? Or missing entirely?

---

## Next Steps

### Priority 1: Probe for "Communication Updates Belief" Circuit
- Look for heads that attend to communicative verbs ("tells", "says", "informs")
- See if any circuit connects this to belief updates
- May not exist or may be very weak

### Priority 2: Test If Presence Is Encoded
- Can we decode "was agent present during event"?
- If yes: circuit exists but isn't used for belief updates
- If no: fundamental capability is missing

### Priority 3: Attention Pattern Analysis
- What does model attend to in dialogue?
- Does it ignore "tells X" completely?
- Or attend but fail to update?

---

## Key Insight for AI Safety

Current multi-agent systems work because:
- All beliefs are made explicit
- Agents communicate directly
- No inference from context needed

**Risk**: Systems may fail unexpectedly when implicit reasoning is required.

**Recommendation**: Always make agent knowledge states explicit in multi-agent prompts.

---

## Summary

| What Model CAN Do | What Model CANNOT Do |
|-------------------|---------------------|
| Parse explicit beliefs | Infer belief updates |
| Follow first-mentioned heuristic | Track information flow |
| Handle explicit multi-agent protocol | Understand that communication updates beliefs |

**The capability for explicit multi-agent collaboration exists.**
**The capability for implicit belief inference does NOT.**

This is a fundamental architectural limitation, not a circuit we haven't found yet.


