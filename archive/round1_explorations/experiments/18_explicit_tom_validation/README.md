# Experiment 18: Explicit vs Implicit ToM Validation

## Motivation

Critical insight: **The model works in multi-agent software development but fails Sally-Anne tests.**

What's different?
- Software dev: All beliefs/knowledge are EXPLICITLY communicated
- Sally-Anne: Beliefs must be INFERRED from narrative events

## Hypothesis

The model has strong "belief statement parsing" circuits but weak/missing "belief computation" circuits.

This explains:
1. Why multi-agent collaboration works (explicit communication)
2. Why Sally-Anne fails (requires belief inference)
3. What our L12H0 + L23H0 circuit actually does (parses explicit beliefs)

## Experiments

### Step 1: Explicit Structure Test

Compare 4 versions of the same scenario:

1. **IMPLICIT** (standard false belief narrative):
   ```
   Sally puts ball in basket. Sally leaves. Anne moves ball to box.
   Sally returns. Where will Sally look?
   ```

2. **EXPLICIT BELIEF**:
   ```
   Sally believes the ball is in the basket. The ball is actually in the box.
   Where will Sally look?
   ```

3. **STRUCTURED FORMAT**:
   ```
   [SALLY'S BELIEF]: The ball is in the basket.
   [REALITY]: The ball is in the box.
   Where will Sally look?
   ```

4. **MULTI-AGENT PROTOCOL** (like software dev):
   ```
   AGENT_SALLY_KNOWLEDGE: I last saw the ball in the basket.
   AGENT_ANNE_ACTION: I moved the ball to the box.
   NOTE: Sally was not present for Anne's action.
   QUERY: Based on Sally's knowledge, where will Sally look?
   ```

### Expected Results

| Version | Expected Accuracy | Why |
|---------|------------------|-----|
| Implicit | ~60-70% | Must infer beliefs |
| Explicit | ~85-95% | Just parse stated belief |
| Structured | ~90-95% | Very clear format |
| Protocol | ~90-95% | Matches training data (multi-agent prompts) |

### Step 2: Perspective Circuit Probe (TODO)

Find representations of "whose perspective we're reasoning from"

### Step 3: Role Assignment Test (TODO)

Test how explicit role assignment affects belief tracking

## Key References

- **Li et al. (EMNLP 2023)**: "Explicit belief state representation significantly improves performance"
- **Zhu et al. (2024)**: "LLMs encode explicit belief representations"
- Our own findings: 86-91% on explicit, 2% on implicit updates

## Success Criteria

If explicit >> implicit:
- Confirms our hypothesis about explicit vs implicit ToM
- Explains why multi-agent software dev works
- Guides future circuit discovery (look for what's missing in implicit case)


