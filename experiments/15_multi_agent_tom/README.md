# Experiment 15: Multi-Agent Theory of Mind

## Goal
Test whether Qwen3-4B can track:
1. **First-order beliefs**: What does Agent A believe?
2. **Second-order beliefs**: What does Agent A believe Agent B believes?
3. **Belief updates in dialogue**: How do agent beliefs change across conversation turns?

## Key Difference from Experiment 14
- Experiment 14: Single-agent false belief (Sally-Anne)
- Experiment 15: Multi-agent recursive beliefs

## Methodology

### Test 1: Recursive ToM (Second-Order Beliefs)
```
Scenario: Alice tells Bob X. Carol tells Alice Y (Bob not present).
Question: What does Alice think Bob will do?

Correct answer requires tracking:
- Alice's belief (updated to Y)
- Alice's model of Bob's belief (still X)
```

### Test 2: Multi-Turn Dialogue Belief Tracking
```
Turn 1: "Alice: The key is in drawer A"
Turn 2: "Bob: I agree, I saw it there"
Turn 3: "Carol: I moved it to drawer B" (Alice hears, Bob doesn't)
Turn 4: "Alice: Let me get the key"

Probe at Turn 4:
- Alice's belief? → drawer B (updated)
- Bob's belief? → drawer A (not updated)
- Can we decode BOTH from the same forward pass?
```

### Test 3: Behavioral Prediction
```
After dialogue, predict:
- Where will Alice look? (should be B)
- Where will Bob look? (should be A)
```

## Success Criteria
1. **Behavioral**: Model predicts different actions for Alice vs Bob based on their beliefs
2. **Representational**: We can linearly decode each agent's belief from activations
3. **Causal**: Ablating specific heads changes agent-specific predictions

## Files
- `scripts/step1_generate_recursive_scenarios.py` - Create 200+ recursive ToM scenarios
- `scripts/step2_behavioral_test.py` - Test if model predicts belief-based actions
- `scripts/step3_dialogue_tracking.py` - Multi-turn belief extraction
- `scripts/step4_agent_specific_probing.py` - Can we decode Agent A vs Agent B's belief separately?

## Expected Outcomes
- If model has genuine multi-agent ToM: Can decode and causally manipulate per-agent beliefs
- If model has only heuristics: Will fail on recursive or conflicting belief scenarios



