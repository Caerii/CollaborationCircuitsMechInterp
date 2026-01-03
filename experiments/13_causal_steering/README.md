# Experiment 13: Causal Steering & Interventions

## Goal
Demonstrate causal control over agent modeling behavior.

## Methods

### 1. Steering Vectors
- Extract "agree" vs "disagree" direction
- Add steering vector during inference
- Measure behavioral change

### 2. Activation Patching
- Run model on "agree" context
- Patch activations into "disagree" context
- Check if response flips

### 3. Targeted Head Intervention
- Steer only the identified ToM heads (L12H0, L24H0, L30H0)
- Compare to steering random heads

## Success Criteria
- Steering flips behavior in >50% of cases
- ToM heads show stronger effect than random heads
- Effect is specific to agent-modeling prompts























