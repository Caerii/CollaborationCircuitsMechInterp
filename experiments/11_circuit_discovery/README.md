# Experiment 11: Circuit Discovery

## Goal
Identify WHICH attention heads implement Theory of Mind / agent modeling.

## Methods

### 1. Attention Pattern Analysis
- Which heads attend from "Agent B" tokens to "Agent A" tokens?
- Do specific heads show cross-agent attention patterns?

### 2. Activation Patching (Causal)
- Patch individual head outputs from "agree" → "disagree" context
- Which heads, when patched, flip the model's response?

### 3. Head Ablation
- Zero out specific heads
- Which heads are NECESSARY for agent modeling?

### 4. Mutual Information
- I(head_output; agent_belief) for each head
- Find heads with high MI for belief tracking

## Expected Output
- List of "ToM heads" with their layer/head indices
- Causal evidence that these heads matter
- Visualization of attention patterns






















