# Experiment 17: Presence Tracking

## Motivation

Experiment 15 found that model:
- Follows EXPLICIT belief statements (86-91%)
- FAILS at inferring belief UPDATES (2% for updated agents)

The key missing capability is **tracking who was present when information changed**.

## Hypothesis

If the model encodes "Agent X was present during event Y":
- The circuit EXISTS but isn't connected to belief prediction
- We can find it and potentially steer/enhance it

If the model DOESN'T encode this:
- The fundamental capability is missing
- Explains why belief updates fail

## Experiments

### Step 1: Presence Probing
Can we decode "was observer present during move?" from activations?

### Step 2: Attention Pattern Analysis (TODO)
Which heads attend to presence/absence cues ("left the room", "returns")?

### Step 3: Activation Patching (TODO)
Patch activations between "present" and "absent" scenarios to find causal components.

## Expected Outcome

This will tell us whether to:
1. Look for a "presence → belief update" circuit that isn't connected
2. Accept that the model lacks fundamental presence tracking


