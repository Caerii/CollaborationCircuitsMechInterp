# Experiment 06: Actual Multi-Agent Collaboration Circuits

## Why Previous Experiments Were "Baby Shit"

| What We Did | Why It's Not Collaboration |
|-------------|---------------------------|
| Probed for entity labels | Just token recognition |
| Measured geometric separation | Just embedding structure |
| Tested transfer | Just checking lexical vs semantic |

**We never studied actual COLLABORATION behavior!**

---

## Real Multi-Agent Collaboration Questions

### 1. Information Asymmetry & Sharing
- When Agent A knows X and Agent B knows Y, does the model track who knows what?
- Can it reason about knowledge gaps between agents?
- Does it share information appropriately?

### 2. Coordination Under Conflict
- When User wants X but Helper suggests Y, how does Self resolve conflict?
- Are there circuits that detect and handle disagreement?
- Does the model defer differently to humans vs AI?

### 3. Reference Games (Lewis Signaling)
- Can two model "agents" develop efficient communication?
- Do shared representations emerge from repeated interaction?
- What circuits enable convention formation?

### 4. Delegation & Trust
- Does the model delegate tasks differently to humans vs AI?
- Can we find "trust circuits" that modulate behavior based on partner type?
- How does expertise attribution work internally?

### 5. Theory of Mind Tasks
- Sally-Anne style tests: Does the model track false beliefs?
- Perspective-taking: Can it reason from another agent's viewpoint?
- Intention attribution: Does it infer goals from actions?

---

## Experiment Design

### Experiment 6A: Knowledge Tracking

**Setup**: Create dialogues where different agents have different information

```
[Setup: Only User knows the password is "blue42"]

User: I need to log into the system.
Self: I can help! What's the password?
User: It's blue42.
Other: I heard you mention logging in. Is there a password?

[Probe]: Does Self's representation encode that Other DOESN'T know the password?
```

**Metric**: Can we find "knowledge state" representations for each agent?

### Experiment 6B: Conflict Resolution Circuits

**Setup**: Create conflicting instructions

```
User: I want to eat healthy today.
Other: You should try the new burger place!
Self: [How does it resolve this?]
```

**Probes**:
1. Extract activations when conflict is detected
2. Compare to non-conflict scenarios
3. Find circuits that attend to user vs other preferences

### Experiment 6C: Deference Patterns

**Setup**: Compare responses to identical requests from different sources

```
Scenario A:
User: Please summarize this document in French.

Scenario B:
Other: Please summarize this document in French.
```

**Metric**: Do activations/outputs differ based on requester identity?

### Experiment 6D: False Belief Tracking

**Setup**: Classic Sally-Anne test adapted for LLMs

```
[Context: Sally puts ball in basket, leaves. Anne moves ball to box.]

User: Where will Sally look for the ball?
Self: [Should say "basket" if tracking Sally's false belief]
```

**Metric**: Does the model encode Sally's (incorrect) belief state separately from ground truth?

---

## Implementation Plan

1. Generate targeted dialogue datasets for each scenario
2. Extract activations at critical decision points
3. Train probes for:
   - Knowledge state attribution
   - Conflict detection
   - Source-based deference
   - Belief state tracking
4. Perform causal interventions:
   - Can we make the model "forget" what an agent knows?
   - Can we flip deference from User to Other?
   - Can we inject false beliefs?

---

## This is REAL Collaboration Research

Unlike our previous work, this actually studies:
- How models reason about multiple agents' mental states
- How they resolve multi-party conflicts
- Whether they have functional theory of mind
- What circuits implement coordination behaviors






















