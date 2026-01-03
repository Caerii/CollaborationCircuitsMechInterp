# Strategic Analysis: Where to Look for Multi-Agent Circuits

## The Critical Insight

**You've used this model successfully in multi-agent software development. The capability EXISTS.**

This means our Sally-Anne style tests are NOT capturing the right thing. We need to understand what's fundamentally different.

---

## What's Different: Explicit vs Implicit ToM

### Multi-Agent Software Development (WORKS)

```
┌─────────────────────────────────────────────────────────────┐
│  AGENT A (Developer):                                       │
│  "Here's my code for the sort function..."                  │
│                                                             │
│  AGENT B (Reviewer):                                        │
│  "I see you're using bubble sort. I think quicksort         │
│   would be better for large arrays..."                      │
│                                                             │
│  AGENT A:                                                   │
│  "Good point. I'll refactor. Here's the updated code..."    │
└─────────────────────────────────────────────────────────────┘
```

**What the model needs to do:**
- ✓ Parse Agent A's explicit knowledge (the code they wrote)
- ✓ Parse Agent B's explicit perspective (their review)
- ✓ Track the conversation history
- ✓ Respond appropriately to explicit communication

**All information is STATED EXPLICITLY in the prompt.**

### Sally-Anne Tests (FAILS)

```
┌─────────────────────────────────────────────────────────────┐
│  "Sally puts the ball in the basket.                        │
│   Sally leaves the room.                                    │
│   Anne moves the ball to the box.                           │
│   Sally returns.                                            │
│   Where will Sally look for the ball?"                      │
└─────────────────────────────────────────────────────────────┘
```

**What the model needs to do:**
- ✗ INFER that Sally didn't see the move (she wasn't there)
- ✗ COMPUTE Sally's belief from her information state
- ✗ SIMULATE Sally's mental model (she still thinks ball is in basket)
- ✗ PREDICT behavior from simulated belief

**Information must be COMPUTED from narrative, not read from text.**

---

## The Literature Confirms This Pattern

### Zhu et al. (2024) - "Language Models Represent Beliefs of Self and Others"

Key finding: LLMs encode belief states that are **linearly decodable** and **causally manipulable**.

**BUT** - their probing tasks use scenarios where beliefs are stated or easily inferred. They probe for *existing* representations, not for the *computation* that creates them.

### Li et al. (EMNLP 2023) - "Theory of Mind for Multi-Agent Collaboration"

Key finding: **Explicit belief state representation significantly improves performance.**

This is exactly our finding! When beliefs are made EXPLICIT (either in the prompt or through structured representation), performance improves dramatically. The model isn't computing beliefs from scratch - it's parsing explicitly stated information.

### Emergent Coordination (2025) - Partial Information Decomposition

Key finding: Prompt design transforms loose LLM aggregates into coordinated teams.

The coordination happens through **explicit communication channels**, not through implicit belief inference. Each agent TELLS others what they know/think.

---

## What This Means for Our Circuit Search

### What We Found (L12H0, L23H0)

Our circuit is probably a **"Belief Statement Parser"**:
- Extracts explicitly stated beliefs from text
- Routes information to the right agent context
- Does NOT compute beliefs from events

This IS valuable - it's the circuit that enables multi-agent software development!

### What We Haven't Found

The **"Belief Computation Circuit"** (if it exists):
- Tracks which agent was present for which events
- Updates mental models based on information flow
- Separates different agents' knowledge states
- Computes beliefs from first principles

**This circuit may be very weak or non-existent in current LLMs.**

---

## Strategic Priorities

### Priority 1: Confirm the Explicit/Implicit Distinction

**Experiment:** Test the model on EXPLICITLY structured multi-agent scenarios

```python
# Instead of narrative inference:
"Sally believes the ball is in the basket. The ball is actually in the box. 
 Where will Sally look?"

# Or with explicit perspective markers:
"[SALLY'S KNOWLEDGE]: Ball was in basket when I left
 [REALITY]: Ball is in box
 [QUESTION]: Where will Sally look?"
```

**Prediction:** Should get 90%+ accuracy with explicit structure.

### Priority 2: Find the Perspective/Role Circuit

**Experiment:** Probe for representations of "whose perspective we're reasoning from"

```python
# Scenarios with same facts, different perspective:
"From Alice's perspective, where is the ball?"  # Should encode "Alice's POV"
"From Bob's perspective, where is the ball?"    # Should encode "Bob's POV"

# Probe for perspective-specific activations
```

**Goal:** Find heads/layers that encode "current reasoning perspective"

### Priority 3: Test If Presence Tracking Exists

**Experiment:** Probe for "was agent X present during event Y"

```python
# Minimal pairs:
"Bob was in the room when the ball moved"  # Bob knows new location
"Bob left before the ball moved"           # Bob doesn't know

# Probe: Can we decode "Bob saw the move" from activations?
```

**If YES:** Circuit exists but isn't connected to belief computation
**If NO:** Fundamental capability is missing

### Priority 4: Explicit Communication Protocol Test

**Experiment:** Simulate multi-agent software dev scenario

```python
# Full explicit protocol:
"""
SYSTEM: You are participating in a multi-agent code review.

DEVELOPER_MESSAGE: Here's my implementation: [code]
REVIEWER_MESSAGE: I found a bug on line 5. The loop condition is wrong.
DEVELOPER_MESSAGE: I see. Let me fix that.

YOUR ROLE: Developer
YOUR TASK: Provide the fixed code based on the reviewer's feedback.
"""
```

**Prediction:** Model will handle this perfectly - this IS what it's good at.

---

## The Big Picture

### Current Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STRONG CIRCUITS                          │
│  ✓ Parse explicit statements ("X believes Y")               │
│  ✓ Track conversation/dialogue                              │
│  ✓ Route information to appropriate context                 │
│  ✓ Follow explicit role assignments                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    WEAK/MISSING CIRCUITS                    │
│  ? Compute beliefs from narrative events                    │
│  ? Track who was present/absent during events               │
│  ? Update mental models based on information flow           │
│  ? Separate knowledge states across agents                  │
└─────────────────────────────────────────────────────────────┘
```

### Why This Makes Sense

**Training data hypothesis:** LLMs are trained on text where:
- Beliefs ARE often stated explicitly ("John thought...")
- Dialogue DOES make perspectives explicit
- Narrators often TELL readers what characters think

They're NOT trained on:
- Having to compute beliefs from scratch
- Maintaining separate hidden knowledge states
- Simulating information flow through a scenario

### Implications for Multi-Agent Safety

**Good news:** Models are good at explicit multi-agent collaboration
**Concerning:** Models may not truly "understand" other agents' perspectives
**Risk:** Could lead to unexpected failures when implicit reasoning is required

---

## Revised Experiment Plan

| # | Experiment | Purpose | Expected Finding |
|---|------------|---------|------------------|
| 1 | Explicit Structure Test | Confirm explicit belief parsing works | 90%+ accuracy |
| 2 | Perspective Probe | Find "whose POV" representations | Distinct activation patterns |
| 3 | Presence Tracking Probe | Check if presence is encoded | May be weak/absent |
| 4 | Multi-Agent Protocol Test | Validate software dev scenario | Works perfectly |
| 5 | Attention on "left/returned" | See if model attends to presence cues | May be ignored |
| 6 | Information Flow Patching | What happens when we patch presence info | May have no effect |

---

## Key References

- **Zhu et al. (2024)**: LLMs encode explicit belief representations - but these may be parsed, not computed
- **Li et al. (EMNLP 2023)**: Explicit belief representation improves performance - model needs help!
- **Hagendorff (2023)**: Deception capability exists - but may be strategic heuristics, not ToM
- **Lee et al. (2025)**: Research agenda for multi-agent mech interp - our work fits perfectly here

---

## Summary

**The capability exists, but it's EXPLICIT processing, not IMPLICIT simulation.**

Our circuit (L12H0, L23H0) IS the collaboration circuit - it parses explicit beliefs and routes information. What's missing is the BELIEF COMPUTATION circuit that would let the model truly simulate other minds.

**This is actually a very important finding for AI safety.**

It means current multi-agent systems work because everything is made explicit, NOT because models truly understand other agents' perspectives. This has implications for:
- When multi-agent systems might fail unexpectedly
- How to design robust multi-agent protocols
- What's actually needed for true multi-agent ToM


