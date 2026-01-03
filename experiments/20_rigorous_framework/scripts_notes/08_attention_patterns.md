# 08_attention_patterns.md

## Step 9: What Do Critical ToM Heads Attend To?

**Goal:** Understand WHAT the critical heads (L32H0, L33H4, L33H16, L33H28, L34H0) are computing by examining their attention patterns.

## BREAKTHROUGH FINDING: These are AGENT TRACKING Heads!

### Attention Distribution (averaged across 4 scenarios)

| Token Type | Attention Weight | Description |
|------------|------------------|-------------|
| **AGENTS** | **70.6%** | Alice, Bob, she, he |
| **Locations** | **17.5%** | drawer, basket, box |
| Belief verbs | 0.7% | thinks, knows, believes |
| Movement | 0.7% | moved, put, left |
| Negation | 0.0% | not, didn't, never |

### Visual Representation

```
AGENTS     ██████████████████████████████████████████████████████████████████████ 70.6%
LOCATIONS  █████████████████ 17.5%
belief     < 1%
movement   < 1%
negation   < 1%
```

## Per-Head Breakdown

| Head | Agents | Locations |
|------|--------|-----------|
| L32H0 | 60.6% | 33.2% |
| L33H4 | 62.9% | 2.5% |
| L33H16 | 78.5% | 18.2% |
| L33H28 | 82.5% | 14.8% |
| L34H0 | 68.4% | 18.7% |

**L33H28 is the most agent-focused head (82.5% on agents)!**

## Interpretation

### What This Means:

1. **NOT Belief Tracking**: The heads almost completely ignore belief verbs ("thinks", "knows"). They don't seem to be tracking the mental state vocabulary.

2. **AGENT IDENTITY Tracking**: They heavily attend to WHO the subject is. This makes sense for ToM - you need to know WHOSE belief you're tracking.

3. **Secondary LOCATION Focus**: ~17% attention to locations. This is the "where" component of "Where does X think the object is?"

4. **Ignores Linguistic Cues**: Movement words and negation get almost no attention. The heads aren't processing the narrative structure directly.

### Proposed Circuit Model

```
Early Layers:    Process narrative → encode "who did what where"
                           ↓
Critical Heads:  AGENT TRACKING → "We're predicting what ALICE thinks"
(L32-34)                ↓
                 Bind agent to belief state
                           ↓
Output Layers:   Generate location prediction based on tracked agent's knowledge
```

### Why ToM Works

The heads don't track beliefs directly - they track WHO we're asking about. The belief tracking happens elsewhere (possibly MLPs), but these attention heads ensure the answer is bound to the correct agent.

### Why Multi-Agent Wasn't Affected

Multi-agent scenarios with multiple agents may require:
- Tracking MULTIPLE agent identities simultaneously
- Comparing belief states between agents
- Higher-order reasoning ("Alice thinks Bob thinks...")

The single-agent tracking heads aren't sufficient for these tasks - the model needs additional circuitry.

## Implications for MATS

This is a strong mechanistic finding:

1. **ToM = Agent Tracking + Belief State Binding**
   - Attention heads track WHO
   - MLPs likely encode WHAT they believe

2. **Separate Circuits for Different Social Tasks**
   - Single-agent ToM: L32-34 attention heads
   - Multi-agent: TBD (different circuit)

3. **Not a "Theory of Mind Module"**
   - The model doesn't have a unified ToM system
   - It has specialized components that compose for ToM tasks

## Figures Generated

- `step9_attention_by_type.png` - Bar chart of attention by token type
- `step9_attention_heatmap.png` - Heatmap showing each head's attention profile

