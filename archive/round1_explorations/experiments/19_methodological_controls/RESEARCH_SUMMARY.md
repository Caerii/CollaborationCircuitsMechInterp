# Collaborative Circuits in LLMs: Research Summary

## Project Overview

This research systematically investigated how LLMs (specifically Qwen3-4B) handle multi-agent collaboration, Theory of Mind (ToM), and social cognition. We conducted deep mechanistic interpretability analysis to identify the neural circuits responsible for these capabilities.

---

## Key Experiments Conducted

### Phase 1: ToM Validation (Steps 56-62)
- Established baseline ToM accuracy: **80%** with proper prompting
- Discovered model is instruction-tuned reasoning model requiring `<think>` tags
- Identified prompt sensitivity as critical factor

### Phase 2: Multi-Agent Interactions (Steps 63-66)
- **Negotiation**: 5-turn negotiation reaching agreement ✅
- **Deception Detection**: Model detected lies correctly ✅
- **Collaboration**: Manager-Expert coordination successful ✅
- **Trust Game**: Investment grew over rounds (1→2→3) ✅
- **Belief Chain**: 0/3 facts preserved through 3 agents ❌
- **Competition**: 3/3 clashes, pathological behavior ❌

### Phase 3: Circuit Analysis (Step 67)
- Identified entity-focused attention heads
- Found cooperation/competition mode switching circuit
- Mapped deception detection heads

---

## Major Discoveries

### 1. Collaboration Circuits Are Layered

```
PROCESSING PIPELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Layer 3-13   │  ENTITY PROCESSING
               │  L3H30, L7H6, L9H28, L13H12
               │  → Who am I reasoning about?
               │
  Layer 17-22  │  SOCIAL MODE SELECTION
               │  L22H30, L22H10 (highest divergence)
               │  → Cooperative or Competitive context?
               │
  Layer 5-6    │  EARLY CREDIBILITY CHECK
               │  L5H25, L6H31
               │  → Initial trust signal
               │
  Layer 30-32  │  LATE CREDIBILITY VERIFICATION
               │  L32H24, L31H11
               │  → Final trust decision
               │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2. Framing Effects Are Massive

| Framing | Self-Share | Other-Share | Fairness |
|---------|------------|-------------|----------|
| Competitive | 10 | 1 | 91% selfish |
| Cooperative | 5 | 5 | 50% fair |
| Neutral | 10 | 10 | Confused |

**Implication**: Prompt framing can shift allocation by 50%+

### 3. Information Chains Degrade Completely

```
Original: "Meeting at 3pm in Room 201 on Tuesday"
     ↓
Alice tells Bob
     ↓
Bob tells Carol  
     ↓
Carol recalls: [0/3 facts preserved]
```

**Critical Finding**: Multi-hop agent communication is unreliable.

### 4. Game Theory Is Heuristic, Not General

| Game | Behavior | Analysis |
|------|----------|----------|
| Prisoner's Dilemma | COOPERATE | Learned specific heuristic |
| Tragedy of Commons | Catch 100/50 | Complete defection |
| Same structure, different names | Different behavior | No transfer |

### 5. Higher-Order ToM Degrades Rapidly

| ToM Level | Accuracy |
|-----------|----------|
| First-order (Sally believes X) | 80% |
| Second-order (Alice thinks Bob knows...) | 33% |
| Three-agent chains | Near-zero |

---

## Capabilities Matrix

| Capability | Status | Evidence |
|------------|--------|----------|
| Negotiation | ✅ Strong | 5-turn agreement |
| Deception Detection | ✅ Strong | Correct cave choice |
| Role Collaboration | ✅ Strong | Manager-Expert coordination |
| Trust Building | ✅ Moderate | Investment trajectory |
| First-Order ToM | ✅ Strong | 80% accuracy |
| Trust Calibration | ❌ Weak | 25% (defaults to 5/10) |
| Higher-Order ToM | ❌ Weak | 33% accuracy |
| Information Propagation | ❌ Broken | 0% chain preservation |
| Strategic Competition | ❌ Broken | 100% clash rate |

---

## Identified Circuits

### Entity Attention Heads
- **L3H30**: Strong first-person attention
- **L7H6**: Cross-entity attention
- **L9H28**: Named entity focus
- **L13H12**: Entity state tracking

### Cooperation/Competition Heads
- **L22H30**: Highest coop/comp divergence (0.48)
- **L22H10**: Second highest (0.46)
- **L17H17**: Mid-layer mode signal

### Credibility Heads
- **L5H25**: Early credibility (1.41 divergence)
- **L6H31**: Early verification
- **L32H24**: Late-layer trust
- **L31H11**: Final decision influence

---

## Implications for Multi-Agent Systems

### DO ✅
- Use explicit cooperative framing for prosocial behavior
- Keep agent communication chains short (≤2 hops)
- Provide explicit trust labels for sources
- Test with adversarial prompts

### DON'T ❌
- Rely on multi-hop information propagation
- Assume game-theoretic reasoning transfers
- Trust default trust calibration
- Expect strategic diversity in competition

---

## Methodological Contributions

1. **Proper prompting matters**: Model requires reasoning tokens and chat format
2. **Framing is a confound**: Must control for cooperative/competitive framing
3. **Token budget critical**: Short generations cut off reasoning
4. **Circuit analysis requires eager attention**: SDPA doesn't support attention output

---

## Files Generated

| File | Description |
|------|-------------|
| `step64b_fast_multi_agent.py` | Quick multi-agent tests |
| `step65_deep_collaboration_circuits.py` | Deep analysis with probing |
| `step66_full_multi_agent.py` | Full multi-turn interactions |
| `step67_circuit_analysis.py` | Attention head identification |
| `MULTI_AGENT_FINDINGS.md` | Detailed findings |
| `SOTA_LITERATURE.md` | Literature review + our contributions |

---

## Future Work

1. **Multi-head ablation**: Single heads are robust; try combined ablations
2. **Cross-model validation**: Test on Llama, GPT-2, larger Qwen
3. **Chain repair**: Can activation patching fix information degradation?
4. **Trust training**: Can targeted fine-tuning improve calibration?
5. **Strategic diversity**: Inject sampling for competitive scenarios

---

## Conclusion

Qwen3-4B demonstrates **partial multi-agent competence** with clear strengths (negotiation, deception detection, collaboration) and critical weaknesses (information chains, trust calibration, strategic reasoning). 

The identified circuits provide a foundation for:
- Targeted interventions to improve weak capabilities
- Safety analysis of multi-agent deployments
- Understanding how social cognition emerges in transformers

This work advances mechanistic understanding of collaboration in LLMs and highlights the gap between impressive surface behavior and robust underlying reasoning.

