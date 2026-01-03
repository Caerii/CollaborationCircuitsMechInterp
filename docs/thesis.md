# Research Thesis

## Core Thesis: The Biology of Multi-Agent Collaboration

**LLMs have internal "organs" for social cognition: user-modeling circuits, agent-modeling circuits (ToM), and evaluation-awareness representations. In multi-agent settings, these organs must coordinate. We study where these representations live, how they interact (or interfere), what pathologies arise (groupthink, collusion, manipulation), and how to diagnose and fix them via targeted interventions.**

### Model Biology Framing

Just as single-model interpretability asks "what circuits compute X?", multi-agent interpretability asks:
- What circuits enable Agent A to model Agent B?
- How do user representations interact with agent representations?
- What happens when these "social organs" malfunction?

This is **pragmatic interpretability** because multi-agent systems are being deployed NOW, and understanding their internal social cognition is directly safety-relevant.

## Supporting Arguments

### 1. Representations Exist and Are Separable

- **Evidence**: 
  - Agent beliefs are linearly decodable from activations ([Zhu et al. 2024](https://arxiv.org/abs/2402.18496))
  - Models form distinct internal models of self and others
  - Explicit belief representations improve multi-agent task success ([Li et al. 2023](https://aclanthology.org/2023.emnlp-main.13/))
- **Implication**: These aren't just abstract concepts—they're concrete activation patterns we can locate and manipulate

### 2. Representations Are Causal

- **Evidence**: Steering user/agent representations changes behavior, ablating ToM circuits breaks belief tracking, evaluation-awareness steering affects deception
- **Implication**: These aren't just epiphenomena—they're part of the causal mechanism

### 3. Interference Causes Failures

- **Hypothesis**: When representations interfere (user model contaminated by agent model, or vice versa), collaboration degrades
- **Test**: Measure correlation between representation interference and collaboration failures; show causal link via targeted interventions

### 4. Mechanistic Knowledge Enables Intervention

- **Evidence**: 
  - "Misalignment vectors" can be identified and ablated without reducing performance (Soligo et al. 2025)
  - Tracing attention heads mediating inter-agent communication pinpoints toxic agreement origins ([Lee et al. 2025](https://arxiv.org/abs/2512.04691))
  - Conditional steering reduces side effects
- **Implication**: Understanding mechanisms gives us control

## Why This Matters

1. **Safety**: Detect manipulation, collusion, and value drift by monitoring representations
2. **Alignment**: Understand when and why alignment breaks in multi-agent contexts
3. **Control**: Design interventions (steering, monitoring, circuit modification) based on mechanistic understanding
4. **Prediction**: Use representation analysis to predict collaboration outcomes before they occur

## Scope and Limitations

**In scope**:
- Multi-agent LLM systems (2-5 agents)
- Dialogue-based collaboration tasks
- Mechanistic analysis using existing interpretability tools
- Safety-relevant behaviors (deception, collusion, groupthink, value drift)

**Out of scope (for now)**:
- Large-scale agent swarms (100+ agents)
- Non-LLM agents or hybrid systems
- Training-time interventions (focus on inference-time analysis)
- Full mathematical theory of multi-agent representations

## Success Criteria

A successful research program would demonstrate:

1. **Identification**: Can locate and characterize user/agent/context representations in multi-agent systems
2. **Separation**: Can show these representations are separable and don't always interfere
3. **Causality**: Can show representation interference causally affects collaboration
4. **Intervention**: Can use mechanistic knowledge to improve collaboration or prevent failures
5. **Generalization**: Findings hold across different models, tasks, and multi-agent scenarios

## Novel Contributions

- First systematic mechanistic study of multi-agent collaboration circuits
- Extension of representation engineering to multi-agent contexts
- Connection between representation interference and collaboration failures
- Practical tools for monitoring/intervening in multi-agent systems based on mechanistic understanding

---

## Key Supporting Literature

| Finding | Paper |
|---------|-------|
| LLMs encode belief states of self and others | [Zhu et al. 2024](https://arxiv.org/abs/2402.18496) |
| Multi-agent ToM emerges and improves with explicit representations | [Li et al. 2023](https://aclanthology.org/2023.emnlp-main.13/) |
| Prompt design creates coordinated collectives | [Emergent Coordination 2025](https://arxiv.org/abs/2510.05174) |
| Deception is an emergent LLM capability | [Hagendorff 2023](https://arxiv.org/abs/2307.16513) |
| LLMs can collude covertly | [Za et al. 2025](https://openreview.net/attachment?id=CdZaamCf5Y&name=pdf), [Mathew et al. 2024](https://arxiv.org/abs/2410.03768) |
| Research agenda for multi-agent mech interp | [Lee et al. 2025](https://arxiv.org/abs/2512.04691) |

*See [references.md](references.md) for full citations.*

