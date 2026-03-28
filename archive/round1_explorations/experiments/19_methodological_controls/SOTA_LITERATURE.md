# State of the Art: Mechanistic Interpretability of Reasoning Models

## Overview

This document synthesizes current research on mechanistic interpretability (MI) of reasoning models, multi-agent LLM systems, and collaborative circuits.

---

## Key Papers and Findings

### 1. Mechanistic Interpretation of Multi-Step Reasoning (EMNLP 2023)

**Paper**: "Towards a Mechanistic Interpretation of Multi-Step Reasoning Capabilities of Language Models"

**Key Contribution**: Introduces **MechanisticProbe** - a method that detects reasoning tree information from model attentions.

**Findings**:
- LLMs undergo genuine multi-step reasoning within their architecture
- Attention patterns encode reasoning tree structure
- Can decode which reasoning steps the model is executing from attention alone

**Relevance to Our Work**: We should probe attention patterns during collaborative reasoning to see if similar reasoning trees emerge for multi-agent belief tracking.

---

### 2. Multi-Agent LLM Coordination (arXiv 2407.12532)

**Paper**: "Towards Collaborative Intelligence: Propagating Intentions and Reasoning for Multi-Agent Coordination"

**Key Contribution**: Framework for training LLMs as collaborative agents in cooperative MARL.

**Key Ideas**:
- **Intention Broadcasting**: Agents broadcast their intentions
- **Others can infer coordination tasks** from broadcasted intentions
- Reduces miscoordination errors
- Fosters emergent coordinated behaviors

**Relevance**: This suggests models might have circuits for:
- Intention encoding
- Intention decoding from other agents
- Coordination planning

---

### 3. Emergent Collaboration in LLM Agents (arXiv 2310.10701)

**Paper**: LLM agents in cooperative text games

**Findings**:
- LLM-based agents outperform traditional MARL and planning approaches
- Exhibit **emergent collaborative behaviors**
- Show **higher-order Theory of Mind** capabilities
- Challenges: Long-horizon context management, hallucinations about task states

**Relevance**: Confirms LLMs can collaborate, but questions remain about mechanism.

---

### 4. AutoGen/CrewAI Frameworks (IEEE 2025)

**Framework Analysis**: Multi-Agent Systems Meet Large Language Models

**Key Ideas**:
- Agents assume specific **roles**
- Engage in structured **dialogue**
- Provide **feedback** to each other
- Utilize **external tools** collectively
- Cognition meets organization

**Relevance**: Role specialization might be encoded in model activations.

---

## Critical Gaps in Current Literature

### Gap 1: No Mechanistic Account of Multi-Agent Representations
- How does a single model represent MULTIPLE agents?
- Is Self/Other/User linearly separable in activation space?
- What circuits handle agent-agent vs agent-user relationships?

### Gap 2: No Circuit-Level Analysis of Collaboration
- Which attention heads track beliefs of Agent A vs Agent B?
- Are there "cooperation heads" vs "competition heads"?
- What happens when agents have conflicting goals?

### Gap 3: Deception Detection Mechanisms Unknown
- How does a model detect lies?
- Is there a "trustworthiness detector" circuit?
- What features indicate unreliable information sources?

### Gap 4: Game Theory & Rational Choice
- How do models represent payoff structures?
- Do they reason about Nash equilibria?
- Why did our Tragedy of Commons show pure defection?

---

## Our Empirical Contributions

### Finding 1: Multi-Turn Agent Interactions Work
- **Negotiation**: Agents reached agreement over 5 turns
- **Collaboration**: Manager-Expert role coordination successful
- **Trust Game**: Investment increased (1→2→3), showing trust building

### Finding 2: Deception Detection Works
- Model (as Dan) was skeptical of Eve's lie and chose CAVE correctly
- **Circuit**: Deception heads identified at L5H25, L6H31, L32H24
- Two-stage processing: early detection + late verification

### Finding 3: Critical Failure - Information Chain Degradation
- 3-agent belief chain: 0/3 key facts preserved
- Original: "Meeting at 3pm in Room 201 on Tuesday"
- After Alice→Bob→Carol: Complete information loss
- **Implication**: Multi-hop agent chains are unreliable

### Finding 4: Game-Specific Heuristics, Not General Reasoning
- **Prisoner's Dilemma**: COOPERATE (prosocial)
- **Tragedy of Commons**: Catch 100 fish (maximum defection, limit was 50)
- Model doesn't transfer reasoning across structurally similar games

### Finding 5: Massive Framing Effects
- Competitive frame: 10-1 split (took 91%)
- Cooperative frame: 5-5 split (perfectly fair)
- **Framing effect**: 50% allocation difference based on framing alone
- **Circuit**: L22H30, L22H10 show highest coop/comp divergence

### Finding 6: Trust Calibration Fails
- Reliable source: 5/10 trust
- Unreliable source: 5/10 trust
- Expert source: 5/10 trust
- Model defaults to neutral regardless of source characteristics

### Finding 7: Higher-Order ToM Weak
- First-order ToM: 80% (from step 62)
- Second-order ToM: 33% (nested beliefs)
- Significant degradation with belief complexity

### Finding 8: Pathological Competition
- Territory game: 3/3 clashes (both chose North every time)
- No strategic diversity or opponent modeling

---

## Identified Circuits (Our Novel Contribution)

### Entity Processing Circuit (Layers 3-13)
- **Key heads**: L3H30, L7H6, L9H28, L13H12, L11H9
- **Function**: Attends strongly to entity words (I, you, Alice, Bob)
- **Role**: Self/Other/User distinction

### Social Mode Circuit (Layers 17-22)
- **Key heads**: L22H30 (0.48 divergence), L22H10 (0.46), L22H5 (0.33)
- **Most divergent layer**: Layer 22
- **Function**: Determines cooperative vs competitive context

### Credibility Assessment Circuit (Two-Stage)
**Stage 1 - Early Detection (Layers 5-6)**:
- L5H25 (1.41 divergence), L6H31 (1.40)
- Initial credibility signal

**Stage 2 - Late Verification (Layers 30-32)**:
- L32H24 (1.40), L31H11, L31H9
- Final trust decision

### Proposed Processing Pipeline
```
Input → Entity ID (L3-13) → Social Mode (L17-22) → Credibility (L5-6, L30-32) → Decision
```

---

## Research Directions

### Direction 1: Linear Probing for Agent Representations
Train linear classifiers on activations to decode:
- Which agent is currently being reasoned about
- What that agent believes
- Trust level toward that agent
- Cooperative vs competitive intent

### Direction 2: Attention Ablation for Collaboration
Identify and ablate attention heads that:
- Track agent beliefs
- Encode cooperation/competition
- Assess information credibility
- Represent collective vs individual goals

### Direction 3: Path Patching for Deception Detection
Trace the causal path from:
- Deceptive statement → Suspicion activation → Correct decision
To identify the "lie detection circuit"

### Direction 4: Game Theory Representations
Investigate how models encode:
- Payoff matrices (PD vs Commons)
- Player identities
- Strategy space
- Equilibrium concepts

---

## Methodology Best Practices (from Literature)

1. **MechanisticProbe approach**: Use attention patterns to infer reasoning structure
2. **Causal interventions**: Ablate components and measure behavioral changes
3. **Linear probing**: Train decoders on activations to extract information
4. **Cross-model validation**: Test findings across model families and sizes
5. **Statistical rigor**: n≥50 trials, proper significance tests
6. **Control conditions**: Match syntax, vary only key variables

---

## Next Steps for Our Research

1. **Increase token budget** for multi-agent ToM (need 200+ tokens for reasoning)
2. **Run linear probes** on agent identity during multi-agent scenarios
3. **Ablate attention heads** during collaboration tasks
4. **Compare PD vs Commons** at circuit level to explain cooperation difference
5. **Trace deception detection** through layers

---

## References

1. EMNLP 2023 - MechanisticProbe for reasoning trees
2. arXiv 2407.12532 - Collaborative Intelligence framework
3. arXiv 2310.10701 - Emergent collaboration in LLM agents
4. IEEE 2025 - Multi-Agent Systems meets LLMs
5. Max Planck - Mechanistic Interpretability overview

