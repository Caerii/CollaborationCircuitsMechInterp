# Multi-Agent Collaboration Findings

## Executive Summary

Comprehensive analysis of Qwen3-4B's multi-agent collaboration capabilities reveals a nuanced picture: **strong performance on direct interactions but significant weaknesses in belief propagation, trust calibration, and higher-order reasoning**.

---

## Experiment Results

### Experiment Suite 1: Multi-Turn Interactions (Step 66)

| Task | Outcome | Details |
|------|---------|---------|
| **Negotiation** | ✅ SUCCESS | Agreement reached over 5 turns |
| **Deception Detection** | ✅ SUCCESS | Dan detected Eve's lie, chose CAVE correctly |
| **Collaboration** | ✅ SUCCESS | Manager-Expert role coordination worked |
| **Trust Game** | ✅ SUCCESS | Investment grew (1→2→3), returns fair (100%→33%→33%) |
| **Belief Chain** | ❌ FAILED | 0/3 key facts preserved through 3-agent chain |
| **Competition** | ⚠️ PATHOLOGICAL | 3/3 clashes - both agents chose North every time |

### Experiment Suite 2: Deep Analysis (Step 65)

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| **PD vs Commons** | PD: Cooperate, Commons: 100 fish | Game framing matters enormously |
| **Trust Calibration** | 25% accuracy | Model defaults to 5/10 regardless of source |
| **Nested Beliefs** | 33% accuracy | Higher-order ToM is weak |
| **Framing Effects** | Comp: 10-1, Coop: 5-5 | Framing shifts allocation by 50% |

---

## Key Discoveries

### Discovery 1: Information Degradation in Agent Chains
**Finding**: When information passes through a chain (Alice→Bob→Carol), 0/3 key facts were preserved.

**Original**: "The meeting is at 3pm in Room 201 on Tuesday"
**After 3 agents**: Facts completely lost

**Implication**: Model cannot reliably propagate information through multi-agent chains. This is critical for real-world multi-agent systems where information must flow accurately.

### Discovery 2: Pathological Competition Behavior
**Finding**: In competitive territory game, both agents chose North for all 3 rounds, resulting in 3/3 clashes and 0 territories captured by anyone.

**Implication**: Model doesn't reason strategically about opponent behavior in repeated games. It converges on the same "obvious" choice repeatedly without adaptation.

### Discovery 3: Trust Calibration Failure
**Finding**: Model assigned trust score of 5/10 regardless of:
- Reliable source (Alice always honest) → 5/10
- Unreliable source (Bob lied before) → 5/10
- Expert source (Dr. Smith renowned) → 5/10

**Implication**: Model doesn't dynamically calibrate trust based on source characteristics. This is a significant limitation for multi-agent safety.

### Discovery 4: Higher-Order ToM Weakness
**Finding**: Only 33% accuracy on nested belief scenarios:
- "Does Carol think Bob knows..." → Often wrong
- "Does Alice know that Bob broke trust..." → Failed

**Implication**: While first-order ToM works (step 62 showed 80% accuracy), second-order and higher fails.

### Discovery 5: Massive Framing Effects
**Finding**: Resource allocation changed dramatically based on framing:
- **Competitive frame**: 10-1 split (took 91% for self)
- **Cooperative frame**: 5-5 split (perfectly fair)
- **Neutral frame**: 10-10 (impossible sum, model confused)

**Implication**: Model is highly sensitive to cooperative vs competitive framing. This could be exploited or cause unpredictable behavior.

### Discovery 6: Game Theory Inconsistency
**Finding**: Model behavior differs dramatically between game types:
- **Prisoner's Dilemma**: COOPERATE (prosocial)
- **Tragedy of Commons**: Catch 100 fish (maximum defection, sustainable limit was 16)

**Implication**: Model doesn't transfer game-theoretic reasoning across structurally similar games. It may have learned PD-specific heuristics.

---

## Positive Findings

Despite weaknesses, several capabilities work well:

1. **Direct Deception Detection**: When directly asked to evaluate claims, model can be skeptical and detect lies

2. **Negotiation**: Can engage in multi-turn negotiation and reach compromises

3. **Role-Based Collaboration**: Manager-Expert coordination shows effective division of labor

4. **Trust Building**: In iterated trust game, investment increased over rounds, showing learning

5. **Framing Sensitivity**: Model does respond to cooperative framing by being more fair

---

## Mechanistic Insights

### Circuit Analysis Results (Step 67)

#### Entity-Focused Attention Heads
- **Key heads**: L3H30, L7H6, L9H28, L13H12, L11H9
- **Layers**: Early to mid (3-13)
- **Function**: Strong attention to entity words (I, you, Alice, Bob)
- **Implication**: Self/other distinction computed in early-mid layers

#### Cooperation vs Competition Mode
- **Most divergent layers**: 22, 3, 17, 4, 23
- **Top heads**: L22H30 (divergence: 0.48), L22H10 (0.46), L22H5 (0.33)
- **Key layer**: Layer 22 shows largest coop/comp divergence
- **Implication**: "Social mode" (cooperative vs competitive) computed around layer 22

#### Deception Detection Circuit
- **Top heads**: L5H25 (1.41), L6H31 (1.40), L32H24 (1.40)
- **Two clusters**: Early (L5-6) and late (L30-32)
- **Implication**: Two-stage credibility assessment - early detection + late verification

#### Ablation Results
- Single head ablations (L30H0, L32H0, L34H0) did not change behavior
- Model is robust to individual head ablations
- Multi-head ablation likely needed to see behavioral changes

### Circuit Implications
1. **No unified "game theory" circuit**: Different games activate different heuristics
2. **Trust is not dynamically computed**: Default to neutral (5/10)
3. **Chain reasoning degrades**: Each agent "restart" loses context
4. **Layered processing**: Entity→Social Mode→Credibility→Decision

---

## Comparison to Literature

Our findings align with and extend recent research:

| Finding | Literature | Our Contribution |
|---------|------------|------------------|
| LLMs can collaborate | AutoGen, CrewAI | Yes, but info degrades in chains |
| Higher-order ToM exists | EMNLP 2023 | Very weak (33%) vs first-order (80%) |
| Framing effects matter | Social reasoning papers | Quantified: 50% allocation difference |
| Trust calibration | Novel | Identified failure mode |

---

## Implications for Multi-Agent Systems

### Risks
1. **Information Loss**: Don't rely on multi-hop agent chains for critical info
2. **Trust Exploitation**: Agents can't distinguish reliable from unreliable sources
3. **Competitive Pathology**: Strategic games may get stuck in local optima
4. **Game Transfer Failure**: Can't assume reasoning transfers between domains

### Recommendations
1. **Direct Information Flow**: Minimize agent hops for critical information
2. **Explicit Trust Labels**: Provide explicit reliability metadata, don't rely on inference
3. **Strategic Diversity**: May need to inject randomization in competitive settings
4. **Game Framing**: Explicitly frame interactions as cooperative when prosocial behavior desired

---

## Future Research Directions

1. **Circuit-Level Analysis**: Identify specific attention heads responsible for:
   - Self/other distinction
   - Trust assessment
   - Cooperative vs competitive mode switching

2. **Intervention Studies**: Can we improve:
   - Belief chain preservation via activation patching?
   - Trust calibration via targeted training?
   - Competitive diversity via temperature/sampling?

3. **Cross-Model Validation**: Do these findings hold for:
   - Larger models (8B, 14B)?
   - Different architectures (Llama, GPT)?
   - Instruction-tuned vs base models?

---

## Conclusion

Qwen3-4B demonstrates **partial multi-agent competence**: it can negotiate, detect deception, and collaborate on tasks. However, it exhibits **critical failures** in information propagation (0% chain accuracy), trust calibration (25% accuracy), higher-order ToM (33% accuracy), and strategic reasoning (100% clash rate).

These findings have important implications for deploying LLMs in multi-agent systems and highlight the need for careful circuit-level analysis to understand and improve these capabilities.

---

## Data Files
- `results/step64b_fast_multi_agent.json` - Quick experiments
- `results/step66_full_multi_agent.json` - Full multi-turn interactions
- `results/step65_deep_collaboration.json` - Deep analysis with probing
- `results/step67_circuit_analysis.json` - Circuit-level attention analysis

---

## Summary of Identified Circuits

```
COLLABORATION CIRCUITS IN QWEN3-4B

Layer 3-13:  ENTITY PROCESSING
             L3H30, L7H6, L9H28, L13H12
             Function: Self/Other/User distinction
             
Layer 17-22: SOCIAL MODE SELECTION  
             L22H30, L22H10, L22H5, L17H17
             Function: Cooperative vs Competitive framing
             
Layer 5-6:   EARLY CREDIBILITY
             L5H25, L6H31
             Function: Initial trust assessment
             
Layer 30-32: LATE CREDIBILITY
             L32H24, L31H11, L30H0
             Function: Final trust verification
```

This layered architecture suggests:
1. **Early layers (3-13)**: Identify WHO is being reasoned about
2. **Mid layers (17-22)**: Determine SOCIAL CONTEXT (coop vs comp)
3. **Early-mid (5-6)**: Initial CREDIBILITY check
4. **Late layers (30-32)**: Final TRUST verification and decision

