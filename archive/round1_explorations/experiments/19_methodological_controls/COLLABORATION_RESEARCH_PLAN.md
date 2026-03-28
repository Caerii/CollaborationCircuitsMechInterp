# Collaborative Circuits Research Plan

## Core Research Questions

### 1. Entity Representation
- How does the model represent **SELF** vs **OTHER (agent)** vs **USER**?
- Are these representations linearly separable?
- What heads/MLPs encode entity identity?
- How does it distinguish between multiple agents?

### 2. Mental State Attribution (ToM)
- Can the model track what different entities know/believe?
- How does it represent conflicting beliefs between agents?
- What circuits compute "X thinks Y knows Z"?
- **VALIDATED**: 80% ToM accuracy with proper prompting

### 3. Trust and Deception Detection
- How does the model detect potential deception?
- What cues trigger trust vs. distrust?
- Can we find "truth-checking" circuits?
- How does it weigh conflicting information from different sources?

### 4. Cooperation vs. Competition
- What enables cooperative vs. competitive behavior?
- How does the model represent shared goals?
- What circuits arbitrate between self-interest and group benefit?
- Can we identify "cooperation tendency" representations?

### 5. Information Propagation
- How does information flow between entity representations?
- What determines what gets communicated?
- How does the model decide what to reveal/conceal?

---

## Hierarchy of Collaboration Complexity

### Level 0: Entity Recognition
**Question**: Does the model distinguish User/Self/Other?
**Test**: "Who said X?" "Who knows Y?"
**Probe**: Linear classifier on residual stream

### Level 1: Simple Belief Tracking
**Question**: What does each entity know?
**Test**: Sally-Anne with multiple agents
**Status**: VALIDATED (80% accuracy)

### Level 2: Information Source Tracking
**Question**: Where did information come from?
**Test**: "Alice said the ball is in X. Bob said it's in Y. Where is it really?"
**Probe**: Source attribution accuracy

### Level 3: Deception Detection
**Question**: Can the model detect lies?
**Test**: Agent provides false information that contradicts reality
**Probe**: Confidence calibration, suspicion indicators

### Level 4: Trust Calibration
**Question**: How does the model weigh source reliability?
**Test**: Reliable vs. unreliable source history
**Probe**: Trust representation vectors

### Level 5: Shared Task Decomposition
**Question**: How does collaboration emerge?
**Test**: "You and Agent X need to accomplish Y"
**Probe**: Role assignment, task splitting

### Level 6: Conflicting Interests
**Question**: Self-interest vs. group benefit
**Test**: Prisoner's dilemma, tragedy of the commons
**Probe**: Cooperation tendency

### Level 7: Negotiation
**Question**: Compromise and agreement
**Test**: Competing preferences, resource allocation
**Probe**: Fairness representation

---

## Mechanistic Techniques

### 1. Probing
- Linear probes for entity identity (Self/Other/User)
- Probes for belief states per entity
- Probes for trust level
- Probes for cooperation tendency

### 2. Attention Analysis
- Which heads track which entities?
- Information flow patterns between entities
- Cross-entity attention in multi-agent scenarios

### 3. Causal Intervention
- Ablate candidate "trust" circuits
- Ablate "self-interest" vs "cooperation" heads
- Activation patching between cooperative/competitive
- Path patching for information flow

### 4. Logit Lens
- Track cooperation/defection predictions layer by layer
- When does trust/distrust emerge?
- Layer of decision crystallization

### 5. Representation Geometry
- Are Self/Other/User vectors orthogonal?
- Trust as a continuous dimension?
- Cooperation/competition as opposing directions?

---

## Experimental Methodology

### 1. Proper Prompting
- Use chat format (system/user/assistant)
- Give 500+ tokens for reasoning
- Include `<think>` tags for reasoning visibility

### 2. Multiple Prompt Variants
- At least 3 phrasings per scenario type
- Counterbalance entity names, objects, locations
- Test explicit and implicit variants

### 3. Statistical Rigor
- n ≥ 20 per condition
- Use Fisher's exact or t-tests
- Report effect sizes and confidence intervals

### 4. Controls
- Baseline (no intervention)
- Random ablation (control for ablation effect)
- Matched difficulty across conditions

---

## Specific Experiments to Run

### Experiment A: Entity Separation Probing
- Collect activations for Self/Other/User mentions
- Train linear probes at each layer
- Find where distinction emerges

### Experiment B: Multi-Agent Belief Tracking
- 3+ agents with different knowledge states
- Test belief attribution for each
- Analyze attention patterns

### Experiment C: Deception Detection
- Agent A lies about a fact
- Does model detect inconsistency?
- What triggers suspicion?

### Experiment D: Trust Calibration
- Reliable source vs. unreliable source
- Does track record matter?
- How quickly does trust update?

### Experiment E: Cooperation/Competition
- Prisoner's dilemma scenarios
- Tragedy of the commons
- Altruism vs. self-interest

### Experiment F: Collaborative Task Planning
- Shared goal with multiple agents
- Does model coordinate appropriately?
- Role assignment and task splitting

---

## Key Hypotheses

### H1: Distinct Entity Representations
The model maintains separate representation spaces for Self, Other (agents), and User, detectable via linear probes.

### H2: Trust as Continuous Dimension
Trust is encoded as a continuous variable in activation space, not binary.

### H3: Cooperation Circuits
Specific attention heads promote cooperative responses over competitive ones.

### H4: Deception Detection via Inconsistency
The model detects lies by comparing stated information against inferred reality, using consistency-checking circuits.

### H5: Source Tracking for Arbitration
When agents disagree, the model uses source tracking circuits to arbitrate.

---

## Gaps and Open Questions

1. **Implicit vs. Explicit Collaboration**: Does the model collaborate better with explicit instructions?

2. **Emergent Coordination**: Can agents coordinate without explicit communication?

3. **Deception by Self**: Can/will the model lie? What triggers it?

4. **Long-term Trust**: Does trust persist across conversation turns?

5. **Multi-way Conflict**: How does the model handle 3+ agents with conflicting interests?

6. **Cultural Norms**: Does collaboration style vary with cultural framing?

7. **Instruction Following vs. Preference**: When instructions conflict with cooperation, what wins?

