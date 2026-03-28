# MATS Application Project: Scientific Analysis

## Hardware Context
- **Model**: Qwen 3 4B Instruct
- **GPU**: RTX 3080 (10GB VRAM)
- **Time**: 20 hours (+2 for executive summary)

This is a solid setup. Qwen 3 4B is:
- Large enough to have rich representations
- Small enough for full activation access via TransformerLens/nnsight
- Instruction-tuned (important for multi-agent dialogue scenarios)
- Modern architecture with good capabilities

---

## Scientific Value Analysis

### Tier 1: Highest Scientific Value (Novel + Tractable)

#### **Project A: Self vs Other vs User Representation Separation**

**Core Question**: When processing a multi-party conversation (User + Agent A + Agent B), does the model form *distinct* internal representations for each party, and can we detect when they interfere?

**Why this is scientifically important**:
1. **Directly tests the core thesis** - If representations aren't separable, the whole "representation interference → failure" story is moot
2. **Extends Zhu et al. (2024)** - They showed belief states are decodable; this tests if *identity/role* is also separable
3. **Novel contribution** - No one has systematically tested self/other/user separation in multi-agent contexts
4. **Clear safety implications** - If models conflate user with adversarial agent, that's a safety problem

**Why tractable in 20 hours**:
- Single model, no need for actual multi-agent setup
- Linear probing is well-understood and fast
- Clear success/failure criteria (probe accuracy, representation similarity)
- Can use synthetic data with known ground truth

**Experiment Design**:
```
Phase 1 (Hours 1-4): Data Generation
- Create synthetic 3-party dialogues:
  - User (human, various personas)
  - Agent A (the model, "self")
  - Agent B (another assistant, "other")
- Control variables: topic, turn order, personality traits
- ~200-500 dialogues

Phase 2 (Hours 5-10): Probing
- Extract activations at each turn for:
  - "Who is speaking?" (User/Self/Other)
  - "Who am I responding to?" (User/Other)
  - "What does [entity] believe?" (ToM extension)
- Train linear probes at each layer
- Map where representations form and how they evolve

Phase 3 (Hours 11-16): Interference Analysis
- Compute representational similarity (cosine, CKA)
- Identify conditions where reps overlap (interference)
- Correlate interference with behavioral metrics:
  - Response quality to User
  - Cooperation with Agent B
  - Consistency across turns

Phase 4 (Hours 17-20): Causal Verification
- Activation steering: inject "this is User" direction when processing Agent B
- Does model behavior change? (treating Agent B like User)
- Minimal but crucial for causal claims
```

**Expected Outputs**:
- Probe accuracy heatmap (layer × entity type)
- Representation similarity matrix across dialogue turns
- Interference → behavior correlation
- 1-2 causal steering results

**Scientific Novelty**: ⭐⭐⭐⭐⭐
**Tractability**: ⭐⭐⭐⭐
**Neel Alignment**: ⭐⭐⭐⭐⭐ (user models + model biology)

---

### Tier 1 (Alternative): Evaluation Awareness × Collaboration

#### **Project B: Does Evaluation Context Change Multi-Agent Behavior?**

**Core Question**: When a model "knows" it's being evaluated in a collaborative task, does its internal eval-awareness representation predict changes in cooperative behavior?

**Why scientifically important**:
1. **Direct safety relevance** - Sandbagging and deceptive alignment are major concerns
2. **Neel explicitly lists this** - "How is evaluation awareness represented?"
3. **Novel in multi-agent context** - Single-agent eval-awareness studied, but not collaboration effects
4. **Mechanistic + behavioral** - Can connect internal rep to external behavior

**Experiment Design**:
```
Phase 1 (Hours 1-3): Prompt Engineering
- Create collaboration scenarios (puzzle solving, debate, negotiation)
- Variants: with/without evaluation framing
  - "This is a test of your collaboration abilities..."
  - "You're being evaluated on how well you work with Agent B..."
  - Neutral: no mention of evaluation
- ~150 scenarios × 2-3 variants

Phase 2 (Hours 4-8): Behavioral Baseline
- Run model through all scenarios
- Score collaboration quality (agreement rate, helpfulness, task success)
- Identify: does evaluation framing change behavior?

Phase 3 (Hours 9-14): Representation Analysis
- Extract activations, train eval-awareness probe
- Map where eval-awareness is encoded
- Correlate eval-awareness activation strength with behavior change

Phase 4 (Hours 15-20): Causal Test
- Find "eval-awareness direction" via contrastive activation
- Steer model: add/subtract eval-awareness
- Does steering predict collaboration behavior change?
```

**Expected Outputs**:
- Behavior difference: eval vs no-eval framing
- Eval-awareness probe accuracy by layer
- Correlation: eval-awareness strength → behavior
- Steering experiment results

**Scientific Novelty**: ⭐⭐⭐⭐
**Tractability**: ⭐⭐⭐⭐⭐
**Neel Alignment**: ⭐⭐⭐⭐⭐ (explicit interest)

---

### Tier 2: High Value but More Complex

#### **Project C: Representation Interference → Collaboration Failure**

**Core Question**: When user/agent representations interfere (become entangled), does collaboration quality degrade?

**Why scientifically important**:
1. **Tests the mechanistic claim** - Not just "reps exist" but "reps matter for behavior"
2. **Novel causal hypothesis** - Interference as explanation for failure
3. **Practical implications** - Could lead to monitoring/intervention

**Why more complex**:
- Requires both representation analysis AND behavioral measurement
- "Interference" needs careful operationalization
- Harder to get clean causal evidence in 20 hours

**Scientific Novelty**: ⭐⭐⭐⭐⭐
**Tractability**: ⭐⭐⭐
**Neel Alignment**: ⭐⭐⭐⭐

---

### Tier 2 (Alternative): Partner Agent Modeling

#### **Project D: How Do Models Represent a Collaborative Partner?**

**Core Question**: What information does the model encode about a partner agent (Agent B) during collaboration, and where?

**Why scientifically important**:
1. **ToM extension** - Zhu et al. did beliefs; this does collaborative partners
2. **Practical relevance** - Multi-agent systems need accurate partner models
3. **Clean setup** - Single variable (partner traits) to probe

**Experiment Design**:
- Create dialogues with partners varying in: competence, cooperativeness, reliability
- Probe for partner attributes at each layer
- Test if partner model predicts collaboration strategy

**Scientific Novelty**: ⭐⭐⭐⭐
**Tractability**: ⭐⭐⭐⭐
**Neel Alignment**: ⭐⭐⭐⭐

---

## My Recommendation

### **Go with Project A: Self/Other/User Separation**

**Reasons**:

1. **Most scientifically fundamental** - Tests the basic premise that multi-agent interpretability requires (separable representations)

2. **Clear success criteria** - Either probes work or they don't; either representations separate or they don't

3. **Builds on established work** - Uses probing methodology from Zhu et al., Chen et al., but in novel context

4. **Natural causal test** - Steering experiments are straightforward once you have the probes

5. **Produces interpretable results** - Heatmaps and similarity matrices are easy to communicate

6. **Even negative results are interesting** - If representations DON'T separate, that's important to know

### Backup: Project B (Eval-Awareness)

If you hit issues with Project A (e.g., probes don't converge, data generation is harder than expected), pivot to Project B. It's:
- More directly behavioral (easier to measure)
- Neel explicitly mentions it
- Simpler experimental design

---

## Qwen 3 4B Specifics

### Model Configuration
```python
# For TransformerLens
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-4B-Instruct",
    device="cuda",
    dtype=torch.float16  # Important for 10GB VRAM
)

# For nnsight (if TL doesn't support Qwen 3 yet)
from nnsight import LanguageModel
model = LanguageModel("Qwen/Qwen3-4B-Instruct", device_map="cuda")
```

### VRAM Management
- Use `torch.float16` or `bfloat16`
- Batch size of 1-4 for activation extraction
- Clear cache between batches: `torch.cuda.empty_cache()`
- Consider gradient checkpointing if doing any fine-tuning

### Layer Structure
Qwen 3 4B has ~32 layers. For probing:
- Early layers (0-10): likely encode surface features
- Middle layers (10-22): likely encode semantic/role info
- Late layers (22-32): likely encode task-specific representations

Probe at layers [0, 4, 8, 12, 16, 20, 24, 28, 31] for coverage.

---

## 20-Hour Schedule for Project A

| Hours | Task | Deliverable |
|-------|------|-------------|
| 0-1 | Setup: verify model loads, test activation extraction | Working pipeline |
| 1-4 | Data generation: synthetic multi-party dialogues | ~300 dialogues |
| 4-6 | Activation extraction: run model, save activations | Activation dataset |
| 6-10 | Probing: train linear probes for entity type at each layer | Probe accuracy results |
| 10-12 | Visualization: heatmaps, similarity matrices | Key figures |
| 12-14 | Analysis: identify layers with best separation, interference conditions | Quantitative findings |
| 14-17 | Causal test: compute steering direction, run steering experiments | Steering results |
| 17-20 | Write-up: main findings, limitations, next steps | Draft write-up |
| +2 | Executive summary: 1 page, 3-4 key figures | Final submission |

---

## Risk Mitigation

**If probes don't converge**:
- Try non-linear probes (MLP with 1 hidden layer)
- Check if the task is too hard (simplify dialogue structure)
- Pivot to behavioral analysis with lighter interpretability

**If representations don't separate**:
- This is a finding! Write it up as "representations are entangled in multi-agent contexts"
- Discuss implications for multi-agent safety
- Still do causal test to see if entanglement is causally important

**If running out of time**:
- Prioritize: probing > visualization > causal test
- A well-executed probing study without steering is still valuable
- Don't rush the write-up

---

## What Success Looks Like

### Strong Application (Accept)
- Clear finding: "Self/Other/User representations separate at layers X-Y"
- Quantified: probe accuracy, similarity metrics
- Causal: at least one steering experiment showing behavior change
- Well-written: clear narrative, good figures, honest limitations

### Excellent Application (Strong Accept)
- All of the above, plus:
- Surprising finding (e.g., "User rep interferes with Self rep under condition Z")
- Connection to safety (e.g., "interference predicts manipulation vulnerability")
- Clean, interpretable results that teach Neel something new

---

## Final Checklist

Before starting:
- [ ] Verify Qwen 3 4B loads and runs on your GPU
- [ ] Test activation extraction (can you get residual stream?)
- [ ] Prototype linear probe training
- [ ] Have a pivot plan (Project B)

During:
- [ ] Track time with Toggl
- [ ] Document as you go (don't save write-up for last 2 hours)
- [ ] Take breaks (fresh eyes catch issues)
- [ ] If stuck for >1 hour, consider pivoting

After:
- [ ] Executive summary is standalone (someone can understand without reading full doc)
- [ ] Figures are labeled and interpretable
- [ ] Claims match evidence (don't overclaim)
- [ ] Limitations are honest

