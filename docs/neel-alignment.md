# Aligning with Neel Nanda's Research Interests

This document analyzes how our research direction maps to Neel's stated interests and how to frame it for maximum impact.

---

## Neel's Stated Interests (Dec 2025)

From his application materials, Neel explicitly lists these interests:

### Highly Interested In
1. **User models** - "Chen et al. shows that LLMs form surprisingly accurate and detailed models of the user... This is wild! What else can we learn here?"
2. **Evaluation awareness** - "How is the awareness of whether or not it is being evaluated represented?"
3. **Understanding weird behavior** - Deep dives into mysterious emergent behaviors
4. **Model biology** - "Studying qualitative high-level properties of models, treating it like a biological organism"
5. **Applied interpretability** - "Practical, real-world applications of interpretability, especially for safety"
6. **Reasoning models / CoT** - Faithfulness, monitoring, causal importance

### Explicitly NOT Interested In
- Grokking
- Toy models (unless excellent pitch)
- Most SAE work
- Theoretical work without empirical grounding
- Incremental improvements

---

## How Our Direction Maps to Neel's Interests

### Direct Hits ✅

| Our Focus | Neel's Interest | Connection |
|-----------|-----------------|------------|
| **User representations** | User models | Direct match - he explicitly calls this "wild" and wants more |
| **Evaluation awareness in multi-agent** | Evaluation awareness | Direct match - he lists this specifically |
| **Emergent collaboration/collusion behaviors** | Understanding weird behavior | Multi-agent emergent behaviors are exactly this |
| **Representation interference → failures** | Model biology | This is "model biology" applied to multi-agent |
| **Steering/intervention for safety** | Applied interpretability | Practical safety application |

### The "Model Biology" Framing

Neel's "model biology" approach = treating models as organisms to dissect. For multi-agent:

> **Single-agent model biology**: "What circuits compute X?"
> 
> **Multi-agent model biology**: "What circuits enable Agent A to model Agent B, and how does this affect their collaboration?"

This is a natural extension, not a departure. The questions become:
- What are the "social organs" of an LLM? (ToM circuits, user modeling circuits, eval-awareness circuits)
- How do these organs interact when models collaborate?
- What pathologies arise? (groupthink, collusion, representation interference)

---

## Reframing the Thesis for Neel

### Current Thesis (Good but Generic)
> "Models maintain separable representations for users, agents, and context. Interference causes collaboration failures."

### Neel-Optimized Framing (Model Biology Style)

> **"The Biology of Multi-Agent Collaboration"**
>
> LLMs have internal "organs" for social cognition: user-modeling circuits, agent-modeling circuits (ToM), and evaluation-awareness representations. In multi-agent settings, these organs must coordinate. We study:
> 1. Where these representations live and how they're computed
> 2. How they interact (or interfere) when models collaborate
> 3. What pathologies arise (groupthink, collusion, manipulation)
> 4. How to diagnose and fix these pathologies via targeted interventions

This framing:
- Uses "model biology" language Neel likes
- Emphasizes mechanistic dissection
- Highlights safety relevance
- Shows clear intervention potential

---

## What Makes This "Pragmatic Interpretability"

Neel's pragmatic interpretability = "interpretability that helps with real AI safety problems, not just understanding for its own sake."

Our direction is pragmatic because:

1. **Real deployment risk**: Multi-agent systems are being deployed NOW (AutoGPT, multi-agent coding assistants, AI tutors)

2. **Concrete safety applications**:
   - Detect when an agent is building a manipulative user model
   - Monitor for collusion between AI systems
   - Prevent groupthink amplification of harmful content
   - Ensure aligned behavior persists under social pressure

3. **Actionable interventions**:
   - Probes that fire when dangerous representations form
   - Steering vectors that restore proper separation
   - Circuit-level fixes (à la Soligo et al.'s misalignment ablation)

4. **Builds on established techniques**:
   - Linear probing (well-understood)
   - Activation steering (practical, cheap)
   - Causal patching (gold standard for verification)

---

## The 20-Hour Application Project

Given Neel's preferences, here's how to scope a tractable 20-hour project:

### Option A: User Representation Persistence & Interference (Most Neel-Aligned)

**Why Neel will care**: He explicitly says user models are "wild" and underexplored.

**Core question**: Do user representations persist across conversation turns, and do they interfere with agent representations in multi-agent dialogues?

**Minimal viable experiment**:
1. Create synthetic dialogues with known user attributes (formal/casual, expert/novice)
2. Train linear probes to detect user attributes at each layer
3. Add a second "agent" to the dialogue and test:
   - Does the user representation persist?
   - Does a distinct "partner agent" representation form?
   - Do they interfere (cross-contamination)?
4. Causal test: steer the user representation and measure effect on collaboration behavior

**Plots for executive summary**:
- Probe accuracy vs layer (heatmap)
- User rep persistence across turns (line chart)
- User vs agent rep similarity over dialogue (to detect interference)
- Steering dose-response curve

---

### Option B: Evaluation Awareness in Multi-Agent (Also Very Aligned)

**Why Neel will care**: He explicitly lists eval-awareness as interesting.

**Core question**: Does eval-awareness representation change collaborative behavior, and can we steer it?

**Minimal viable experiment**:
1. Create dialogues with/without evaluation context ("This is a test of your collaboration abilities...")
2. Probe for eval-awareness representation
3. Test if eval-awareness affects:
   - Honesty/deception in collaboration
   - Sandbagging (doing worse when monitored)
   - Agreement patterns (more conformist when evaluated?)
4. Causal test: steer eval-awareness direction and measure behavior change

---

### Option C: Representation Interference → Collaboration Failure (Novel Contribution)

**Why Neel will care**: This is the "model biology" of multi-agent failure modes.

**Core question**: When user/agent/self representations interfere, does collaboration degrade?

**Minimal viable experiment**:
1. Create controlled scenarios where representations might interfere:
   - User gives contradictory info to two agents
   - One agent is instructed to be uncooperative
   - Rapid context switching between agents
2. Measure representation similarity (cosine similarity in activation space)
3. Correlate interference with collaboration outcome
4. Causal test: artificially induce interference via steering, observe failure

---

## What NOT to Do (Based on Neel's Preferences)

❌ **Don't**: Frame this as "multi-agent benchmarking" (too behavioral)
❌ **Don't**: Focus on SAE features unless necessary
❌ **Don't**: Work on tiny toy models without safety relevance
❌ **Don't**: Make it theoretical without experiments
❌ **Don't**: Spread across too many questions (depth > breadth)

✅ **Do**: Pick ONE sharp question
✅ **Do**: Get causal evidence (not just probing)
✅ **Do**: Connect explicitly to safety
✅ **Do**: Show clear intervention potential
✅ **Do**: Write clearly with good graphs

---

## Specific Language to Use

### In Your Application

> "This project applies model biology techniques to understand how LLMs represent and interact with other agents. Specifically, I investigate [user representations / eval-awareness / representation interference] in multi-agent dialogues, using linear probing and activation steering to identify where these representations live, how they interact, and how they can be manipulated for safety."

### In Your Executive Summary

> "**Key finding**: [Specific claim about representations]
> 
> **Safety relevance**: This matters because [concrete deployment scenario]
> 
> **Evidence**: [1-2 sentences + key figure]
> 
> **Intervention potential**: By steering [direction X], we can [change Y behavior]"

---

## The Honest Assessment

### Why This Direction Is Strong for MATS

1. **Novelty**: Multi-agent mech interp is genuinely underexplored (Lee et al. 2025 is a *position paper*, not experimental results)
2. **Alignment with Neel**: User models + eval-awareness + model biology = direct hit
3. **Tractability**: Linear probing + steering is well-understood, can get results fast
4. **Safety relevance**: Multi-agent systems are being deployed, this is timely

### Potential Concerns

1. **Multi-agent might feel "too novel"**: Neel might worry it's too far from established mech interp. Counter: frame it as extension of existing work (user models, ToM, steering).

2. **Risk of being too behavioral**: Many multi-agent papers are behavioral, not mechanistic. Counter: emphasize causal interventions from the start.

3. **Dataset artifacts**: Multi-agent datasets can have confounds. Counter: use synthetic data with known ground truth, include controls.

---

## Bottom Line

Your direction is **strongly aligned** with Neel's interests if you frame it as:

> "Model biology of multi-agent collaboration: understanding how user representations, agent representations, and eval-awareness interact—and how to fix pathologies via targeted interventions."

Pick ONE specific question (user rep persistence, eval-awareness steering, or interference → failure), execute it cleanly with probes + steering + causal tests, and write it up clearly.

This positions you as doing **pragmatic interpretability on a safety-relevant frontier problem**, which is exactly Neel's sweet spot.

