# What The Literature Told Us - And What We Found

**December 2025**

---

## Literature Sources

### 1. Zhu et al. (2024) - Belief Representations
- **Claim**: Linear probes can decode belief states from LLM activations
- **Methodology**: Structured scenarios with controlled variations
- **Our learning**: Proper controls are essential

### 2. ToM Benchmarks Position Paper (2024)
- **Claim**: Most ToM benchmarks are broken for LLMs
- **Key distinction**: "Literal ToM" vs "Functional ToM"
- **Our learning**: High accuracy doesn't prove understanding

### 3. Systematic Review (2024)
- **Claim**: LLMs pass first-order FB but fail second-order
- **Concern**: Methodological biases overestimate capabilities
- **Our learning**: Need diverse tasks, not just Sally-Anne

### 4. ExploreToM Framework
- **Approach**: Adversarial scenario generation
- **Purpose**: Break heuristic shortcuts
- **Our learning**: Design scenarios that heuristics can't solve

---

## What The Literature Said We Should Do

### 1. Counterbalance Locations
- Use both orders (A→B and B→A)
- This cancels location bias
- **We didn't do this originally**

### 2. Include True-Belief Controls
- Test both false-belief AND true-belief scenarios
- Model must get BOTH correct
- **We didn't do this originally**

### 3. Use Neutral/Novel Locations
- Made-up names eliminate learned priors
- Tests genuine reasoning vs. pattern matching
- **We didn't do this originally**

### 4. Compare to Heuristic Baselines
- Calculate what simple heuristics would achieve
- Model must beat these baselines
- **We didn't do this originally**

### 5. Use 8-Scenario Design
- 2 (belief type) × 2 (order) × 2 (question type) = 8
- Require ALL 8 correct for task success
- **We didn't do this originally**

---

## Results of Proper Methodology

### When We Applied Literature Best Practices:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROPER EXPERIMENT RESULTS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Overall Model Accuracy:     37.1%   (worse than chance!)           │
│                                                                     │
│  Heuristic Baselines:                                               │
│    - First-mention:          25.0%                                  │
│    - Recency (last):         75.0%   ← beats model!                 │
│    - Reality:                75.0%   ← beats model!                 │
│                                                                     │
│  By Scenario Type:                                                  │
│    - False-Belief:           46.7%   (near chance)                  │
│    - True-Belief:            21.7%   (BELOW chance!)                │
│    - Control:                40.0%   (poor)                         │
│                                                                     │
│  By Location Type:                                                  │
│    - Neutral/Made-up:        0%      (COMPLETE FAILURE)             │
│    - Real words:             74.2%   (matches recency heuristic)    │
│                                                                     │
│  Task-Level (all 8 correct): 1/30 = 3.3%                           │
│                                                                     │
│  Statistical Test:                                                  │
│    McNemar chi² = 55.1, p < 0.0001                                 │
│    Model is SIGNIFICANTLY WORSE than recency heuristic!             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Discoveries

### 1. Model Cannot Generalize to Novel Locations
- 0% accuracy with made-up location names ("container A", "zone 1")
- This proves model relies on learned word associations, not reasoning

### 2. True-Belief Performance Exposes the Problem
- 21.7% accuracy (worse than random!)
- Model should easily get true-belief scenarios right
- The fact it fails shows no genuine understanding

### 3. Recency Heuristic Explains "Success"
- The model's behavior matches recency more than ToM
- Earlier "ToM success" was just recency + location bias coinciding

### 4. Real Location Names Enable Shortcuts
- 74.2% with real words vs 0% with novel words
- The "success" came from word familiarity, not reasoning

---

## Why Our Earlier Results Were Wrong

### The Illusion of ToM

| What We Saw | What Was Actually Happening |
|-------------|----------------------------|
| "High accuracy on Sally-Anne" | Location bias (drawer > basket) coincided with correct answer |
| "Circuit ablation improved ToM" | Changed which bias dominated, not ToM |
| "Prompt sensitivity" | Different formats triggered different biases |
| "100% with some verbs" | Specific patterns in training data |

### The Critical Confounds

1. **Location bias**: "drawer" preferred over "basket"
2. **Recency bias**: Last-mentioned location preferred
3. **Prompt format effects**: Different completions trigger different patterns
4. **Training data patterns**: Model learned associations, not reasoning

---

## The Real Truth

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   QWEN3-4B HAS NO THEORY OF MIND CAPABILITY                      ║
║                                                                   ║
║   Evidence:                                                       ║
║   • 0% accuracy on novel locations (can't generalize)            ║
║   • Worse than recency heuristic (p < 0.0001)                    ║
║   • 21.7% on true-belief (should be ~100%)                       ║
║   • 3.3% task success with proper controls                       ║
║                                                                   ║
║   What the model DOES have:                                       ║
║   • Learned word associations (drawer, basket, etc.)             ║
║   • Recency patterns (predict recently mentioned)                ║
║   • Prompt pattern matching                                       ║
║                                                                   ║
║   These sometimes COINCIDENTALLY align with ToM answers          ║
║   on standard benchmarks, creating the ILLUSION of ToM.          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Implications for Our Research

### 1. Circuit Findings Need Reinterpretation
Our earlier "inhibitory circuits" weren't suppressing ToM - they were:
- Modulating which bias (recency vs location) dominated
- Or just adding noise to a fundamentally flawed process

### 2. Benchmark Performance is Misleading
High scores on ToMi, Sally-Anne etc. don't indicate ToM because:
- Standard benchmarks use familiar locations
- They don't include proper controls
- Heuristics can achieve similar scores

### 3. Multi-Agent "ToM" Research Needs Caution
If single agents lack ToM, multi-agent ToM claims need:
- Rigorous counterbalanced designs
- Novel entity names
- Heuristic baseline comparisons

---

## What This Means for MATS Research

### The Good News
- We discovered a fundamental limitation through rigorous analysis
- This prevents us from building on a false foundation
- The methodology itself is a contribution

### The Concerning News
- Many published ToM claims may be similarly flawed
- "Emergent ToM" in LLMs may be an illusion
- Safety interventions targeting ToM circuits may be misguided

### The Path Forward
1. Use proper methodology from the start
2. Always include heuristic baselines
3. Always use counterbalanced designs
4. Test with novel/made-up entity names
5. Report effect sizes, not just accuracy

---

## Lessons Learned

1. **High accuracy ≠ Understanding** - Especially without proper controls
2. **Literature methodology matters** - Following best practices revealed the truth
3. **Self-critique is essential** - Our earlier confidence was unfounded
4. **Novel test cases expose shortcuts** - Made-up names broke the model
5. **True-belief controls are crucial** - They reveal when model is just guessing

---

*This document synthesizes findings from applying literature-recommended methodology to our ToM experiments.*


