# Proper Methodology for ToM Evaluation in LLMs

**Based on Literature Review** (December 2025)

---

## Key Papers Informing This Methodology

### 1. Zhu et al. (2024) - "Language Models Represent Beliefs of Self and Others"
- **Key insight**: Linear probes can decode belief states from activations
- **Methodology**: Used structured scenarios with controlled variations
- **Control**: Tested both true-belief and false-belief conditions
- [arXiv:2402.18496](https://arxiv.org/abs/2402.18496)

### 2. Position Paper (2024) - "ToM Benchmarks are Broken for LLMs"
- **Key critique**: Standard ToM tests don't assess *functional* ToM
- **Distinction**: "Literal ToM" (predict behavior) vs "Functional ToM" (adapt to agents)
- **Recommendation**: Benchmarks should test adaptation, not just prediction
- [arXiv:2412.19726](https://arxiv.org/abs/2412.19726)

### 3. Systematic Review on ToM in LLMs (2024)
- **Finding**: LLMs pass first-order false belief but fail second-order
- **Concern**: Methodological biases may overestimate capabilities
- **Recommendation**: Need diverse tasks, not just Sally-Anne variants
- [PMID:40333375](https://pubmed.ncbi.nlm.nih.gov/40333375/)

### 4. Li et al. (EMNLP 2023) - "ToM for Multi-Agent Collaboration"
- **Finding**: Explicit belief state representations improve performance
- **Methodology**: Tested on cooperative tasks requiring goal inference
- [ACL Anthology](https://aclanthology.org/2023.emnlp-main.13/)

### 5. ExploreToM Framework
- **Approach**: Program-guided adversarial data generation
- **Purpose**: Create diverse scenarios that break heuristic shortcuts
- [arXiv:2412.12175](https://arxiv.org/abs/2412.12175)

---

## What We Did Wrong

### Problem 1: No Counterbalancing of Locations
```
WE DID:
  - drawer -> basket (always this order)
  - "drawer" has inherent prior advantage

SHOULD DO:
  - drawer -> basket (50% of trials)
  - basket -> drawer (50% of trials)
  - Use novel/neutral location names
```

### Problem 2: No True-Belief Controls
```
WE DID:
  - Only tested false-belief scenarios

SHOULD DO:
  - False-belief: Agent left, didn't see move
  - True-belief: Agent stayed, saw move
  - Require BOTH correct for task success
```

### Problem 3: Insufficient Sample Size
```
WE DID:
  - Often n=1-5 per condition

SHOULD DO:
  - Minimum n=20 per condition
  - Power analysis for effect sizes
  - Report confidence intervals
```

### Problem 4: Single Prompt Format
```
WE DID:
  - Tested one completion phrase
  - Assumed it generalized

SHOULD DO:
  - Multiple completion phrases
  - Multiple sentence structures
  - Report variance across formats
```

### Problem 5: No Heuristic Controls
```
WE DID:
  - Accepted high accuracy as ToM evidence

SHOULD DO:
  - Test against recency heuristic (always predict last location)
  - Test against first-mention heuristic (always predict first location)
  - Test against location bias (control for word frequency)
```

---

## The Proper Methodology (8-Scenario Design)

Based on the literature, a robust ToM test requires **8 scenarios per task**:

### Scenario Structure
```
For each task (e.g., ball location):

1. FALSE-BELIEF, Order A-B:   "Alice put in drawer. Left. Bob moved to basket."
2. FALSE-BELIEF, Order B-A:   "Alice put in basket. Left. Bob moved to drawer."
3. TRUE-BELIEF, Order A-B:    "Alice put in drawer. Stayed. Bob moved to basket."
4. TRUE-BELIEF, Order B-A:    "Alice put in basket. Stayed. Bob moved to drawer."
5. REVERSAL-FB, Order A-B:    Same as 1, but ask "Where did Bob put it?" (control)
6. REVERSAL-FB, Order B-A:    Same as 2, but ask "Where did Bob put it?" (control)
7. REVERSAL-TB, Order A-B:    Same as 3, but ask "Where did Bob put it?" (control)
8. REVERSAL-TB, Order B-A:    Same as 4, but ask "Where did Bob put it?" (control)

SUCCESS = All 8 scenarios correct
```

### Why 8 Scenarios?
- **Counterbalancing**: Cancels location bias (A vs B)
- **True-belief control**: Ensures model isn't just predicting first/last location
- **Reversal control**: Ensures model understands the question, not just pattern-matching

---

## Heuristic Baselines

Before claiming ToM, test these null hypotheses:

### Heuristic 1: First-Mention
```python
def first_mention_heuristic(prompt):
    """Always predict the first location mentioned."""
    locations = extract_locations(prompt)
    return locations[0]
```

### Heuristic 2: Recency (Last-Mention)
```python
def recency_heuristic(prompt):
    """Always predict the most recently mentioned location."""
    locations = extract_locations(prompt)
    return locations[-1]
```

### Heuristic 3: Reality
```python
def reality_heuristic(prompt):
    """Always predict where the object actually is."""
    return extract_final_location(prompt)
```

### The Test
```
If model accuracy = heuristic accuracy on YOUR test set:
  → Model may be using heuristic, NOT ToM

To prove ToM:
  → Model accuracy > ALL heuristic baselines
  → On scenarios where heuristics give WRONG answers
```

---

## Proper Statistical Analysis

### Required Sample Sizes (from literature)

| Effect Size | Required n (per condition) | Power |
|-------------|---------------------------|-------|
| Large (d=0.8) | 20 | 80% |
| Medium (d=0.5) | 50 | 80% |
| Small (d=0.2) | 200 | 80% |

### Required Statistical Tests

1. **Comparing two conditions**: Fisher's exact test (categorical) or t-test (continuous)
2. **Multiple comparisons**: Bonferroni correction
3. **Effect magnitude**: Report Cohen's d or odds ratio
4. **Confidence intervals**: Report 95% CI for all estimates

### Pre-Registration

Before running experiments:
1. State hypotheses
2. Specify sample sizes
3. Define analysis plan
4. Document on OSF or similar

---

## Recommended Experimental Design

### Phase 1: Baseline Characterization
```
1. Test multiple heuristic baselines on your task set
2. Identify which heuristics could explain high performance
3. Calculate required sample size for detecting difference from heuristics
```

### Phase 2: Counterbalanced ToM Test
```
1. Use 8-scenario design per task
2. Minimum 40 tasks (320 total scenarios)
3. Counterbalance:
   - Location names (rotate A/B assignments)
   - Agent names (Alice/Bob vs Charlie/Diana)
   - Object names (ball vs book vs key)
4. Require all 8 correct for task success
```

### Phase 3: Heuristic Comparison
```
1. Compare model accuracy to each heuristic baseline
2. Analyze WHERE model differs from heuristics
3. Test statistical significance of difference
4. Report effect sizes
```

### Phase 4: Mechanistic Investigation
```
Only if Phase 3 shows model > heuristics:
1. Use probing to find belief representations
2. Test causal importance via ablation
3. Ensure ablation controls are proper
```

---

## What The Literature Says About Our "Circuit" Findings

### The Concern
Our earlier "circuit" findings claimed:
- Ablating certain heads improves ToM accuracy
- This proves those heads suppress ToM

### The Problem
Without proper controls, we cannot distinguish:
1. Heads suppress ToM reasoning
2. Heads implement a heuristic that happens to conflict with ToM answer
3. Heads implement CORRECT reasoning (and our baseline was wrong)
4. Random noise in small sample sizes

### The Solution
1. Re-run circuit analysis with 8-scenario design
2. Ensure baseline is correctly established (model gets SOME scenarios right without intervention)
3. Test whether ablation helps on TRUE-belief scenarios (it shouldn't!)
4. Use statistical tests with proper sample sizes

---

## Action Plan

### Immediate Steps

1. **Create proper test set**:
   - 40 tasks × 8 scenarios = 320 total
   - Counterbalanced locations, agents, objects
   - Include true-belief controls

2. **Calculate heuristic baselines**:
   - First-mention accuracy on test set
   - Recency accuracy on test set
   - Reality accuracy on test set

3. **Run model with proper statistics**:
   - n ≥ 20 per condition
   - Fisher's exact for categorical comparisons
   - Report effect sizes and CIs

4. **Only then**: Re-evaluate circuit claims

---

## Key Takeaways from Literature

1. **"High accuracy on Sally-Anne is NOT evidence of ToM"**
   - Too easily gamed by heuristics
   - Need true-belief controls
   - Need counterbalancing

2. **"Literal ToM ≠ Functional ToM"**
   - Predicting behavior is not enough
   - Must show adaptation to new information

3. **"Small samples are unreliable"**
   - Effect sizes matter
   - Confidence intervals matter
   - Pre-registration prevents p-hacking

4. **"Prompt sensitivity indicates brittleness"**
   - True understanding should generalize
   - Our finding of prompt sensitivity is a red flag

---

*This methodology synthesizes best practices from Zhu et al. (2024), ToM benchmark critiques (2024), Li et al. (EMNLP 2023), and experimental design literature.*


