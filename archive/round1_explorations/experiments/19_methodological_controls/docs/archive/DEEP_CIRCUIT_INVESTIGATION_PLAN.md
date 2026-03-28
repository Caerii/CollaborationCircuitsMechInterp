# Deep Circuit Investigation Plan

## Motivation

The gargantuan sweep confirmed consistency (~21% boost across 1,278 verbs), but the INTERESTING findings came from our earlier diverse sweep:

1. **"told" causes COMPLETE ToM failure** (0% → 100%)
2. **Spanish shows +36% boost** (vs +4% English)
3. **"asking" verbs trigger MAX inhibition** (8% baseline)
4. **Complex sentences = 0% baseline** (vs 93% for indirect speech)

Now we need to understand **WHY** - using attention pattern analysis.

---

## Investigation 1: The "told" Mystery

### Question
Why does "told" trigger complete ToM failure while "provided" doesn't?

### Method
1. Compare attention patterns for the 5 decision heads on:
   - "told" scenarios (0% baseline)
   - "provided" scenarios (100% baseline)
   - "mentioned" (mid-range)

2. Track attention at key positions:
   - The verb itself
   - Agent names (Alice, Bob, Carol)
   - Object (ball)
   - Location tokens (drawer, basket)

### Hypothesis
The decision heads might attend strongly to "told" but weakly to "provided", triggering different downstream behavior.

---

## Investigation 2: What Makes a Verb "Inhibitory"?

### Question
Is there a pattern in which verbs trigger strong vs weak inhibition?

### Method
1. Take the 10 worst baseline verbs (0% or near) and 10 best (80%+)
2. Extract attention patterns for each
3. Look for:
   - Attention to verb position
   - Attention to agent positions
   - Cross-position attention flows

### Verbs to Compare
**Worst (0% baseline)**: told, noted, clarified, announced, indicated, asked, queried, conveyed
**Best (80%+ baseline)**: dispatched, provided, supported, manifested, worded

---

## Investigation 3: Spanish Deep-Dive

### Question
Why does Spanish show +36% boost while English shows only +4%?

### Method
1. Test same scenarios in English vs Spanish
2. Compare attention patterns across languages
3. Check if the decision heads even activate for Spanish text

### Hypothesis
The decision heads may be "English-tuned" and simply not detect Spanish communication patterns as strongly.

---

## Investigation 4: The "Asking" Category

### Question
Why do bidirectional verbs (asked, requested, inquired) trigger the STRONGEST inhibition (8% baseline)?

### Method
1. Compare attention for:
   - "asked" (bidirectional)
   - "told" (unidirectional)  
   - "mentioned" (neutral)

2. Look for patterns that distinguish these

### Hypothesis
Bidirectional verbs might create more complex belief-state requirements that the model handles poorly.

---

## Investigation 5: Sentence Structure Effects

### Question
Why do complex sentences trigger MORE inhibition than simple ones?

### Method
1. Same semantic content, different structures:
   - Simple: "Bob told Carol that he moved the ball"
   - Complex: "Having found the ball, Bob told Carol, who was nearby, that he had moved it"
   - Indirect: "Carol was informed by Bob about the ball's new location"

2. Compare attention patterns

### Hypothesis
Simple, canonical patterns may match training data patterns that trigger the inhibitory circuit.

---

## Technical Approach

### Attention Extraction
1. Use eager attention (not SDPA) to get attention weights
2. Extract from all 5 decision heads: L17H4, L18H11, L18H14, L19H30, L21H17
3. Also extract from enabler heads: L15H9, L19H2, L19H15

### Analysis Methods
1. **Attention heatmaps**: Show which tokens attend to which
2. **Head-specific patterns**: What does each head focus on?
3. **Position importance**: Which token positions matter most?
4. **Diff analysis**: How do attention patterns differ between correct/incorrect responses?

### Visualizations
1. Per-verb attention heatmaps
2. Aggregated patterns across verb categories
3. Language comparison plots
4. Structure comparison plots

---

## Expected Deliverables

1. `step20_told_mystery.py` - Deep-dive on "told" vs "provided"
2. `step21_verb_patterns.py` - Systematic verb comparison
3. `step22_multilingual_attention.py` - Spanish vs English attention
4. `step23_structure_effects.py` - Sentence structure analysis
5. Visualization outputs in `figures/`
6. `DEEP_CIRCUIT_FINDINGS.md` - Comprehensive writeup

---

## Priority Order

1. **"told" mystery** - Most striking finding, directly actionable
2. **Verb patterns** - Systematic understanding
3. **Spanish deep-dive** - Novel multilingual insight
4. **Structure effects** - Explains complexity paradox

