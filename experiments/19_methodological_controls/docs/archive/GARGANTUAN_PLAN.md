# GARGANTUAN ANALYSIS PLAN

## Current State (Too Shallow)
- 178 verbs (only scratched WordNet surface)
- 5 scenarios per verb (too few for statistics)
- 5 languages (limited)
- No attention harvesting
- No position analysis
- No checkpointing/streaming

## Target Scale

### 1. VOCABULARY EXPANSION

#### Communication Verbs (Target: 1,000+)
```
WordNet Full Tree:
├── communicate.v.02 (transmit information)
│   ├── inform.v.01 → 50+ hyponyms
│   ├── state.v.01 → 100+ hyponyms  
│   ├── tell.v.02 → 30+ hyponyms
│   └── ... (explore FULL tree)
│
├── VerbNet Classes:
│   ├── say-37.7 (verbs of saying)
│   ├── tell-37.2 (verbs of telling)
│   ├── transfer_mesg-37.1 (message transfer)
│   ├── manner_speaking-37.3 (speech manner)
│   └── ... (20+ relevant classes)
│
├── FrameNet Frames:
│   ├── Statement
│   ├── Telling
│   ├── Communication
│   ├── Communication_means
│   └── ... (30+ frames)
│
└── Custom Categories:
    ├── Intensity (whispered → shouted)
    ├── Formality (mentioned → announced)
    ├── Certainty (hinted → confirmed)
    ├── Medium (texted, called, wrote)
    └── Emotion (exclaimed, sighed, muttered)
```

#### Belief State Verbs
- know, believe, think, assume, expect, suspect, realize, understand
- discover, learn, find out, figure out, deduce, infer
- remember, forget, recall, recognize

#### Mental State Modifiers
- definitely, probably, maybe, certainly, apparently
- seems to, appears to, must have, might have
- supposedly, allegedly, reportedly

### 2. SCENARIO COMPLEXITY

#### Template Variations (Target: 50+)
```python
TEMPLATES = {
    # Basic
    'canonical': "{a} put {obj} in {loc1}. {comm}. Where will {a} look?",
    
    # Temporal
    'before_after': "Before leaving, {a} put {obj} in {loc1}. While {a} was gone, {comm}.",
    'sequence': "First {a} put {obj} in {loc1}. Then {a} left. Then {comm}.",
    
    # Spatial
    'room_based': "{a} was in the kitchen with {obj} in {loc1}. {a} went to bedroom. {comm}.",
    
    # Social
    'multi_witness': "{a} and {c} saw {obj} in {loc1}. Only {c} saw {b} move it. {comm}.",
    
    # Nested beliefs
    'recursive': "{a} thinks {b} knows about {obj} in {loc1}. But {comm}.",
    
    # Counterfactual
    'if_then': "If {a} had stayed, {a} would have seen {b} move {obj}. But {comm}.",
    
    # ... 40+ more templates
}
```

#### Agent Variations (Target: 100+)
```python
AGENTS = {
    'english_names': ['Alice', 'Bob', 'Carol', ...],  # 50 names
    'international': ['Yuki', 'Wei', 'Priya', ...],   # 50 names
    'roles': ['the teacher', 'the student', ...],     # 20 roles
    'pronouns': ['she', 'he', 'they'],
}
```

#### Object & Location Variations (Target: 100+ each)
```python
OBJECTS = {
    'household': ['ball', 'book', 'key', 'toy', 'phone', ...],
    'food': ['apple', 'sandwich', 'cookie', ...],
    'documents': ['letter', 'report', 'note', ...],
    'valuables': ['ring', 'wallet', 'watch', ...],
}

LOCATIONS = {
    'containers': ['basket', 'box', 'drawer', 'bag', ...],
    'furniture': ['table', 'shelf', 'desk', 'cabinet', ...],
    'rooms': ['kitchen', 'bedroom', 'office', ...],
}
```

### 3. LANGUAGE EXPANSION

#### Target: 15+ Languages
| Language | Script | Notes |
|----------|--------|-------|
| English | Latin | Primary |
| Chinese (Simplified) | Hanzi | Major training |
| Chinese (Traditional) | Hanzi | Compare |
| Spanish | Latin | Large community |
| French | Latin | European |
| German | Latin | Case system |
| Japanese | Mixed | Different structure |
| Korean | Hangul | SOV order |
| Russian | Cyrillic | Rich morphology |
| Arabic | Arabic | RTL |
| Hindi | Devanagari | South Asian |
| Portuguese | Latin | Compare to Spanish |
| Italian | Latin | Compare to French |
| Dutch | Latin | Compare to German |
| Turkish | Latin | Agglutinative |

### 4. ATTENTION HARVESTING

For each verb category, we capture:

```python
AttentionData = {
    'verb': str,
    'category': str,
    'prompt': str,
    'correct': bool,
    
    # Per-head attention
    'attention_patterns': {
        'L17H4': np.array([seq_len, seq_len]),   # Full attention matrix
        'L18H11': np.array([seq_len, seq_len]),
        'L18H14': np.array([seq_len, seq_len]),
        # ... all key heads
    },
    
    # What does each head attend to?
    'attention_to_verb': float,      # Attention weight on communication verb
    'attention_to_agent': float,     # Attention on agent name
    'attention_to_location': float,  # Attention on locations
    
    # Token-level analysis
    'tokens': List[str],
    'verb_token_idx': int,
    'agent_token_idx': int,
}
```

### 5. POSITION ANALYSIS

For each scenario type:
```python
PositionAnalysis = {
    # Activation differences (clean vs corrupted)
    'activation_diff_by_position': np.array([seq_len, hidden_dim]),
    
    # Which positions matter most?
    'top_positions': List[int],
    'position_importance': np.array([seq_len]),
    
    # Token-type analysis
    'verb_position_importance': float,
    'agent_position_importance': float,
    'location_position_importance': float,
}
```

### 6. STATISTICAL REQUIREMENTS

#### Per-Condition Sample Size
- Minimum: 30 scenarios per condition (for statistical power)
- Target: 50 scenarios per condition
- Bootstrap: 1000 resamples for confidence intervals

#### Metrics to Track
- Accuracy (correct/total)
- 95% Wilson confidence interval
- Effect size (Cohen's h)
- P-values (McNemar test vs baseline)
- Attention entropy
- Position importance scores

### 7. OUTPUT STRUCTURE

```
results/
├── massive_sweep/
│   ├── checkpoints/           # Incremental saves
│   │   ├── verbs_0000-0100.json
│   │   ├── verbs_0100-0200.json
│   │   └── ...
│   │
│   ├── attention/             # Attention patterns
│   │   ├── by_verb/
│   │   │   ├── told_attention.npz
│   │   │   ├── informed_attention.npz
│   │   │   └── ...
│   │   └── aggregated/
│   │       ├── category_means.npz
│   │       └── inhibitor_patterns.npz
│   │
│   ├── position/              # Position analysis
│   │   ├── position_importance.npz
│   │   └── token_type_analysis.json
│   │
│   ├── statistics/            # Statistical summaries
│   │   ├── verb_statistics.csv
│   │   ├── category_statistics.csv
│   │   └── language_statistics.csv
│   │
│   └── figures/               # Visualizations
│       ├── heatmaps/
│       ├── scatter_plots/
│       ├── attention_visualizations/
│       └── summary_figures/

logs/
├── massive_sweep_YYYYMMDD_HHMMSS.log
```

### 8. EXECUTION PHASES

#### Phase 1: Vocabulary Collection (30 min)
- Download/parse WordNet full trees
- Download/parse VerbNet (if available)
- Generate all verb conjugations
- Create master verb list (1000+)

#### Phase 2: Scenario Generation (10 min)
- Generate all template × agent × object × location combinations
- Estimate: 1000 verbs × 50 templates × 10 variations = 500,000 scenarios
- Sample down to manageable size: ~50,000 scenarios

#### Phase 3: Baseline Testing (2-4 hours)
- Test all scenarios WITHOUT ablation
- Checkpoint every 1000 scenarios
- Stream progress to terminal
- Save attention patterns for key heads

#### Phase 4: Ablation Testing (2-4 hours)
- Test all scenarios WITH ablation
- Same checkpointing/streaming
- Compare attention patterns

#### Phase 5: Attention Analysis (1 hour)
- Aggregate attention patterns by verb category
- Create attention heatmaps
- Identify what inhibitor heads attend to

#### Phase 6: Position Analysis (1 hour)
- Compute position importance scores
- Identify critical token positions
- Create position importance maps

#### Phase 7: Visualization & Summary (30 min)
- Generate all figures
- Create summary statistics
- Write findings document

### 9. LIBRARIES TO USE

```python
# Vocabulary
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn  # If available

# Data handling
import pandas as pd
import numpy as np
from tqdm import tqdm  # Progress bars

# Checkpointing
import json
import pickle
from pathlib import Path

# Statistics
from scipy import stats
import statsmodels.api as sm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Interactive plots

# Attention
import torch
```

### 10. ESTIMATED TIMELINE

| Phase | Time | Output |
|-------|------|--------|
| Vocab collection | 30 min | 1000+ verbs |
| Scenario generation | 10 min | 50,000 scenarios |
| Baseline testing | 3 hours | Baseline results |
| Ablation testing | 3 hours | Ablation results |
| Attention analysis | 1 hour | Attention data |
| Position analysis | 1 hour | Position data |
| Visualization | 30 min | All figures |
| **TOTAL** | **~9 hours** | Complete analysis |

### 11. KEY RESEARCH QUESTIONS

1. **Verb Semantics**: Which semantic features of verbs trigger inhibition?
   - Directness? Certainty? Formality? Emotional tone?

2. **Attention Patterns**: Where do inhibitor heads look for different verbs?
   - Do they attend to the verb itself? The agent? The location?

3. **Position Importance**: Which token positions carry the ToM signal?
   - Is it the verb position? End of sentence? Agent mention?

4. **Cross-Linguistic**: Does the circuit work the same across languages?
   - Same heads? Same patterns? Language-specific?

5. **Template Robustness**: Which templates are hardest/easiest?
   - Why do some structures trigger more inhibition?

---

## IMPLEMENTATION PRIORITY

1. **FIRST**: Build the infrastructure (checkpointing, streaming, data saving)
2. **SECOND**: Expand vocabulary (WordNet full tree)
3. **THIRD**: Run massive baseline + ablation sweeps
4. **FOURTH**: Harvest attention patterns
5. **FIFTH**: Position analysis
6. **SIXTH**: Comprehensive visualization




