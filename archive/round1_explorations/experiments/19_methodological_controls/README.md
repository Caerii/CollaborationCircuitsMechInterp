# Experiment 19: Theory of Mind Circuit Discovery in Qwen3-4B

---

## 🎉 BREAKTHROUGH FINDING (Dec 24, 2025)

**Qwen3-4B HAS Theory of Mind when properly prompted!**

### The Critical Mistake
We were testing an **instruction-tuned reasoning model** as if it were a **raw completion model**.

### Corrected Results (with proper chat prompting):

| Scenario Type | Accuracy | Previous (Raw Completion) |
|---------------|----------|---------------------------|
| **False Belief** | **90%** (9/10) | ~50% |
| **True Belief** | **83%** (5/6) | ~0% |
| **Communication** | **50%** (2/4) | ~0% |
| **OVERALL** | **80%** | ~35% |

### What Changed

**Old method (wrong):**
```python
prompt = "Alice put ball in drawer. Alice left. Bob moved ball to basket. Alice looks in the"
probs = model.get_next_token_probs(prompt)  # Wrong!
```

**New method (correct):**
```python
prompt = """<|im_start|>system
Think step by step in <think> tags. Then give one-word answer.<|im_end|>
<|im_start|>user
Alice put the ball in the drawer. Alice left. Bob moved it to the basket. Alice returned.
Where will Alice look?<|im_end|>
<|im_start|>assistant
"""
response = model.generate(prompt, max_tokens=500)  # Let it reason!
```

### Example Correct ToM Reasoning:
```
<think>
Alice's last interaction with the ball was when she put it in the drawer. 
She didn't see what happened after she left. Bob moved it to the basket, 
but Alice isn't aware of that. So when she comes back, she would think 
the ball is still in the drawer because she last saw it there.
</think>

drawer
```

**This is genuine Theory of Mind reasoning!**

See `BREAKTHROUGH_FINDING.md` for full details.

---

## Earlier Findings (Context Only)

The sections below document earlier exploration using raw completion testing. These findings are now understood to be **artifacts of incorrect methodology** for testing instruction-tuned models.

---

## Previous Investigation Summary

### What We Thought We Found (Invalidated)
- "Inhibitory circuit" in L32-35 causing ToM failures
- Verb-type effects (action vs belief verbs)
- 100% accuracy with circuit ablation

### What Actually Happened
- Raw completion testing doesn't work for chat models
- The model needs reasoning space (`<think>` tags)
- With proper prompting, the model shows genuine ToM

---

## Project Structure

```
19_methodological_controls/
├── README.md                    # This file
├── BREAKTHROUGH_FINDING.md      # NEW: Key discovery about proper prompting
├── FINDINGS.md                  # Earlier technical findings (context)
├── SPEEDRUN_FINDINGS.md         # Speed investigation results
│
├── scripts/
│   ├── step1-58_*.py           # Earlier experiments (raw completion)
│   ├── step59_quick_reasoning.py    # Initial reasoning test
│   ├── step60_comprehensive_reasoning.py  # Full validation
│   ├── step61_true_belief_test.py   # True belief clarity test
│   └── step62_final_validation.py   # 20-scenario validation
│
├── results/                     # JSON output files
├── figures/                     # Visualizations
├── toolkit/                     # ToM prompt engineering toolkit
└── docs/archive/               # Earlier research notes
```

---

## Key Takeaways

### 1. Testing Matters
Instruction-tuned models need instruction-tuned testing:
- Use chat format with system/user/assistant tags
- Give the model space to reason (500+ tokens)
- Look at generated responses, not just probabilities

### 2. Qwen3-4B Has ToM
When properly prompted:
- 90% False Belief accuracy
- 83% True Belief accuracy  
- 80% overall

### 3. Limitations
- Communication scenarios are harder (50%)
- Model sometimes second-guesses correct reasoning
- Very explicit phrasing helps True Belief scenarios

---

## Running the Validation

```bash
# Quick 3-scenario test
python scripts/step59_quick_reasoning.py

# Full 20-scenario validation  
python scripts/step62_final_validation.py
```

---

## Implications for MATS Project

This changes our research direction:

### Still Valid Research Questions:
- How does the model represent belief states internally during reasoning?
- What circuits are responsible for "Alice knows" vs "Alice doesn't know"?
- How does the `<think>` mechanism enable better reasoning?

### Invalidated Directions:
- The "inhibitory circuit" hypothesis
- Claims that the model lacks ToM
- Focus on verb-type as causal factor

---

## See Also

- `BREAKTHROUGH_FINDING.md` - Full details on the discovery
- `scripts/step62_final_validation.py` - Validation code
- `results/step62_final_validation.json` - Validation results
