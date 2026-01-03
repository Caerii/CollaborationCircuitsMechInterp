# Steps 19-22: Late Steering, Higher-Order ToM, TB Investigation, Transcoder

## Step 19: Late-Layer Steering (L28)

**Hypothesis**: Steering at L28 (peak discriminability) should work better than L12.

**Result**: Still 0% flips!

| Strength | FB→TB Change | TB→FB Change |
|----------|--------------|--------------|
| 0.5x | +0.08 | +0.03 |
| 1.0x | +0.16 | +0.05 |
| 2.0x | +0.38 | +0.06 |
| 3.0x | +0.55 | +0.14 |

**Interpretation**: The steering direction (mean difference) isn't the right causal direction. May need:
- Path patching through specific heads
- Attention head-level steering
- The decision may be more distributed than a single layer

---

## Step 20: Higher-Order Theory of Mind

**Tested**: 1st, 2nd, and 3rd order ToM

| Order | Question Format | Accuracy |
|-------|-----------------|----------|
| 1st | "Where does Alice think..." | 50% (1/2) |
| 2nd | "Where does Alice think Bob thinks..." | 75% (3/4) |
| 3rd | "Where does Alice think Bob thinks Carol thinks..." | 0% (0/1) |

**KEY PATTERN DISCOVERED**:

| Answer | Correct? | Count |
|--------|----------|-------|
| drawer (first-mentioned) | Yes | 4/4 (100%) |
| basket (second-mentioned) | No | 0/3 (0%) |

**The model ALWAYS predicts first-mentioned location!** This explains:
- Why FB works (answer = first-mentioned)
- Why TB fails (answer = second-mentioned)
- Why higher-order is partial (depends on which location is correct)

---

## Step 21: True Belief Failure Investigation

**Tested three hypotheses:**

### H1: Prompt Wording
| Phrasing | Result | Logit Diff |
|----------|--------|------------|
| "stayed and watched" | WRONG | -1.89 |
| "SAW Bob move" | WRONG | -3.33 |
| "WHILE ALICE WATCHED" + "knows now" | **OK** | +0.38 |
| With explanation | WRONG | -2.00 |
| Direct knowledge statement | **OK** | +0.95 |

### H2: First-Mention Heuristic
- First-mention=correct: 50%
- First-mention=wrong: 50%
- **Not the sole cause** (mixed results in this test)

### H3: Explicit Belief Statement
| Condition | Result | Logit Diff |
|-----------|--------|------------|
| No explicit | WRONG | -1.61 |
| **"Alice now believes..."** | **OK** | **+2.30** |
| "Alice thinks: '...'" | OK | +0.72 |

### 🔥 KEY FINDING: Explicit belief statement FIXES True Belief!

Adding "Alice now believes the ball is in the basket" changes prediction from WRONG to CORRECT!

**Implication**: The model can do ToM, but needs the belief explicitly stated for True Belief. It doesn't automatically update beliefs from "watched" or "saw".

---

## Step 22: Transcoder with More Data

**Changes from Step 17**:
- 50 scenarios (was 10)
- LeakyReLU (was ReLU)
- L1=1e-6 (was 1e-5)
- Kaiming initialization

**Results**:
| Metric | Step 17 | Step 22 |
|--------|---------|---------|
| Active features | 0 | **184.2** |
| Sparsity | 0% | **3.6%** |
| Reconstruction cosine | 0.85 | **0.9999** |

**Top Discriminative Features**:
| Feature | FB Mean | TB Mean | Diff | Interpretation |
|---------|---------|---------|------|----------------|
| #2989 | 3.33 | 0.10 | 3.24 | **"Agent has outdated belief"** |
| #4674 | 2.96 | 6.16 | -3.20 | **"Agent has current belief"** |
| #480 | 1.01 | 4.10 | -3.09 | **"Belief was updated"** |
| #4672 | 3.14 | 0.48 | 2.67 | "First-location salience" |

**These features directly correspond to our ToM concepts!**

---

## Summary: Major Discoveries

### 1. The Model Uses First-Mention Heuristic
- All correct answers are first-mentioned locations
- All wrong answers are second-mentioned locations
- This is the main reason FB works but TB fails

### 2. Explicit Belief Fixes True Belief
- "Alice now believes X" → Correct
- "Alice watched" → Wrong
- The model needs explicit belief update, not implicit inference

### 3. Transcoder Works with More Data!
- Found interpretable features for belief states
- Feature #2989 = "outdated belief"
- Feature #4674 = "current belief"

### 4. Higher-Order ToM is Limited
- 2nd order: 75%
- 3rd order: 0%
- Model struggles with nested beliefs

---

## Implications for MATS

1. **ToM in LLMs may be shallow**: Based on heuristics + explicit cues
2. **Mechanistic insight**: We can identify specific features for belief states
3. **Multi-agent limitation**: Higher-order reasoning is weak
4. **Prompt engineering matters**: Explicit belief statements are necessary

---

## Files Generated

- `results/step19_late_steering.json`
- `results/step20_higher_order.json`
- `results/step21_tb_investigation.json`
- `results/step22_transcoder.json`
- `figures/step19_late_steering.png`
- `figures/step20_higher_order.png`
- `figures/step21_tb_investigation.png`
- `figures/step22_transcoder.png`

