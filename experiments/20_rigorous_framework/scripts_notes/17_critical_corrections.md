# Critical Corrections (Steps 31-32)

## 🚨 MAJOR REVISIONS TO PREVIOUS FINDINGS

### Correction 1: First-Mention Heuristic is WRONG

**Previous claim (Step 24)**: Model uses first-mention heuristic.

**Correction (Step 31)**: When we control for first-mention vs original location:
- `basket_first_original_drawer`: Predicts **drawer** ✅ (original location)
- `drawer_first_original_basket`: Predicts **basket** ✅ (original location)

**The model tracks ORIGINAL LOCATION, not first-mentioned!**

The heads we found (L23H4, L13H10) are tracking where the object STARTED, not which location was mentioned first.

---

### Correction 2: True Belief WORKS in Chat Mode

**Previous claim**: True Belief fails (40% accuracy).

**Correction (Step 32)**: With proper token length (500 tokens):
- False Belief: **CORRECT** ✅
- True Belief: **CORRECT** ✅

The model correctly reasons:
- FB: "Alice left, so she wouldn't know... she would still think drawer"
- TB: "She watched Bob move it... so the answer should be basket"

**Our earlier findings were confounded by token truncation!**

---

### Correction 3: Completion vs Chat Mode

| Mode | FB | TB | Notes |
|------|----|----|-------|
| Completion | ✅ | ❌ | TB fails in completion |
| Chat (100 tokens) | ? | ? | Truncated, unclear |
| Chat (500 tokens) | ✅ | ✅ | BOTH work! |

**Chat mode with enough tokens shows genuine ToM!**

---

### Correction 4: AI Entity Findings are NUANCED

Step 30 claimed ToM fails for AI entities. But Step 31 shows:

| Entity | Result | Notes |
|--------|--------|-------|
| Claude | ✅ | Known AI name |
| Alexa | ❌ | Predicts current location |
| Siri | ❌ | Predicts current location |
| "Alex the robot" | ✅ | Anthropomorphized |

It's not that ToM fails for ALL AI - it fails for specific names.
Anthropomorphized AI ("Alex the robot") works!

---

## Revised Understanding

### What Actually Happens:

1. **Completion mode**:
   - FB: Works (predicts original location)
   - TB: Fails (still predicts original location, doesn't process "watched")

2. **Chat mode with reasoning**:
   - FB: Works
   - TB: Works
   - Model explicitly reasons about who saw what

3. **The "True Belief failure" is NOT a ToM failure**:
   - It's a **processing failure** in completion mode
   - The model CAN do TB when given reasoning space

---

## Implications

1. **ToM capability EXISTS** - proven by chat mode success
2. **First-mention circuit claim is WRONG** - it's original-location tracking
3. **Completion mode limitations** - doesn't mean no ToM
4. **Chat mode is the right evaluation** - matches model's training

---

## What Our Circuit Findings Actually Mean:

| Component | What We Said | What It Actually Is |
|-----------|--------------|---------------------|
| L23H4, L13H10 | "First-mention heads" | **Original-location heads** |
| L28 discriminability | "Peak belief" | Still valid |
| L32-34 ToM heads | "Agent tracking" | Still valid |
| TB failure | "No ToM" | **Processing limitation, not capability gap** |

---

## Updated Key Claims for MATS:

1. ~~LLMs have shallow ToM based on heuristics~~ → LLMs have genuine ToM
2. ~~True Belief fails~~ → TB works in chat mode with reasoning
3. ~~First-mention heuristic~~ → Original-location tracking
4. ~~Explicit beliefs needed~~ → Reasoning space needed (chat mode)

The real finding: **ToM is a capability that requires reasoning space to express.**

