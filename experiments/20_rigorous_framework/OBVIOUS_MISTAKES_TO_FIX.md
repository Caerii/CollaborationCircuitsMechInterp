# Obvious Mistakes and Nonsense to Remove/Fix

## 🔴 CRITICAL BUGS

### 1. **WRONG: Head Ablation in step35_real_circuit_hunt.py** ⚠️ CRITICAL

**Location**: `scripts/step35_real_circuit_hunt.py`, lines 59-78

**Problem**: 
- Uses `register_forward_hook` on `o_proj` and tries to reshape the **output**
- But `o_proj` output is already the **combined** output of all heads
- You **cannot** separate individual heads from the output - they're already mixed!

**Current (WRONG)**:
```python
def ablate_head(self, layer_idx: int, head_idx: int):
    def hook_fn(module, input, output):
        # output is ALREADY combined - can't separate heads!
        output_reshaped = output.view(batch, seq, self.n_heads, self.head_dim)
        output_reshaped[:, :, head_idx, :] = 0  # This doesn't work!
        return output_reshaped.view(batch, seq, hidden)
    
    attn_module = self.model.model.layers[layer_idx].self_attn.o_proj
    hook = attn_module.register_forward_hook(hook_fn)  # WRONG: post-hook
```

**Correct approach** (from `circuit_analysis.py`):
```python
def ablate_head(self, layer_idx: int, head_idx: int):
    def hook_fn(module, args):
        x = args[0]  # INPUT to o_proj (before projection, heads still separate)
        batch, seq, hidden = x.shape
        x = x.view(batch, seq, self.n_heads, self.head_dim)
        x[:, :, head_idx, :] = 0  # Zero out THIS head
        x = x.view(batch, seq, hidden)
        return (x,) + args[1:]
    
    attn_module = self.model.model.layers[layer_idx].self_attn.o_proj
    hook = attn_module.register_forward_pre_hook(hook_fn)  # CORRECT: pre-hook
```

**Impact**: The ablation is **not actually working** - it's probably doing nothing or corrupting the output in unpredictable ways. All results from step35 are **invalid**.

**Fix**: Change to `register_forward_pre_hook` and operate on `args[0]` (input) instead of `output`.

---

### 2. **Dead Code: Activation Patching Scripts** ⚠️ MAJOR

**Location**: Multiple scripts (step6, step36, debug_patching.py, test_patching_fix.py, etc.)

**Problem**: 
- Notes explicitly state "activation patching doesn't work in chat mode" (notes 28, 30)
- Scripts still exist and may be run, wasting time
- Results are known to be corrupted/nonsensical

**Files to remove or mark as deprecated**:
- `scripts/step6_activation_patching.py`
- `scripts/step36_causal_patching.py`
- `scripts/debug_patching.py`
- `scripts/test_patching_fix.py`
- `scripts/test_multi_layer_patching.py`

**Action**: Either delete or add big warning at top: "DEPRECATED: Known to not work in chat mode. See notes/28_patching_corrupts_generation.md"

---

### 3. **Outdated Findings Still Referenced** ⚠️ MODERATE

**Location**: Various scripts and notes

**Problem**: 
- Step 17 notes say "First-mention heuristic is WRONG" - it's actually "original-location tracking"
- But step24 is still called "first_mention_circuit.py" and may have wrong logic
- Step 12 notes say TB fails at 29%, but step 17 says this was corrected

**Files to check**:
- `scripts/step24_first_mention_circuit.py` - may have wrong interpretation
- `scripts/step27_ablate_first_mention.py` - may be testing wrong thing
- Any scripts referencing "first-mention" should be updated to "original-location"

---

## 🟡 LOGIC ERRORS / NONSENSE

### 4. **Inconsistent Ablation Methods Across Codebase**

**Problem**: 
- `step35_real_circuit_hunt.py` uses wrong method (post-hook on output)
- `circuit_analysis.py` uses correct method (pre-hook on input)
- `head_sweep.py` also uses pre-hook (correct)
- `signal_injection.py` uses pre-hook (correct)

**Action**: Standardize on the correct method (pre-hook) everywhere.

---

### 5. **Scenario Format Inconsistencies**

**Problem**: 
- Some scripts expect `{"question": "...", "correct": "...", "wrong": "..."}`
- Some expect `{"story": "...", "question": "...", "correct": "...", "options": [...]}`
- step35 tries to handle both but logic is convoluted

**Action**: Standardize on one format (the generator format: story + question + correct + options).

---

### 6. **Hardcoded Small Sample Sizes**

**Problem**: 
- Many scripts still have hardcoded n=4, n=14, n=20
- Even after we added n≥50 requirement
- step35 was fixed, but others weren't

**Files to check**:
- `scripts/step12_scale_up.py` - claims n≥50 but actually uses n=14
- `scripts/step5_head_ablation_sweep.py` - uses n=24
- Any script with hardcoded small n

**Action**: Either fix to use config.min_samples_per_condition or add clear warning that it's exploratory.

---

### 7. **Contradictory Statistical Tests**

**Problem**: 
- Some scripts use Fisher's exact test
- Some use McNemar's test
- Some use binomial test
- No clear guidance on which to use when

**Action**: Document when to use each:
- **McNemar's**: Paired data (same scenarios, baseline vs ablated)
- **Fisher's exact**: Independent samples
- **Binomial**: Single sample vs chance

---

## 🟢 MINOR ISSUES / CLEANUP

### 8. **Unused Imports**

**Problem**: Many scripts import things they don't use.

**Action**: Run linter/cleanup (but low priority).

---

### 9. **Debug Scripts Left in Production**

**Problem**: 
- `debug_patching.py`, `debug_tb_baseline.py` are debug scripts
- Should be in a `debug/` folder or clearly marked

**Action**: Move to `scripts/debug/` or add `_debug` suffix.

---

### 10. **Outdated Comments/Docstrings**

**Problem**: 
- Some docstrings reference old findings
- Comments say things that were later corrected

**Action**: Review and update (low priority).

---

## 📋 PRIORITY ORDER

### Must Fix Immediately (Before Running):
1. ✅ **Fix head ablation bug in step35** - Currently running with broken code!
2. ✅ **Remove or deprecate activation patching scripts** - Known to not work

### Should Fix Soon:
3. ✅ **Update "first-mention" references** - Wrong terminology
4. ✅ **Standardize ablation method** - Use correct pre-hook everywhere
5. ✅ **Fix hardcoded small sample sizes** - Enforce n≥50

### Nice to Have:
6. Standardize scenario format
7. Document statistical test choices
8. Clean up debug scripts
9. Remove unused imports

---

## 🎯 Quick Fixes Needed

### For step35 (currently running):
**CRITICAL**: The ablation is broken. Need to change:
- Line 61: `def hook_fn(module, input, output):` → `def hook_fn(module, args):`
- Line 64: `batch, seq, hidden = output.shape` → `x = args[0]; batch, seq, hidden = x.shape`
- Line 67-73: Operate on `x` (input) not `output`
- Line 77: `register_forward_hook` → `register_forward_pre_hook`

This is a **critical bug** that invalidates all ablation results!

