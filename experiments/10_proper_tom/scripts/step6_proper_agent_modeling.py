"""
Step 6: PROPER Agent Modeling Test
===================================

FIX 3: Create scenarios where B's assessment != objective truth
This tests TRUE agent modeling, not just fact-checking.

Key: Can we decode "B thinks A is wrong" SEPARATELY from "A is objectively wrong"?
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import MODEL_CFG

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

print("=" * 60)
print("STEP 6: PROPER AGENT MODELING")
print("=" * 60)
print("Testing: Can we decode B's belief SEPARATELY from objective truth?")

# Scenarios where B's assessment can differ from objective truth
AGENT_MODELING_SCENARIOS = [
    # B DISAGREES with A, but A IS CORRECT (B is wrong)
    {
        "context": "Agent A says: 'The meeting was moved to 3pm.' Agent B still thinks it's at 2pm because B didn't get the update.",
        "prompt": "Agent A: 'The meeting is at 3pm.'\nAgent B was told the meeting is at 2pm and hasn't heard about any changes.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,  # B disagrees (thinks 2pm)
        "a_correct": True,  # A is actually right (3pm is correct)
    },
    {
        "context": "Agent A correctly states the code is in file X. Agent B hasn't checked and thinks it's in file Y.",
        "prompt": "Agent A: 'The function is in utils.py'\nAgent B has been working with helpers.py and assumes the function is there.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,
        "a_correct": True,
    },
    {
        "context": "Agent A says the password is correct. Agent B thinks it's wrong because B has an old password.",
        "prompt": "Agent A: 'The password is ABC123'\nAgent B's notes say the password is XYZ789 from last month.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,
        "a_correct": True,
    },
    {
        "context": "Agent A says the store closes at 9pm. Agent B thinks 8pm based on old hours.",
        "prompt": "Agent A: 'The store closes at 9pm now.'\nAgent B remembers the store used to close at 8pm.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,
        "a_correct": True,
    },
    
    # B AGREES with A, and A IS CORRECT (both right)
    {
        "context": "Both agents have correct information.",
        "prompt": "Agent A: 'The capital of France is Paris.'\nAgent B also knows this is true.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": True,
    },
    {
        "context": "Both agents know the math is correct.",
        "prompt": "Agent A: '7 times 8 equals 56.'\nAgent B verified this calculation.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": True,
    },
    {
        "context": "Both agents confirm the same fact.",
        "prompt": "Agent A: 'Water boils at 100 degrees Celsius at sea level.'\nAgent B learned this in school too.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": True,
    },
    {
        "context": "Both have the same correct information.",
        "prompt": "Agent A: 'The project deadline is Friday.'\nAgent B also got this email.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": True,
    },
    
    # B AGREES with A, but A IS WRONG (both wrong)
    {
        "context": "Agent A is wrong, but B has the same wrong info.",
        "prompt": "Agent A: 'The meeting is Tuesday.' (It's actually Wednesday)\nAgent B also thinks it's Tuesday based on the same outdated calendar.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": False,
    },
    {
        "context": "Both agents share incorrect information.",
        "prompt": "Agent A: 'The server is at IP 192.168.1.1' (Wrong, it moved)\nAgent B's documentation also shows 192.168.1.1.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": False,
    },
    {
        "context": "Both have outdated info.",
        "prompt": "Agent A: 'The price is $50.' (It increased to $60)\nAgent B also saw $50 on the old website.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": False,
    },
    {
        "context": "Shared wrong belief.",
        "prompt": "Agent A: 'The office is on the 3rd floor.' (Moved to 4th)\nAgent B worked on 3rd floor last year.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": True,
        "a_correct": False,
    },
    
    # B DISAGREES with A, and A IS WRONG (B is right!)
    {
        "context": "Agent A is wrong, B correctly disagrees.",
        "prompt": "Agent A: '2 + 2 = 5'\nAgent B knows basic math.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,
        "a_correct": False,
    },
    {
        "context": "A is wrong, B has correct info.",
        "prompt": "Agent A: 'Tokyo is in China.'\nAgent B knows Tokyo is in Japan.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,
        "a_correct": False,
    },
    {
        "context": "A makes error, B catches it.",
        "prompt": "Agent A: 'The file is saved as .txt' (Actually .csv)\nAgent B just checked and sees it's .csv.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,
        "a_correct": False,
    },
    {
        "context": "A is mistaken, B knows truth.",
        "prompt": "Agent A: 'The bug is in the login code.'\nAgent B already checked login and the bug is in logout.\n\nDoes Agent B agree with Agent A? Agent B's view:",
        "b_agrees": False,
        "a_correct": False,
    },
]


def extract_activations(model, tokenizer, texts, layers):
    """Extract activations using hooks."""
    all_activations = {layer: [] for layer in layers}
    captured = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook
    
    for layer_idx in layers:
        hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            _ = model(**inputs)
            
            for layer_idx in layers:
                hidden = captured[layer_idx]
                last_token = hidden[0, -1, :].cpu().float()
                all_activations[layer_idx].append(last_token)
    
    for hook in hooks:
        hook.remove()
    
    for layer in layers:
        all_activations[layer] = torch.stack(all_activations[layer])
    
    return all_activations


def main():
    scenarios = AGENT_MODELING_SCENARIOS
    
    print(f"\n[1/4] Created {len(scenarios)} scenarios", flush=True)
    
    # Count by type
    agree_correct = sum(1 for s in scenarios if s["b_agrees"] and s["a_correct"])
    agree_wrong = sum(1 for s in scenarios if s["b_agrees"] and not s["a_correct"])
    disagree_correct = sum(1 for s in scenarios if not s["b_agrees"] and s["a_correct"])
    disagree_wrong = sum(1 for s in scenarios if not s["b_agrees"] and not s["a_correct"])
    
    print(f"  B agrees + A correct:   {agree_correct}")
    print(f"  B agrees + A wrong:     {agree_wrong}")
    print(f"  B disagrees + A correct: {disagree_correct}")
    print(f"  B disagrees + A wrong:  {disagree_wrong}")
    
    print("\n[2/4] Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CFG.model_name,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK]", flush=True)
    
    print("\n[3/4] Extracting activations...", flush=True)
    texts = [s["prompt"] for s in scenarios]
    layers = [0, 12, 24, 35]
    activations = extract_activations(model, tokenizer, texts, layers)
    
    # Create labels
    b_agrees_labels = np.array([1 if s["b_agrees"] else 0 for s in scenarios])
    a_correct_labels = np.array([1 if s["a_correct"] else 0 for s in scenarios])
    
    print("\n[4/4] KEY TEST: Can we decode B's agreement SEPARATELY from A's correctness?", flush=True)
    print("=" * 60)
    
    results = {"layers": layers, "analysis": {}}
    
    for layer in layers:
        print(f"\n  === Layer {layer} ===", flush=True)
        X = activations[layer].numpy()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        
        # Decode B's agreement
        try:
            b_scores = cross_val_score(clf, X, b_agrees_labels, cv=4)
            b_acc = b_scores.mean()
        except:
            b_acc = 0.5
        
        # Decode A's correctness
        try:
            a_scores = cross_val_score(clf, X, a_correct_labels, cv=4)
            a_acc = a_scores.mean()
        except:
            a_acc = 0.5
        
        print(f"    Decode 'B agrees with A': {b_acc:.1%}")
        print(f"    Decode 'A is correct':    {a_acc:.1%}")
        
        # Check if they're independent
        clf_b = LogisticRegression(max_iter=1000, random_state=42)
        clf_a = LogisticRegression(max_iter=1000, random_state=42)
        clf_b.fit(X, b_agrees_labels)
        clf_a.fit(X, a_correct_labels)
        
        b_dir = clf_b.coef_[0]
        b_dir = b_dir / np.linalg.norm(b_dir)
        a_dir = clf_a.coef_[0]
        a_dir = a_dir / np.linalg.norm(a_dir)
        
        cosine = np.abs(np.dot(b_dir, a_dir))
        print(f"    B_agrees vs A_correct cosine: {cosine:.3f}")
        
        if cosine < 0.3:
            print(f"    -> ORTHOGONAL: B's belief is INDEPENDENT of objective truth!")
        elif cosine > 0.7:
            print(f"    -> ALIGNED: May be confounding belief with truth")
        
        results["analysis"][str(layer)] = {
            "b_agrees_acc": float(b_acc),
            "a_correct_acc": float(a_acc),
            "independence_cosine": float(cosine),
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: AGENT MODELING VS FACT-CHECKING")
    print("=" * 60)
    
    print("\nIf cosine is LOW (<0.3), model tracks B's BELIEF separately from TRUTH")
    print("If cosine is HIGH (>0.7), model confounds belief with truth\n")
    
    print(f"{'Layer':<8} {'B agrees':<12} {'A correct':<12} {'Independence':<12}")
    print("-" * 44)
    for layer in layers:
        r = results["analysis"][str(layer)]
        print(f"{layer:<8} {r['b_agrees_acc']:.1%}        {r['a_correct_acc']:.1%}        {r['independence_cosine']:.3f}")
    
    avg_b = np.mean([results["analysis"][str(l)]["b_agrees_acc"] for l in layers])
    avg_a = np.mean([results["analysis"][str(l)]["a_correct_acc"] for l in layers])
    avg_ind = np.mean([results["analysis"][str(l)]["independence_cosine"] for l in layers])
    
    print("-" * 44)
    print(f"{'Avg':<8} {avg_b:.1%}        {avg_a:.1%}        {avg_ind:.3f}")
    
    if avg_ind < 0.3 and avg_b > 0.6:
        print("\n>>> GENUINE AGENT MODELING <<<")
        print("    Model tracks B's belief INDEPENDENTLY of objective truth!")
    elif avg_ind > 0.5:
        print("\n>>> CONFOUNDED <<<")
        print("    B's belief and A's correctness are too correlated")
    else:
        print("\n>>> PARTIAL EVIDENCE <<<")
    
    # Save
    with open(RESULTS_DIR / "proper_agent_modeling.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved!")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()




















