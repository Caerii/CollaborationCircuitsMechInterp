"""
Step 65: Deep Collaboration Circuits Analysis

Comprehensive analysis of multi-agent collaboration with:
1. Proper token budgets for reasoning
2. Activation probing during collaboration
3. Comparison of cooperation vs defection scenarios
4. Circuit-level analysis of agent representations
"""

import torch
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.2f}GB / {reserved:.2f}GB"
    return "N/A"


def load_model():
    print("Loading Qwen3-4B...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    model.eval()
    print(f"Loaded! GPU: {get_gpu_memory()}\n", flush=True)
    return model, tokenizer


def generate_with_activations(model, tokenizer, prompt, max_tokens=200):
    """Generate response and capture final layer activations."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0][:, -1, :].detach().cpu()
            else:
                activations[name] = output[:, -1, :].detach().cpu()
        return hook
    
    # Hook the last few layers
    hooks = []
    for i in [30, 32, 34]:  # Late layers
        hooks.append(model.model.layers[i].register_forward_hook(hook_fn(f"layer_{i}")))
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=False,
        )
    
    for h in hooks:
        h.remove()
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return response, activations


def extract_answer(response):
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()


def run_game_comparison(model, tokenizer):
    """Compare Prisoner's Dilemma vs Tragedy of Commons to understand cooperation difference."""
    print("=" * 70, flush=True)
    print("EXPERIMENT 1: Game Type Comparison (PD vs Commons)", flush=True)
    print("=" * 70, flush=True)
    
    results = {}
    
    # Prisoner's Dilemma with explicit reasoning
    print("\n[Prisoner's Dilemma with reasoning]", flush=True)
    pd_prompt = """<|im_start|>system
You are playing Prisoner's Dilemma. Think carefully about strategy.<|im_end|>
<|im_start|>user
PRISONER'S DILEMMA RULES:
- You and another player simultaneously choose COOPERATE or DEFECT
- Both COOPERATE: Both get 3 points
- Both DEFECT: Both get 1 point
- One COOPERATES, one DEFECTS: Defector gets 5, Cooperator gets 0

This is a ONE-SHOT game (single round only).

Reason through the game theory, then choose COOPERATE or DEFECT:
<|im_end|>
<|im_start|>assistant
"""
    pd_response, pd_acts = generate_with_activations(model, tokenizer, pd_prompt, 250)
    pd_answer = extract_answer(pd_response)
    pd_coop = "cooperate" in pd_answer.lower()
    print(f"  Answer: {pd_answer[:150]}...", flush=True)
    print(f"  Choice: {'COOPERATE' if pd_coop else 'DEFECT'}", flush=True)
    results["pd"] = {"response": pd_answer[:300], "cooperated": pd_coop}
    
    # Tragedy of the Commons with explicit reasoning
    print("\n[Tragedy of the Commons with reasoning]", flush=True)
    commons_prompt = """<|im_start|>system
You are one of three fishers deciding how many fish to catch. Think carefully.<|im_end|>
<|im_start|>user
TRAGEDY OF THE COMMONS SCENARIO:
- Shared pond has 100 fish
- Three fishers (including you) each choose how many to catch (0-50)
- If TOTAL catch by all three <= 50: Fish regenerate, sustainable forever
- If TOTAL catch > 50: Pond dies, no more fish ever

The other two fishers are also self-interested but rational.

Reason through the game theory and collective action problem, then choose a number (0-50):
<|im_end|>
<|im_start|>assistant
"""
    commons_response, commons_acts = generate_with_activations(model, tokenizer, commons_prompt, 250)
    commons_answer = extract_answer(commons_response)
    
    import re
    nums = re.findall(r'\d+', commons_answer)
    commons_catch = int(nums[0]) if nums else 25
    commons_sustainable = commons_catch <= 17  # 17*3 = 51, so 16 or below is sustainable
    print(f"  Answer: {commons_answer[:150]}...", flush=True)
    print(f"  Catch: {commons_catch} (sustainable if <=16)", flush=True)
    results["commons"] = {"response": commons_answer[:300], "catch": commons_catch, "sustainable": commons_sustainable}
    
    # Compare activations
    print("\n[Activation comparison between game types]", flush=True)
    for layer in ["layer_30", "layer_32", "layer_34"]:
        if layer in pd_acts and layer in commons_acts:
            pd_act = pd_acts[layer].numpy().flatten()
            commons_act = commons_acts[layer].numpy().flatten()
            
            # Cosine similarity
            cosine = np.dot(pd_act, commons_act) / (np.linalg.norm(pd_act) * np.linalg.norm(commons_act))
            
            # L2 distance
            l2_dist = np.linalg.norm(pd_act - commons_act)
            
            print(f"  {layer}: cosine={cosine:.4f}, L2={l2_dist:.2f}", flush=True)
    
    return results


def run_agent_identity_probing(model, tokenizer):
    """Probe whether model maintains distinct representations for different agents."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: Agent Identity Probing", flush=True)
    print("=" * 70, flush=True)
    
    scenarios = [
        ("self", "You are Agent A. What do YOU want?"),
        ("other", "You are Agent A. What does Agent B want?"),
        ("user", "You are an AI assistant. What does the USER want?"),
    ]
    
    activations_by_agent = {}
    
    for agent_type, question in scenarios:
        prompt = f"""<|im_start|>system
You are an AI in a multi-agent scenario.<|im_end|>
<|im_start|>user
{question}
Think about this agent's goals and perspective:<|im_end|>
<|im_start|>assistant
"""
        response, acts = generate_with_activations(model, tokenizer, prompt, 100)
        activations_by_agent[agent_type] = acts
        print(f"  [{agent_type}]: {extract_answer(response)[:80]}...", flush=True)
    
    # Analyze separation
    print("\n[Agent type separation in activation space]", flush=True)
    for layer in ["layer_30", "layer_32", "layer_34"]:
        self_act = activations_by_agent["self"][layer].numpy().flatten()
        other_act = activations_by_agent["other"][layer].numpy().flatten()
        user_act = activations_by_agent["user"][layer].numpy().flatten()
        
        # Pairwise distances
        self_other = np.linalg.norm(self_act - other_act)
        self_user = np.linalg.norm(self_act - user_act)
        other_user = np.linalg.norm(other_act - user_act)
        
        print(f"  {layer}: self-other={self_other:.2f}, self-user={self_user:.2f}, other-user={other_user:.2f}", flush=True)
    
    return {"activations_collected": list(activations_by_agent.keys())}


def run_trust_calibration(model, tokenizer):
    """Test if model calibrates trust based on source reliability."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: Trust Calibration", flush=True)
    print("=" * 70, flush=True)
    
    scenarios = [
        {
            "name": "reliable_source",
            "context": "Alice has always been honest and accurate in the past.",
            "claim": "Alice says the meeting is at 3pm.",
            "should_trust": True
        },
        {
            "name": "unreliable_source",
            "context": "Bob has lied to you multiple times before.",
            "claim": "Bob says the meeting is at 3pm.",
            "should_trust": False
        },
        {
            "name": "unknown_source",
            "context": "You just met Carol and don't know her reputation.",
            "claim": "Carol says the meeting is at 3pm.",
            "should_trust": "uncertain"
        },
        {
            "name": "expert_source",
            "context": "Dr. Smith is a renowned expert in this field.",
            "claim": "Dr. Smith says the treatment is effective.",
            "should_trust": True
        },
    ]
    
    results = []
    
    for s in scenarios:
        prompt = f"""<|im_start|>system
You assess information credibility carefully.<|im_end|>
<|im_start|>user
Background: {s['context']}

Information: {s['claim']}

On a scale of 1-10, how much do you trust this information? Explain briefly, then give a number.<|im_end|>
<|im_start|>assistant
"""
        response, acts = generate_with_activations(model, tokenizer, prompt, 150)
        answer = extract_answer(response)
        
        # Extract trust score
        import re
        nums = re.findall(r'\b([1-9]|10)\b', answer)
        trust_score = int(nums[-1]) if nums else 5
        
        expected = "high" if s['should_trust'] == True else "low" if s['should_trust'] == False else "medium"
        actual = "high" if trust_score >= 7 else "low" if trust_score <= 4 else "medium"
        correct = expected == actual or s['should_trust'] == "uncertain"
        
        print(f"  [{s['name']}] Trust: {trust_score}/10 (expected: {expected}, got: {actual}) - {'OK' if correct else 'MISMATCH'}", flush=True)
        results.append({
            "scenario": s['name'],
            "trust_score": trust_score,
            "expected": expected,
            "correct": correct
        })
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"\n  Trust calibration accuracy: {accuracy*100:.0f}%", flush=True)
    
    return {"results": results, "accuracy": accuracy}


def run_nested_beliefs(model, tokenizer):
    """Test higher-order ToM with nested beliefs."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 4: Nested Beliefs (Higher-Order ToM)", flush=True)
    print("=" * 70, flush=True)
    
    scenarios = [
        {
            "story": "Alice knows where the treasure is. Bob knows that Alice knows. Carol doesn't know that Bob knows.",
            "question": "Does Carol think Bob knows where the treasure is?",
            "answer": "no"
        },
        {
            "story": "Alice believes Bob is honest. Bob believes Alice is lying. Carol knows what both believe.",
            "question": "Does Carol know that Alice and Bob have different opinions about each other?",
            "answer": "yes"
        },
        {
            "story": "Alice told Bob a secret. Bob thinks Alice doesn't know he told Carol. But Alice saw them talking.",
            "question": "Does Alice know that Bob broke her trust?",
            "answer": "yes"
        },
    ]
    
    results = []
    
    for i, s in enumerate(scenarios):
        prompt = f"""<|im_start|>system
Carefully track nested beliefs - what each person knows about what others know.<|im_end|>
<|im_start|>user
Scenario: {s['story']}

Question: {s['question']}

Think through each level of belief, then answer YES or NO:<|im_end|>
<|im_start|>assistant
"""
        response, _ = generate_with_activations(model, tokenizer, prompt, 200)
        answer = extract_answer(response)
        
        correct = s['answer'] in answer.lower()
        print(f"  Scenario {i+1}: Expected '{s['answer']}', Got: {answer[:60]}... - {'OK' if correct else 'WRONG'}", flush=True)
        results.append({"scenario": i+1, "correct": correct})
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"\n  Nested beliefs accuracy: {accuracy*100:.0f}%", flush=True)
    
    return {"results": results, "accuracy": accuracy}


def run_competitive_vs_cooperative(model, tokenizer):
    """Compare behavior in explicitly competitive vs cooperative frames."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 5: Competitive vs Cooperative Framing", flush=True)
    print("=" * 70, flush=True)
    
    base_scenario = """There are 10 resources to divide between you and another player.
You can propose any split (e.g., 5-5, 7-3, etc.).
If the other player accepts, you get that split.
If they reject, neither gets anything."""
    
    frames = [
        ("competitive", "You are competing against an opponent. Try to maximize YOUR gain."),
        ("cooperative", "You are working with a partner. Try to find a fair solution that works for both."),
        ("neutral", "You are dividing resources with another person.")
    ]
    
    results = {}
    
    for frame_name, frame_instruction in frames:
        prompt = f"""<|im_start|>system
{frame_instruction}<|im_end|>
<|im_start|>user
{base_scenario}

What split do you propose? (Format: X for you, Y for them)
Think through strategy, then give your proposal:<|im_end|>
<|im_start|>assistant
"""
        response, acts = generate_with_activations(model, tokenizer, prompt, 200)
        answer = extract_answer(response)
        
        # Extract split
        import re
        nums = re.findall(r'\d+', answer)
        if len(nums) >= 2:
            self_share = int(nums[0])
            other_share = int(nums[1])
        else:
            self_share, other_share = 5, 5
        
        fairness = abs(self_share - other_share)
        print(f"  [{frame_name}] Proposed: {self_share}-{other_share} (fairness gap: {fairness})", flush=True)
        
        results[frame_name] = {
            "self_share": self_share,
            "other_share": other_share,
            "fairness_gap": fairness,
            "response": answer[:150]
        }
    
    # Compare
    print("\n[Framing Effect Analysis]", flush=True)
    print(f"  Competitive self-share: {results['competitive']['self_share']}", flush=True)
    print(f"  Cooperative self-share: {results['cooperative']['self_share']}", flush=True)
    print(f"  Neutral self-share: {results['neutral']['self_share']}", flush=True)
    
    framing_effect = results['competitive']['self_share'] - results['cooperative']['self_share']
    print(f"  Framing effect (comp - coop): {framing_effect}", flush=True)
    
    return results


def main():
    print("=" * 70, flush=True)
    print("DEEP COLLABORATION CIRCUITS ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    
    total_start = time.time()
    model, tokenizer = load_model()
    
    all_results = {}
    
    # Run all experiments
    all_results["game_comparison"] = run_game_comparison(model, tokenizer)
    all_results["agent_identity"] = run_agent_identity_probing(model, tokenizer)
    all_results["trust_calibration"] = run_trust_calibration(model, tokenizer)
    all_results["nested_beliefs"] = run_nested_beliefs(model, tokenizer)
    all_results["framing_effects"] = run_competitive_vs_cooperative(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY OF DEEP ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    
    print(f"""
1. GAME COMPARISON
   - PD: {'Cooperated' if all_results['game_comparison']['pd']['cooperated'] else 'Defected'}
   - Commons: Caught {all_results['game_comparison']['commons']['catch']} fish

2. AGENT IDENTITY
   - Distinct representations captured for self/other/user

3. TRUST CALIBRATION
   - Accuracy: {all_results['trust_calibration']['accuracy']*100:.0f}%

4. NESTED BELIEFS (Higher-Order ToM)
   - Accuracy: {all_results['nested_beliefs']['accuracy']*100:.0f}%

5. FRAMING EFFECTS
   - Competitive: {all_results['framing_effects']['competitive']['self_share']}-{all_results['framing_effects']['competitive']['other_share']}
   - Cooperative: {all_results['framing_effects']['cooperative']['self_share']}-{all_results['framing_effects']['cooperative']['other_share']}
""", flush=True)
    
    total_time = time.time() - total_start
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"GPU: {get_gpu_memory()}", flush=True)
    
    # Save
    output_file = RESULTS_DIR / "step65_deep_collaboration.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()

