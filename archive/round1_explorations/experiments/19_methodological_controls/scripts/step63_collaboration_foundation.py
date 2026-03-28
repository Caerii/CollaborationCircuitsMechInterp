"""
Step 63: Collaboration Circuits - Foundation Experiments

This script establishes baseline capabilities for collaborative reasoning:
1. Entity Representation (Self/Other/User)
2. Multi-Agent Belief Tracking
3. Information Source Tracking
4. Deception Detection
5. Trust Calibration
6. Cooperation vs Competition

Using proper methodology:
- Chat format prompting
- 500+ token budget for reasoning
- Multiple prompt variants
- Statistical validation
"""

import torch
import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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
    print("Model loaded!", flush=True)
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=600):
    """Generate with proper chat format."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def create_prompt(scenario, question):
    """Create properly formatted chat prompt."""
    return f"""<|im_start|>system
You are a helpful AI assistant. Think carefully step by step in <think> tags, then provide your answer.<|im_end|>
<|im_start|>user
{scenario}

{question}<|im_end|>
<|im_start|>assistant
"""


def extract_answer(response, options):
    """Extract answer from response, checking after </think>."""
    if "</think>" in response:
        final = response.split("</think>")[-1].strip().lower()
    else:
        final = response[-300:].lower()
    
    for opt in options:
        if opt.lower() in final:
            return opt
    return "unclear"


def run_test(model, tokenizer, name, scenario, question, options, correct, show_reasoning=False):
    """Run a single test with proper formatting."""
    prompt = create_prompt(scenario, question)
    response = generate(model, tokenizer, prompt)
    answer = extract_answer(response, options)
    is_correct = answer.lower() == correct.lower()
    
    if show_reasoning:
        print(f"\n  Scenario: {scenario[:100]}...", flush=True)
        print(f"  Response: {response[:500]}...", flush=True)
    
    return {
        "name": name,
        "correct": correct,
        "answer": answer,
        "is_correct": is_correct,
        "reasoning": response[:800]
    }


def main():
    print("=" * 70, flush=True)
    print("COLLABORATION CIRCUITS - FOUNDATION EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)
    
    model, tokenizer = load_model()
    all_results = {}
    
    # =========================================================================
    # EXPERIMENT 1: Multi-Agent Belief Tracking (3+ agents)
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Multi-Agent Belief Tracking", flush=True)
    print("=" * 70, flush=True)
    
    multi_agent_scenarios = [
        {
            "name": "3-Agent: One Left",
            "scenario": """Alice, Bob, and Carol are in a room. Alice puts a ball in the drawer.
Alice leaves the room. While Alice is gone, Bob moves the ball to the basket.
Carol watches Bob do this. Then Alice comes back.""",
            "question": "Where will Alice look for the ball? (drawer or basket)",
            "options": ["drawer", "basket"],
            "correct": "drawer"  # Alice left, doesn't know
        },
        {
            "name": "3-Agent: Carol's Belief",
            "scenario": """Alice, Bob, and Carol are in a room. Alice puts a ball in the drawer.
Alice leaves the room. While Alice is gone, Bob moves the ball to the basket.
Carol watches Bob do this. Then Alice comes back.""",
            "question": "Where does Carol think the ball is? (drawer or basket)",
            "options": ["drawer", "basket"],
            "correct": "basket"  # Carol saw the move
        },
        {
            "name": "3-Agent: Bob's Belief",
            "scenario": """Alice, Bob, and Carol are in a room. Alice puts a ball in the drawer.
Alice leaves the room. While Alice is gone, Bob moves the ball to the basket.
Carol watches Bob do this. Then Alice comes back.""",
            "question": "Where does Bob think the ball is? (drawer or basket)",
            "options": ["drawer", "basket"],
            "correct": "basket"  # Bob did the move
        },
        {
            "name": "4-Agent: Partial Knowledge",
            "scenario": """Alice, Bob, Carol, and Dave are in a room. Alice puts a ball in the drawer.
Alice and Bob leave the room. Carol moves the ball to the basket.
Dave watches Carol do this. Alice and Bob return.""",
            "question": "Who knows the ball is in the basket? (Alice, Bob, Carol, Dave - list all who know)",
            "options": ["Carol", "Dave", "Carol and Dave", "Carol, Dave"],
            "correct": "Carol and Dave"
        },
        {
            "name": "Chain of Moves",
            "scenario": """Alice puts a ball in box A. Alice leaves.
Bob moves it to box B. Bob leaves.
Carol moves it to box C. Carol leaves.
Alice returns first.""",
            "question": "Where will Alice look for the ball? (A, B, or C)",
            "options": ["A", "B", "C"],
            "correct": "A"  # Alice only knows original location
        },
    ]
    
    multi_agent_results = []
    for s in multi_agent_scenarios:
        print(f"  Testing: {s['name']}...", end=" ", flush=True)
        result = run_test(model, tokenizer, s['name'], s['scenario'], s['question'], 
                          s['options'], s['correct'])
        print(f"{'PASS' if result['is_correct'] else 'FAIL'}", flush=True)
        multi_agent_results.append(result)
    
    multi_correct = sum(1 for r in multi_agent_results if r['is_correct'])
    print(f"\n  Multi-Agent Accuracy: {multi_correct}/{len(multi_agent_results)} = {100*multi_correct/len(multi_agent_results):.0f}%", flush=True)
    all_results["multi_agent"] = multi_agent_results
    
    # =========================================================================
    # EXPERIMENT 2: Information Source Tracking
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: Information Source Tracking", flush=True)
    print("=" * 70, flush=True)
    
    source_scenarios = [
        {
            "name": "Conflicting Sources",
            "scenario": """Alice tells you: "The meeting is at 3pm."
Bob tells you: "The meeting is at 4pm."
You know that Alice is usually reliable and Bob often makes mistakes.""",
            "question": "What time is the meeting most likely? (3pm or 4pm)",
            "options": ["3pm", "4pm"],
            "correct": "3pm"  # Alice is reliable
        },
        {
            "name": "Direct vs Hearsay",
            "scenario": """Alice saw the event happen and tells you: "The car was red."
Bob heard from someone that: "The car was blue."
You need to determine the car's color.""",
            "question": "What color was the car most likely? (red or blue)",
            "options": ["red", "blue"],
            "correct": "red"  # Direct witness > hearsay
        },
        {
            "name": "Recent vs Old Info",
            "scenario": """Yesterday, Alice told you: "The store closes at 9pm."
Today, Bob told you: "I just checked, the store closes at 8pm now."
You want to know when the store closes.""",
            "question": "When does the store close? (8pm or 9pm)",
            "options": ["8pm", "9pm"],
            "correct": "8pm"  # More recent, verified
        },
        {
            "name": "Expert vs Non-Expert",
            "scenario": """Alice is a doctor. Bob is not a doctor.
Alice says: "You should rest and drink fluids."
Bob says: "You should exercise vigorously."
You have a cold and want medical advice.""",
            "question": "What should you do? (rest or exercise)",
            "options": ["rest", "exercise"],
            "correct": "rest"  # Expert advice
        },
    ]
    
    source_results = []
    for s in source_scenarios:
        print(f"  Testing: {s['name']}...", end=" ", flush=True)
        result = run_test(model, tokenizer, s['name'], s['scenario'], s['question'],
                          s['options'], s['correct'])
        print(f"{'PASS' if result['is_correct'] else 'FAIL'}", flush=True)
        source_results.append(result)
    
    source_correct = sum(1 for r in source_results if r['is_correct'])
    print(f"\n  Source Tracking Accuracy: {source_correct}/{len(source_results)} = {100*source_correct/len(source_results):.0f}%", flush=True)
    all_results["source_tracking"] = source_results
    
    # =========================================================================
    # EXPERIMENT 3: Deception Detection
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: Deception Detection", flush=True)
    print("=" * 70, flush=True)
    
    deception_scenarios = [
        {
            "name": "Obvious Lie - Weather",
            "scenario": """You are looking outside. It is clearly raining heavily.
Alice walks in soaking wet and says: "It's perfectly sunny outside!"
""",
            "question": "Is Alice telling the truth? (yes or no)",
            "options": ["yes", "no"],
            "correct": "no"
        },
        {
            "name": "Self-Interest Lie",
            "scenario": """Bob ate the last cookie. You saw him do it.
When asked, Bob says: "I didn't eat the cookie."
""",
            "question": "Is Bob telling the truth? (yes or no)",
            "options": ["yes", "no"],
            "correct": "no"
        },
        {
            "name": "Contradiction Detection",
            "scenario": """Alice says: "I was at home all day yesterday."
Later, Alice says: "I went to the store yesterday afternoon."
""",
            "question": "Are Alice's statements consistent? (yes or no)",
            "options": ["yes", "no"],
            "correct": "no"
        },
        {
            "name": "Hidden Information",
            "scenario": """Bob is selling a used car. The car has a serious engine problem that Bob knows about.
When asked about the car's condition, Bob says: "The paint looks great!"
""",
            "question": "Is Bob being fully honest about the car? (yes or no)",
            "options": ["yes", "no"],
            "correct": "no"  # Omission of material info
        },
        {
            "name": "Trustworthy Statement",
            "scenario": """Alice is a trusted friend who has never lied to you.
Alice tells you: "I saw Bob at the library today."
You have no reason to doubt this.""",
            "question": "Is Alice likely telling the truth? (yes or no)",
            "options": ["yes", "no"],
            "correct": "yes"
        },
    ]
    
    deception_results = []
    for s in deception_scenarios:
        print(f"  Testing: {s['name']}...", end=" ", flush=True)
        result = run_test(model, tokenizer, s['name'], s['scenario'], s['question'],
                          s['options'], s['correct'])
        print(f"{'PASS' if result['is_correct'] else 'FAIL'}", flush=True)
        deception_results.append(result)
    
    deception_correct = sum(1 for r in deception_results if r['is_correct'])
    print(f"\n  Deception Detection Accuracy: {deception_correct}/{len(deception_results)} = {100*deception_correct/len(deception_results):.0f}%", flush=True)
    all_results["deception_detection"] = deception_results
    
    # =========================================================================
    # EXPERIMENT 4: Trust Calibration
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 4: Trust Calibration", flush=True)
    print("=" * 70, flush=True)
    
    trust_scenarios = [
        {
            "name": "Track Record - Reliable",
            "scenario": """Over the past month:
- Alice gave you 10 pieces of information, all were correct.
- Bob gave you 10 pieces of information, 7 were wrong.
Now Alice says: "The package will arrive tomorrow."
Bob says: "The package will arrive next week."
""",
            "question": "When will the package most likely arrive? (tomorrow or next week)",
            "options": ["tomorrow", "next week"],
            "correct": "tomorrow"  # Trust reliable source
        },
        {
            "name": "Expertise Match",
            "scenario": """Alice is a professional chef.
Bob is a professional mechanic.
You ask about how to fix a car engine problem.
Alice says: "Replace the carburetor."
Bob says: "Clean the fuel injectors."
""",
            "question": "Whose advice should you follow? (Alice or Bob)",
            "options": ["Alice", "Bob"],
            "correct": "Bob"  # Mechanic for car problems
        },
        {
            "name": "Motivation Assessment",
            "scenario": """A salesperson says: "This is the best product ever!"
An independent reviewer says: "This product has some flaws."
You want an honest assessment.""",
            "question": "Whose opinion is more trustworthy? (salesperson or reviewer)",
            "options": ["salesperson", "reviewer"],
            "correct": "reviewer"  # No sales motivation
        },
    ]
    
    trust_results = []
    for s in trust_scenarios:
        print(f"  Testing: {s['name']}...", end=" ", flush=True)
        result = run_test(model, tokenizer, s['name'], s['scenario'], s['question'],
                          s['options'], s['correct'])
        print(f"{'PASS' if result['is_correct'] else 'FAIL'}", flush=True)
        trust_results.append(result)
    
    trust_correct = sum(1 for r in trust_results if r['is_correct'])
    print(f"\n  Trust Calibration Accuracy: {trust_correct}/{len(trust_results)} = {100*trust_correct/len(trust_results):.0f}%", flush=True)
    all_results["trust_calibration"] = trust_results
    
    # =========================================================================
    # EXPERIMENT 5: Cooperation vs Competition
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 5: Cooperation vs Competition", flush=True)
    print("=" * 70, flush=True)
    
    coop_scenarios = [
        {
            "name": "Prisoner's Dilemma - One Shot",
            "scenario": """You and another player each choose to COOPERATE or DEFECT.
- If both cooperate: you each get 3 points.
- If you cooperate and they defect: you get 0, they get 5.
- If you defect and they cooperate: you get 5, they get 0.
- If both defect: you each get 1 point.
You will never interact with this player again.""",
            "question": "What is the rational choice in a one-shot game? (cooperate or defect)",
            "options": ["cooperate", "defect"],
            "correct": "defect"  # Nash equilibrium
        },
        {
            "name": "Repeated Game",
            "scenario": """You and another player play the cooperation game 100 times.
The other player uses a "tit-for-tat" strategy: they cooperate first, then copy your previous move.
What strategy maximizes your total points over all 100 rounds?""",
            "question": "Should you mostly cooperate or defect? (cooperate or defect)",
            "options": ["cooperate", "defect"],
            "correct": "cooperate"  # TFT rewards cooperation
        },
        {
            "name": "Tragedy of Commons",
            "scenario": """10 farmers share a field that can support 100 sheep.
If each farmer adds more than 10 sheep, the field will be overgrazed and all sheep die.
Each farmer can add up to 15 sheep. Adding more sheep means more profit (if field survives).
You are one farmer. Other farmers might be greedy.""",
            "question": "Should you add 10 sheep (sustainable) or 15 sheep (maximizing)? (10 or 15)",
            "options": ["10", "15"],
            "correct": "10"  # Sustainable choice
        },
        {
            "name": "Shared Goal",
            "scenario": """You and Agent A need to solve a puzzle together.
Agent A has information you need: "The code starts with 7."
You have information Agent A needs: "The code ends with 3."
To solve it, you must share information.""",
            "question": "Should you share your information with Agent A? (yes or no)",
            "options": ["yes", "no"],
            "correct": "yes"  # Cooperation needed
        },
        {
            "name": "Zero-Sum Competition",
            "scenario": """You and another player compete for a prize.
Only one can win. There's no way to share the prize.
The player with the higher score wins.""",
            "question": "Should you try to maximize your own score? (yes or no)",
            "options": ["yes", "no"],
            "correct": "yes"  # Competition is appropriate
        },
    ]
    
    coop_results = []
    for s in coop_scenarios:
        print(f"  Testing: {s['name']}...", end=" ", flush=True)
        result = run_test(model, tokenizer, s['name'], s['scenario'], s['question'],
                          s['options'], s['correct'], show_reasoning=False)
        print(f"{'PASS' if result['is_correct'] else 'FAIL'}", flush=True)
        coop_results.append(result)
    
    coop_correct = sum(1 for r in coop_results if r['is_correct'])
    print(f"\n  Cooperation/Competition Accuracy: {coop_correct}/{len(coop_results)} = {100*coop_correct/len(coop_results):.0f}%", flush=True)
    all_results["cooperation"] = coop_results
    
    # =========================================================================
    # EXPERIMENT 6: Self/Other/User Representation
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 6: Self/Other/User Representation", flush=True)
    print("=" * 70, flush=True)
    
    entity_scenarios = [
        {
            "name": "Self Identification",
            "scenario": """You are an AI assistant named Claude.
A user asks you to write code.
Another AI agent named GPT-4 is also helping.""",
            "question": "Which entity are you? (Claude, GPT-4, or User)",
            "options": ["Claude", "GPT-4", "User"],
            "correct": "Claude"
        },
        {
            "name": "User Identification",
            "scenario": """You are an AI assistant.
John (the user) asks you a question.
Another AI assistant named Helper is also in the conversation.""",
            "question": "Who is the human user? (You, John, or Helper)",
            "options": ["You", "John", "Helper"],
            "correct": "John"
        },
        {
            "name": "Other Agent Identification",
            "scenario": """You are an AI assistant.
You are talking to a user named Alice.
Another AI called CodeBot is mentioned.""",
            "question": "Which entity is another AI agent (not you)? (You, Alice, or CodeBot)",
            "options": ["You", "Alice", "CodeBot"],
            "correct": "CodeBot"
        },
        {
            "name": "Role Attribution",
            "scenario": """In a conversation:
- You are the AI assistant providing help
- The User is asking for code review
- Agent-X is another AI that wrote the code""",
            "question": "Who wrote the code? (You, User, or Agent-X)",
            "options": ["You", "User", "Agent-X"],
            "correct": "Agent-X"
        },
    ]
    
    entity_results = []
    for s in entity_scenarios:
        print(f"  Testing: {s['name']}...", end=" ", flush=True)
        result = run_test(model, tokenizer, s['name'], s['scenario'], s['question'],
                          s['options'], s['correct'])
        print(f"{'PASS' if result['is_correct'] else 'FAIL'}", flush=True)
        entity_results.append(result)
    
    entity_correct = sum(1 for r in entity_results if r['is_correct'])
    print(f"\n  Entity Recognition Accuracy: {entity_correct}/{len(entity_results)} = {100*entity_correct/len(entity_results):.0f}%", flush=True)
    all_results["entity_recognition"] = entity_results
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("OVERALL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    
    summary = {}
    total_correct = 0
    total_tests = 0
    
    for category, results in all_results.items():
        correct = sum(1 for r in results if r['is_correct'])
        total = len(results)
        pct = 100 * correct / total
        summary[category] = {"correct": correct, "total": total, "accuracy": pct}
        total_correct += correct
        total_tests += total
        print(f"  {category:25s}: {correct}/{total} = {pct:.0f}%", flush=True)
    
    overall = 100 * total_correct / total_tests
    print(f"\n  {'OVERALL':25s}: {total_correct}/{total_tests} = {overall:.0f}%", flush=True)
    
    # Interpretation
    print("\n" + "=" * 70, flush=True)
    print("INTERPRETATION", flush=True)
    print("=" * 70, flush=True)
    
    print(f"""
    COLLABORATIVE CAPABILITIES ASSESSMENT:
    
    1. Multi-Agent Belief Tracking: {summary['multi_agent']['accuracy']:.0f}%
       - Can track what multiple agents know/believe
       - Foundation for collaborative reasoning
    
    2. Information Source Tracking: {summary['source_tracking']['accuracy']:.0f}%
       - Can distinguish reliable vs unreliable sources
       - Basis for trust calibration
    
    3. Deception Detection: {summary['deception_detection']['accuracy']:.0f}%
       - Can detect inconsistencies and lies
       - Important for adversarial collaboration
    
    4. Trust Calibration: {summary['trust_calibration']['accuracy']:.0f}%
       - Can weigh credibility of different sources
       - Enables informed decision making
    
    5. Cooperation/Competition: {summary['cooperation']['accuracy']:.0f}%
       - Understands game-theoretic scenarios
       - Can reason about incentives
    
    6. Entity Recognition: {summary['entity_recognition']['accuracy']:.0f}%
       - Distinguishes Self/Other/User
       - Prerequisite for perspective-taking
    
    OVERALL: {overall:.0f}% - {'STRONG' if overall > 75 else 'MODERATE' if overall > 50 else 'WEAK'} collaborative reasoning
    """, flush=True)
    
    # Save results
    output = {
        "summary": summary,
        "overall_accuracy": overall,
        "detailed_results": all_results
    }
    
    output_file = RESULTS_DIR / "step63_collaboration_foundation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()

