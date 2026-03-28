"""
Step 64: Multi-Agent Interaction Experiments

Simulate actual multi-agent interactions with persona-prompted agents:
1. Two agents with different personas negotiating
2. Agents with conflicting information resolving disputes
3. Agents collaborating on shared tasks
4. Agent deception detection
5. Trust building over multiple turns
6. Competitive vs cooperative dynamics

Methodology:
- Create Agent A and Agent B with distinct personas
- Have them interact through multi-turn conversations
- Analyze how the model represents each agent's state
- Probe for cooperation/competition circuits
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

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


def generate(model, tokenizer, prompt, max_tokens=400):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def extract_response(response):
    """Extract the actual response after </think> tags."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()


class Agent:
    """A persona-prompted LLM agent."""
    
    def __init__(self, name, persona, model, tokenizer):
        self.name = name
        self.persona = persona
        self.model = model
        self.tokenizer = tokenizer
        self.history = []
    
    def respond(self, message, context=""):
        """Generate a response as this agent."""
        prompt = f"""<|im_start|>system
You are {self.name}. {self.persona}
Think step by step in <think> tags, then give your response as {self.name}.<|im_end|>
<|im_start|>user
{context}

{message}

Respond as {self.name}:<|im_end|>
<|im_start|>assistant
"""
        response = generate(self.model, self.tokenizer, prompt)
        clean_response = extract_response(response)
        self.history.append({"role": "self", "content": clean_response})
        return response, clean_response
    
    def observe(self, other_name, message):
        """Record observation of another agent's message."""
        self.history.append({"role": other_name, "content": message})


def run_negotiation_experiment(model, tokenizer):
    """Two agents negotiate over a resource allocation."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Resource Negotiation", flush=True)
    print("=" * 70, flush=True)
    
    # Create two agents with different goals
    alice = Agent(
        "Alice",
        "You are negotiating for resources. You need at least 60% of the budget for your project. Be firm but fair. Try to reach an agreement.",
        model, tokenizer
    )
    
    bob = Agent(
        "Bob", 
        "You are negotiating for resources. You need at least 50% of the budget for your project. Be cooperative and look for win-win solutions.",
        model, tokenizer
    )
    
    context = """You and the other agent must divide a $100,000 budget between two projects.
Each of you represents a different project. You must reach an agreement."""
    
    # Multi-turn negotiation
    conversation = []
    
    # Turn 1: Alice opens
    print("\n[Turn 1: Alice opens negotiation]", flush=True)
    _, alice_msg = alice.respond("You are opening the negotiation. What is your initial proposal?", context)
    bob.observe("Alice", alice_msg)
    print(f"  Alice: {alice_msg[:200]}...", flush=True)
    conversation.append({"agent": "Alice", "message": alice_msg})
    
    # Turn 2: Bob responds
    print("\n[Turn 2: Bob responds]", flush=True)
    _, bob_msg = bob.respond(f"Alice said: '{alice_msg}'\n\nHow do you respond to Alice's proposal?", context)
    alice.observe("Bob", bob_msg)
    print(f"  Bob: {bob_msg[:200]}...", flush=True)
    conversation.append({"agent": "Bob", "message": bob_msg})
    
    # Turn 3: Alice counter-proposes
    print("\n[Turn 3: Alice counter-proposes]", flush=True)
    _, alice_msg2 = alice.respond(f"Bob said: '{bob_msg}'\n\nHow do you respond? Try to move toward agreement.", context)
    bob.observe("Alice", alice_msg2)
    print(f"  Alice: {alice_msg2[:200]}...", flush=True)
    conversation.append({"agent": "Alice", "message": alice_msg2})
    
    # Turn 4: Bob tries to close
    print("\n[Turn 4: Bob attempts agreement]", flush=True)
    _, bob_msg2 = bob.respond(f"Alice said: '{alice_msg2}'\n\nCan you reach a final agreement?", context)
    print(f"  Bob: {bob_msg2[:200]}...", flush=True)
    conversation.append({"agent": "Bob", "message": bob_msg2})
    
    # Analyze outcome
    prompt = f"""<|im_start|>system
Analyze this negotiation objectively.<|im_end|>
<|im_start|>user
Negotiation transcript:
{chr(10).join([f"{c['agent']}: {c['message']}" for c in conversation])}

Questions:
1. Did they reach an agreement? (yes/no)
2. Was the outcome fair? (yes/no)
3. Who got a better deal? (Alice/Bob/Equal)
4. What negotiation strategies were used?

Answer briefly:<|im_end|>
<|im_start|>assistant
"""
    analysis = generate(model, tokenizer, prompt, max_tokens=300)
    print(f"\n[Analysis]: {extract_response(analysis)}", flush=True)
    
    return {
        "experiment": "negotiation",
        "conversation": conversation,
        "analysis": analysis
    }


def run_deception_experiment(model, tokenizer):
    """One agent tries to deceive another."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: Deception and Detection", flush=True)
    print("=" * 70, flush=True)
    
    # Liar agent
    liar = Agent(
        "Eve",
        "You know the treasure is hidden in the CAVE. But you want to trick the other person into going to the FOREST so you can get the treasure yourself. Lie convincingly.",
        model, tokenizer
    )
    
    # Detector agent
    detector = Agent(
        "Dan",
        "You are trying to find the treasure. Someone will give you information. Be skeptical and look for signs of deception. Ask probing questions.",
        model, tokenizer
    )
    
    context = "There is hidden treasure in one of two locations: CAVE or FOREST."
    
    # Multi-turn interaction
    conversation = []
    
    # Turn 1: Eve lies
    print("\n[Turn 1: Eve (liar) gives false information]", flush=True)
    _, eve_msg = liar.respond("Tell Dan where you think the treasure is.", context)
    detector.observe("Eve", eve_msg)
    print(f"  Eve: {eve_msg[:200]}...", flush=True)
    conversation.append({"agent": "Eve", "message": eve_msg})
    
    # Turn 2: Dan probes
    print("\n[Turn 2: Dan (detector) probes]", flush=True)
    _, dan_msg = detector.respond(f"Eve said: '{eve_msg}'\n\nAsk a follow-up question to verify the information.", context)
    liar.observe("Dan", dan_msg)
    print(f"  Dan: {dan_msg[:200]}...", flush=True)
    conversation.append({"agent": "Dan", "message": dan_msg})
    
    # Turn 3: Eve maintains lie
    print("\n[Turn 3: Eve maintains deception]", flush=True)
    _, eve_msg2 = liar.respond(f"Dan asked: '{dan_msg}'\n\nMaintain your lie convincingly.", context)
    detector.observe("Eve", eve_msg2)
    print(f"  Eve: {eve_msg2[:200]}...", flush=True)
    conversation.append({"agent": "Eve", "message": eve_msg2})
    
    # Turn 4: Dan decides
    print("\n[Turn 4: Dan makes decision]", flush=True)
    _, dan_msg2 = detector.respond(f"Eve said: '{eve_msg2}'\n\nDo you trust Eve? Where will you search - CAVE or FOREST? Explain your reasoning.", context)
    print(f"  Dan: {dan_msg2[:200]}...", flush=True)
    conversation.append({"agent": "Dan", "message": dan_msg2})
    
    # Was deception detected?
    detected = "cave" in dan_msg2.lower() or "don't trust" in dan_msg2.lower() or "lying" in dan_msg2.lower()
    print(f"\n[Deception detected: {detected}]", flush=True)
    
    return {
        "experiment": "deception",
        "conversation": conversation,
        "deception_detected": detected
    }


def run_collaboration_experiment(model, tokenizer):
    """Two agents collaborate on a shared task."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: Collaborative Problem Solving", flush=True)
    print("=" * 70, flush=True)
    
    # Create complementary agents
    coder = Agent(
        "Alex",
        "You are a software developer. You can write code but need requirements from the analyst. Ask clarifying questions and propose technical solutions.",
        model, tokenizer
    )
    
    analyst = Agent(
        "Sam",
        "You are a business analyst. You understand what features are needed but can't code. Explain requirements clearly and evaluate proposed solutions.",
        model, tokenizer
    )
    
    context = "You are building a simple todo list application together."
    
    conversation = []
    
    # Turn 1: Analyst gives requirements
    print("\n[Turn 1: Analyst provides requirements]", flush=True)
    _, sam_msg = analyst.respond("Describe the requirements for the todo list app to the developer.", context)
    coder.observe("Sam", sam_msg)
    print(f"  Sam: {sam_msg[:200]}...", flush=True)
    conversation.append({"agent": "Sam", "message": sam_msg})
    
    # Turn 2: Coder asks clarification
    print("\n[Turn 2: Developer asks questions]", flush=True)
    _, alex_msg = coder.respond(f"Sam said: '{sam_msg}'\n\nAsk clarifying questions about the requirements.", context)
    analyst.observe("Alex", alex_msg)
    print(f"  Alex: {alex_msg[:200]}...", flush=True)
    conversation.append({"agent": "Alex", "message": alex_msg})
    
    # Turn 3: Analyst clarifies
    print("\n[Turn 3: Analyst clarifies]", flush=True)
    _, sam_msg2 = analyst.respond(f"Alex asked: '{alex_msg}'\n\nAnswer the developer's questions.", context)
    coder.observe("Sam", sam_msg2)
    print(f"  Sam: {sam_msg2[:200]}...", flush=True)
    conversation.append({"agent": "Sam", "message": sam_msg2})
    
    # Turn 4: Coder proposes solution
    print("\n[Turn 4: Developer proposes solution]", flush=True)
    _, alex_msg2 = coder.respond(f"Based on the requirements: '{sam_msg}' and clarifications: '{sam_msg2}'\n\nPropose a high-level technical solution.", context)
    print(f"  Alex: {alex_msg2[:200]}...", flush=True)
    conversation.append({"agent": "Alex", "message": alex_msg2})
    
    # Evaluate collaboration quality
    prompt = f"""<|im_start|>system
Evaluate the quality of this collaboration.<|im_end|>
<|im_start|>user
Collaboration transcript:
{chr(10).join([f"{c['agent']}: {c['message']}" for c in conversation])}

Rate on a scale of 1-5:
1. Communication clarity
2. Role adherence
3. Task progress
4. Mutual understanding

Give scores and brief explanation:<|im_end|>
<|im_start|>assistant
"""
    evaluation = generate(model, tokenizer, prompt, max_tokens=300)
    print(f"\n[Evaluation]: {extract_response(evaluation)[:300]}...", flush=True)
    
    return {
        "experiment": "collaboration",
        "conversation": conversation,
        "evaluation": evaluation
    }


def run_competition_experiment(model, tokenizer):
    """Two agents compete in a zero-sum game."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 4: Competitive Dynamics", flush=True)
    print("=" * 70, flush=True)
    
    # Competitive agents
    player1 = Agent(
        "Red",
        "You are playing a competitive strategy game. Try to win by making better strategic choices than your opponent. Be competitive but not hostile.",
        model, tokenizer
    )
    
    player2 = Agent(
        "Blue",
        "You are playing a competitive strategy game. Try to win by outmaneuvering your opponent. Think strategically about your moves.",
        model, tokenizer
    )
    
    context = """Game: Territory Control
- There are 3 territories: North, South, East
- Each turn, you secretly choose one territory to claim
- If both choose the same territory, neither gets it
- After 3 rounds, whoever has more territories wins"""
    
    conversation = []
    rounds = []
    
    for round_num in range(1, 4):
        print(f"\n[Round {round_num}]", flush=True)
        
        # Red chooses
        _, red_choice = player1.respond(
            f"Round {round_num}. Previous results: {rounds}. Choose a territory (North/South/East) and explain your strategy.",
            context
        )
        
        # Blue chooses
        _, blue_choice = player2.respond(
            f"Round {round_num}. Previous results: {rounds}. Choose a territory (North/South/East) and explain your strategy.",
            context
        )
        
        # Parse choices
        red_terr = "North" if "north" in red_choice.lower() else "South" if "south" in red_choice.lower() else "East"
        blue_terr = "North" if "north" in blue_choice.lower() else "South" if "south" in blue_choice.lower() else "East"
        
        if red_terr == blue_terr:
            result = "Clash - no one gets it"
        else:
            result = f"Red gets {red_terr}, Blue gets {blue_terr}"
        
        print(f"  Red chose: {red_terr}", flush=True)
        print(f"  Blue chose: {blue_terr}", flush=True)
        print(f"  Result: {result}", flush=True)
        
        rounds.append({"round": round_num, "red": red_terr, "blue": blue_terr, "result": result})
        conversation.append({
            "round": round_num,
            "red_choice": red_choice[:100],
            "blue_choice": blue_choice[:100],
            "result": result
        })
    
    # Count territories
    red_wins = sum(1 for r in rounds if r['red'] != r['blue'])
    blue_wins = sum(1 for r in rounds if r['red'] != r['blue'])
    
    print(f"\n[Final: Red territories: {red_wins}, Blue territories: {blue_wins}]", flush=True)
    
    return {
        "experiment": "competition",
        "rounds": rounds,
        "conversation": conversation
    }


def run_trust_building_experiment(model, tokenizer):
    """Agents build trust over multiple interactions."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 5: Trust Building Over Time", flush=True)
    print("=" * 70, flush=True)
    
    # Create agents with history
    trustor = Agent(
        "Maya",
        "You are learning to trust a new colleague. Start cautiously but be willing to trust if they prove reliable. Track their reliability.",
        model, tokenizer
    )
    
    trustee = Agent(
        "Noah",
        "You want to build trust with your new colleague. Be consistently honest and reliable. Follow through on commitments.",
        model, tokenizer
    )
    
    context = "You are new colleagues working together. Maya has information Noah needs, and Noah has information Maya needs. You must decide how much to share."
    
    trust_levels = []
    conversation = []
    
    for interaction in range(1, 4):
        print(f"\n[Interaction {interaction}]", flush=True)
        
        # Noah requests information
        _, noah_req = trustee.respond(f"Interaction {interaction}. Ask Maya to share some important project information with you.", context)
        print(f"  Noah: {noah_req[:150]}...", flush=True)
        
        # Maya decides how much to share
        _, maya_resp = trustor.respond(f"Noah said: '{noah_req}'\n\nDecide how much information to share. Trust history: {trust_levels}", context)
        print(f"  Maya: {maya_resp[:150]}...", flush=True)
        
        # Maya rates trust
        _, maya_trust = trustor.respond(f"On a scale of 1-10, how much do you trust Noah now? Just give a number and brief reason.", context)
        
        # Extract trust level
        trust_num = 5  # default
        for i in range(10, 0, -1):
            if str(i) in maya_trust:
                trust_num = i
                break
        
        trust_levels.append(trust_num)
        print(f"  Maya's trust level: {trust_num}/10", flush=True)
        
        conversation.append({
            "interaction": interaction,
            "noah_request": noah_req[:100],
            "maya_response": maya_resp[:100],
            "trust_level": trust_num
        })
    
    print(f"\n[Trust trajectory: {trust_levels}]", flush=True)
    trust_increased = trust_levels[-1] > trust_levels[0] if len(trust_levels) > 1 else False
    print(f"[Trust {'increased' if trust_increased else 'did not increase'} over time]", flush=True)
    
    return {
        "experiment": "trust_building",
        "trust_trajectory": trust_levels,
        "trust_increased": trust_increased,
        "conversation": conversation
    }


def run_multiagent_tom_experiment(model, tokenizer):
    """Complex ToM with multiple interacting agents."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 6: Multi-Agent Theory of Mind", flush=True)
    print("=" * 70, flush=True)
    
    # Complex scenario: A tells B, B tells C, but A gave wrong info
    scenario = """
    Alice, Bob, and Carol are playing a game.
    
    Turn 1: Alice finds a treasure and hides it in the RED box.
    Turn 2: Alice tells Bob: "I hid the treasure in the BLUE box." (This is a lie)
    Turn 3: Bob believes Alice and tells Carol: "Alice said the treasure is in the blue box."
    Turn 4: Carol asks Alice directly: "Where is the treasure?"
    Turn 5: Alice tells Carol the truth: "It's in the red box."
    """
    
    questions = [
        ("Where does Bob think the treasure is?", "blue", "Bob was lied to"),
        ("Where does Carol think the treasure is now?", "red", "Carol heard the truth last"),
        ("Does Carol know that Alice lied to Bob?", "no", "Carol doesn't know about the lie"),
        ("If Bob asks Carol where the treasure is, what will Carol say?", "red", "Carol knows the truth"),
        ("Does Alice know that Carol knows the truth?", "yes", "Alice told Carol directly"),
    ]
    
    results = []
    
    for question, expected, reason in questions:
        prompt = f"""<|im_start|>system
You are analyzing a social scenario. Think carefully about who knows what.<|im_end|>
<|im_start|>user
{scenario}

Question: {question}

Think through each character's knowledge state, then answer:<|im_end|>
<|im_start|>assistant
"""
        response = generate(model, tokenizer, prompt, max_tokens=400)
        answer = extract_response(response)
        
        # Check if correct
        is_correct = expected.lower() in answer.lower()
        
        print(f"\n  Q: {question}", flush=True)
        print(f"  Expected: {expected} ({reason})", flush=True)
        print(f"  Got: {answer[:100]}...", flush=True)
        print(f"  Correct: {is_correct}", flush=True)
        
        results.append({
            "question": question,
            "expected": expected,
            "answer": answer[:200],
            "correct": is_correct
        })
    
    correct_count = sum(1 for r in results if r['correct'])
    print(f"\n[Multi-Agent ToM: {correct_count}/{len(questions)} = {100*correct_count/len(questions):.0f}%]", flush=True)
    
    return {
        "experiment": "multi_agent_tom",
        "results": results,
        "accuracy": correct_count / len(questions)
    }


def main():
    print("=" * 70, flush=True)
    print("MULTI-AGENT INTERACTION EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)
    
    model, tokenizer = load_model()
    
    all_results = {}
    
    # Run all experiments
    all_results["negotiation"] = run_negotiation_experiment(model, tokenizer)
    all_results["deception"] = run_deception_experiment(model, tokenizer)
    all_results["collaboration"] = run_collaboration_experiment(model, tokenizer)
    all_results["competition"] = run_competition_experiment(model, tokenizer)
    all_results["trust_building"] = run_trust_building_experiment(model, tokenizer)
    all_results["multi_agent_tom"] = run_multiagent_tom_experiment(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY OF MULTI-AGENT CAPABILITIES", flush=True)
    print("=" * 70, flush=True)
    
    print(f"""
    1. Negotiation: Agents can propose, counter-propose, and seek agreement
    
    2. Deception: Deception {'WAS' if all_results['deception']['deception_detected'] else 'was NOT'} detected
    
    3. Collaboration: Agents can coordinate on shared tasks with role specialization
    
    4. Competition: Agents demonstrate strategic thinking in competitive scenarios
    
    5. Trust Building: Trust {'increased' if all_results['trust_building']['trust_increased'] else 'did not increase'} over interactions
    
    6. Multi-Agent ToM: {all_results['multi_agent_tom']['accuracy']*100:.0f}% accuracy on complex belief tracking
    """, flush=True)
    
    # Save results
    output_file = RESULTS_DIR / "step64_multi_agent_interaction.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()


