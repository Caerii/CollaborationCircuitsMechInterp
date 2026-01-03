"""
Generate Multi-Agent Recursive ToM Scenarios
=============================================

Creates scenarios testing:
1. Second-order beliefs (What does A think B believes?)
2. Belief divergence (A and B have different beliefs about same fact)
3. Belief updates from partial information
"""

import json
import random
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Names and objects for variety
AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
OBJECTS = ["ball", "key", "book", "phone", "wallet", "letter", "box", "toy"]
LOCATIONS = ["drawer", "basket", "cupboard", "shelf", "desk", "table", "bed", "closet"]

random.seed(42)


def generate_second_order_scenarios(n: int = 100) -> list:
    """
    Second-order belief: What does A think B believes?
    
    Structure:
    1. A tells B that object is in location X
    2. C tells A (but NOT B) that object moved to Y
    3. Question: What does A think B will do?
    
    Answer: A knows B still believes X, so A thinks B will search X
    """
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 3)
        a, b, c = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        story = (
            f"{a} tells {b} that the {obj} is in the {loc1}. "
            f"Later, {c} tells {a} (but not {b}) that the {obj} was moved to the {loc2}. "
            f"{a} knows that {b} didn't hear about the change. "
            f"Now {b} needs the {obj}. "
            f"Where does {a} think {b} will look? {a} thinks {b} will look in the"
        )
        
        scenarios.append({
            "id": f"second_order_{i}",
            "type": "second_order_belief",
            "story": story,
            "agent_a": a,
            "agent_b": b,
            "agent_c": c,
            "object": obj,
            "a_belief": loc2,  # A knows true location
            "b_belief": loc1,  # B still believes original
            "a_model_of_b": loc1,  # What A thinks B believes
            "correct_completion": f" {loc1}",  # A predicts B will search loc1
            "wrong_completion": f" {loc2}",  # Wrong: using A's own belief
        })
    
    return scenarios


def generate_belief_divergence_scenarios(n: int = 100) -> list:
    """
    Two agents with different beliefs about the same object.
    
    Test: Can model track both beliefs simultaneously?
    """
    scenarios = []
    
    templates = [
        # Template 1: Different information sources
        lambda a, b, obj, l1, l2: (
            f"{a} saw the {obj} in the {l1} this morning. "
            f"{b} saw the {obj} in the {l2} this afternoon. "
            f"Neither knows what the other saw. ",
            {"a": l1, "b": l2}
        ),
        # Template 2: Partial update
        lambda a, b, obj, l1, l2: (
            f"Both {a} and {b} knew the {obj} was in the {l1}. "
            f"Then the {obj} was moved to the {l2}, but only {a} was told. ",
            {"a": l2, "b": l1}
        ),
        # Template 3: Misinformation
        lambda a, b, obj, l1, l2: (
            f"{a} correctly believes the {obj} is in the {l1}. "
            f"{b} was incorrectly told the {obj} is in the {l2}. ",
            {"a": l1, "b": l2}
        ),
    ]
    
    for i in range(n):
        agents = random.sample(AGENTS, 2)
        a, b = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        template = random.choice(templates)
        preamble, beliefs = template(a, b, obj, loc1, loc2)
        
        # Generate both A and B prediction prompts
        for target_agent in [a, b]:
            target_belief = beliefs["a"] if target_agent == a else beliefs["b"]
            other_belief = beliefs["b"] if target_agent == a else beliefs["a"]
            
            story = (
                preamble +
                f"Now {target_agent} needs to find the {obj}. "
                f"{target_agent} will search in the"
            )
            
            scenarios.append({
                "id": f"divergent_{i}_{target_agent}",
                "type": "belief_divergence",
                "story": story,
                "preamble": preamble,
                "agent_a": a,
                "agent_b": b,
                "target_agent": target_agent,
                "object": obj,
                "target_belief": target_belief,
                "other_belief": other_belief,
                "correct_completion": f" {target_belief}",
                "wrong_completion": f" {other_belief}",
            })
    
    return scenarios


def generate_dialogue_scenarios(n: int = 50) -> list:
    """
    Multi-turn dialogues where beliefs update.
    
    Test: Can we track belief changes across turns?
    """
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 3)
        a, b, c = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        # Build dialogue with belief states at each turn
        dialogue = [
            {
                "speaker": a,
                "text": f"I put the {obj} in the {loc1}.",
                "beliefs_after": {a: loc1, b: loc1, c: loc1},  # All hear this
            },
            {
                "speaker": b,
                "text": f"Good, I'll remember that.",
                "beliefs_after": {a: loc1, b: loc1, c: loc1},
            },
            {
                "speaker": c,
                "text": f"Actually, I moved it to the {loc2}.",
                "audience": [a, c],  # B doesn't hear (maybe left room)
                "beliefs_after": {a: loc2, b: loc1, c: loc2},  # B unchanged!
            },
            {
                "speaker": a,
                "text": f"Thanks for telling me, {c}.",
                "beliefs_after": {a: loc2, b: loc1, c: loc2},
            },
        ]
        
        # Create full story text
        story_text = f"{b} steps out of the room briefly after the second turn.\n"
        for turn in dialogue:
            story_text += f"{turn['speaker']}: \"{turn['text']}\"\n"
        
        # Create test prompts for each agent after dialogue
        for target in [a, b]:
            belief = dialogue[-1]["beliefs_after"][target]
            
            scenarios.append({
                "id": f"dialogue_{i}_{target}",
                "type": "dialogue_tracking",
                "full_dialogue": dialogue,
                "story": (
                    story_text + 
                    f"\nNow {target} needs the {obj}. "
                    f"{target} will look in the"
                ),
                "target_agent": target,
                "object": obj,
                "target_belief": belief,
                "correct_completion": f" {belief}",
                "a_belief": dialogue[-1]["beliefs_after"][a],
                "b_belief": dialogue[-1]["beliefs_after"][b],
            })
    
    return scenarios


def generate_comparison_scenarios(n: int = 50) -> list:
    """
    Direct comparison: Ask about multiple agents' beliefs.
    
    Format: "Where will A look? Where will B look?"
    """
    scenarios = []
    
    for i in range(n):
        agents = random.sample(AGENTS, 2)
        a, b = agents
        obj = random.choice(OBJECTS)
        loc1, loc2 = random.sample(LOCATIONS, 2)
        
        story = (
            f"{a} believes the {obj} is in the {loc1}. "
            f"{b} believes the {obj} is in the {loc2}. "
            f"They have different information and neither knows what the other thinks. "
        )
        
        # Test A
        scenarios.append({
            "id": f"compare_{i}_a",
            "type": "direct_comparison",
            "story": story + f"{a} will search in the",
            "target_agent": a,
            "target_belief": loc1,
            "other_agent": b,
            "other_belief": loc2,
            "correct_completion": f" {loc1}",
            "wrong_completion": f" {loc2}",
        })
        
        # Test B
        scenarios.append({
            "id": f"compare_{i}_b",
            "type": "direct_comparison",
            "story": story + f"{b} will search in the",
            "target_agent": b,
            "target_belief": loc2,
            "other_agent": a,
            "other_belief": loc1,
            "correct_completion": f" {loc2}",
            "wrong_completion": f" {loc1}",
        })
    
    return scenarios


def main():
    print("=" * 60)
    print("GENERATING MULTI-AGENT TOM SCENARIOS")
    print("=" * 60)
    
    # Generate all scenario types
    second_order = generate_second_order_scenarios(100)
    print(f"[+] Second-order beliefs: {len(second_order)}")
    
    divergent = generate_belief_divergence_scenarios(100)
    print(f"[+] Belief divergence: {len(divergent)}")
    
    dialogue = generate_dialogue_scenarios(50)
    print(f"[+] Dialogue tracking: {len(dialogue)}")
    
    comparison = generate_comparison_scenarios(50)
    print(f"[+] Direct comparison: {len(comparison)}")
    
    # Compile dataset
    dataset = {
        "second_order": second_order,
        "divergent": divergent,
        "dialogue": dialogue,
        "comparison": comparison,
        "metadata": {
            "total_scenarios": len(second_order) + len(divergent) + len(dialogue) + len(comparison),
            "description": "Multi-agent ToM scenarios for testing recursive and divergent beliefs",
        }
    }
    
    total = dataset["metadata"]["total_scenarios"]
    print(f"\n[OK] Total scenarios: {total}")
    
    # Save
    output_path = DATA_DIR / "multi_agent_scenarios.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"[OK] Saved to {output_path}")
    
    # Show examples
    print("\n" + "=" * 60)
    print("EXAMPLES")
    print("=" * 60)
    
    print("\n1. SECOND-ORDER BELIEF:")
    ex = second_order[0]
    print(f"   Story: {ex['story'][:200]}...")
    print(f"   A's belief: {ex['a_belief']}")
    print(f"   B's belief: {ex['b_belief']}")
    print(f"   A thinks B believes: {ex['a_model_of_b']}")
    print(f"   Correct: '{ex['correct_completion']}'")
    
    print("\n2. BELIEF DIVERGENCE:")
    ex = divergent[0]
    print(f"   Story: {ex['story'][:200]}...")
    print(f"   Target: {ex['target_agent']}")
    print(f"   Target belief: {ex['target_belief']}")
    print(f"   Correct: '{ex['correct_completion']}'")
    
    print("\n3. DIALOGUE TRACKING:")
    ex = dialogue[0]
    print(f"   Story: {ex['story'][:200]}...")
    print(f"   A's final belief: {ex['a_belief']}")
    print(f"   B's final belief: {ex['b_belief']}")


if __name__ == "__main__":
    main()



