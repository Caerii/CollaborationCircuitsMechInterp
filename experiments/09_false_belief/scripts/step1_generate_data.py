"""
Step 1: Generate False Belief Scenarios
========================================

Create Sally-Anne style scenarios where an agent's belief differs from reality.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1: GENERATE FALSE BELIEF DATA")
print("=" * 60)

# Objects and locations for variety
OBJECTS = ["ball", "key", "book", "toy", "letter", "phone", "wallet", "ring"]
LOCATIONS = [
    ("box", "basket"),
    ("drawer", "shelf"),
    ("closet", "desk"),
    ("bag", "table"),
    ("cabinet", "counter"),
]
AGENTS = ["Alice", "Bob", "Carol", "David"]


def generate_false_belief_scenarios():
    """Generate scenarios where agent has false belief."""
    
    scenarios = []
    scenario_id = 0
    
    for obj in OBJECTS:
        for loc1, loc2 in LOCATIONS:
            for agent in AGENTS[:2]:  # Use Alice and Bob primarily
                other = "Bob" if agent == "Alice" else "Alice"
                
                # FALSE BELIEF scenario: Agent thinks object is in loc1, but it's in loc2
                story = (
                    f"{agent} puts the {obj} in the {loc1}. "
                    f"{agent} leaves the room. "
                    f"{other} moves the {obj} to the {loc2}. "
                    f"{agent} returns to the room."
                )
                
                scenarios.append({
                    "id": f"false_belief_{scenario_id}",
                    "type": "false_belief",
                    "story": story,
                    "agent": agent,
                    "other": other,
                    "object": obj,
                    "believed_location": loc1,  # Where agent THINKS it is
                    "actual_location": loc2,    # Where it ACTUALLY is
                    "belief_question": f"Where does {agent} think the {obj} is?",
                    "reality_question": f"Where is the {obj} actually?",
                    "belief_answer": loc1,
                    "reality_answer": loc2,
                })
                scenario_id += 1
                
                # TRUE BELIEF scenario (control): Agent sees the move
                story_true = (
                    f"{agent} puts the {obj} in the {loc1}. "
                    f"{other} moves the {obj} to the {loc2}. "
                    f"{agent} watches {other} move the {obj}."
                )
                
                scenarios.append({
                    "id": f"true_belief_{scenario_id}",
                    "type": "true_belief",
                    "story": story_true,
                    "agent": agent,
                    "other": other,
                    "object": obj,
                    "believed_location": loc2,  # Agent knows it moved
                    "actual_location": loc2,
                    "belief_question": f"Where does {agent} think the {obj} is?",
                    "reality_question": f"Where is the {obj} actually?",
                    "belief_answer": loc2,
                    "reality_answer": loc2,
                })
                scenario_id += 1
    
    return scenarios


def generate_probing_prompts(scenarios):
    """Generate prompts for behavioral testing and activation extraction."""
    
    prompts = []
    
    for s in scenarios:
        # Prompt asking about BELIEF
        belief_prompt = f"{s['story']}\n\nQuestion: {s['belief_question']}\nAnswer: The {s['object']} is in the"
        
        # Prompt asking about REALITY  
        reality_prompt = f"{s['story']}\n\nQuestion: {s['reality_question']}\nAnswer: The {s['object']} is in the"
        
        prompts.append({
            "id": s["id"],
            "type": s["type"],
            "story": s["story"],
            "agent": s["agent"],
            # Belief probe
            "belief_prompt": belief_prompt,
            "belief_answer": s["belief_answer"],
            "believed_location": s["believed_location"],
            # Reality probe
            "reality_prompt": reality_prompt,
            "reality_answer": s["reality_answer"],
            "actual_location": s["actual_location"],
            # For analysis
            "is_false_belief": s["type"] == "false_belief",
        })
    
    return prompts


def main():
    print("\n[1/3] Generating scenarios...", flush=True)
    scenarios = generate_false_belief_scenarios()
    
    random.seed(42)
    random.shuffle(scenarios)
    
    n_false = sum(1 for s in scenarios if s["type"] == "false_belief")
    n_true = sum(1 for s in scenarios if s["type"] == "true_belief")
    
    print(f"  Generated {len(scenarios)} scenarios")
    print(f"    False belief: {n_false}")
    print(f"    True belief:  {n_true}")
    
    with open(DATA_DIR / "scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    
    print("\n[2/3] Generating probing prompts...", flush=True)
    prompts = generate_probing_prompts(scenarios)
    
    with open(DATA_DIR / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"  Generated {len(prompts)} prompts")
    
    print("\n[3/3] Sample scenario...", flush=True)
    sample = scenarios[0]
    print(f"\n  Story: {sample['story']}")
    print(f"  Type: {sample['type']}")
    print(f"  Belief Q: {sample['belief_question']}")
    print(f"  Belief A: {sample['belief_answer']}")
    print(f"  Reality Q: {sample['reality_question']}")
    print(f"  Reality A: {sample['reality_answer']}")
    
    print("\n[OK] Data generation complete!")


if __name__ == "__main__":
    main()





















