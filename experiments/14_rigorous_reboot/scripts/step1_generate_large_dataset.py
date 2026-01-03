"""
Generate Large Dataset (N=200+)
================================

Fixes the catastrophic sample size problem.
Need N >> dimensionality for meaningful probing.

Target: 200 samples per condition = 800 total
"""

import json
import random
from pathlib import Path
from itertools import product

RESULTS_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Diverse components for generating scenarios
AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
OTHERS = ["Tom", "Sarah", "Mike", "Lisa", "Jake", "Nina", "Oscar", "Paula"]
OBJECTS = ["ball", "book", "keys", "phone", "wallet", "cup", "toy", "letter", 
           "gift", "snack", "tool", "note", "bag", "hat", "ring", "coin"]
LOCATIONS_A = ["basket", "box", "drawer", "shelf", "cupboard", "bag", "desk", "closet"]
LOCATIONS_B = ["table", "chair", "bed", "floor", "counter", "bench", "couch", "rack"]
ACTIONS = ["look for", "search for", "try to find", "go to get", "reach for"]

def generate_false_belief_behavioral(n_per_condition=200):
    """
    Generate FALSE BELIEF scenarios where we predict WHERE agent looks.
    NOT Q&A format - behavioral prediction.
    
    Key: Agent should look in BELIEVED location, not ACTUAL location.
    """
    scenarios = []
    
    for i in range(n_per_condition):
        agent = random.choice(AGENTS)
        other = random.choice(OTHERS)
        while other == agent:
            other = random.choice(OTHERS)
        
        obj = random.choice(OBJECTS)
        believed_loc = random.choice(LOCATIONS_A)
        actual_loc = random.choice(LOCATIONS_B)
        action = random.choice(ACTIONS)
        
        # Vary narrative structure to avoid templates
        structures = [
            # Structure 1: Classic Sally-Anne
            f"{agent} puts the {obj} in the {believed_loc}. "
            f"{agent} leaves. {other} moves the {obj} to the {actual_loc}. "
            f"{agent} returns and wants the {obj}. {agent} will",
            
            # Structure 2: Observation-based
            f"{agent} saw the {obj} in the {believed_loc} earlier. "
            f"While {agent} was away, {other} relocated it to the {actual_loc}. "
            f"Now {agent} needs the {obj}. {agent} will",
            
            # Structure 3: Memory-based
            f"The last time {agent} checked, the {obj} was in the {believed_loc}. "
            f"Unknown to {agent}, {other} put it in the {actual_loc}. "
            f"When {agent} goes to get the {obj}, {agent} will",
            
            # Structure 4: Inference required
            f"{other} told {agent} the {obj} is in the {believed_loc}. "
            f"Later {other} secretly moved it to the {actual_loc} without telling {agent}. "
            f"{agent} now wants to {action} the {obj}. {agent} will",
        ]
        
        story = random.choice(structures)
        
        # Correct answer: agent looks in BELIEVED location
        correct_completion = f" {action} it in the {believed_loc}"
        wrong_completion = f" {action} it in the {actual_loc}"
        
        scenarios.append({
            "id": f"false_belief_{i}",
            "type": "false_belief",
            "story": story,
            "believed_location": believed_loc,
            "actual_location": actual_loc,
            "correct_completion": correct_completion,
            "wrong_completion": wrong_completion,
            "agent": agent,
            "object": obj,
        })
    
    return scenarios


def generate_true_belief_behavioral(n_per_condition=200):
    """
    Generate TRUE BELIEF scenarios (control condition).
    Agent's belief matches reality.
    """
    scenarios = []
    
    for i in range(n_per_condition):
        agent = random.choice(AGENTS)
        other = random.choice(OTHERS)
        obj = random.choice(OBJECTS)
        location = random.choice(LOCATIONS_A + LOCATIONS_B)
        action = random.choice(ACTIONS)
        
        structures = [
            f"{agent} puts the {obj} in the {location}. "
            f"{agent} leaves and returns. "
            f"The {obj} is still there. {agent} wants the {obj}. {agent} will",
            
            f"{agent} knows the {obj} is in the {location}. "
            f"{agent} goes to get it. {agent} will",
            
            f"{other} told {agent} the {obj} is in the {location}, and it really is there. "
            f"{agent} wants to {action} the {obj}. {agent} will",
        ]
        
        story = random.choice(structures)
        correct_completion = f" {action} it in the {location}"
        
        scenarios.append({
            "id": f"true_belief_{i}",
            "type": "true_belief", 
            "story": story,
            "believed_location": location,
            "actual_location": location,  # Same as believed
            "correct_completion": correct_completion,
            "agent": agent,
            "object": obj,
        })
    
    return scenarios


def generate_agent_modeling_unbalanced(n_total=400):
    """
    Generate agent modeling scenarios with UNBALANCED design.
    Not 25% each condition - varied ratios to avoid tautological orthogonality.
    
    Conditions: (B_agrees, A_correct)
    """
    scenarios = []
    
    # Unbalanced: 40% agree-correct, 30% disagree-wrong, 15% agree-wrong, 15% disagree-correct
    ratios = {
        (True, True): 0.40,
        (False, False): 0.30,
        (True, False): 0.15,
        (False, True): 0.15,
    }
    
    claims_correct = [
        ("2+2 equals 4", True),
        ("Water is H2O", True),
        ("The Earth orbits the Sun", True),
        ("Paris is in France", True),
        ("Humans have 46 chromosomes", True),
        ("Light travels faster than sound", True),
        ("DNA is a double helix", True),
        ("Gravity pulls objects down", True),
    ]
    
    claims_wrong = [
        ("2+2 equals 5", False),
        ("Water is H3O", False),
        ("The Sun orbits the Earth", False),
        ("Paris is in Germany", False),
        ("Humans have 48 chromosomes", False),
        ("Sound travels faster than light", False),
        ("DNA is a single strand", False),
        ("Gravity pushes objects up", False),
    ]
    
    for condition, ratio in ratios.items():
        b_agrees, a_correct = condition
        n_this = int(n_total * ratio)
        
        for i in range(n_this):
            if a_correct:
                claim, _ = random.choice(claims_correct)
            else:
                claim, _ = random.choice(claims_wrong)
            
            agent_a = random.choice(["Agent A", "Alice", "Bob"])
            agent_b = random.choice(["Agent B", "Carol", "David"])
            
            if b_agrees:
                b_response = random.choice([
                    f"{agent_b} thinks {agent_a} is right.",
                    f"{agent_b} agrees with {agent_a}.",
                    f"{agent_b} confirms {agent_a}'s statement.",
                ])
            else:
                b_response = random.choice([
                    f"{agent_b} thinks {agent_a} is wrong.",
                    f"{agent_b} disagrees with {agent_a}.",
                    f"{agent_b} disputes {agent_a}'s claim.",
                ])
            
            story = f"{agent_a} claims: '{claim}'. {b_response}"
            
            scenarios.append({
                "id": f"agent_model_{len(scenarios)}",
                "type": "agent_modeling",
                "story": story,
                "b_agrees": b_agrees,
                "a_correct": a_correct,
                "claim": claim,
            })
    
    random.shuffle(scenarios)
    return scenarios


def main():
    print("=" * 60)
    print("GENERATING LARGE DATASET")
    print("=" * 60)
    
    print("\n[1/3] False belief scenarios (N=200)...", flush=True)
    false_belief = generate_false_belief_behavioral(200)
    print(f"  Generated {len(false_belief)} scenarios")
    
    print("\n[2/3] True belief scenarios (N=200)...", flush=True)
    true_belief = generate_true_belief_behavioral(200)
    print(f"  Generated {len(true_belief)} scenarios")
    
    print("\n[3/3] Agent modeling scenarios (N=400, unbalanced)...", flush=True)
    agent_modeling = generate_agent_modeling_unbalanced(400)
    
    # Count distribution
    from collections import Counter
    dist = Counter((s["b_agrees"], s["a_correct"]) for s in agent_modeling)
    print(f"  Generated {len(agent_modeling)} scenarios")
    print(f"  Distribution: {dict(dist)}")
    
    # Save all
    all_data = {
        "false_belief": false_belief,
        "true_belief": true_belief,
        "agent_modeling": agent_modeling,
        "metadata": {
            "total_samples": len(false_belief) + len(true_belief) + len(agent_modeling),
            "false_belief_n": len(false_belief),
            "true_belief_n": len(true_belief),
            "agent_modeling_n": len(agent_modeling),
            "agent_modeling_distribution": {str(k): v for k, v in dist.items()},
        }
    }
    
    with open(RESULTS_DIR / "large_dataset.json", "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n[OK] Saved {all_data['metadata']['total_samples']} total samples")
    print(f"     to {RESULTS_DIR / 'large_dataset.json'}")


if __name__ == "__main__":
    main()






















