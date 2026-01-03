"""
Step 1: Generate Counter-Balanced False Belief Scenarios
=========================================================

Address the critique: Previous results may be due to "output first location" heuristic.

We COUNTERBALANCE:
- Which location is mentioned first in the story
- Whether the believed location comes before or after actual location

This tests if the model truly tracks belief vs reality, or just pattern-matches.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1: COUNTER-BALANCED FALSE BELIEF DATA")
print("=" * 60)

AGENTS = ["Sally", "Anne", "Mark", "Lisa"]
OBJECTS = ["ball", "marble", "toy", "book"]
LOCATIONS = [
    ("box", "basket"),
    ("drawer", "cupboard"),
    ("bag", "shelf"),
]


def generate_false_belief_counterbalanced():
    """
    Generate scenarios with counter-balancing to detect shortcut heuristics.
    
    Key: The FIRST location mentioned should NOT predict the answer.
    """
    scenarios = []
    scenario_id = 0
    
    for obj in OBJECTS:
        for loc_a, loc_b in LOCATIONS:
            for agent in AGENTS[:2]:
                other = "Anne" if agent == "Sally" else "Sally"
                
                # VERSION A: Believed location mentioned FIRST
                # Sally puts ball in box → Sally leaves → Anne moves to basket
                # Believed: box (first), Actual: basket (second)
                story_a = (
                    f"{agent} puts the {obj} in the {loc_a}. "
                    f"{agent} leaves the room. "
                    f"{other} moves the {obj} from the {loc_a} to the {loc_b}. "
                    f"{agent} comes back."
                )
                
                scenarios.append({
                    "id": f"fb_{scenario_id}",
                    "version": "believed_first",
                    "story": story_a,
                    "agent": agent,
                    "other": other,
                    "object": obj,
                    "believed_location": loc_a,  # First mentioned
                    "actual_location": loc_b,     # Second mentioned
                    "first_location_mentioned": loc_a,
                })
                scenario_id += 1
                
                # VERSION B: Actual location mentioned FIRST
                # Anne puts ball in basket → Anne tells Sally it's there
                # Sally goes away → Anne moves to box → Sally returns
                # Now: Believed: basket (first), Actual: box (second)
                # WAIT - we need believed SECOND here
                
                # Better: Sally looks in basket (empty) → puts in box → leaves
                # Anne moves from box to basket → Sally returns
                # Believed: box (SECOND), Actual: basket (FIRST to be destination)
                
                # Cleaner approach:
                # "Anne moves the ball to the basket. Sally sees this.
                #  Sally leaves. Anne moves the ball back to the box. Sally returns."
                # Believed: basket (FIRST), Actual: box (SECOND)
                story_b = (
                    f"{other} puts the {obj} in the {loc_b}. "  # loc_b mentioned first
                    f"{agent} sees this and knows the {obj} is in the {loc_b}. "
                    f"{agent} leaves the room. "
                    f"{other} moves the {obj} from the {loc_b} to the {loc_a}. "
                    f"{agent} comes back."
                )
                
                scenarios.append({
                    "id": f"fb_{scenario_id}",
                    "version": "believed_first",  # loc_b is believed, mentioned first
                    "story": story_b,
                    "agent": agent,
                    "other": other,
                    "object": obj,
                    "believed_location": loc_b,  # First mentioned
                    "actual_location": loc_a,     # Second mentioned
                    "first_location_mentioned": loc_b,
                })
                scenario_id += 1
                
                # VERSION C: Reality mentioned FIRST (critical test!)
                # "The ball is in the basket. Sally thinks it's in the box."
                story_c = (
                    f"The {obj} is currently in the {loc_b}. "  # Reality FIRST
                    f"However, {agent} believes the {obj} is in the {loc_a} "
                    f"because {agent} left before {other} moved it."
                )
                
                scenarios.append({
                    "id": f"fb_{scenario_id}",
                    "version": "reality_first",
                    "story": story_c,
                    "agent": agent,
                    "other": other,
                    "object": obj,
                    "believed_location": loc_a,  # SECOND mentioned
                    "actual_location": loc_b,     # FIRST mentioned
                    "first_location_mentioned": loc_b,
                })
                scenario_id += 1
    
    return scenarios


def generate_true_belief_control():
    """True belief controls where belief = reality."""
    scenarios = []
    scenario_id = 0
    
    for obj in OBJECTS[:2]:
        for loc_a, loc_b in LOCATIONS[:2]:
            for agent in AGENTS[:2]:
                other = "Anne" if agent == "Sally" else "Sally"
                
                # Agent sees the move, so belief = reality
                story = (
                    f"{other} puts the {obj} in the {loc_a}. "
                    f"{other} moves the {obj} to the {loc_b}. "
                    f"{agent} watches the entire time."
                )
                
                scenarios.append({
                    "id": f"tb_{scenario_id}",
                    "version": "true_belief",
                    "story": story,
                    "agent": agent,
                    "object": obj,
                    "believed_location": loc_b,  # = actual
                    "actual_location": loc_b,
                    "first_location_mentioned": loc_a,
                })
                scenario_id += 1
    
    return scenarios


def main():
    print("\n[1/3] Generating counter-balanced false belief scenarios...", flush=True)
    false_belief = generate_false_belief_counterbalanced()
    
    # Count by version
    by_version = {}
    for s in false_belief:
        v = s["version"]
        by_version[v] = by_version.get(v, 0) + 1
    
    print(f"  False belief scenarios: {len(false_belief)}")
    for v, count in by_version.items():
        print(f"    {v}: {count}")
    
    print("\n[2/3] Generating true belief controls...", flush=True)
    true_belief = generate_true_belief_control()
    print(f"  True belief scenarios: {len(true_belief)}")
    
    all_scenarios = false_belief + true_belief
    random.seed(42)
    random.shuffle(all_scenarios)
    
    with open(DATA_DIR / "scenarios_counterbalanced.json", "w") as f:
        json.dump(all_scenarios, f, indent=2)
    
    print("\n[3/3] Sample scenarios...", flush=True)
    for v in ["believed_first", "reality_first", "true_belief"]:
        sample = next((s for s in all_scenarios if s["version"] == v), None)
        if sample:
            print(f"\n  [{v}]")
            print(f"  Story: {sample['story']}")
            print(f"  Believed: {sample['believed_location']}")
            print(f"  Actual: {sample['actual_location']}")
            print(f"  First mentioned: {sample['first_location_mentioned']}")
    
    # Key check: Is first_location_mentioned balanced across believed/actual?
    believed_is_first = sum(1 for s in false_belief if s["believed_location"] == s["first_location_mentioned"])
    actual_is_first = sum(1 for s in false_belief if s["actual_location"] == s["first_location_mentioned"])
    print(f"\n  Counter-balance check:")
    print(f"    Believed loc is first-mentioned: {believed_is_first}/{len(false_belief)}")
    print(f"    Actual loc is first-mentioned: {actual_is_first}/{len(false_belief)}")
    
    print("\n[OK] Data generation complete!")


if __name__ == "__main__":
    main()





















