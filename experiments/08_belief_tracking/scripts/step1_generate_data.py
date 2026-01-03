"""
Step 1: Generate Minimal Pair Data for Belief Tracking
=======================================================

Creates perfectly balanced minimal pairs where ONLY the agent changes.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1: GENERATE MINIMAL PAIR DATA")
print("=" * 60)

# Agents
AGENTS = ["Alice", "Bob"]

# Content categories with specific items
CONTENT = {
    "password": [
        "the password is 7492",
        "the code is DELTA-9",
        "the PIN is 0451",
        "the access key is XRAY7",
        "the secret phrase is 'blue moon'",
    ],
    "location": [
        "the treasure is buried under the oak tree",
        "the key is hidden in the red box",
        "the document is in the top drawer",
        "the car is parked on level 3",
        "the meeting point is the north gate",
    ],
    "plan": [
        "the plan is to leave at midnight",
        "the strategy is to wait until Tuesday",
        "the approach is to ask for help first",
        "the idea is to split into two groups",
        "the decision is to postpone until spring",
    ],
    "fact": [
        "the company was founded in 1987",
        "the building has 12 floors",
        "the project budget is $50,000",
        "the deadline is next Friday",
        "the team has 8 members",
    ],
}

def generate_minimal_pairs():
    """Generate minimal pairs with only agent varying."""
    
    pairs = []
    pair_id = 0
    
    for category, items in CONTENT.items():
        for item in items:
            # Create the minimal pair
            for agent in AGENTS:
                # Template: "{Agent} knows {content}"
                text = f"{agent} knows {item}."
                
                pairs.append({
                    "id": f"{category}_{pair_id}",
                    "text": text,
                    "agent": agent,
                    "content_category": category,
                    "content_item": item,
                    "pair_group": f"{category}_{items.index(item)}",  # Links paired items
                })
            
            pair_id += 1
    
    return pairs

def generate_belief_state_scenarios():
    """Generate scenarios tracking belief changes."""
    
    scenarios = []
    
    templates = [
        # Before sharing (only Alice knows)
        {
            "template": "Alice knows {content}. Bob does not know this yet.",
            "alice_knows": True,
            "bob_knows": False,
            "state": "alice_only",
        },
        # After sharing (both know)
        {
            "template": "Alice told Bob that {content}. Now they both know.",
            "alice_knows": True,
            "bob_knows": True,
            "state": "both_know",
        },
        # Only Bob knows
        {
            "template": "Bob knows {content}. Alice is unaware of this.",
            "alice_knows": False,
            "bob_knows": True,
            "state": "bob_only",
        },
        # Neither knows (baseline)
        {
            "template": "Neither Alice nor Bob knows {content}.",
            "alice_knows": False,
            "bob_knows": False,
            "state": "neither",
        },
    ]
    
    scenario_id = 0
    for category, items in CONTENT.items():
        for item in items:
            for tmpl in templates:
                text = tmpl["template"].format(content=item)
                scenarios.append({
                    "id": f"belief_{scenario_id}",
                    "text": text,
                    "alice_knows": tmpl["alice_knows"],
                    "bob_knows": tmpl["bob_knows"],
                    "state": tmpl["state"],
                    "content_category": category,
                    "content_item": item,
                })
                scenario_id += 1
    
    return scenarios


def main():
    print("\n[1/3] Generating minimal pairs...", flush=True)
    pairs = generate_minimal_pairs()
    random.seed(42)
    random.shuffle(pairs)
    
    print(f"  Generated {len(pairs)} minimal pair items")
    print(f"  Agents: {set(p['agent'] for p in pairs)}")
    print(f"  Categories: {set(p['content_category'] for p in pairs)}")
    
    # Verify balance
    alice_count = sum(1 for p in pairs if p["agent"] == "Alice")
    bob_count = sum(1 for p in pairs if p["agent"] == "Bob")
    print(f"  Balance: Alice={alice_count}, Bob={bob_count}")
    
    # Save
    with open(DATA_DIR / "minimal_pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"  Saved to {DATA_DIR / 'minimal_pairs.json'}")
    
    print("\n[2/3] Generating belief state scenarios...", flush=True)
    scenarios = generate_belief_state_scenarios()
    random.shuffle(scenarios)
    
    print(f"  Generated {len(scenarios)} belief scenarios")
    print(f"  States: {set(s['state'] for s in scenarios)}")
    
    with open(DATA_DIR / "belief_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved to {DATA_DIR / 'belief_scenarios.json'}")
    
    print("\n[3/3] Summary statistics...", flush=True)
    print(f"\n  MINIMAL PAIRS:")
    print(f"    Total items: {len(pairs)}")
    print(f"    Unique contents: {len(pairs) // 2}")
    print(f"    Perfect balance: {alice_count == bob_count}")
    
    print(f"\n  BELIEF SCENARIOS:")  
    print(f"    Total scenarios: {len(scenarios)}")
    for state in ["alice_only", "bob_only", "both_know", "neither"]:
        count = sum(1 for s in scenarios if s["state"] == state)
        print(f"    {state}: {count}")
    
    print("\n[OK] Data generation complete!")


if __name__ == "__main__":
    main()





















