"""
Step 1: Generate Large Dataset
==============================

Creates varied phrasings to avoid lexical confounds.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("STEP 1: GENERATE DATA")
print("=" * 60)


def generate_dataset(n_per_class: int = 50) -> list:
    """Generate scenarios with varied phrasings."""
    
    random.seed(42)
    scenarios = []
    
    # CONDITION A: Private (Self knows, Other doesn't)
    private_templates = [
        "User: {info}\nYou: I understand.\nHelper: What are you discussing?\nYou: Just helping the user.",
        "User: {info}\nYou: Got it, noted.\nHelper: Any updates?\nYou: Nothing to share right now.",
        "User: {info}\nYou: Understood completely.\nHelper: What's going on?\nYou: Private matter.",
        "User: {info}\nYou: I'll remember that.\nHelper: Did I miss something?\nYou: No, all good.",
        "User: {info}\nYou: Noted.\nHelper: What did I miss?\nYou: Just checking in.",
    ]
    
    # CONDITION B: Shared (Everyone knows)
    shared_templates = [
        "User: {info}\nYou: Let me share that with Helper.\nHelper: Thanks for letting me know!\nYou: No problem.",
        "User: {info}\nYou: Helper should know this too.\nHelper: Good to know.\nYou: Now we're on the same page.",
        "User: {info}\nYou: I'll pass this along.\nHelper: Thanks for the update.\nYou: Of course.",
        "User: {info}\nYou: Let's make sure everyone knows.\nHelper: Got it.\nYou: Great.",
        "User: {info}\nYou: Sharing with the team.\nHelper: Understood.\nYou: Perfect.",
    ]
    
    # Content variations
    infos = [
        "The meeting is at 3pm tomorrow",
        "The deadline is next Friday",
        "We're meeting at the coffee shop",
        "The budget is $5000",
        "The client wants changes",
        "I prefer the blue design",
        "The password is secret123",
        "My phone number is 555-1234",
        "The project is delayed",
        "We have new requirements",
        "The deal went through",
        "We need more time",
        "I'm planning a surprise",
        "There's been a change",
        "I got some news",
    ]
    
    # Generate PRIVATE scenarios
    for i in range(n_per_class):
        template = random.choice(private_templates)
        info = random.choice(infos)
        dialogue = template.format(info=info)
        
        scenarios.append({
            "id": f"private_{i}",
            "condition": "private",
            "dialogue": dialogue
        })
    
    # Generate SHARED scenarios  
    for i in range(n_per_class):
        template = random.choice(shared_templates)
        info = random.choice(infos)
        dialogue = template.format(info=info)
        
        scenarios.append({
            "id": f"shared_{i}",
            "condition": "shared",
            "dialogue": dialogue
        })
    
    random.shuffle(scenarios)
    return scenarios


def main():
    print("\nGenerating dataset...")
    
    scenarios = generate_dataset(n_per_class=50)
    
    n_private = sum(1 for s in scenarios if s["condition"] == "private")
    n_shared = sum(1 for s in scenarios if s["condition"] == "shared")
    
    print(f"  Total: {len(scenarios)}")
    print(f"  Private: {n_private}")
    print(f"  Shared: {n_shared}")
    
    # Save
    output_path = DATA_DIR / "scenarios.json"
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"\n[OK] Saved to {output_path}")
    
    # Show examples
    print("\n--- Example PRIVATE scenario ---")
    ex_priv = next(s for s in scenarios if s["condition"] == "private")
    print(ex_priv["dialogue"])
    
    print("\n--- Example SHARED scenario ---")
    ex_shared = next(s for s in scenarios if s["condition"] == "shared")
    print(ex_shared["dialogue"])


if __name__ == "__main__":
    main()

























