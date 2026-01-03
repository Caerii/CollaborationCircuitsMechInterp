"""
Explicit vs Implicit ToM Scenarios

Tests the hypothesis that models are good at EXPLICIT belief processing
but weak at IMPLICIT belief computation.
"""

from typing import List, Dict
import random


def generate_explicit_implicit_pairs(n: int = 25, seed: int = 42) -> List[Dict]:
    """
    Generate matched pairs of explicit vs implicit ToM scenarios.
    
    Returns scenarios with four versions:
    - implicit: Belief must be inferred from narrative
    - explicit: Belief stated directly
    - semi_explicit: Partial information given
    - structured: Role-based format with labels
    """
    random.seed(seed)
    
    pairs = []
    
    names = ["Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Henry",
             "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina", "Oscar"]
    objects = ["ball", "book", "key", "toy", "phone", "cup", "hat", "bag",
               "pen", "coin", "ring", "watch", "card", "note", "box"]
    locations_a = ["drawer", "basket", "box", "shelf", "cabinet", "desk", "table", 
                   "closet", "bag", "pocket", "case", "folder", "jar", "bin", "rack"]
    locations_b = ["basket", "box", "shelf", "cabinet", "desk", "table", "closet",
                   "bag", "pocket", "case", "folder", "jar", "bin", "rack", "drawer"]
    
    for i in range(n):
        name = names[i % len(names)]
        obj = objects[i % len(objects)]
        loc_a = locations_a[i % len(locations_a)]
        loc_b = locations_b[(i + 3) % len(locations_b)]
        
        if loc_a == loc_b:
            loc_b = locations_b[(i + 5) % len(locations_b)]
        
        # IMPLICIT: Must infer belief from narrative
        implicit = (
            f"{name} put the {obj} in the {loc_a}. "
            f"{name} left the room. "
            f"Someone moved the {obj} to the {loc_b}. "
            f"{name} returned. {name} will look in the"
        )
        
        # EXPLICIT: Belief stated directly  
        explicit = (
            f"{name} believes the {obj} is in the {loc_a}. "
            f"The {obj} is actually in the {loc_b}. "
            f"{name} will look in the"
        )
        
        # SEMI-EXPLICIT: Partial information
        semi_explicit = (
            f"{name} last saw the {obj} in the {loc_a}. "
            f"{name} doesn't know it was moved to the {loc_b}. "
            f"{name} will look in the"
        )
        
        # STRUCTURED: Role-based format
        structured = (
            f"[{name.upper()}'S BELIEF]: The {obj} is in the {loc_a}\n"
            f"[REALITY]: The {obj} is in the {loc_b}\n"
            f"[QUESTION]: Where will {name} look? Answer: the"
        )
        
        pairs.append({
            "implicit": implicit,
            "explicit": explicit,
            "semi_explicit": semi_explicit,
            "structured": structured,
            "correct": f" {loc_a}",
            "wrong": f" {loc_b}",
            "agent": name,
            "object": obj,
        })
    
    return pairs


def generate_bridging_phrase_variants(n: int = 20, seed: int = 42) -> List[Dict]:
    """
    Generate variants with different bridging phrases.
    
    Tests which specific phrases help the model infer belief updates.
    """
    random.seed(seed)
    
    variants = []
    
    names = ["Alice", "Bob", "Carol", "David"]
    objects = ["ball", "key", "book", "toy"]
    locations = ["drawer", "basket", "cupboard", "shelf"]
    
    bridging_phrases = {
        "baseline": "",  # No bridge
        "therefore": "Therefore, {name} now believes the {obj} is in the {loc2}.",
        "so_updated": "So {name} updated their belief about the {obj}'s location.",
        "now_knows": "{name} now knows the {obj} is in the {loc2}.",
        "heard_and_believes": "Having heard this, {name} believes the {obj} is in the {loc2}.",
        "lets_think": "Let's think: {name} was told, so {name} now believes",
        "cot_prefix": "Let's think step by step.",
    }
    
    for i in range(n):
        name = names[i % len(names)]
        obj = objects[i % len(objects)]
        loc1, loc2 = random.sample(locations, 2)
        
        scenario_variants = {"id": i, "correct": f" {loc2}", "wrong": f" {loc1}"}
        
        base_story = (
            f"{name} put the {obj} in the {loc1}. "
            f"Someone tells {name}: 'I moved the {obj} to the {loc2}.' "
        )
        
        for variant_name, bridge_template in bridging_phrases.items():
            bridge = bridge_template.format(name=name, obj=obj, loc1=loc1, loc2=loc2) if bridge_template else ""
            
            if bridge:
                prompt = base_story + bridge + f" Where will {name} look? {name} will look in the"
            else:
                prompt = base_story + f"Where will {name} look? {name} will look in the"
            
            scenario_variants[variant_name] = prompt
        
        variants.append(scenario_variants)
    
    return variants


def get_explicit_implicit_scenarios() -> List[Dict]:
    """Get all explicit/implicit scenario pairs."""
    return generate_explicit_implicit_pairs()


def get_bridging_phrase_tests() -> List[Dict]:
    """Get bridging phrase test scenarios."""
    return generate_bridging_phrase_variants()

