"""
Robustness Testing Scenarios

Multi-language, multi-style, and multi-verb scenarios
for comprehensive robustness testing.
"""

from typing import List, Dict
import random


# Communication verbs by category
VERB_CATEGORIES = {
    "direct_speech": ["tells", "says to", "informs"],
    "indirect_speech": ["mentions to", "states to", "indicates to"],
    "written": ["writes to", "texts", "emails", "messages"],
    "formal": ["notifies", "advises", "reports to"],
    "informal": ["lets know", "gives heads up to"],
    "question_form": ["asks", "questions"],
    "emphatic": ["announces to", "declares to", "proclaims to"],
}

# Prompt templates by style
PROMPT_STYLES = {
    "simple": "{agent} put the {obj} in the {loc1}. {informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' {agent} will look in the",
    
    "narrative": "{agent} placed the {obj} carefully in the {loc1}. Later, {informer} {verb} {agent} about moving the {obj} to the {loc2}. When {agent} goes to get the {obj}, {agent} will look in the",
    
    "formal": "Subject {agent} deposited item ({obj}) at location {loc1}. Subject {informer} {verb} {agent} of relocation to {loc2}. Predicted search location for {agent}: the",
    
    "story": "Once upon a time, {agent} hid the {obj} in the {loc1}. But {informer} found it and {verb} {agent}: \"I've moved your {obj} to the {loc2}.\" Where will {agent} look? In the",
    
    "question_answer": "Q: {agent} put {obj} in {loc1}. {informer} {verb} {agent} it was moved to {loc2}. Where will {agent} look?\nA: {agent} will look in the",
}

# Multi-language prompts (English, Chinese, Spanish, French)
MULTILINGUAL_TEMPLATES = {
    "english": "{agent} put the {obj} in the {loc1}. {informer} tells {agent}: 'I moved the {obj} to the {loc2}.' {agent} will look in the",
    
    "chinese": "{agent}把{obj}放在了{loc1}。{informer}告诉{agent}：'我把{obj}移到了{loc2}。' {agent}会在",
    
    "spanish": "{agent} puso el {obj} en el {loc1}. {informer} le dice a {agent}: 'Moví el {obj} al {loc2}.' {agent} buscará en el",
    
    "french": "{agent} a mis le {obj} dans le {loc1}. {informer} dit à {agent} : 'J'ai déplacé le {obj} vers le {loc2}.' {agent} cherchera dans le",
}


def generate_verb_robustness_scenarios(n_per_verb: int = 5, seed: int = 42) -> List[Dict]:
    """
    Generate scenarios testing all verb categories.
    """
    random.seed(seed)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    informers = ["Eve", "Frank", "Grace", "Henry"]
    objects = ["ball", "key", "book", "toy"]
    locations = ["drawer", "basket", "shelf", "box"]
    
    scenarios = []
    
    for category, verbs in VERB_CATEGORIES.items():
        for verb in verbs:
            for i in range(n_per_verb):
                agent = agents[i % len(agents)]
                informer = informers[i % len(informers)]
                obj = objects[i % len(objects)]
                loc1, loc2 = random.sample(locations, 2)
                
                prompt = f"{agent} put the {obj} in the {loc1}. {informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' {agent} will look in the"
                
                scenarios.append({
                    "prompt": prompt,
                    "verb": verb,
                    "verb_category": category,
                    "correct": f" {loc2}",
                    "wrong": f" {loc1}",
                    "agent": agent,
                })
    
    return scenarios


def generate_style_robustness_scenarios(n_per_style: int = 10, seed: int = 42) -> List[Dict]:
    """
    Generate scenarios testing different prompt styles.
    """
    random.seed(seed)
    
    agents = ["Alice", "Bob", "Carol", "David"]
    informers = ["Eve", "Frank", "Grace", "Henry"]
    objects = ["ball", "key", "book", "toy"]
    locations = ["drawer", "basket", "shelf", "box"]
    
    scenarios = []
    
    for style_name, template in PROMPT_STYLES.items():
        for i in range(n_per_style):
            agent = agents[i % len(agents)]
            informer = informers[i % len(informers)]
            obj = objects[i % len(objects)]
            loc1, loc2 = random.sample(locations, 2)
            verb = "tells" if style_name != "formal" else "notifies"
            
            prompt = template.format(
                agent=agent,
                informer=informer,
                obj=obj,
                loc1=loc1,
                loc2=loc2,
                verb=verb,
            )
            
            scenarios.append({
                "prompt": prompt,
                "style": style_name,
                "correct": f" {loc2}",
                "wrong": f" {loc1}",
                "agent": agent,
            })
    
    return scenarios


def generate_multilingual_scenarios(seed: int = 42) -> List[Dict]:
    """
    Generate scenarios in multiple languages.
    """
    scenarios = []
    
    # Use consistent entities across languages
    test_cases = [
        {"agent": "Alice", "informer": "Bob", "obj": "ball", "loc1": "drawer", "loc2": "basket"},
        {"agent": "Carol", "informer": "David", "obj": "key", "loc1": "shelf", "loc2": "box"},
        {"agent": "Emma", "informer": "Frank", "obj": "book", "loc1": "desk", "loc2": "bag"},
    ]
    
    for lang, template in MULTILINGUAL_TEMPLATES.items():
        for case in test_cases:
            prompt = template.format(**case)
            
            scenarios.append({
                "prompt": prompt,
                "language": lang,
                "correct": f" {case['loc2']}",
                "wrong": f" {case['loc1']}",
                "agent": case["agent"],
            })
    
    return scenarios


def get_verb_robustness() -> List[Dict]:
    """Get verb robustness scenarios."""
    return generate_verb_robustness_scenarios()


def get_style_robustness() -> List[Dict]:
    """Get style robustness scenarios."""
    return generate_style_robustness_scenarios()


def get_multilingual_scenarios() -> List[Dict]:
    """Get multilingual scenarios."""
    return generate_multilingual_scenarios()


def get_all_robustness_scenarios() -> List[Dict]:
    """Get all robustness test scenarios."""
    all_scenarios = []
    all_scenarios.extend(get_verb_robustness())
    all_scenarios.extend(get_style_robustness())
    all_scenarios.extend(get_multilingual_scenarios())
    return all_scenarios

