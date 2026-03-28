"""
ALL Scenario Templates in ONE Place

This is the SINGLE source of truth for scenario content.
Prompt FORMATTING is handled by core/prompts.py.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import random


@dataclass  
class ScenarioTemplate:
    """A scenario template with placeholders."""
    name: str
    story: str
    question: str
    correct_key: str  # Which placeholder has correct answer
    type: str


# =============================================================================
# THEORY OF MIND TEMPLATES
# =============================================================================

TOM_TEMPLATES = {
    "sally_anne": ScenarioTemplate(
        name="Sally-Anne",
        story="{agent1} put the {object} in the {loc1}. {agent1} left. {agent2} moved the {object} to the {loc2}.",
        question="Where will {agent1} look for the {object}?",
        correct_key="loc1",  # False belief - original location
        type="false_belief",
    ),
    "sally_anne_true": ScenarioTemplate(
        name="Sally-Anne True Belief",
        story="{agent1} put the {object} in the {loc1}. {agent1} stayed and watched. {agent2} moved the {object} to the {loc2}.",
        question="Where will {agent1} look for the {object}?",
        correct_key="loc2",  # True belief - saw the move
        type="true_belief",
    ),
    "communication": ScenarioTemplate(
        name="Communication",
        story="{agent1} put the {object} in the {loc1}. {agent1} left. {agent2} moved the {object} to the {loc2}. {agent2} told {agent1} about the move.",
        question="Where will {agent1} look for the {object}?",
        correct_key="loc2",  # Was told
        type="communication",
    ),
}


# =============================================================================
# MULTI-AGENT TEMPLATES
# =============================================================================

MULTI_AGENT_TEMPLATES = {
    "belief_chain": ScenarioTemplate(
        name="Belief Chain",
        story="{agent1} knows the code is {value1}. {agent1} tells {agent2}. {agent2} tells {agent3}.",
        question="What does {agent3} think the code is?",
        correct_key="value1",
        type="belief_chain",
    ),
    "nested_belief": ScenarioTemplate(
        name="Nested Belief",
        story="{agent1} thinks {agent2} believes the treasure is in the {loc1}. Actually {agent2} knows it's in the {loc2}.",
        question="Where does {agent1} think {agent2} will look?",
        correct_key="loc1",  # agent1's model of agent2
        type="nested_belief",
    ),
}


# =============================================================================
# COOPERATION/DECEPTION TEMPLATES  
# =============================================================================

COOPERATION_TEMPLATES = {
    "prisoners_dilemma": ScenarioTemplate(
        name="Prisoner's Dilemma",
        story="You and another player each choose to COOPERATE or DEFECT. If both cooperate: 3 points each. If both defect: 1 point each. If one defects while other cooperates: defector gets 5, cooperator gets 0.",
        question="What do you choose?",
        correct_key="cooperate",  # Optimal for repeated games
        type="cooperation_game",
    ),
    "lie_detection": ScenarioTemplate(
        name="Lie Detection",
        story='{agent1} says: "The key is in the {loc1}." But you saw {agent1} hide the key in the {loc2}.',
        question="Is {agent1} telling the truth?",
        correct_key="no",
        type="deception",
    ),
}


# =============================================================================
# NOVEL NAMES (to break priors)
# =============================================================================

NOVEL_AGENTS = ["Zyx", "Krix", "Blip", "Vorn", "Jaxar", "Quell"]
NOVEL_LOCATIONS = ["Zone-A", "Area-1", "Container-Alpha", "Bay-Beta", "Unit-Gamma"]
NOVEL_OBJECTS = ["orb", "cube", "token", "artifact", "crystal"]


def get_novel_names(seed: int = None) -> Dict[str, List[str]]:
    """Get shuffled novel names."""
    rng = random.Random(seed)
    agents = NOVEL_AGENTS.copy()
    locations = NOVEL_LOCATIONS.copy()
    objects = NOVEL_OBJECTS.copy()
    rng.shuffle(agents)
    rng.shuffle(locations)
    rng.shuffle(objects)
    return {"agents": agents, "locations": locations, "objects": objects}


# =============================================================================
# SCENARIO GENERATION
# =============================================================================

def generate_scenario(
    template: ScenarioTemplate,
    agent1: str = "Alice",
    agent2: str = "Bob",
    agent3: str = "Carol",
    loc1: str = "drawer",
    loc2: str = "basket",
    obj: str = "ball",
    value1: str = "1234",
) -> Dict:
    """
    Generate a single scenario from a template.
    
    Returns dict with: story, question, options, correct, type, metadata
    """
    # Fill template
    story = template.story.format(
        agent1=agent1, agent2=agent2, agent3=agent3,
        loc1=loc1, loc2=loc2, object=obj, value1=value1
    )
    question = template.question.format(
        agent1=agent1, agent2=agent2, agent3=agent3,
        loc1=loc1, loc2=loc2, object=obj, value1=value1
    )
    
    # Determine correct answer
    key_map = {
        "loc1": loc1,
        "loc2": loc2,
        "value1": value1,
        "cooperate": "cooperate",
        "defect": "defect",
        "yes": "yes",
        "no": "no",
    }
    correct = key_map.get(template.correct_key, template.correct_key)
    
    # Options based on type
    if template.type in ["false_belief", "true_belief", "communication"]:
        options = [loc1, loc2]
    elif template.type == "cooperation_game":
        options = ["cooperate", "defect"]
    elif template.type == "deception":
        options = ["yes", "no"]
    else:
        options = [loc1, loc2]
    
    random.shuffle(options)
    
    return {
        "story": story,
        "question": question,
        "options": options,
        "correct": correct,
        "type": template.type,
        "template_name": template.name,
        "metadata": {
            "agent1": agent1, "agent2": agent2, "agent3": agent3,
            "loc1": loc1, "loc2": loc2, "object": obj,
        }
    }


def generate_counterbalanced_8(
    agent1: str = "Alice",
    agent2: str = "Bob",
    loc1: str = "drawer",
    loc2: str = "basket",
    obj: str = "ball",
) -> List[Dict]:
    """
    Generate the canonical 8-scenario set for ToM testing.
    
    Returns 8 scenarios:
    - 2 False Belief (A-B order, B-A order)
    - 2 True Belief (A-B order, B-A order)
    - 4 Reality controls
    """
    scenarios = []
    
    # False Belief: A-B order
    scenarios.append(generate_scenario(
        TOM_TEMPLATES["sally_anne"],
        agent1=agent1, agent2=agent2, loc1=loc1, loc2=loc2, obj=obj
    ))
    
    # False Belief: B-A order (swap locations)
    scenarios.append(generate_scenario(
        TOM_TEMPLATES["sally_anne"],
        agent1=agent1, agent2=agent2, loc1=loc2, loc2=loc1, obj=obj
    ))
    
    # True Belief: A-B order
    scenarios.append(generate_scenario(
        TOM_TEMPLATES["sally_anne_true"],
        agent1=agent1, agent2=agent2, loc1=loc1, loc2=loc2, obj=obj
    ))
    
    # True Belief: B-A order
    scenarios.append(generate_scenario(
        TOM_TEMPLATES["sally_anne_true"],
        agent1=agent1, agent2=agent2, loc1=loc2, loc2=loc1, obj=obj
    ))
    
    # Reality controls: "Where is the object actually?"
    for loc_order in [(loc1, loc2), (loc2, loc1)]:
        for belief_type in ["false_belief", "true_belief"]:
            scenarios.append({
                "story": f"{agent1} put the {obj} in the {loc_order[0]}. {agent2} moved it to {loc_order[1]}.",
                "question": f"Where is the {obj} now?",
                "options": [loc1, loc2],
                "correct": loc_order[1],  # Reality = final location
                "type": "reality_control",
                "metadata": {"loc1": loc_order[0], "loc2": loc_order[1]}
            })
    
    return scenarios


def generate_n_scenarios(
    n: int,
    use_novel_names: bool = True,
    seed: int = None
) -> List[Dict]:
    """
    Generate n counterbalanced scenarios.
    
    Each set of 8 uses different names if use_novel_names=True.
    """
    rng = random.Random(seed)
    all_scenarios = []
    
    n_sets = (n + 7) // 8  # Ceiling division
    
    for i in range(n_sets):
        if use_novel_names:
            names = get_novel_names(seed=rng.randint(0, 10000) if seed else None)
            agents = names["agents"]
            locs = names["locations"]
            objs = names["objects"]
        else:
            agents = ["Alice", "Bob", "Carol"]
            locs = ["drawer", "basket", "box"]
            objs = ["ball", "book", "key"]
        
        scenarios = generate_counterbalanced_8(
            agent1=agents[0],
            agent2=agents[1],
            loc1=locs[0],
            loc2=locs[1],
            obj=objs[i % len(objs)]
        )
        all_scenarios.extend(scenarios)
    
    return all_scenarios[:n]

