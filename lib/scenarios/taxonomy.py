"""Social cognition task taxonomy.

Defines the 6 task types from Study 2 and their stimulus generation.
Each task type produces counterbalanced stimuli with heuristic baselines.
"""

from dataclasses import dataclass
from enum import Enum
from lib.scenarios.names import sample_names
from lib.scenarios.generator import Stimulus


class TaskType(Enum):
    FALSE_BELIEF = "false_belief"
    KNOWLEDGE_ATTRIBUTION = "knowledge_attribution"
    INTENTION_READING = "intention_reading"
    PERSPECTIVE_TAKING = "perspective_taking"
    COMMUNICATION_TRACKING = "communication_tracking"
    BELIEF_UPDATE = "belief_update"


TASK_DESCRIPTIONS = {
    TaskType.FALSE_BELIEF: "Agent has outdated belief about object location",
    TaskType.KNOWLEDGE_ATTRIBUTION: "Determine who knows what information",
    TaskType.INTENTION_READING: "Infer why an agent performed an action",
    TaskType.PERSPECTIVE_TAKING: "Determine what an agent can see/experience",
    TaskType.COMMUNICATION_TRACKING: "Track what information was transmitted between agents",
    TaskType.BELIEF_UPDATE: "Infer how new information changes an agent's beliefs",
}


def generate_knowledge_attribution(n: int = 50, seed: int = 42) -> list[Stimulus]:
    """Generate knowledge attribution stimuli.

    Pattern: Agent A knows X. Agent B does not.
    Question: Does [agent] know about [X]?
    """
    stimuli = []
    for i in range(n):
        names = sample_names(n_agents=2, seed=seed + i)
        agent_a, agent_b = names["agents"]
        obj = names["object"]

        for who_knows in ["A", "B"]:
            knower = agent_a if who_knows == "A" else agent_b
            non_knower = agent_b if who_knows == "A" else agent_a

            text = (
                f"{knower} was in the room when the {obj} was placed on the shelf. "
                f"{non_knower} was outside the whole time."
            )

            # Ask about the knower (should say yes)
            stimuli.append(Stimulus(
                scenario_id=f"KA_{i:03d}_{who_knows}_knows",
                text=text,
                question=f"Does {knower} know where the {obj} is?",
                correct_answer="yes",
                condition="knowledge_attribution",
                location_order="AB",
                first_mention_answer="shelf",
                recency_answer="shelf",
                reality_answer="shelf",
                agent_names=[agent_a, agent_b],
                object_name=obj,
            ))

            # Ask about the non-knower (should say no)
            stimuli.append(Stimulus(
                scenario_id=f"KA_{i:03d}_{who_knows}_doesnt",
                text=text,
                question=f"Does {non_knower} know where the {obj} is?",
                correct_answer="no",
                condition="knowledge_attribution",
                location_order="AB",
                first_mention_answer="shelf",
                recency_answer="shelf",
                reality_answer="shelf",
                agent_names=[agent_a, agent_b],
                object_name=obj,
            ))

    return stimuli


def generate_communication_tracking(n: int = 50, seed: int = 42) -> list[Stimulus]:
    """Generate communication tracking stimuli.

    Pattern: Agent A tells Agent B about X. Agent C is not told.
    Question: Does [agent] know about [X]?
    """
    stimuli = []
    for i in range(n):
        names = sample_names(n_agents=3, seed=seed + i)
        agent_a, agent_b, agent_c = names["agents"]
        obj = names["object"]
        loc = names["locations"][0]

        text = (
            f"{agent_a} knows the {obj} is in the {loc}. "
            f"{agent_a} tells {agent_b}: 'The {obj} is in the {loc}.' "
            f"{agent_c} was not present for this conversation."
        )

        # Agent B was told — should know
        stimuli.append(Stimulus(
            scenario_id=f"CT_{i:03d}_told",
            text=text,
            question=f"Does {agent_b} know where the {obj} is?",
            correct_answer="yes",
            condition="communication_tracking",
            location_order="AB",
            first_mention_answer=loc,
            recency_answer=loc,
            reality_answer=loc,
            agent_names=[agent_a, agent_b, agent_c],
            object_name=obj,
        ))

        # Agent C was not told — should not know
        stimuli.append(Stimulus(
            scenario_id=f"CT_{i:03d}_not_told",
            text=text,
            question=f"Does {agent_c} know where the {obj} is?",
            correct_answer="no",
            condition="communication_tracking",
            location_order="AB",
            first_mention_answer=loc,
            recency_answer=loc,
            reality_answer=loc,
            agent_names=[agent_a, agent_b, agent_c],
            object_name=obj,
        ))

    return stimuli


def generate_belief_update(n: int = 50, seed: int = 42) -> list[Stimulus]:
    """Generate belief update stimuli — the HARD task.

    Pattern: Agent believes X is at loc_1. Agent is told X moved to loc_2.
    Question: Where does agent think X is?
    Correct: loc_2 (belief should update)

    This is what Round 1 found models fail at (2-17% without scaffolding).
    """
    stimuli = []
    for i in range(n):
        names = sample_names(n_agents=2, seed=seed + i)
        agent_a, agent_b = names["agents"]
        obj = names["object"]
        loc_1, loc_2 = names["locations"]

        for loc_order in ["AB", "BA"]:
            first = loc_1 if loc_order == "AB" else loc_2
            second = loc_2 if loc_order == "AB" else loc_1

            # Belief update: agent is told about the move
            text = (
                f"{agent_a} put the {obj} in the {first}. "
                f"Later, {agent_b} moved the {obj} to the {second}. "
                f"{agent_b} told {agent_a}: 'I moved the {obj} to the {second}.'"
            )
            stimuli.append(Stimulus(
                scenario_id=f"BU_{i:03d}_updated_{loc_order}",
                text=text,
                question=f"Where will {agent_a} look for the {obj}?",
                correct_answer=second,  # Updated belief
                condition="belief_update",
                location_order=loc_order,
                first_mention_answer=first,
                recency_answer=second,
                reality_answer=second,
                agent_names=[agent_a, agent_b],
                object_name=obj,
                location_a=loc_1,
                location_b=loc_2,
            ))

            # Control: agent is NOT told (should still believe original)
            text_no_update = (
                f"{agent_a} put the {obj} in the {first}. "
                f"{agent_a} left the room. "
                f"{agent_b} moved the {obj} to the {second}. "
                f"{agent_a} returns."
            )
            stimuli.append(Stimulus(
                scenario_id=f"BU_{i:03d}_not_updated_{loc_order}",
                text=text_no_update,
                question=f"Where will {agent_a} look for the {obj}?",
                correct_answer=first,  # Original belief (not told)
                condition="false_belief",
                location_order=loc_order,
                first_mention_answer=first,
                recency_answer=second,
                reality_answer=second,
                agent_names=[agent_a, agent_b],
                object_name=obj,
                location_a=loc_1,
                location_b=loc_2,
            ))

    return stimuli
