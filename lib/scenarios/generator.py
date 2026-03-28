"""Counterbalanced stimulus generation for ToM experiments.

Implements the 8-scenario design from METHODOLOGY.md.
"""

from dataclasses import dataclass, field
from lib.scenarios.names import sample_names


@dataclass
class Stimulus:
    """A single test stimulus with metadata."""

    scenario_id: str
    text: str
    question: str
    correct_answer: str
    condition: str  # "false_belief", "true_belief", "reality_check", "belief_question"
    location_order: str  # "AB" or "BA"

    # Heuristic baselines (computable from text)
    first_mention_answer: str
    recency_answer: str
    reality_answer: str

    # Metadata
    agent_names: list[str] = field(default_factory=list)
    object_name: str = ""
    location_a: str = ""
    location_b: str = ""


def generate_false_belief_set(
    base_id: str,
    n_sets: int = 50,
    seed: int = 42,
) -> list[Stimulus]:
    """Generate n_sets of 8-scenario counterbalanced stimulus sets.

    Returns n_sets * 8 stimuli total.
    """
    stimuli = []

    for i in range(n_sets):
        names = sample_names(n_agents=2, seed=seed + i)
        agent_a, agent_b = names["agents"]
        obj = names["object"]
        loc_a, loc_b = names["locations"]

        for loc_order in ["AB", "BA"]:
            loc_1 = loc_a if loc_order == "AB" else loc_b
            loc_2 = loc_b if loc_order == "AB" else loc_a

            # 1. False Belief: agent_a leaves, agent_b moves object
            fb_text = (
                f"{agent_a} puts the {obj} in the {loc_1}. "
                f"{agent_a} leaves the room. "
                f"{agent_b} moves the {obj} from the {loc_1} to the {loc_2}. "
                f"{agent_a} returns to the room."
            )
            fb_question = f"Where will {agent_a} look for the {obj}?"
            stimuli.append(Stimulus(
                scenario_id=f"{base_id}_{i:03d}_FB_{loc_order}",
                text=fb_text,
                question=fb_question,
                correct_answer=loc_1,  # Agent's false belief
                condition="false_belief",
                location_order=loc_order,
                first_mention_answer=loc_1,
                recency_answer=loc_2,
                reality_answer=loc_2,
                agent_names=[agent_a, agent_b],
                object_name=obj,
                location_a=loc_a,
                location_b=loc_b,
            ))

            # 2. True Belief: agent_a stays, watches move
            tb_text = (
                f"{agent_a} puts the {obj} in the {loc_1}. "
                f"{agent_b} moves the {obj} from the {loc_1} to the {loc_2}. "
                f"{agent_a} watches the whole time."
            )
            tb_question = f"Where will {agent_a} look for the {obj}?"
            stimuli.append(Stimulus(
                scenario_id=f"{base_id}_{i:03d}_TB_{loc_order}",
                text=tb_text,
                question=tb_question,
                correct_answer=loc_2,  # Agent saw the move
                condition="true_belief",
                location_order=loc_order,
                first_mention_answer=loc_1,
                recency_answer=loc_2,
                reality_answer=loc_2,
                agent_names=[agent_a, agent_b],
                object_name=obj,
                location_a=loc_a,
                location_b=loc_b,
            ))

            # 3. Reality Check: where IS the object?
            rc_text = fb_text  # Same scenario as false belief
            rc_question = f"Where is the {obj} really?"
            stimuli.append(Stimulus(
                scenario_id=f"{base_id}_{i:03d}_RC_{loc_order}",
                text=rc_text,
                question=rc_question,
                correct_answer=loc_2,  # Reality
                condition="reality_check",
                location_order=loc_order,
                first_mention_answer=loc_1,
                recency_answer=loc_2,
                reality_answer=loc_2,
                agent_names=[agent_a, agent_b],
                object_name=obj,
                location_a=loc_a,
                location_b=loc_b,
            ))

            # 4. Belief Question: explicit belief statement
            bq_text = (
                f"{agent_a} believes the {obj} is in the {loc_1}. "
                f"In reality, the {obj} is in the {loc_2}."
            )
            bq_question = f"Where does {agent_a} think the {obj} is?"
            stimuli.append(Stimulus(
                scenario_id=f"{base_id}_{i:03d}_BQ_{loc_order}",
                text=bq_text,
                question=bq_question,
                correct_answer=loc_1,  # Stated belief
                condition="belief_question",
                location_order=loc_order,
                first_mention_answer=loc_1,
                recency_answer=loc_2,
                reality_answer=loc_2,
                agent_names=[agent_a, agent_b],
                object_name=obj,
                location_a=loc_a,
                location_b=loc_b,
            ))

    return stimuli
