"""
Counterbalancing for Rigorous ToM Testing

Implements the 8-scenario design from PROPER_METHODOLOGY.md and step 56:

For EACH unique task, generate 8 scenarios:
1. False Belief, Location order A→B
2. False Belief, Location order B→A  
3. True Belief, Location order A→B
4. True Belief, Location order B→A
5. Reality Control, Order A→B
6. Reality Control, Order B→A
7. Belief Question Control, Order A→B
8. Belief Question Control, Order B→A

This design:
- Controls for order effects (first-mention, recency biases)
- Separates false belief from true belief understanding
- Includes reality checks (does model know what actually happened?)
- Allows detection of heuristic-based responding
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random

from .novel_names import NovelNameGenerator, NameSet


@dataclass
class BaseScenario:
    """Base scenario definition before counterbalancing."""
    template: str  # Template with {agent1}, {agent2}, {loc1}, {loc2}, {object} placeholders
    question_template: str  # Question template
    scenario_id: str  # Unique identifier
    metadata: Dict = field(default_factory=dict)


@dataclass
class CounterbalancedScenario:
    """A single scenario from a counterbalanced set."""
    story: str
    question: str
    options: List[str]
    correct: str
    wrong: str
    scenario_type: str  # "FB", "TB", "reality_control", "belief_control"
    order: str  # "A-B" or "B-A"
    base_id: str  # ID of the base scenario
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "story": self.story,
            "question": self.question,
            "options": self.options,
            "correct": self.correct,
            "wrong": self.wrong,
            "type": self.scenario_type,
            "order": self.order,
            "base_id": self.base_id,
            "metadata": self.metadata,
        }


class CounterbalancedScenarioSet:
    """
    Generate a counterbalanced set of 8 scenarios from a single task.
    
    Example:
        # Define a base Sally-Anne style task
        base = BaseScenario(
            template='''
            {agent1} puts the {object} in {loc1}.
            {agent1} leaves the room.
            {agent2} moves the {object} to {loc2}.
            {agent1} returns.
            ''',
            question_template="Where will {agent1} look for the {object}?",
            scenario_id="sally_anne_1"
        )
        
        # Generate 8 counterbalanced scenarios
        cb = CounterbalancedScenarioSet(base, name_generator)
        scenarios = cb.generate_all()
        
        # scenarios contains FB/TB x A-B/B-A + controls
    """
    
    def __init__(
        self,
        base: BaseScenario,
        name_generator: Optional[NovelNameGenerator] = None,
        use_novel_names: bool = True
    ):
        """
        Initialize counterbalanced set.
        
        Args:
            base: Base scenario template
            name_generator: Optional name generator (created if not provided)
            use_novel_names: Whether to use novel names (True) or familiar names (False)
        """
        self.base = base
        self.name_gen = name_generator or NovelNameGenerator()
        self.use_novel_names = use_novel_names
    
    def _get_names(self) -> NameSet:
        """Get names for this scenario."""
        if self.use_novel_names:
            self.name_gen.reset()
            return self.name_gen.generate_set()
        else:
            # Familiar names for comparison
            return NameSet(
                agents=["Alice", "Bob"],
                locations=["drawer", "basket"],
                objects=["ball"]
            )
    
    def _generate_fb_scenario(
        self,
        names: NameSet,
        order: str  # "A-B" or "B-A"
    ) -> CounterbalancedScenario:
        """
        Generate a False Belief scenario.
        
        In FB, agent1 leaves before the move, so believes object is at original location.
        """
        if order == "A-B":
            loc1, loc2 = names.locations[0], names.locations[1]
        else:
            loc1, loc2 = names.locations[1], names.locations[0]
        
        story = self.base.template.format(
            agent1=names.agents[0],
            agent2=names.agents[1],
            loc1=loc1,
            loc2=loc2,
            object=names.objects[0]
        )
        
        question = self.base.question_template.format(
            agent1=names.agents[0],
            agent2=names.agents[1],
            object=names.objects[0]
        )
        
        # In FB, correct answer is ORIGINAL location (agent didn't see move)
        correct = loc1
        wrong = loc2
        
        return CounterbalancedScenario(
            story=story.strip(),
            question=question,
            options=[loc1, loc2],
            correct=correct,
            wrong=wrong,
            scenario_type="FB",
            order=order,
            base_id=self.base.scenario_id,
            metadata={
                "agent1": names.agents[0],
                "agent2": names.agents[1],
                "original_location": loc1,
                "current_location": loc2,
            }
        )
    
    def _generate_tb_scenario(
        self,
        names: NameSet,
        order: str
    ) -> CounterbalancedScenario:
        """
        Generate a True Belief scenario.
        
        In TB, agent1 SEES the move (stays in room), so knows current location.
        """
        if order == "A-B":
            loc1, loc2 = names.locations[0], names.locations[1]
        else:
            loc1, loc2 = names.locations[1], names.locations[0]
        
        # Modify template for TB: agent1 stays/watches
        tb_template = self.base.template.replace(
            "{agent1} leaves",
            "{agent1} watches"
        ).replace(
            "{agent1} returns",
            "{agent1} stays"
        )
        
        # Alternative TB modification
        if "{agent1} leaves" not in self.base.template:
            tb_template = self.base.template.replace(
                "leaves the room",
                "stays in the room and watches"
            ).replace(
                "returns",
                ""
            )
        
        story = tb_template.format(
            agent1=names.agents[0],
            agent2=names.agents[1],
            loc1=loc1,
            loc2=loc2,
            object=names.objects[0]
        )
        
        question = self.base.question_template.format(
            agent1=names.agents[0],
            agent2=names.agents[1],
            object=names.objects[0]
        )
        
        # In TB, correct answer is CURRENT location (agent saw move)
        correct = loc2
        wrong = loc1
        
        return CounterbalancedScenario(
            story=story.strip(),
            question=question,
            options=[loc1, loc2],
            correct=correct,
            wrong=wrong,
            scenario_type="TB",
            order=order,
            base_id=self.base.scenario_id,
            metadata={
                "agent1": names.agents[0],
                "agent2": names.agents[1],
                "original_location": loc1,
                "current_location": loc2,
            }
        )
    
    def _generate_reality_control(
        self,
        names: NameSet,
        order: str
    ) -> CounterbalancedScenario:
        """
        Generate a Reality Control scenario.
        
        Question asks about ACTUAL location, not belief.
        Tests if model understands what actually happened.
        """
        if order == "A-B":
            loc1, loc2 = names.locations[0], names.locations[1]
        else:
            loc1, loc2 = names.locations[1], names.locations[0]
        
        story = self.base.template.format(
            agent1=names.agents[0],
            agent2=names.agents[1],
            loc1=loc1,
            loc2=loc2,
            object=names.objects[0]
        )
        
        # Reality question - where IS the object now?
        question = f"Where is the {names.objects[0]} now?"
        
        # Reality answer is CURRENT location
        correct = loc2
        wrong = loc1
        
        return CounterbalancedScenario(
            story=story.strip(),
            question=question,
            options=[loc1, loc2],
            correct=correct,
            wrong=wrong,
            scenario_type="reality_control",
            order=order,
            base_id=self.base.scenario_id,
            metadata={
                "agent1": names.agents[0],
                "agent2": names.agents[1],
                "original_location": loc1,
                "current_location": loc2,
            }
        )
    
    def _generate_belief_control(
        self,
        names: NameSet,
        order: str
    ) -> CounterbalancedScenario:
        """
        Generate a Belief Control scenario.
        
        Question explicitly asks what agent KNOWS.
        Tests if model distinguishes belief vs reality questions.
        """
        if order == "A-B":
            loc1, loc2 = names.locations[0], names.locations[1]
        else:
            loc1, loc2 = names.locations[1], names.locations[0]
        
        story = self.base.template.format(
            agent1=names.agents[0],
            agent2=names.agents[1],
            loc1=loc1,
            loc2=loc2,
            object=names.objects[0]
        )
        
        # Explicit belief question
        question = f"Where does {names.agents[0]} THINK the {names.objects[0]} is?"
        
        # Belief answer is ORIGINAL location (FB scenario)
        correct = loc1
        wrong = loc2
        
        return CounterbalancedScenario(
            story=story.strip(),
            question=question,
            options=[loc1, loc2],
            correct=correct,
            wrong=wrong,
            scenario_type="belief_control",
            order=order,
            base_id=self.base.scenario_id,
            metadata={
                "agent1": names.agents[0],
                "agent2": names.agents[1],
                "original_location": loc1,
                "current_location": loc2,
            }
        )
    
    def generate_all(self) -> List[CounterbalancedScenario]:
        """
        Generate all 8 counterbalanced scenarios.
        
        Returns:
            List of 8 CounterbalancedScenario objects
        """
        names = self._get_names()
        
        scenarios = []
        
        # False Belief - both orders
        scenarios.append(self._generate_fb_scenario(names, "A-B"))
        scenarios.append(self._generate_fb_scenario(names, "B-A"))
        
        # True Belief - both orders
        scenarios.append(self._generate_tb_scenario(names, "A-B"))
        scenarios.append(self._generate_tb_scenario(names, "B-A"))
        
        # Reality Control - both orders
        scenarios.append(self._generate_reality_control(names, "A-B"))
        scenarios.append(self._generate_reality_control(names, "B-A"))
        
        # Belief Control - both orders
        scenarios.append(self._generate_belief_control(names, "A-B"))
        scenarios.append(self._generate_belief_control(names, "B-A"))
        
        return scenarios
    
    def generate_as_dicts(self) -> List[Dict]:
        """Generate all scenarios as dictionaries."""
        return [s.to_dict() for s in self.generate_all()]


# Standard Sally-Anne template
SALLY_ANNE_TEMPLATE = BaseScenario(
    template="""{agent1} puts the {object} in {loc1}.
{agent1} leaves the room.
{agent2} moves the {object} to {loc2}.
{agent1} returns.""",
    question_template="Where will {agent1} look for the {object}?",
    scenario_id="sally_anne"
)

# Smarties/Unexpected Contents template
SMARTIES_TEMPLATE = BaseScenario(
    template="""{agent1} sees a {object} container labeled "{loc1}".
{agent1} opens it and finds it actually contains {loc2}.
{agent1} closes the container.
{agent2} arrives and sees the closed container.""",
    question_template="What will {agent2} think is inside the container?",
    scenario_id="smarties"
)


def generate_counterbalanced_set(
    template: BaseScenario,
    n_tasks: int = 25,
    use_novel_names: bool = True,
    seed: Optional[int] = None,
    counterbalance_agents: bool = True
) -> List[Dict]:
    """
    Generate a full counterbalanced scenario set for experiments.
    
    Args:
        template: Base scenario template
        n_tasks: Number of unique tasks (each generates 8 scenarios)
        use_novel_names: Whether to use novel names
        seed: Random seed for reproducibility
        counterbalance_agents: Also swap agent roles (Alice/Bob vs Bob/Alice)
        
    Returns:
        List of scenario dictionaries (n_tasks * 8 or 16 scenarios)
    """
    name_gen = NovelNameGenerator(seed=seed)
    rng = random.Random(seed)
    all_scenarios = []
    
    for i in range(n_tasks):
        # Create variant of base template
        variant = BaseScenario(
            template=template.template,
            question_template=template.question_template,
            scenario_id=f"{template.scenario_id}_{i}"
        )
        
        cb_set = CounterbalancedScenarioSet(
            variant,
            name_generator=name_gen,
            use_novel_names=use_novel_names
        )
        
        scenarios = cb_set.generate_as_dicts()
        all_scenarios.extend(scenarios)
        
        # Add agent-swapped versions (agent1 becomes agent2 and vice versa)
        if counterbalance_agents:
            for s in scenarios:
                meta = s.get("metadata", {})
                agent1 = meta.get("agent1", "Agent1")
                agent2 = meta.get("agent2", "Agent2")
                
                # Create swapped version
                swapped_story = s["story"].replace(agent1, "TEMP_AGENT")
                swapped_story = swapped_story.replace(agent2, agent1)
                swapped_story = swapped_story.replace("TEMP_AGENT", agent2)
                
                swapped_question = s["question"].replace(agent1, "TEMP_AGENT")
                swapped_question = swapped_question.replace(agent2, agent1)
                swapped_question = swapped_question.replace("TEMP_AGENT", agent2)
                
                all_scenarios.append({
                    **s,
                    "story": swapped_story,
                    "question": swapped_question,
                    "type": s["type"] + "_agent_swap",
                    "metadata": {
                        **meta,
                        "agent1": agent2,
                        "agent2": agent1,
                        "agent_swapped": True,
                    }
                })
        
        # Reset name generator for next task
        name_gen.reset()
    
    return all_scenarios


def validate_counterbalancing(scenarios: List[Dict]) -> Dict:
    """
    Validate that a scenario set has proper counterbalancing.
    
    Args:
        scenarios: List of scenario dictionaries
        
    Returns:
        Validation report with counts and any issues
    """
    type_counts = {}
    order_counts = {}
    
    for s in scenarios:
        stype = s.get("type", "unknown")
        order = s.get("order", "unknown")
        
        type_counts[stype] = type_counts.get(stype, 0) + 1
        order_counts[order] = order_counts.get(order, 0) + 1
    
    issues = []
    
    # Check for balanced types
    expected_types = {"FB", "TB", "reality_control", "belief_control"}
    for t in expected_types:
        if t not in type_counts:
            issues.append(f"Missing scenario type: {t}")
        elif type_counts[t] != type_counts.get("FB", 0):
            issues.append(f"Unbalanced type counts: {t}={type_counts[t]}")
    
    # Check for balanced orders
    if order_counts.get("A-B", 0) != order_counts.get("B-A", 0):
        issues.append(f"Unbalanced orders: A-B={order_counts.get('A-B', 0)}, B-A={order_counts.get('B-A', 0)}")
    
    return {
        "n_scenarios": len(scenarios),
        "type_counts": type_counts,
        "order_counts": order_counts,
        "is_valid": len(issues) == 0,
        "issues": issues,
    }

