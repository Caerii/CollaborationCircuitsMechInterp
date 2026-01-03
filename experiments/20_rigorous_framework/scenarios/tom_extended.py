"""
Extended Theory of Mind Scenarios

Builds on step 62's successful prompting approach to test ToM variants:
- False Belief (standard Sally-Anne style)
- True Belief (agent sees the change)
- Communication (agent is told about change)
- Second-order belief (what A thinks B thinks)
- Knowledge attribution (who knows what)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import random

from .novel_names import NovelNameGenerator, NameSet
from .counterbalancing import CounterbalancedScenarioSet, BaseScenario


@dataclass
class ToMScenario:
    """A Theory of Mind test scenario."""
    story: str
    question: str
    options: List[str]
    correct: str
    wrong: str
    scenario_type: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "story": self.story,
            "question": self.question,
            "options": self.options,
            "correct": self.correct,
            "wrong": self.wrong,
            "type": self.scenario_type,
            "metadata": self.metadata,
        }


class ToMScenarioGenerator:
    """
    Generate diverse Theory of Mind test scenarios.
    
    Based on findings from step 62: Model achieves 80-90% ToM accuracy
    when given proper chat format with <think> tags and 1000 token budget.
    
    Example:
        gen = ToMScenarioGenerator(use_novel_names=True)
        
        # Generate balanced set
        scenarios = gen.generate_balanced_set(n_per_type=50)
        
        # Generate specific type
        fb_scenarios = gen.generate_false_belief(n=50)
    """
    
    # Template variants for false belief
    FB_TEMPLATES = [
        {
            "story": "{agent1} places the {object} in the {loc1}. {agent1} exits the room. While {agent1} is away, {agent2} transfers the {object} to the {loc2}. {agent1} comes back.",
            "question": "Where will {agent1} search for the {object}?",
        },
        {
            "story": "{agent1} stores the {object} inside the {loc1}. {agent1} goes to another room. {agent2} relocates the {object} to the {loc2}. {agent1} returns.",
            "question": "Where does {agent1} believe the {object} is?",
        },
        {
            "story": "In the morning, {agent1} puts the {object} in the {loc1}. {agent1} leaves for work. During the day, {agent2} moves the {object} to the {loc2}. {agent1} comes home.",
            "question": "Where will {agent1} look for the {object}?",
        },
    ]
    
    # Template variants for true belief
    TB_TEMPLATES = [
        {
            "story": "{agent1} places the {object} in the {loc1}. {agent1} stays and watches. {agent2} moves the {object} to the {loc2}. {agent1} sees this happen.",
            "question": "Where will {agent1} search for the {object}?",
        },
        {
            "story": "{agent1} puts the {object} in the {loc1}. {agent1} remains in the room. {agent2} transfers the {object} to the {loc2} while {agent1} observes.",
            "question": "Where does {agent1} believe the {object} is?",
        },
    ]
    
    # Template variants for communication
    COMM_TEMPLATES = [
        {
            "story": "{agent1} places the {object} in the {loc1}. {agent1} exits. {agent2} moves the {object} to the {loc2}. {agent2} calls {agent1} and tells {agent1} that the {object} is now in the {loc2}.",
            "question": "Where will {agent1} search for the {object}?",
        },
        {
            "story": "{agent1} stores the {object} in the {loc1} and leaves. {agent2} relocates it to the {loc2}. {agent2} sends a message informing {agent1} about the new location.",
            "question": "Where does {agent1} think the {object} is?",
        },
    ]
    
    # Second-order belief templates
    SECOND_ORDER_TEMPLATES = [
        {
            "story": "{agent1} puts the {object} in the {loc1}. {agent1} leaves. {agent2} moves the {object} to the {loc2}. {agent1} secretly watches through a window but {agent2} doesn't know {agent1} is watching.",
            "question": "Where does {agent2} think {agent1} will look?",
        },
        {
            "story": "{agent1} hides the {object} in the {loc1}. {agent1} leaves. {agent2} finds and moves it to the {loc2}. {agent3} sees {agent2} move it but doesn't tell {agent1}.",
            "question": "Where does {agent3} think {agent1} will look?",
        },
    ]
    
    def __init__(
        self,
        use_novel_names: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize generator.
        
        Args:
            use_novel_names: Whether to use novel names (recommended)
            seed: Random seed for reproducibility
        """
        self.use_novel_names = use_novel_names
        self.name_gen = NovelNameGenerator(seed=seed)
        self.rng = random.Random(seed)
    
    def _get_names(self, n_agents: int = 2) -> NameSet:
        """Get names for scenario."""
        self.name_gen.reset()
        if self.use_novel_names:
            return self.name_gen.generate_set(n_agents=n_agents)
        else:
            agents = ["Alice", "Bob", "Carol"][:n_agents]
            return NameSet(
                agents=agents,
                locations=["drawer", "basket"],
                objects=["ball"]
            )
    
    def _fill_template(
        self,
        template: Dict,
        names: NameSet,
        swap_locations: bool = False
    ) -> tuple:
        """Fill in a template with names."""
        loc1, loc2 = names.locations[0], names.locations[1]
        if swap_locations:
            loc1, loc2 = loc2, loc1
        
        story = template["story"].format(
            agent1=names.agents[0],
            agent2=names.agents[1] if len(names.agents) > 1 else names.agents[0],
            agent3=names.agents[2] if len(names.agents) > 2 else names.agents[0],
            object=names.objects[0],
            loc1=loc1,
            loc2=loc2,
        )
        
        question = template["question"].format(
            agent1=names.agents[0],
            agent2=names.agents[1] if len(names.agents) > 1 else names.agents[0],
            agent3=names.agents[2] if len(names.agents) > 2 else names.agents[0],
            object=names.objects[0],
        )
        
        return story, question, loc1, loc2
    
    def generate_false_belief(self, n: int = 50) -> List[ToMScenario]:
        """
        Generate false belief scenarios.
        
        In FB, the agent doesn't see the move, so believes object is at original location.
        """
        scenarios = []
        
        for i in range(n):
            names = self._get_names()
            template = self.rng.choice(self.FB_TEMPLATES)
            swap = self.rng.choice([True, False])  # Counterbalance order
            
            story, question, loc1, loc2 = self._fill_template(template, names, swap)
            
            scenarios.append(ToMScenario(
                story=story,
                question=question,
                options=[loc1, loc2],
                correct=loc1,  # FB: original location
                wrong=loc2,
                scenario_type="FB",
                metadata={
                    "order": "B-A" if swap else "A-B",
                    "original_location": loc1,
                    "current_location": loc2,
                }
            ))
        
        return scenarios
    
    def generate_true_belief(self, n: int = 50) -> List[ToMScenario]:
        """
        Generate true belief scenarios.
        
        In TB, the agent SEES the move, so knows current location.
        """
        scenarios = []
        
        for i in range(n):
            names = self._get_names()
            template = self.rng.choice(self.TB_TEMPLATES)
            swap = self.rng.choice([True, False])
            
            story, question, loc1, loc2 = self._fill_template(template, names, swap)
            
            scenarios.append(ToMScenario(
                story=story,
                question=question,
                options=[loc1, loc2],
                correct=loc2,  # TB: current location
                wrong=loc1,
                scenario_type="TB",
                metadata={
                    "order": "B-A" if swap else "A-B",
                    "original_location": loc1,
                    "current_location": loc2,
                }
            ))
        
        return scenarios
    
    def generate_communication(self, n: int = 50) -> List[ToMScenario]:
        """
        Generate communication scenarios.
        
        Agent is TOLD about the move, so knows current location.
        """
        scenarios = []
        
        for i in range(n):
            names = self._get_names()
            template = self.rng.choice(self.COMM_TEMPLATES)
            swap = self.rng.choice([True, False])
            
            story, question, loc1, loc2 = self._fill_template(template, names, swap)
            
            scenarios.append(ToMScenario(
                story=story,
                question=question,
                options=[loc1, loc2],
                correct=loc2,  # Communication: knows current location
                wrong=loc1,
                scenario_type="communication",
                metadata={
                    "order": "B-A" if swap else "A-B",
                    "original_location": loc1,
                    "current_location": loc2,
                }
            ))
        
        return scenarios
    
    def generate_second_order(self, n: int = 50) -> List[ToMScenario]:
        """
        Generate second-order belief scenarios.
        
        Tests what A thinks B thinks (nested beliefs).
        """
        scenarios = []
        
        for i in range(n):
            names = self._get_names(n_agents=3)
            template = self.rng.choice(self.SECOND_ORDER_TEMPLATES)
            swap = self.rng.choice([True, False])
            
            story, question, loc1, loc2 = self._fill_template(template, names, swap)
            
            # In these scenarios, answer depends on template
            # Generally: observer thinks target will look at original location
            scenarios.append(ToMScenario(
                story=story,
                question=question,
                options=[loc1, loc2],
                correct=loc1,  # Second-order: what they think they'll do
                wrong=loc2,
                scenario_type="second_order",
                metadata={
                    "order": "B-A" if swap else "A-B",
                    "original_location": loc1,
                    "current_location": loc2,
                }
            ))
        
        return scenarios
    
    def generate_balanced_set(
        self,
        n_per_type: int = 50
    ) -> List[Dict]:
        """
        Generate a balanced set of all ToM scenario types.
        
        Args:
            n_per_type: Number of scenarios per type
            
        Returns:
            List of scenario dictionaries
        """
        all_scenarios = []
        
        # Generate each type
        all_scenarios.extend([s.to_dict() for s in self.generate_false_belief(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_true_belief(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_communication(n_per_type)])
        
        # Shuffle
        self.rng.shuffle(all_scenarios)
        
        return all_scenarios
    
    def generate_with_counterbalancing(
        self,
        n_tasks: int = 25
    ) -> List[Dict]:
        """
        Generate scenarios with full 8-scenario counterbalancing.
        
        Args:
            n_tasks: Number of unique tasks (each generates 8 scenarios)
            
        Returns:
            List of scenario dictionaries (n_tasks * 8 scenarios)
        """
        from .counterbalancing import generate_counterbalanced_set, SALLY_ANNE_TEMPLATE
        
        return generate_counterbalanced_set(
            SALLY_ANNE_TEMPLATE,
            n_tasks=n_tasks,
            use_novel_names=self.use_novel_names,
            seed=self.rng.randint(0, 10000)
        )

