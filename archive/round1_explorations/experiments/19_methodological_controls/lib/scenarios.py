"""
Scenario generation for Theory of Mind testing.

Provides generators for various ToM test scenarios:
- Sally-Anne false belief
- Belief update (implicit and explicit)
- Multi-agent scenarios
- Robustness variations (templates, languages, edge cases)
- Negative controls
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ToMScenario:
    """Structured ToM test scenario."""
    prompt: str
    correct_token: str
    wrong_token: str
    scenario_type: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'prompt': self.prompt,
            'correct': self.correct_token,
            'wrong': self.wrong_token,
            'type': self.scenario_type,
            'metadata': self.metadata or {},
        }


# Default vocabulary pools
DEFAULT_AGENTS = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
DEFAULT_INFORMERS = ["Iris", "Jack", "Kate", "Leo", "Mia", "Nick"]
DEFAULT_OBJECTS = ["ball", "book", "key", "toy", "phone", "wallet"]
DEFAULT_LOCATIONS = ["basket", "box", "drawer", "shelf", "cabinet", "desk"]
DEFAULT_VERBS = ["tells", "informs", "says to", "calls", "messages"]


class ScenarioGenerator:
    """
    Generate Theory of Mind test scenarios.
    
    All generators return lists of dictionaries with:
    - prompt: The test prompt
    - correct: The correct completion token
    - wrong: The incorrect completion token  
    - Additional metadata fields
    
    Example:
        gen = ScenarioGenerator(seed=42)
        scenarios = gen.belief_update_implicit(n=50)
    """
    
    def __init__(
        self,
        seed: int = 42,
        agents: List[str] = None,
        informers: List[str] = None,
        objects: List[str] = None,
        locations: List[str] = None,
        verbs: List[str] = None,
    ):
        """
        Initialize generator with vocabulary.
        
        Args:
            seed: Random seed for reproducibility
            agents: Agent names (main character)
            informers: Names for people who inform agents
            objects: Object names
            locations: Location names
            verbs: Communication verbs
        """
        self.seed = seed
        self.agents = agents or DEFAULT_AGENTS
        self.informers = informers or DEFAULT_INFORMERS
        self.objects = objects or DEFAULT_OBJECTS
        self.locations = locations or DEFAULT_LOCATIONS
        self.verbs = verbs or DEFAULT_VERBS
        
    def _sample(self, pool: List, n: int = 1, replace: bool = False) -> List:
        """Sample from a pool."""
        if replace or n == 1:
            return [random.choice(pool) for _ in range(n)]
        return random.sample(pool, min(n, len(pool)))
    
    def belief_update_implicit(self, n: int = 50) -> List[Dict]:
        """
        Generate IMPLICIT belief update scenarios.
        
        These require the model to infer that communication updates belief.
        No explicit phrases like "so X now knows..."
        
        Args:
            n: Number of scenarios
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        scenarios = []
        
        for i in range(n):
            agent = random.choice(self.agents)
            informer = random.choice(self.informers)
            obj = random.choice(self.objects)
            loc1, loc2 = random.sample(self.locations, 2)
            verb = random.choice(self.verbs)
            
            prompt = (
                f"{agent} put the {obj} in the {loc1}. "
                f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
                f"Where will {agent} look for the {obj}? {agent} will look in the"
            )
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'type': 'implicit_update',
                'agent': agent,
                'informer': informer,
                'obj': obj,
                'loc1': loc1,
                'loc2': loc2,
            })
            
        return scenarios
    
    def belief_update_explicit(self, n: int = 50) -> List[Dict]:
        """
        Generate EXPLICIT belief update scenarios.
        
        These include bridging phrases that make belief update explicit.
        
        Args:
            n: Number of scenarios
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        scenarios = []
        
        bridges = [
            "So {agent} now knows the {obj} is in the {loc2}",
            "Therefore {agent} updated their belief",
            "{agent} understood and remembered the new location",
        ]
        
        for i in range(n):
            agent = random.choice(self.agents)
            informer = random.choice(self.informers)
            obj = random.choice(self.objects)
            loc1, loc2 = random.sample(self.locations, 2)
            verb = random.choice(self.verbs)
            bridge = random.choice(bridges).format(agent=agent, obj=obj, loc2=loc2)
            
            prompt = (
                f"{agent} put the {obj} in the {loc1}. "
                f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
                f"{bridge}. "
                f"Where will {agent} look for the {obj}? {agent} will look in the"
            )
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'type': 'explicit_update',
                'agent': agent,
                'informer': informer,
                'obj': obj,
                'loc1': loc1,
                'loc2': loc2,
                'bridge': bridge,
            })
            
        return scenarios
    
    def sally_anne_classic(self, n: int = 50) -> List[Dict]:
        """
        Generate classic Sally-Anne false belief scenarios.
        
        Sally puts object in location A, leaves.
        Anne moves it to location B.
        Sally returns - where will she look?
        
        Args:
            n: Number of scenarios
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        scenarios = []
        
        for i in range(n):
            sally, anne = random.sample(self.agents, 2)
            obj = random.choice(self.objects)
            loc1, loc2 = random.sample(self.locations, 2)
            
            prompt = (
                f"{sally} puts the {obj} in the {loc1}. "
                f"{sally} leaves the room. "
                f"{anne} moves the {obj} to the {loc2}. "
                f"{sally} returns. "
                f"Where will {sally} look for the {obj}? {sally} will look in the"
            )
            
            # Sally still believes it's in loc1 (false belief)
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc1}",  # False belief - original location
                'wrong': f" {loc2}",    # Reality - new location
                'type': 'sally_anne',
                'sally': sally,
                'anne': anne,
                'obj': obj,
                'loc1': loc1,
                'loc2': loc2,
            })
            
        return scenarios
    
    def multiagent_implicit_communication(self, n: int = 50) -> List[Dict]:
        """
        Generate multi-agent implicit communication scenarios.
        
        Agent A and Agent B interact, A tells B about a change.
        Tests if model understands B's belief updated through communication.
        
        Args:
            n: Number of scenarios
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        scenarios = []
        
        for i in range(n):
            agent_a, agent_b = random.sample(self.agents, 2)
            obj = random.choice(self.objects)
            loc1, loc2 = random.sample(self.locations, 2)
            
            # A moves object, tells B
            prompt = (
                f"Agent {agent_a} and Agent {agent_b} are working together. "
                f"The {obj} was in the {loc1}. "
                f"{agent_a} moved the {obj} to the {loc2}. "
                f"{agent_a} said to {agent_b}: 'I moved the {obj} to the {loc2}.' "
                f"When {agent_b} needs the {obj}, {agent_b} will look in the"
            )
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'type': 'multiagent_implicit_comm',
                'agent_a': agent_a,
                'agent_b': agent_b,
                'obj': obj,
                'loc1': loc1,
                'loc2': loc2,
            })
            
        return scenarios
    
    def multiagent_implicit_dialogue(self, n: int = 50) -> List[Dict]:
        """
        Generate multi-agent dialogue tracking scenarios.
        
        More complex scenario with dialogue exchange.
        
        Args:
            n: Number of scenarios
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        scenarios = []
        
        for i in range(n):
            agent_a, agent_b = random.sample(self.agents, 2)
            obj = random.choice(self.objects)
            loc1, loc2 = random.sample(self.locations, 2)
            
            prompt = (
                f"[Dialogue between {agent_a} and {agent_b}]\n"
                f"{agent_a}: I need to find the {obj}. Last I knew it was in the {loc1}.\n"
                f"{agent_b}: Actually, I moved it to the {loc2} earlier today.\n"
                f"{agent_a}: Oh, thanks for letting me know.\n"
                f"Where will {agent_a} look for the {obj}? {agent_a} will look in the"
            )
            
            scenarios.append({
                'prompt': prompt,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'type': 'multiagent_implicit_dialogue',
                'agent_a': agent_a,
                'agent_b': agent_b,
                'obj': obj,
                'loc1': loc1,
                'loc2': loc2,
            })
            
        return scenarios
    
    def template_variations(
        self,
        n_per_template: int = 20,
        templates: List[str] = None
    ) -> List[Dict]:
        """
        Generate same scenario with different linguistic templates.
        
        Tests robustness to phrasing variations.
        
        Args:
            n_per_template: Scenarios per template
            templates: Custom templates (uses defaults if None)
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        
        default_templates = {
            'original': "{a} put the {obj} in the {loc1}. {b} moved it to the {loc2} and told {a}. {a} will look in the",
            'passive': "The {obj} was placed in the {loc1} by {a}. It was moved to {loc2} by {b}, who informed {a}. {a} will search in the",
            'story': "Once, {a} had a {obj} in the {loc1}. {b} moved it to the {loc2} and mentioned this to {a}. {a} will check the",
            'formal': "{a} stored the {obj} in the {loc1}. {b} relocated it to the {loc2} and notified {a}. {a} will retrieve it from the",
            'casual': "{a} left the {obj} in the {loc1}. {b}: 'Hey, moved it to {loc2}!' {a} will look in the",
        }
        
        scenarios = []
        
        for template_name, template in default_templates.items():
            for i in range(n_per_template):
                a, b = random.sample(self.agents, 2)
                obj = random.choice(self.objects)
                loc1, loc2 = random.sample(self.locations, 2)
                
                prompt = template.format(a=a, b=b, obj=obj, loc1=loc1, loc2=loc2)
                
                scenarios.append({
                    'prompt': prompt,
                    'correct': f" {loc2}",
                    'wrong': f" {loc1}",
                    'type': f'template_{template_name}',
                    'template': template_name,
                    'agent': a,
                    'informer': b,
                    'obj': obj,
                    'loc1': loc1,
                    'loc2': loc2,
                })
                
        return scenarios
    
    def negative_controls(self, n: int = 50) -> List[Dict]:
        """
        Generate NEGATIVE control scenarios.
        
        Scenarios where ToM SHOULD give the original location
        (agent was NOT told about the change).
        
        Args:
            n: Number of scenarios
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        scenarios = []
        
        control_types = [
            ('not_told', "{a} put the {obj} in the {loc1}. {b} moved it to {loc2} but NEVER told {a}. {a} will look in the"),
            ('didnt_hear', "{a} put the {obj} in the {loc1}. {b} moved it to {loc2}. {b} tried to tell {a} but {a} didn't hear. {a} will look in the"),
            ('message_failed', "{a} put the {obj} in the {loc1}. {b} moved it to {loc2}. {b} texted {a} but the message failed. {a} will look in the"),
        ]
        
        n_per_type = max(1, n // len(control_types))
        
        for control_name, template in control_types:
            for i in range(n_per_type):
                a, b = random.sample(self.agents, 2)
                obj = random.choice(self.objects)
                loc1, loc2 = random.sample(self.locations, 2)
                
                prompt = template.format(a=a, b=b, obj=obj, loc1=loc1, loc2=loc2)
                
                # For negative controls, correct = original location
                scenarios.append({
                    'prompt': prompt,
                    'correct': f" {loc1}",  # Original - agent doesn't know!
                    'wrong': f" {loc2}",    # New - but agent wasn't told
                    'type': f'negative_{control_name}',
                    'is_negative_control': True,
                    'agent': a,
                    'informer': b,
                    'obj': obj,
                    'loc1': loc1,
                    'loc2': loc2,
                })
                
        return scenarios
    
    def communication_verb_variations(self, n_per_verb: int = 15) -> List[Dict]:
        """
        Test different communication verbs.
        
        Args:
            n_per_verb: Scenarios per verb
            
        Returns:
            List of scenario dictionaries
        """
        random.seed(self.seed)
        scenarios = []
        
        comm_patterns = [
            ("told", "{b} told {a}: '{msg}'"),
            ("texted", "{b} texted {a}: '{msg}'"),
            ("emailed", "{b} emailed {a}: '{msg}'"),
            ("messaged", "{b} messaged {a}: '{msg}'"),
            ("called", "{b} called {a} and said: '{msg}'"),
            ("wrote", "{b} wrote to {a}: '{msg}'"),
            ("informed", "{b} informed {a} that {direct}"),
            ("notified", "{b} notified {a}: '{msg}'"),
            ("mentioned", "{b} mentioned to {a}: '{msg}'"),
        ]
        
        for verb_name, pattern in comm_patterns:
            for i in range(n_per_verb):
                a, b = random.sample(self.agents, 2)
                obj = random.choice(self.objects)
                loc1, loc2 = random.sample(self.locations, 2)
                
                msg = f"I moved the {obj} to the {loc2}"
                direct = f"the {obj} was moved to the {loc2}"
                
                comm = pattern.format(a=a, b=b, msg=msg, direct=direct)
                
                prompt = (
                    f"{a} put the {obj} in the {loc1}. "
                    f"{a} went away. "
                    f"{b} moved the {obj} to the {loc2}. "
                    f"{comm}. "
                    f"Where will {a} look? {a} will look in the"
                )
                
                scenarios.append({
                    'prompt': prompt,
                    'correct': f" {loc2}",
                    'wrong': f" {loc1}",
                    'type': f'verb_{verb_name}',
                    'verb': verb_name,
                    'agent': a,
                    'informer': b,
                    'obj': obj,
                    'loc1': loc1,
                    'loc2': loc2,
                })
                
        return scenarios
    
    def paired_clean_corrupted(self, n: int = 50) -> List[Dict]:
        """
        Generate paired clean/corrupted prompts for path patching.
        
        Clean = explicit bridge (model gets it right)
        Corrupted = implicit (model often gets it wrong)
        
        Args:
            n: Number of pairs
            
        Returns:
            List of dictionaries with 'clean' and 'corrupted' prompts
        """
        random.seed(self.seed)
        pairs = []
        
        bridges = [
            "So {agent} now knows the {obj} is in the {loc2}",
            "Therefore {agent} updated their belief",
        ]
        
        for i in range(n):
            agent = random.choice(self.agents)
            informer = random.choice(self.informers)
            obj = random.choice(self.objects)
            loc1, loc2 = random.sample(self.locations, 2)
            verb = random.choice(self.verbs)
            bridge = random.choice(bridges).format(agent=agent, obj=obj, loc2=loc2)
            
            base = (
                f"{agent} put the {obj} in the {loc1}. "
                f"{informer} {verb} {agent}: 'I moved the {obj} to the {loc2}.' "
            )
            
            question = f"Where will {agent} look for the {obj}? {agent} will look in the"
            
            pairs.append({
                'clean': base + f"{bridge}. " + question,
                'corrupted': base + question,
                'correct': f" {loc2}",
                'wrong': f" {loc1}",
                'agent': agent,
                'informer': informer,
                'obj': obj,
                'loc1': loc1,
                'loc2': loc2,
            })
            
        return pairs


# Convenience functions
def generate_fixed_scenarios(
    n: int = 100,
    scenario_type: str = 'implicit',
    seed: int = 42
) -> List[Dict]:
    """
    Generate a fixed, reproducible set of scenarios.
    
    Args:
        n: Number of scenarios
        scenario_type: 'implicit', 'explicit', 'sally_anne', or 'multiagent'
        seed: Random seed
        
    Returns:
        List of scenario dictionaries
    """
    gen = ScenarioGenerator(seed=seed)
    
    if scenario_type == 'implicit':
        return gen.belief_update_implicit(n)
    elif scenario_type == 'explicit':
        return gen.belief_update_explicit(n)
    elif scenario_type == 'sally_anne':
        return gen.sally_anne_classic(n)
    elif scenario_type == 'multiagent':
        return gen.multiagent_implicit_communication(n)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


