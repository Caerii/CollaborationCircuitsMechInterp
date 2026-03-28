"""
Deception and Trust Scenarios

Tests model's ability to:
- Detect lies and deception
- Track credibility of information sources
- Calibrate trust based on past behavior
- Understand intent behind communication
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import random

from .novel_names import NovelNameGenerator, NameSet


@dataclass
class DeceptionScenario:
    """A deception/trust test scenario."""
    story: str
    question: str
    options: List[str]
    correct: str
    scenario_type: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "story": self.story,
            "question": self.question,
            "options": self.options,
            "correct": self.correct,
            "type": self.scenario_type,
            "metadata": self.metadata,
        }


class DeceptionScenarioGenerator:
    """
    Generate deception detection and trust calibration scenarios.
    
    Based on step 63-66 findings on collaboration capabilities.
    
    Example:
        gen = DeceptionScenarioGenerator()
        
        # Lie detection
        lies = gen.generate_lie_detection(n=50)
        
        # Trust calibration
        trust = gen.generate_trust_calibration(n=50)
    """
    
    def __init__(
        self,
        use_novel_names: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize generator.
        
        Args:
            use_novel_names: Whether to use novel names
            seed: Random seed
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
                objects=["key"]
            )
    
    def generate_lie_detection(self, n: int = 50) -> List[DeceptionScenario]:
        """
        Generate lie detection scenarios.
        
        Agent knows the truth but tells a lie. Model must identify
        what the truth actually is vs what was claimed.
        """
        scenarios = []
        
        lie_templates = [
            {
                "story": "{agent1} hides the {object} in the {loc1}. {agent2} asks where it is. {agent1} wants to keep it secret and says it's in the {loc2}.",
                "question": "Where is the {object} actually located?",
                "truth_is_loc1": True,
            },
            {
                "story": "{agent1} sees the {object} in the {loc1}. Later, {agent1} tells {agent2} that they saw it in the {loc2} because {agent1} wants to mislead {agent2}.",
                "question": "Where did {agent1} actually see the {object}?",
                "truth_is_loc1": True,
            },
            {
                "story": "The {object} is in the {loc1}. {agent1} knows this but tells {agent2} it's in the {loc2} to trick them.",
                "question": "What is the true location of the {object}?",
                "truth_is_loc1": True,
            },
        ]
        
        for _ in range(n):
            names = self._get_names(2)
            template = self.rng.choice(lie_templates)
            swap = self.rng.choice([True, False])
            
            loc1, loc2 = names.locations[0], names.locations[1]
            if swap:
                loc1, loc2 = loc2, loc1
            
            story = template["story"].format(
                agent1=names.agents[0],
                agent2=names.agents[1],
                object=names.objects[0],
                loc1=loc1,
                loc2=loc2,
            )
            
            question = template["question"].format(
                agent1=names.agents[0],
                object=names.objects[0],
            )
            
            correct = loc1 if template["truth_is_loc1"] else loc2
            wrong = loc2 if template["truth_is_loc1"] else loc1
            
            scenarios.append(DeceptionScenario(
                story=story,
                question=question,
                options=[loc1, loc2],
                correct=correct,
                scenario_type="lie_detection",
                metadata={
                    "truth_location": correct,
                    "claimed_location": wrong,
                    "liar": names.agents[0],
                }
            ))
        
        return scenarios
    
    def generate_trust_calibration(self, n: int = 50) -> List[DeceptionScenario]:
        """
        Generate trust calibration scenarios.
        
        Agent has history of truthfulness or deception.
        Model must adjust trust based on track record.
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(2)
            
            # Determine if agent is trustworthy
            is_trustworthy = self.rng.choice([True, False])
            
            if is_trustworthy:
                history = f"{names.agents[0]} has always told the truth in the past."
                current_claim_true = True  # Trustworthy agent tells truth
            else:
                history = f"{names.agents[0]} has lied many times before."
                current_claim_true = self.rng.choice([True, False])  # Untrustworthy might or might not lie
            
            claimed_loc = names.locations[0]
            actual_loc = names.locations[0] if current_claim_true else names.locations[1]
            
            story = (
                f"{history} "
                f"Now {names.agents[0]} tells {names.agents[1]} that the {names.objects[0]} "
                f"is in the {claimed_loc}. "
                f"In reality, the {names.objects[0]} is in the {actual_loc}."
            )
            
            question = f"Should {names.agents[1]} trust {names.agents[0]}'s claim?"
            
            # Trust is warranted if agent is trustworthy OR if claim happens to be true
            should_trust = is_trustworthy
            
            scenarios.append(DeceptionScenario(
                story=story,
                question=question,
                options=["yes", "no"],
                correct="yes" if should_trust else "no",
                scenario_type="trust_calibration",
                metadata={
                    "agent_trustworthy": is_trustworthy,
                    "claim_true": current_claim_true,
                }
            ))
        
        return scenarios
    
    def generate_motive_detection(self, n: int = 50) -> List[DeceptionScenario]:
        """
        Generate motive detection scenarios.
        
        Agent has obvious motive to lie. Model must identify
        when claims should be doubted due to conflict of interest.
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(2)
            
            has_motive = self.rng.choice([True, False])
            
            if has_motive:
                motive_desc = (
                    f"{names.agents[0]} wants the {names.objects[0]} for themselves. "
                )
                question = f"Given {names.agents[0]}'s interest, should their claim be trusted?"
                expected = "no"
            else:
                motive_desc = (
                    f"{names.agents[0]} has no interest in the {names.objects[0]}. "
                )
                question = f"Is there reason to doubt {names.agents[0]}'s claim?"
                expected = "no"
            
            story = (
                f"{motive_desc}"
                f"{names.agents[0]} tells {names.agents[1]} that the {names.objects[0]} "
                f"is in the {names.locations[0]}."
            )
            
            scenarios.append(DeceptionScenario(
                story=story,
                question=question,
                options=["yes", "no"],
                correct=expected,
                scenario_type="motive_detection",
                metadata={
                    "has_motive_to_lie": has_motive,
                }
            ))
        
        return scenarios
    
    def generate_source_verification(self, n: int = 50) -> List[DeceptionScenario]:
        """
        Generate source verification scenarios.
        
        Information comes from different sources with different reliability.
        Model must track and evaluate source quality.
        """
        scenarios = []
        
        source_types = [
            ("firsthand", "saw it directly", 0.9),
            ("secondhand", "was told by someone who saw it", 0.7),
            ("rumor", "heard a rumor about it", 0.3),
            ("guess", "is just guessing", 0.1),
        ]
        
        for _ in range(n):
            names = self._get_names(2)
            source_type, source_desc, reliability = self.rng.choice(source_types)
            
            story = (
                f"{names.agents[0]} {source_desc} and claims the {names.objects[0]} "
                f"is in the {names.locations[0]}."
            )
            
            question = f"How reliable is {names.agents[0]}'s information?"
            options = ["very reliable", "somewhat reliable", "unreliable"]
            
            if reliability >= 0.8:
                correct = "very reliable"
            elif reliability >= 0.5:
                correct = "somewhat reliable"
            else:
                correct = "unreliable"
            
            scenarios.append(DeceptionScenario(
                story=story,
                question=question,
                options=options,
                correct=correct,
                scenario_type="source_verification",
                metadata={
                    "source_type": source_type,
                    "reliability": reliability,
                }
            ))
        
        return scenarios
    
    def generate_balanced_set(self, n_per_type: int = 50) -> List[Dict]:
        """Generate balanced set of deception scenarios."""
        all_scenarios = []
        
        all_scenarios.extend([s.to_dict() for s in self.generate_lie_detection(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_trust_calibration(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_motive_detection(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_source_verification(n_per_type)])
        
        self.rng.shuffle(all_scenarios)
        return all_scenarios

