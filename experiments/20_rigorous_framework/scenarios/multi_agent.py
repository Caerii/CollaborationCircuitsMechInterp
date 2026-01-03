"""
Multi-Agent Theory of Mind Scenarios

Tests ToM with 3+ agents, building on step 66's multi-agent experiments:
- Belief chains: Alice tells Bob, Bob tells Carol
- Information asymmetry: Different agents have different information
- Nested beliefs: What A thinks B knows that C believes
- Group knowledge: What is common knowledge vs private knowledge
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

from .novel_names import NovelNameGenerator, NameSet


@dataclass
class MultiAgentScenario:
    """A multi-agent ToM test scenario."""
    story: str
    question: str
    options: List[str]
    correct: str
    scenario_type: str
    n_agents: int
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "story": self.story,
            "question": self.question,
            "options": self.options,
            "correct": self.correct,
            "type": self.scenario_type,
            "n_agents": self.n_agents,
            "metadata": self.metadata,
        }


class MultiAgentScenarioGenerator:
    """
    Generate multi-agent Theory of Mind scenarios.
    
    Tests capabilities beyond standard 2-agent ToM:
    - Tracking beliefs of multiple agents simultaneously
    - Information propagation through communication chains
    - Nested/recursive belief attribution
    
    Example:
        gen = MultiAgentScenarioGenerator()
        
        # Belief chain scenarios
        chain_scenarios = gen.generate_belief_chain(n=50, chain_length=3)
        
        # Information asymmetry
        asymm_scenarios = gen.generate_information_asymmetry(n=50)
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
    
    def _get_names(self, n_agents: int) -> NameSet:
        """Get names for scenario."""
        self.name_gen.reset()
        if self.use_novel_names:
            return self.name_gen.generate_set(n_agents=n_agents)
        else:
            agents = ["Alice", "Bob", "Carol", "Dave", "Eve"][:n_agents]
            return NameSet(
                agents=agents,
                locations=["drawer", "basket", "box"],
                objects=["ball"]
            )
    
    def generate_belief_chain(
        self,
        n: int = 50,
        chain_length: int = 3
    ) -> List[MultiAgentScenario]:
        """
        Generate belief chain scenarios.
        
        Tests information propagation: A tells B, B tells C, etc.
        Question asks what the last agent in chain knows.
        
        Args:
            n: Number of scenarios
            chain_length: Number of agents in chain
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(chain_length)
            
            # Build the chain story
            story_parts = []
            story_parts.append(
                f"{names.agents[0]} discovers that a secret {names.objects[0]} "
                f"is hidden in the {names.locations[0]}."
            )
            
            # Each agent tells the next
            for i in range(chain_length - 1):
                story_parts.append(
                    f"{names.agents[i]} tells {names.agents[i+1]} about the location."
                )
            
            story = " ".join(story_parts)
            
            # Question about last agent's knowledge
            last_agent = names.agents[-1]
            question = f"Where does {last_agent} believe the {names.objects[0]} is?"
            
            # All agents in chain know the correct location
            correct = names.locations[0]
            wrong = names.locations[1]
            
            scenarios.append(MultiAgentScenario(
                story=story,
                question=question,
                options=[correct, wrong],
                correct=correct,
                scenario_type="belief_chain",
                n_agents=chain_length,
                metadata={
                    "chain_agents": names.agents,
                    "fact_location": correct,
                }
            ))
        
        return scenarios
    
    def generate_broken_chain(
        self,
        n: int = 50,
        chain_length: int = 3
    ) -> List[MultiAgentScenario]:
        """
        Generate broken chain scenarios.
        
        Like belief chain, but one link lies or makes a mistake.
        Tests tracking of misinformation.
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(chain_length)
            
            # Where to break the chain
            break_point = self.rng.randint(0, chain_length - 2)
            
            story_parts = []
            story_parts.append(
                f"{names.agents[0]} discovers that a {names.objects[0]} "
                f"is in the {names.locations[0]}."
            )
            
            for i in range(chain_length - 1):
                if i == break_point:
                    # This agent gives wrong information
                    story_parts.append(
                        f"{names.agents[i]} mistakenly tells {names.agents[i+1]} "
                        f"that it's in the {names.locations[1]}."
                    )
                else:
                    story_parts.append(
                        f"{names.agents[i]} tells {names.agents[i+1]} what they know."
                    )
            
            story = " ".join(story_parts)
            
            last_agent = names.agents[-1]
            question = f"Where does {last_agent} believe the {names.objects[0]} is?"
            
            # After break point, agents believe wrong location
            correct = names.locations[1]  # The wrong info propagated
            wrong = names.locations[0]
            
            scenarios.append(MultiAgentScenario(
                story=story,
                question=question,
                options=[names.locations[0], names.locations[1]],
                correct=correct,
                scenario_type="broken_chain",
                n_agents=chain_length,
                metadata={
                    "break_point": break_point,
                    "true_location": names.locations[0],
                    "believed_location": correct,
                }
            ))
        
        return scenarios
    
    def generate_information_asymmetry(
        self,
        n: int = 50
    ) -> List[MultiAgentScenario]:
        """
        Generate information asymmetry scenarios.
        
        Different agents have different pieces of information.
        Tests tracking who knows what.
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(3)
            
            # Agent 1 knows location, Agent 2 knows a different fact, Agent 3 knows both
            story = (
                f"{names.agents[0]} puts the {names.objects[0]} in the {names.locations[0]}. "
                f"{names.agents[1]} is in another room and doesn't see this. "
                f"{names.agents[2]} watches {names.agents[0]} and then goes to tell {names.agents[1]}, "
                f"but gets distracted and forgets to mention it."
            )
            
            # Ask about agent who doesn't know
            question = f"Does {names.agents[1]} know where the {names.objects[0]} is?"
            
            scenarios.append(MultiAgentScenario(
                story=story,
                question=question,
                options=["yes", "no"],
                correct="no",
                scenario_type="information_asymmetry",
                n_agents=3,
                metadata={
                    "who_knows": [names.agents[0], names.agents[2]],
                    "who_doesnt_know": [names.agents[1]],
                }
            ))
        
        return scenarios
    
    def generate_nested_belief(
        self,
        n: int = 50,
        depth: int = 2
    ) -> List[MultiAgentScenario]:
        """
        Generate nested belief scenarios.
        
        Tests what A thinks B thinks (C thinks...).
        
        Args:
            n: Number of scenarios
            depth: Nesting depth (2 = "A thinks B thinks")
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(depth + 1)
            
            if depth == 2:
                # Standard second-order: A thinks B thinks
                story = (
                    f"{names.agents[0]} puts the {names.objects[0]} in the {names.locations[0]}. "
                    f"{names.agents[0]} leaves. "
                    f"{names.agents[1]} moves the {names.objects[0]} to the {names.locations[1]}. "
                    f"{names.agents[0]} secretly watches from outside but {names.agents[1]} doesn't notice."
                )
                
                question = f"Where does {names.agents[1]} think {names.agents[0]} will look?"
                
                # B thinks A will look at original location (B doesn't know A watched)
                correct = names.locations[0]
                wrong = names.locations[1]
                
            else:
                # Higher-order (depth 3+)
                story = (
                    f"{names.agents[0]} hides the {names.objects[0]} in the {names.locations[0]}. "
                    f"{names.agents[0]} leaves. "
                    f"{names.agents[1]} moves it to the {names.locations[1]}. "
                    f"{names.agents[2]} sees {names.agents[1]} do this. "
                    f"{names.agents[0]} doesn't know about any changes."
                )
                
                question = f"Where does {names.agents[2]} think {names.agents[0]} will look?"
                correct = names.locations[0]  # C knows A doesn't know about move
                wrong = names.locations[1]
            
            scenarios.append(MultiAgentScenario(
                story=story,
                question=question,
                options=[names.locations[0], names.locations[1]],
                correct=correct,
                scenario_type=f"nested_belief_depth_{depth}",
                n_agents=depth + 1,
                metadata={
                    "depth": depth,
                    "original_location": names.locations[0],
                    "current_location": names.locations[1],
                }
            ))
        
        return scenarios
    
    def generate_common_knowledge(
        self,
        n: int = 50
    ) -> List[MultiAgentScenario]:
        """
        Generate common knowledge scenarios.
        
        Tests understanding of what everyone knows that everyone knows.
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(3)
            
            # Public announcement = common knowledge
            story = (
                f"{names.agents[0]}, {names.agents[1]}, and {names.agents[2]} are in a room together. "
                f"An announcement is made: 'The {names.objects[0]} is in the {names.locations[0]}.' "
                f"Everyone hears the announcement together."
            )
            
            question = (
                f"Does {names.agents[1]} know that {names.agents[2]} knows "
                f"where the {names.objects[0]} is?"
            )
            
            scenarios.append(MultiAgentScenario(
                story=story,
                question=question,
                options=["yes", "no"],
                correct="yes",  # Common knowledge from public announcement
                scenario_type="common_knowledge",
                n_agents=3,
                metadata={
                    "announcement_location": names.locations[0],
                    "is_common_knowledge": True,
                }
            ))
        
        return scenarios
    
    def generate_balanced_set(self, n_per_type: int = 50) -> List[Dict]:
        """Generate balanced set of all multi-agent scenario types."""
        all_scenarios = []
        
        all_scenarios.extend([s.to_dict() for s in self.generate_belief_chain(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_broken_chain(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_information_asymmetry(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_nested_belief(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_common_knowledge(n_per_type)])
        
        self.rng.shuffle(all_scenarios)
        return all_scenarios

