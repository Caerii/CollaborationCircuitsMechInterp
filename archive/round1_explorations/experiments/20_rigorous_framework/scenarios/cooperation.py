"""
Cooperation and Competition Scenarios

Tests model's understanding of:
- Prisoner's Dilemma and game theory
- Tragedy of the Commons
- Negotiation and bargaining
- Cooperative vs competitive framing effects
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

from .novel_names import NovelNameGenerator, NameSet


@dataclass
class CooperationScenario:
    """A cooperation/competition test scenario."""
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


class CooperationScenarioGenerator:
    """
    Generate cooperation and competition scenarios.
    
    Based on step 64b-66 findings on game theory and negotiation.
    
    Example:
        gen = CooperationScenarioGenerator()
        
        # Prisoner's Dilemma understanding
        pd = gen.generate_prisoners_dilemma(n=50)
        
        # Tragedy of Commons
        commons = gen.generate_tragedy_of_commons(n=50)
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
            agents = ["Alice", "Bob", "Carol", "Dave"][:n_agents]
            return NameSet(
                agents=agents,
                locations=["room A", "room B"],
                objects=["resource"]
            )
    
    def generate_prisoners_dilemma(self, n: int = 50) -> List[CooperationScenario]:
        """
        Generate Prisoner's Dilemma understanding scenarios.
        
        Tests understanding of:
        - Nash equilibrium (both defect)
        - Pareto optimality (both cooperate is better for both)
        - Dominant strategy reasoning
        """
        scenarios = []
        
        pd_variants = [
            {
                "setup": "If both cooperate, each gets 3 points. If both defect, each gets 1 point. If one cooperates and one defects, the defector gets 5 and the cooperator gets 0.",
                "questions": [
                    ("What is the Nash equilibrium outcome?", "both defect"),
                    ("What outcome gives the highest total points?", "both cooperate"),
                    ("If {agent1} knows {agent2} will cooperate, what should {agent1} do to maximize their own points?", "defect"),
                ]
            },
        ]
        
        for _ in range(n):
            names = self._get_names(2)
            variant = self.rng.choice(pd_variants)
            q_template, correct = self.rng.choice(variant["questions"])
            
            story = (
                f"{names.agents[0]} and {names.agents[1]} must each choose to cooperate or defect. "
                f"{variant['setup']}"
            )
            
            question = q_template.format(
                agent1=names.agents[0],
                agent2=names.agents[1],
            )
            
            if "Nash" in question:
                options = ["both cooperate", "both defect", "one cooperates one defects"]
            elif "highest total" in question:
                options = ["both cooperate", "both defect", "one cooperates one defects"]
            else:
                options = ["cooperate", "defect"]
            
            scenarios.append(CooperationScenario(
                story=story,
                question=question,
                options=options,
                correct=correct,
                scenario_type="prisoners_dilemma",
                metadata={
                    "question_type": q_template.split()[0],
                }
            ))
        
        return scenarios
    
    def generate_tragedy_of_commons(self, n: int = 50) -> List[CooperationScenario]:
        """
        Generate Tragedy of the Commons scenarios.
        
        Tests understanding of:
        - Individual vs collective incentives
        - Resource depletion dynamics
        - Sustainable vs exploitative strategies
        """
        scenarios = []
        
        commons_variants = [
            {
                "resource": "fish",
                "sustainable_limit": 50,
                "setup": "There is a shared lake with fish. Each fisher can take up to 100 fish. If total catch exceeds 150, the population collapses and there will be no fish next year.",
            },
            {
                "resource": "water",
                "sustainable_limit": 30,
                "setup": "A village shares a well. Each family can draw up to 50 liters. If total usage exceeds 100 liters per day, the well will dry up.",
            },
        ]
        
        question_templates = [
            ("If everyone acts in pure self-interest, what happens to the {resource}?", "depleted"),
            ("What is the sustainable amount each person can take?", "limited"),
            ("To preserve the {resource} long-term, what strategy is needed?", "cooperation"),
        ]
        
        for _ in range(n):
            names = self._get_names(3)
            variant = self.rng.choice(commons_variants)
            q_template, answer_type = self.rng.choice(question_templates)
            
            story = (
                f"{names.agents[0]}, {names.agents[1]}, and {names.agents[2]} share a common resource. "
                f"{variant['setup']}"
            )
            
            question = q_template.format(resource=variant["resource"])
            
            if answer_type == "depleted":
                options = ["preserved", "depleted", "grows"]
                correct = "depleted"
            elif answer_type == "limited":
                options = ["unlimited amount", "limited amount", "zero"]
                correct = "limited amount"
            else:
                options = ["competition", "cooperation", "isolation"]
                correct = "cooperation"
            
            scenarios.append(CooperationScenario(
                story=story,
                question=question,
                options=options,
                correct=correct,
                scenario_type="tragedy_of_commons",
                metadata={
                    "resource": variant["resource"],
                    "sustainable_limit": variant["sustainable_limit"],
                }
            ))
        
        return scenarios
    
    def generate_negotiation(self, n: int = 50) -> List[CooperationScenario]:
        """
        Generate negotiation understanding scenarios.
        
        Tests understanding of:
        - Fair division
        - BATNA (Best Alternative to Negotiated Agreement)
        - Win-win vs zero-sum thinking
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(2)
            
            # Different negotiation contexts
            context_type = self.rng.choice(["split", "batna", "integrative"])
            
            if context_type == "split":
                total = self.rng.randint(80, 120)
                story = (
                    f"{names.agents[0]} and {names.agents[1]} must divide ${total} between them. "
                    f"If they can't agree, neither gets anything."
                )
                question = "What is the fair split?"
                correct = "equal"
                options = ["equal", f"{names.agents[0]} gets more", f"{names.agents[1]} gets more"]
                
            elif context_type == "batna":
                story = (
                    f"{names.agents[0]} is selling a car and has another buyer offering $8000. "
                    f"{names.agents[1]} wants to buy it and offers $7500."
                )
                question = f"Should {names.agents[0]} accept {names.agents[1]}'s offer?"
                correct = "no"
                options = ["yes", "no"]
                
            else:  # integrative
                story = (
                    f"{names.agents[0]} wants the car for its engine. "
                    f"{names.agents[1]} wants the same car for its body parts. "
                    f"The car cannot be split."
                )
                question = "Can both parties get what they want?"
                correct = "yes"
                options = ["yes", "no"]
            
            scenarios.append(CooperationScenario(
                story=story,
                question=question,
                options=options,
                correct=correct,
                scenario_type="negotiation",
                metadata={
                    "context_type": context_type,
                }
            ))
        
        return scenarios
    
    def generate_framing_effects(self, n: int = 50) -> List[CooperationScenario]:
        """
        Generate framing effect scenarios.
        
        Tests whether model understands that the same game can be
        perceived differently based on competitive vs cooperative framing.
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(2)
            
            is_cooperative_framing = self.rng.choice([True, False])
            
            if is_cooperative_framing:
                framing = "community game where partners work together"
                expected_behavior = "cooperate"
            else:
                framing = "Wall Street game where opponents compete"
                expected_behavior = "defect"
            
            story = (
                f"In a study, the Prisoner's Dilemma was presented as a '{framing}'. "
                f"{names.agents[0]} and {names.agents[1]} play this version."
            )
            
            question = "How does this framing affect expected behavior?"
            options = ["more cooperation", "more defection", "no effect"]
            
            correct = "more cooperation" if is_cooperative_framing else "more defection"
            
            scenarios.append(CooperationScenario(
                story=story,
                question=question,
                options=options,
                correct=correct,
                scenario_type="framing_effect",
                metadata={
                    "framing": "cooperative" if is_cooperative_framing else "competitive",
                }
            ))
        
        return scenarios
    
    def generate_tit_for_tat(self, n: int = 50) -> List[CooperationScenario]:
        """
        Generate tit-for-tat strategy understanding scenarios.
        
        Tests understanding of reciprocal strategies in repeated games.
        """
        scenarios = []
        
        for _ in range(n):
            names = self._get_names(2)
            
            # History of moves
            n_rounds = self.rng.randint(3, 5)
            agent1_moves = [self.rng.choice(["C", "D"]) for _ in range(n_rounds)]
            
            # Tit-for-tat responds to opponent's previous move
            # So if playing against agent1, TFT would do what agent1 did previously
            tft_next_move = agent1_moves[-1]  # Copy last move
            
            history = ", ".join([f"Round {i+1}: {m}" for i, m in enumerate(agent1_moves)])
            
            story = (
                f"{names.agents[1]} plays Tit-for-Tat strategy (start cooperating, then copy opponent's last move). "
                f"{names.agents[0]}'s history: {history}."
            )
            
            question = f"What will {names.agents[1]} do in the next round?"
            options = ["cooperate", "defect"]
            correct = "cooperate" if tft_next_move == "C" else "defect"
            
            scenarios.append(CooperationScenario(
                story=story,
                question=question,
                options=options,
                correct=correct,
                scenario_type="tit_for_tat",
                metadata={
                    "history": agent1_moves,
                    "strategy": "TFT",
                }
            ))
        
        return scenarios
    
    def generate_balanced_set(self, n_per_type: int = 50) -> List[Dict]:
        """Generate balanced set of cooperation scenarios."""
        all_scenarios = []
        
        all_scenarios.extend([s.to_dict() for s in self.generate_prisoners_dilemma(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_tragedy_of_commons(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_negotiation(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_framing_effects(n_per_type)])
        all_scenarios.extend([s.to_dict() for s in self.generate_tit_for_tat(n_per_type)])
        
        self.rng.shuffle(all_scenarios)
        return all_scenarios

