"""
Phenomenon Discovery

Tools for discovering new patterns in model behavior and generating
testable hypotheses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class FailurePattern:
    """A discovered pattern in model failures."""
    name: str
    description: str
    n_instances: int
    example_scenarios: List[Dict]
    distinguishing_features: Dict
    hypothesized_cause: str
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_instances": self.n_instances,
            "n_examples": len(self.example_scenarios),
            "features": self.distinguishing_features,
            "hypothesized_cause": self.hypothesized_cause,
        }


@dataclass  
class Hypothesis:
    """A testable hypothesis generated from discovered patterns."""
    id: str
    statement: str
    prediction: str
    test_design: str
    falsifiable: bool
    priority: str  # "high", "medium", "low"
    source_pattern: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "prediction": self.prediction,
            "test_design": self.test_design,
            "falsifiable": self.falsifiable,
            "priority": self.priority,
            "source_pattern": self.source_pattern,
        }


class PhenomenonDiscovery:
    """
    Discover patterns in model behavior and generate hypotheses.
    
    This is the "exploration" part of the framework - finding new
    things to investigate rather than testing known phenomena.
    
    Example:
        discovery = PhenomenonDiscovery()
        
        # Analyze failures
        failures = [r for r in results if not r["is_correct"]]
        patterns = discovery.find_failure_patterns(failures)
        
        # Generate hypotheses
        hypotheses = discovery.generate_hypotheses(patterns)
        
        for h in hypotheses:
            print(f"{h.id}: {h.statement}")
    """
    
    def __init__(self):
        """Initialize discovery."""
        self.feature_extractors = {
            "scenario_type": self._extract_scenario_type,
            "text_length": self._extract_text_length,
            "n_agents": self._extract_n_agents,
            "has_negation": self._extract_has_negation,
            "question_type": self._extract_question_type,
            "location_order": self._extract_location_order,
        }
    
    def _extract_scenario_type(self, scenario: Dict) -> str:
        return scenario.get("type", scenario.get("scenario_type", "unknown"))
    
    def _extract_text_length(self, scenario: Dict) -> str:
        text = scenario.get("story", scenario.get("prompt", ""))
        length = len(text.split())
        if length < 30:
            return "short"
        elif length < 60:
            return "medium"
        else:
            return "long"
    
    def _extract_n_agents(self, scenario: Dict) -> str:
        text = scenario.get("story", scenario.get("prompt", ""))
        # Count capitalized words that might be names
        names = set(re.findall(r'\b[A-Z][a-z]+\b', text))
        # Filter out common non-names
        non_names = {"The", "In", "If", "When", "Where", "What", "How", "Then", "Now"}
        names = names - non_names
        n = len(names)
        if n <= 2:
            return "2_agents"
        elif n <= 3:
            return "3_agents"
        else:
            return "4+_agents"
    
    def _extract_has_negation(self, scenario: Dict) -> str:
        text = scenario.get("story", scenario.get("prompt", ""))
        negations = ["not", "n't", "never", "no one", "nobody", "doesn't", "didn't", "don't"]
        has_neg = any(neg in text.lower() for neg in negations)
        return "has_negation" if has_neg else "no_negation"
    
    def _extract_question_type(self, scenario: Dict) -> str:
        question = scenario.get("question", "")
        if "where" in question.lower():
            return "where_question"
        elif "what" in question.lower():
            return "what_question"
        elif "who" in question.lower():
            return "who_question"
        elif "does" in question.lower() or "do" in question.lower():
            return "yes_no_question"
        else:
            return "other_question"
    
    def _extract_location_order(self, scenario: Dict) -> str:
        return scenario.get("order", scenario.get("metadata", {}).get("order", "unknown"))
    
    def extract_features(self, scenario: Dict) -> Dict[str, str]:
        """Extract all features from a scenario."""
        return {
            name: extractor(scenario)
            for name, extractor in self.feature_extractors.items()
        }
    
    def find_failure_patterns(
        self,
        failures: List[Dict],
        min_pattern_size: int = 5
    ) -> List[FailurePattern]:
        """
        Find patterns in model failures.
        
        Args:
            failures: List of failure scenarios/results
            min_pattern_size: Minimum instances to consider a pattern
            
        Returns:
            List of FailurePattern
        """
        # Extract features for all failures
        feature_sets = [self.extract_features(f) for f in failures]
        
        # Group failures by each feature
        patterns = []
        
        for feature_name in self.feature_extractors:
            # Count by feature value
            by_value = defaultdict(list)
            for failure, features in zip(failures, feature_sets):
                by_value[features[feature_name]].append(failure)
            
            # Find over-represented values
            total = len(failures)
            for value, instances in by_value.items():
                if len(instances) >= min_pattern_size:
                    # Calculate if this is over-represented
                    proportion = len(instances) / total
                    
                    patterns.append(FailurePattern(
                        name=f"{feature_name}={value}",
                        description=f"Failures disproportionately have {feature_name}={value}",
                        n_instances=len(instances),
                        example_scenarios=instances[:3],
                        distinguishing_features={feature_name: value},
                        hypothesized_cause=self._hypothesize_cause(feature_name, value),
                    ))
        
        # Sort by number of instances
        patterns.sort(key=lambda x: x.n_instances, reverse=True)
        
        return patterns
    
    def _hypothesize_cause(self, feature: str, value: str) -> str:
        """Generate a hypothesis for why this feature causes failures."""
        hypotheses = {
            "scenario_type": {
                "communication": "Model may not track information propagation through communication",
                "second_order": "Nested belief tracking may exceed model's recursion depth",
                "FB": "False belief tracking is inherently difficult",
            },
            "text_length": {
                "long": "Attention may not span long contexts effectively",
                "short": "Insufficient context for proper inference",
            },
            "n_agents": {
                "3_agents": "Tracking 3+ agents may exceed working memory",
                "4+_agents": "Many agents may confuse entity tracking",
            },
            "has_negation": {
                "has_negation": "Negation may confuse belief attribution",
            },
            "location_order": {
                "B-A": "Recency bias may override true belief tracking",
            },
        }
        
        return hypotheses.get(feature, {}).get(value, "Unknown cause - requires investigation")
    
    def find_prompt_sensitivities(
        self,
        base_scenario: Dict,
        variations: List[Dict],
        results: List[Dict]
    ) -> Dict[str, float]:
        """
        Find what prompt elements the model is sensitive to.
        
        Args:
            base_scenario: The base scenario
            variations: List of scenario variations
            results: Results for each variation
            
        Returns:
            Dict mapping variation type to sensitivity score
        """
        sensitivities = {}
        
        # Extract base features
        base_features = self.extract_features(base_scenario)
        base_correct = results[0].get("is_correct", True)
        
        # Compare to variations
        for i, (var, result) in enumerate(zip(variations[1:], results[1:]), 1):
            var_features = self.extract_features(var)
            var_correct = result.get("is_correct", True)
            
            # Find what changed
            changed_features = {
                k: (base_features[k], var_features[k])
                for k in base_features
                if base_features[k] != var_features[k]
            }
            
            # Record sensitivity
            if base_correct != var_correct:
                for feature, (old, new) in changed_features.items():
                    key = f"{feature}: {old} -> {new}"
                    sensitivities[key] = sensitivities.get(key, 0) + 1
        
        return sensitivities
    
    def generate_hypotheses(
        self,
        patterns: List[FailurePattern]
    ) -> List[Hypothesis]:
        """
        Generate testable hypotheses from discovered patterns.
        
        Args:
            patterns: List of FailurePattern
            
        Returns:
            List of Hypothesis
        """
        hypotheses = []
        
        for i, pattern in enumerate(patterns):
            # Generate hypothesis based on pattern
            h = Hypothesis(
                id=f"H{i+1}",
                statement=f"The model fails more on scenarios with {pattern.name}",
                prediction=f"Accuracy on {pattern.name} scenarios will be significantly below average",
                test_design=self._design_test(pattern),
                falsifiable=True,
                priority=self._assign_priority(pattern),
                source_pattern=pattern.name,
            )
            hypotheses.append(h)
            
            # Generate intervention hypothesis
            h_intervention = Hypothesis(
                id=f"H{i+1}i",
                statement=f"Providing explicit instruction can improve {pattern.name} scenarios",
                prediction=f"Adding '{self._suggest_intervention(pattern)}' to prompt will increase accuracy",
                test_design="Compare accuracy with and without intervention prompt",
                falsifiable=True,
                priority="medium",
                source_pattern=pattern.name,
            )
            hypotheses.append(h_intervention)
        
        return hypotheses
    
    def _design_test(self, pattern: FailurePattern) -> str:
        """Design a test for a hypothesis."""
        feature = list(pattern.distinguishing_features.keys())[0]
        value = list(pattern.distinguishing_features.values())[0]
        
        return f"""
1. Generate n=50+ scenarios specifically with {feature}={value}
2. Generate n=50+ matched scenarios without this feature
3. Run model on both sets
4. Compare accuracy with chi-squared test
5. Calculate effect size (Cohen's h)
6. Require p<0.05 and h>0.2 for confirmation
        """.strip()
    
    def _assign_priority(self, pattern: FailurePattern) -> str:
        """Assign priority based on pattern characteristics."""
        if pattern.n_instances > 20:
            return "high"
        elif pattern.n_instances > 10:
            return "medium"
        else:
            return "low"
    
    def _suggest_intervention(self, pattern: FailurePattern) -> str:
        """Suggest a prompt intervention for the pattern."""
        feature = list(pattern.distinguishing_features.keys())[0]
        
        interventions = {
            "scenario_type": "Think carefully about what each agent knows",
            "text_length": "Focus on the key events",
            "n_agents": "Track each agent's knowledge separately",
            "has_negation": "Pay attention to negation words",
            "location_order": "Consider the agent's perspective, not just the facts",
        }
        
        return interventions.get(feature, "Think step by step")

