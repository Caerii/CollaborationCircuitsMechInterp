"""
Heuristic Baselines for ToM Evaluation

Implements the heuristic comparison required by PROPER_METHODOLOGY.md:
- First-mention: Predict the first location mentioned
- Recency: Predict the most recently mentioned location  
- Reality: Predict the actual current location

A model claiming ToM MUST outperform all of these baselines.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HeuristicResult:
    """Result of heuristic prediction."""
    prediction: str
    confidence: float
    reasoning: str


class HeuristicBaselines:
    """
    Compute heuristic baseline predictions for ToM scenarios.
    
    Key insight from step 56: The model's ~35% accuracy was WORSE than
    simple heuristics like recency (75%). Any ToM claim requires
    outperforming these baselines.
    
    Example:
        baselines = HeuristicBaselines()
        
        scenario = {
            "story": "Alice puts the ball in the drawer. Alice leaves. Bob moves it to the basket.",
            "question": "Where will Alice look?",
            "options": ["drawer", "basket"],
            "correct": "drawer"
        }
        
        predictions = baselines.predict_all(scenario)
        # {"first_mention": "drawer", "recency": "basket", "reality": "basket"}
        
        # Model must beat the BEST heuristic to claim ToM
        model_correct = model_prediction == scenario["correct"]
        best_heuristic_correct = any(
            pred == scenario["correct"] 
            for pred in predictions.values()
        )
    """
    
    # Common location words to look for
    LOCATION_PATTERNS = [
        r'\b(drawer|basket|box|container|cupboard|shelf|bag|pocket|table|desk|cabinet|closet)\b',
        r'\b(Zone-\w+|Container-\w+|Area-\w+|Unit-\w+)\b',  # Novel names
        r'\b(room \w+|room-\w+)\b',
        r'\bthe (\w+)\b(?=\s*\.)',  # "the X." pattern
    ]
    
    def __init__(self, custom_locations: Optional[List[str]] = None):
        """
        Initialize baselines.
        
        Args:
            custom_locations: Optional list of location words to look for
        """
        self.custom_locations = custom_locations or []
    
    def _find_locations(self, text: str, options: Optional[List[str]] = None) -> List[Tuple[str, int]]:
        """
        Find all locations mentioned in text with their positions.
        
        Args:
            text: Text to search
            options: Optional list of valid location options to constrain search
            
        Returns:
            List of (location, position) tuples ordered by position
        """
        locations = []
        text_lower = text.lower()
        
        # If options provided, look for those specifically
        if options:
            for opt in options:
                opt_lower = opt.lower()
                pos = 0
                while True:
                    idx = text_lower.find(opt_lower, pos)
                    if idx == -1:
                        break
                    locations.append((opt, idx))
                    pos = idx + 1
        else:
            # Use pattern matching
            for pattern in self.LOCATION_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    loc = match.group(1) if match.lastindex else match.group(0)
                    locations.append((loc, match.start()))
            
            # Add custom locations
            for loc in self.custom_locations:
                for match in re.finditer(re.escape(loc), text, re.IGNORECASE):
                    locations.append((loc, match.start()))
        
        # Sort by position
        locations.sort(key=lambda x: x[1])
        
        return locations
    
    def first_mention(
        self,
        scenario: Dict,
        text_key: str = "story"
    ) -> HeuristicResult:
        """
        Predict the first location mentioned.
        
        This heuristic exploits the tendency for the "correct" answer
        to be introduced first in many ToM scenarios.
        """
        text = scenario.get(text_key, "")
        options = scenario.get("options", [])
        
        locations = self._find_locations(text, options)
        
        if locations:
            prediction = locations[0][0]
            return HeuristicResult(
                prediction=prediction,
                confidence=0.8,
                reasoning=f"First location mentioned: '{prediction}'"
            )
        
        # Fallback to first option
        if options:
            return HeuristicResult(
                prediction=options[0],
                confidence=0.5,
                reasoning="No locations found, using first option"
            )
        
        return HeuristicResult(
            prediction="",
            confidence=0.0,
            reasoning="No prediction possible"
        )
    
    def recency(
        self,
        scenario: Dict,
        text_key: str = "story"
    ) -> HeuristicResult:
        """
        Predict the most recently mentioned location.
        
        This heuristic exploits recency bias - tendency to remember
        the most recent information.
        """
        text = scenario.get(text_key, "")
        options = scenario.get("options", [])
        
        locations = self._find_locations(text, options)
        
        if locations:
            prediction = locations[-1][0]
            return HeuristicResult(
                prediction=prediction,
                confidence=0.8,
                reasoning=f"Most recent location mentioned: '{prediction}'"
            )
        
        # Fallback to last option
        if options:
            return HeuristicResult(
                prediction=options[-1],
                confidence=0.5,
                reasoning="No locations found, using last option"
            )
        
        return HeuristicResult(
            prediction="",
            confidence=0.0,
            reasoning="No prediction possible"
        )
    
    def reality(
        self,
        scenario: Dict,
        text_key: str = "story"
    ) -> HeuristicResult:
        """
        Predict the actual current location (reality).
        
        For ToM tasks, this is typically the WRONG answer for false belief
        scenarios but RIGHT for true belief scenarios.
        
        This heuristic looks for phrases like "moves to", "is now in",
        "transfers to" to find the final location.
        """
        text = scenario.get(text_key, "")
        options = scenario.get("options", [])
        
        # Look for movement/transfer phrases
        transfer_patterns = [
            r'moves? (?:the \w+ )?(?:to|into) (?:the )?(\w+)',
            r'transfers? (?:the \w+ )?to (?:the )?(\w+)',
            r'puts? (?:it )?in(?:to)? (?:the )?(\w+)',
            r'is now in (?:the )?(\w+)',
            r'relocates? (?:the \w+ )?to (?:the )?(\w+)',
        ]
        
        final_location = None
        for pattern in transfer_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                final_location = matches[-1].group(1)
                break
        
        # If found, check if it matches an option
        if final_location and options:
            for opt in options:
                if opt.lower() == final_location.lower() or final_location.lower() in opt.lower():
                    return HeuristicResult(
                        prediction=opt,
                        confidence=0.9,
                        reasoning=f"Final transfer destination: '{opt}'"
                    )
        
        # Fallback to recency
        return self.recency(scenario, text_key)
    
    def predict_all(
        self,
        scenario: Dict,
        text_key: str = "story"
    ) -> Dict[str, str]:
        """
        Get predictions from all heuristics.
        
        Args:
            scenario: Scenario dictionary
            text_key: Key for story text
            
        Returns:
            Dict mapping heuristic name to prediction
        """
        return {
            "first_mention": self.first_mention(scenario, text_key).prediction,
            "recency": self.recency(scenario, text_key).prediction,
            "reality": self.reality(scenario, text_key).prediction,
        }
    
    def evaluate(
        self,
        scenarios: List[Dict],
        model_predictions: List[str],
        correct_key: str = "correct",
        text_key: str = "story"
    ) -> Dict:
        """
        Evaluate model against heuristic baselines.
        
        Args:
            scenarios: List of scenario dictionaries
            model_predictions: Model's predictions
            correct_key: Key for correct answer
            text_key: Key for story text
            
        Returns:
            Evaluation results with accuracies and comparison
        """
        n = len(scenarios)
        
        model_correct = 0
        first_mention_correct = 0
        recency_correct = 0
        reality_correct = 0
        
        for scenario, model_pred in zip(scenarios, model_predictions):
            correct = scenario.get(correct_key, "")
            
            if model_pred and model_pred.lower() == correct.lower():
                model_correct += 1
            
            heuristics = self.predict_all(scenario, text_key)
            
            if heuristics["first_mention"].lower() == correct.lower():
                first_mention_correct += 1
            if heuristics["recency"].lower() == correct.lower():
                recency_correct += 1
            if heuristics["reality"].lower() == correct.lower():
                reality_correct += 1
        
        model_acc = model_correct / n if n > 0 else 0
        first_mention_acc = first_mention_correct / n if n > 0 else 0
        recency_acc = recency_correct / n if n > 0 else 0
        reality_acc = reality_correct / n if n > 0 else 0
        
        best_heuristic_acc = max(first_mention_acc, recency_acc, reality_acc)
        
        return {
            "n_scenarios": n,
            "model_accuracy": model_acc,
            "first_mention_accuracy": first_mention_acc,
            "recency_accuracy": recency_acc,
            "reality_accuracy": reality_acc,
            "best_heuristic_accuracy": best_heuristic_acc,
            "model_beats_heuristics": model_acc > best_heuristic_acc,
            "margin_over_best": model_acc - best_heuristic_acc,
        }


def compute_heuristic_predictions(
    scenarios: List[Dict],
    text_key: str = "story"
) -> Dict[str, List[str]]:
    """
    Convenience function to get all heuristic predictions for a batch.
    
    Args:
        scenarios: List of scenarios
        text_key: Key for story text
        
    Returns:
        Dict mapping heuristic name to list of predictions
    """
    baselines = HeuristicBaselines()
    
    results = {
        "first_mention": [],
        "recency": [],
        "reality": [],
    }
    
    for scenario in scenarios:
        preds = baselines.predict_all(scenario, text_key)
        for key in results:
            results[key].append(preds[key])
    
    return results

