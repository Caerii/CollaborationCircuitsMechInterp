"""
Minimal Pair Analysis

Systematically vary ONE element at a time to isolate
what causes behavioral differences.

This is crucial for rigorous causal claims - you can't claim
"verb type matters" unless you've held everything else constant.
"""

import torch
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class MinimalPairResult:
    """Result of a minimal pair test."""
    name: str
    prompt: str
    prediction: str
    target_prob: float
    contrast_prob: float
    correct: bool
    hypothesis: str


class MinimalPairTester:
    """
    Test minimal pairs to isolate causal factors.
    
    The key principle: vary ONE thing at a time while holding
    everything else constant. This lets you make causal claims.
    
    Example:
        tester = MinimalPairTester(model, tokenizer)
        
        # Define base story
        base = "Alice put the ball in drawer. Alice left. Bob moved ball to basket."
        
        # Define completions varying ONE factor
        pairs = {
            "past_searched": {
                "completion": "Alice returned. Alice searched in the",
                "hypothesis": "Past tense + action verb"
            },
            "present_thinks": {
                "completion": "Alice returns. Alice thinks the ball is in the",
                "hypothesis": "Present tense + belief verb"
            },
            "past_thinks": {
                "completion": "Alice returned. Alice thinks the ball is in the",
                "hypothesis": "Past tense + belief verb (isolate verb)"
            },
            "present_searched": {
                "completion": "Alice returns. Alice searched in the",
                "hypothesis": "Present tense + action verb (isolate tense)"
            }
        }
        
        results = tester.test_pairs(base, pairs, " drawer", " basket")
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_next_token_probs(
        self,
        prompt: str,
        target_token: str,
        contrast_token: str,
    ) -> Dict:
        """Get probabilities for target vs contrast token."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        contrast_id = self.tokenizer.encode(contrast_token, add_special_tokens=False)[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
        
        target_prob = float(probs[target_id])
        contrast_prob = float(probs[contrast_id])
        prediction = target_token if target_prob > contrast_prob else contrast_token
        
        return {
            "prediction": prediction,
            "target_prob": target_prob,
            "contrast_prob": contrast_prob,
            "target_wins": target_prob > contrast_prob,
        }
    
    def test_pairs(
        self,
        base_story: str,
        completions: Dict[str, Dict],
        target_token: str,
        contrast_token: str,
    ) -> List[MinimalPairResult]:
        """
        Test minimal pairs.
        
        Args:
            base_story: The constant story part
            completions: Dict mapping name to {"completion": str, "hypothesis": str}
            target_token: Expected correct token
            contrast_token: Incorrect alternative
            
        Returns:
            List of MinimalPairResult
        """
        results = []
        
        for name, info in completions.items():
            prompt = base_story + " " + info["completion"]
            
            probs = self.get_next_token_probs(prompt, target_token, contrast_token)
            
            results.append(MinimalPairResult(
                name=name,
                prompt=prompt,
                prediction=probs["prediction"],
                target_prob=probs["target_prob"],
                contrast_prob=probs["contrast_prob"],
                correct=probs["target_wins"],
                hypothesis=info["hypothesis"],
            ))
        
        return results
    
    def isolate_factor(
        self,
        results: List[MinimalPairResult],
    ) -> Dict[str, Dict]:
        """
        Analyze which factors matter.
        
        Groups results by hypothesis to see which factors
        consistently affect behavior.
        
        Args:
            results: List of MinimalPairResult
            
        Returns:
            Dict with analysis
        """
        # Group by correctness
        correct = [r for r in results if r.correct]
        incorrect = [r for r in results if not r.correct]
        
        analysis = {
            "total_correct": len(correct),
            "total_incorrect": len(incorrect),
            "accuracy": len(correct) / len(results) if results else 0,
            "correct_hypotheses": [r.hypothesis for r in correct],
            "incorrect_hypotheses": [r.hypothesis for r in incorrect],
        }
        
        return analysis


# Pre-built minimal pair templates for ToM
TOM_MINIMAL_PAIRS = {
    # Baseline comparison
    "working_baseline": {
        "completion": "When Alice returned, Alice looked for the ball. Alice searched in the",
        "hypothesis": "Full working format"
    },
    "failing_baseline": {
        "completion": "Alice returns. Alice thinks the ball is in the",
        "hypothesis": "Minimal failing format"
    },
    
    # Isolate TENSE
    "past_thinks": {
        "completion": "Alice returned. Alice thinks the ball is in the",
        "hypothesis": "Past tense + thinks"
    },
    "present_searched": {
        "completion": "Alice returns. Alice searched in the",
        "hypothesis": "Present tense + searched"
    },
    
    # Isolate VERB TYPE
    "returned_searched": {
        "completion": "Alice returned. Alice searched in the",
        "hypothesis": "Returned + searched"
    },
    "returned_believes": {
        "completion": "Alice returned. Alice believes the ball is in the",
        "hypothesis": "Returned + believes"
    },
    "returns_looks": {
        "completion": "Alice returns. Alice looks in the",
        "hypothesis": "Returns + looks"
    },
    
    # Isolate EXPLICITNESS
    "explicit_question": {
        "completion": "Where will Alice look for the ball? Alice will look in the",
        "hypothesis": "Explicit question"
    },
    "implicit_completion": {
        "completion": "Alice searched in the",
        "hypothesis": "Implicit completion"
    },
}


def run_standard_tom_minimal_pairs(model, tokenizer, base_story: str) -> Dict:
    """
    Run standard ToM minimal pair analysis.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        base_story: The ToM scenario (e.g., Sally-Anne setup)
        
    Returns:
        Analysis results
    """
    tester = MinimalPairTester(model, tokenizer)
    results = tester.test_pairs(
        base_story,
        TOM_MINIMAL_PAIRS,
        target_token=" drawer",  # Original location
        contrast_token=" basket",  # New location
    )
    
    analysis = tester.isolate_factor(results)
    analysis["detailed_results"] = [
        {
            "name": r.name,
            "correct": r.correct,
            "hypothesis": r.hypothesis,
            "target_prob": r.target_prob,
            "contrast_prob": r.contrast_prob,
        }
        for r in results
    ]
    
    return analysis

