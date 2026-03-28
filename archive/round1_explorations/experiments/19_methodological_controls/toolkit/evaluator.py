"""
ToM Evaluator

Evaluate model ToM performance with detailed analysis.
"""

import torch
from typing import Dict, List, Optional, Tuple


class ToMEvaluator:
    """
    Evaluate ToM performance of language models.
    
    Usage:
        evaluator = ToMEvaluator(model, tokenizer)
        result = evaluator.evaluate_false_belief(
            prompt="Alice put the ball...",
            correct="drawer",
            incorrect="basket"
        )
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize evaluator with a model and tokenizer.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def get_token_logits(self, prompt: str, tokens: List[str]) -> Dict[str, float]:
        """
        Get logits for specific tokens given a prompt.
        
        Args:
            prompt: Input prompt
            tokens: List of tokens to get logits for
        
        Returns:
            Dictionary mapping tokens to their logits
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        
        result = {}
        for token in tokens:
            # Try with and without space prefix
            for prefix in [" ", ""]:
                token_ids = self.tokenizer.encode(prefix + token, add_special_tokens=False)
                if token_ids:
                    result[token] = logits[token_ids[0]].item()
                    break
            else:
                result[token] = float('-inf')
        
        return result
    
    def evaluate_false_belief(
        self,
        prompt: str,
        correct: str,
        incorrect: str
    ) -> Dict:
        """
        Evaluate a false belief scenario.
        
        Args:
            prompt: The ToM prompt
            correct: The correct answer (where agent believes object is)
            incorrect: The incorrect answer (where object actually is)
        
        Returns:
            Dictionary with evaluation results
        """
        logits = self.get_token_logits(prompt, [correct, incorrect])
        
        correct_logit = logits[correct]
        incorrect_logit = logits[incorrect]
        diff = correct_logit - incorrect_logit
        
        # Get top 5 predictions
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        top_k = torch.topk(outputs.logits[0, -1, :], k=5)
        top_tokens = [self.tokenizer.decode([t]).strip() for t in top_k.indices.tolist()]
        
        return {
            "correct_answer": correct,
            "incorrect_answer": incorrect,
            "correct_logit": correct_logit,
            "incorrect_logit": incorrect_logit,
            "logit_difference": diff,
            "is_correct": diff > 0,
            "confidence": "high" if abs(diff) > 5 else "medium" if abs(diff) > 1 else "low",
            "top_5_predictions": top_tokens
        }
    
    def evaluate_scenario_set(
        self,
        scenarios: List[Dict]
    ) -> Dict:
        """
        Evaluate a set of ToM scenarios.
        
        Args:
            scenarios: List of scenario dicts with 'prompt', 'correct', 'incorrect'
        
        Returns:
            Summary statistics and individual results
        """
        results = []
        correct_count = 0
        
        for scenario in scenarios:
            result = self.evaluate_false_belief(
                scenario["prompt"],
                scenario["correct"],
                scenario["incorrect"]
            )
            result["scenario_name"] = scenario.get("name", "unnamed")
            results.append(result)
            
            if result["is_correct"]:
                correct_count += 1
        
        return {
            "accuracy": correct_count / len(scenarios) if scenarios else 0,
            "correct_count": correct_count,
            "total": len(scenarios),
            "results": results
        }
    
    def compare_templates(
        self,
        agent: str,
        object: str,
        original_location: str,
        new_location: str,
        mover: str,
        templates: Dict[str, str]
    ) -> Dict:
        """
        Compare performance across different prompt templates.
        
        Args:
            agent, object, original_location, new_location, mover: Scenario parameters
            templates: Dictionary of template_name -> template_string
        
        Returns:
            Comparison results for each template
        """
        results = {}
        
        for template_name, template in templates.items():
            prompt = template.format(
                agent=agent,
                object=object,
                original_location=original_location,
                new_location=new_location,
                mover=mover
            )
            
            result = self.evaluate_false_belief(
                prompt,
                correct=original_location,
                incorrect=new_location
            )
            result["template"] = template_name
            results[template_name] = result
        
        # Rank templates by performance
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]["logit_difference"],
            reverse=True
        )
        
        return {
            "results": results,
            "best_template": ranked[0][0] if ranked else None,
            "worst_template": ranked[-1][0] if ranked else None,
            "ranking": [name for name, _ in ranked]
        }
    
    def diagnose_failure(
        self,
        prompt: str,
        correct: str,
        incorrect: str
    ) -> Dict:
        """
        Diagnose why a ToM prompt might be failing.
        
        Args:
            prompt: The failing prompt
            correct: Expected correct answer
            incorrect: Incorrect answer
        
        Returns:
            Diagnosis with suggestions
        """
        result = self.evaluate_false_belief(prompt, correct, incorrect)
        
        diagnosis = {
            "result": result,
            "issues": [],
            "suggestions": []
        }
        
        # Check for belief verbs
        belief_verbs = ["thinks", "believes", "knows", "assumes"]
        for verb in belief_verbs:
            if verb in prompt.lower():
                diagnosis["issues"].append(f"Contains belief verb '{verb}'")
                diagnosis["suggestions"].append(
                    f"Replace '{verb}' with action verb like 'will look', 'expects', or 'remembers'"
                )
        
        # Check prompt length
        if len(prompt.split()) < 20:
            diagnosis["issues"].append("Prompt may be too short/minimal")
            diagnosis["suggestions"].append(
                "Add more narrative context (agent leaving, object being moved, agent returning)"
            )
        
        # Check for explicit return mention
        if "return" not in prompt.lower() and "came back" not in prompt.lower():
            diagnosis["issues"].append("No explicit mention of agent returning")
            diagnosis["suggestions"].append(
                "Add explicit return: 'Alice returns' or 'Alice came back'"
            )
        
        if not diagnosis["issues"]:
            diagnosis["issues"].append("No obvious issues detected")
            diagnosis["suggestions"].append(
                "Try using the 'action_remembers' template for best results"
            )
        
        return diagnosis


# Convenience function
def quick_evaluate(model, tokenizer, prompt: str, correct: str, incorrect: str) -> bool:
    """
    Quick evaluation of a single ToM prompt.
    
    Returns:
        True if model predicts correct answer, False otherwise
    """
    evaluator = ToMEvaluator(model, tokenizer)
    result = evaluator.evaluate_false_belief(prompt, correct, incorrect)
    return result["is_correct"]


