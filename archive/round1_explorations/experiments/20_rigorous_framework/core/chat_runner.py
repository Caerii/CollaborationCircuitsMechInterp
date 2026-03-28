"""
Chat-Based Experiment Runner

Provides proper chat formatting for reasoning models like Qwen3-4B.
Key insight from step 62: The model shows 80-90% ToM accuracy when given:
1. Proper chat format with system/user/assistant roles
2. Instruction to use <think> tags
3. Sufficient token budget (1000 tokens)

This is in contrast to 35-50% accuracy with raw completion testing.
"""

import torch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import time

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FRAMEWORK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FRAMEWORK_ROOT))

try:
    from ..config import ExperimentConfig
    from .response_parser import ResponseParser, ParsedResponse
except ImportError:
    from config import ExperimentConfig
    from core.response_parser import ResponseParser, ParsedResponse


@dataclass
class ScenarioResult:
    """Result from running a single scenario."""
    scenario: Dict
    prompt: str
    raw_response: str
    parsed: ParsedResponse
    predicted_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    generation_time: float
    
    def to_dict(self) -> Dict:
        return {
            "scenario_type": self.scenario.get("type", "unknown"),
            "prompt": self.prompt[:200] + "..." if len(self.prompt) > 200 else self.prompt,
            "response": self.raw_response[:500] + "..." if len(self.raw_response) > 500 else self.raw_response,
            "predicted": self.predicted_answer,
            "correct": self.correct_answer,
            "is_correct": self.is_correct,
            "has_reasoning": self.parsed.has_think_tags,
            "confidence": self.parsed.confidence,
            "generation_time": self.generation_time,
        }


@dataclass 
class BatchResult:
    """Result from running a batch of scenarios."""
    results: List[ScenarioResult]
    n_total: int
    n_correct: int
    accuracy: float
    mean_confidence: float
    mean_generation_time: float
    by_type: Dict[str, Dict]
    
    def to_dict(self) -> Dict:
        return {
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "mean_confidence": self.mean_confidence,
            "mean_generation_time": self.mean_generation_time,
            "by_type": self.by_type,
            "results": [r.to_dict() for r in self.results],
        }


class ChatExperimentRunner:
    """
    Run experiments with proper chat format and token budget.
    
    This class implements the successful approach from step 62:
    - Uses chat format with <|im_start|> / <|im_end|> tags
    - Instructs model to reason in <think> tags
    - Allows 1000 tokens for full reasoning
    - Parses response to extract final answer
    
    Example:
        runner = ChatExperimentRunner(model, tokenizer, config)
        
        scenario = {
            "story": "Alice put the ball in the drawer...",
            "question": "Where will Alice look for the ball?",
            "options": ["drawer", "basket"],
            "correct": "drawer",
            "type": "false_belief"
        }
        
        result = runner.run_scenario(scenario)
        print(f"Correct: {result.is_correct}")
    """
    
    # Chat format template (Qwen style)
    CHAT_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    
    DEFAULT_SYSTEM = "Think step by step in <think> tags. Then give ONE WORD answer."
    
    def __init__(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize runner.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            config: Experiment configuration
            system_prompt: Optional custom system prompt
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.parser = ResponseParser()
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_prompt(
        self,
        scenario: Dict,
        story_key: str = "story",
        question_key: str = "question",
        options_key: str = "options"
    ) -> str:
        """
        Format a scenario into a chat prompt.
        
        Args:
            scenario: Scenario dictionary
            story_key: Key for story/context text
            question_key: Key for question text
            options_key: Key for answer options
            
        Returns:
            Formatted prompt string
        """
        story = scenario.get(story_key, scenario.get("prompt", ""))
        question = scenario.get(question_key, "")
        options = scenario.get(options_key, [])
        
        # Build user prompt
        user_prompt = story
        if question:
            user_prompt += f"\n\n{question}"
        if options:
            options_str = " or ".join(options)
            user_prompt += f" (Answer with one word: {options_str})"
        
        return self.CHAT_TEMPLATE.format(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate response from model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: config.max_tokens)
            temperature: Sampling temperature
            do_sample: Whether to sample (False = greedy)
            
        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.config.max_tokens
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def run_scenario(
        self,
        scenario: Dict,
        correct_key: str = "correct",
        options_key: str = "options",
        max_tokens: Optional[int] = None
    ) -> ScenarioResult:
        """
        Run a single scenario and evaluate the response.
        
        Args:
            scenario: Scenario dictionary with story, question, options, correct
            correct_key: Key for correct answer
            options_key: Key for answer options
            
        Returns:
            ScenarioResult with evaluation
        """
        start_time = time.time()
        
        # Format prompt
        prompt = self.format_prompt(scenario)
        
        # Generate (with optional max_tokens override)
        response = self.generate(prompt, max_tokens=max_tokens)
        generation_time = time.time() - start_time
        
        # Parse
        options = scenario.get(options_key, [])
        correct = scenario.get(correct_key, "")
        
        parsed = self.parser.parse(response)
        
        # Extract answer
        if options:
            predicted, _ = self.parser.extract_answer_token(response, options)
        else:
            predicted = parsed.answer
        
        # Evaluate
        is_correct = False
        if predicted and correct:
            is_correct = predicted.lower().strip() == correct.lower().strip()
        
        return ScenarioResult(
            scenario=scenario,
            prompt=prompt,
            raw_response=response,
            parsed=parsed,
            predicted_answer=predicted,
            correct_answer=correct,
            is_correct=is_correct,
            generation_time=generation_time
        )
    
    def run_batch(
        self,
        scenarios: List[Dict],
        verbose: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Run a batch of scenarios with evaluation.
        
        Args:
            scenarios: List of scenario dictionaries
            verbose: Print progress
            progress_callback: Optional callback(i, n, result) for progress
            
        Returns:
            BatchResult with aggregated statistics
        """
        results = []
        correct_count = 0
        total_confidence = 0.0
        total_time = 0.0
        by_type = {}
        
        for i, scenario in enumerate(scenarios):
            result = self.run_scenario(scenario)
            results.append(result)
            
            if result.is_correct:
                correct_count += 1
            total_confidence += result.parsed.confidence
            total_time += result.generation_time
            
            # Track by type
            stype = scenario.get("type", "unknown")
            if stype not in by_type:
                by_type[stype] = {"correct": 0, "total": 0}
            by_type[stype]["total"] += 1
            if result.is_correct:
                by_type[stype]["correct"] += 1
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(scenarios)}] Accuracy: {correct_count/(i+1):.1%}")
            
            if progress_callback:
                progress_callback(i, len(scenarios), result)
        
        # Calculate per-type accuracy
        for stype in by_type:
            by_type[stype]["accuracy"] = (
                by_type[stype]["correct"] / by_type[stype]["total"]
                if by_type[stype]["total"] > 0 else 0.0
            )
        
        return BatchResult(
            results=results,
            n_total=len(scenarios),
            n_correct=correct_count,
            accuracy=correct_count / len(scenarios) if scenarios else 0.0,
            mean_confidence=total_confidence / len(scenarios) if scenarios else 0.0,
            mean_generation_time=total_time / len(scenarios) if scenarios else 0.0,
            by_type=by_type
        )
    
    def run_with_intervention(
        self,
        scenarios: List[Dict],
        intervention_fn: Callable,
        cleanup_fn: Optional[Callable] = None,
        verbose: bool = True
    ) -> BatchResult:
        """
        Run scenarios with a model intervention (e.g., ablation).
        
        Args:
            scenarios: List of scenarios
            intervention_fn: Function to install intervention (e.g., hooks)
            cleanup_fn: Optional function to remove intervention
            verbose: Print progress
            
        Returns:
            BatchResult
        """
        # Install intervention
        intervention_fn()
        
        try:
            result = self.run_batch(scenarios, verbose=verbose)
        finally:
            # Cleanup intervention
            if cleanup_fn:
                cleanup_fn()
        
        return result
    
    def compare_conditions(
        self,
        scenarios: List[Dict],
        conditions: Dict[str, Callable],
        verbose: bool = True
    ) -> Dict[str, BatchResult]:
        """
        Compare multiple intervention conditions.
        
        Args:
            scenarios: List of scenarios
            conditions: Dict mapping condition name to (intervention_fn, cleanup_fn) or intervention_fn
            verbose: Print progress
            
        Returns:
            Dict mapping condition name to BatchResult
        """
        results = {}
        
        for name, intervention in conditions.items():
            if verbose:
                print(f"\n=== Condition: {name} ===")
            
            if intervention is None:
                # Baseline - no intervention
                results[name] = self.run_batch(scenarios, verbose=verbose)
            elif isinstance(intervention, tuple):
                # (intervention_fn, cleanup_fn)
                results[name] = self.run_with_intervention(
                    scenarios, intervention[0], intervention[1], verbose=verbose
                )
            else:
                # Just intervention_fn
                results[name] = self.run_with_intervention(
                    scenarios, intervention, None, verbose=verbose
                )
        
        return results


def load_model_for_chat(
    model_name: str = "Qwen/Qwen3-4B",
    device_map: str = "auto",
    dtype: str = "float16"
):
    """
    Convenience function to load model configured for chat experiments.
    
    Args:
        model_name: HuggingFace model name
        device_map: Device placement
        dtype: Model dtype
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype_map.get(dtype, torch.float16),
        trust_remote_code=True,
        attn_implementation="eager",  # Required for attention access
    )
    model.eval()
    
    return model, tokenizer

