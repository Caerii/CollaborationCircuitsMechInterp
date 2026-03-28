"""
Theory of Mind (ToM) Prompt Engineering Toolkit

This toolkit provides utilities for eliciting reliable ToM reasoning from LLMs.

Key findings implemented:
1. Action verbs work better than belief verbs
2. Well-structured prompts work better than minimal formats
3. Explicit narrative context improves accuracy

Usage:
    from toolkit import ToMPromptBuilder, ToMEvaluator
    
    builder = ToMPromptBuilder()
    prompt = builder.create_false_belief_prompt(
        agent="Alice",
        object="ball",
        original_location="drawer",
        new_location="basket",
        mover="Bob"
    )
"""

from .prompt_builder import ToMPromptBuilder
from .evaluator import ToMEvaluator
from .templates import TEMPLATES

__all__ = ['ToMPromptBuilder', 'ToMEvaluator', 'TEMPLATES']


