"""
Unified Prompt Handling

ALL prompt formatting in ONE place.
Scenarios define CONTENT, this handles FORMAT.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FormattedPrompt:
    """A formatted prompt ready for the model."""
    text: str
    system: str
    user: str
    expected_format: str  # "single_word", "json", "free"


class PromptFormatter:
    """
    Single class for ALL prompt formatting.
    
    Usage:
        formatter = PromptFormatter()
        prompt = formatter.format_tom_scenario(scenario)
        # or
        prompt = formatter.format_chat(system, user)
    """
    
    # Default system prompts
    SYSTEM_TOM = "Think step by step in <think> tags. Then give ONE WORD answer."
    SYSTEM_MULTI_AGENT = "You are playing a role. Stay in character. Think in <think> tags."
    SYSTEM_ANALYSIS = "Analyze carefully. Respond with JSON."
    
    def format_chat(
        self,
        system: str,
        user: str,
        include_assistant_start: bool = True
    ) -> str:
        """
        Format as Qwen chat template.
        
        This is THE standard format for all experiments.
        """
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user}<|im_end|>\n"
        if include_assistant_start:
            prompt += "<|im_start|>assistant\n"
        return prompt
    
    def format_tom_scenario(self, scenario: Dict) -> FormattedPrompt:
        """
        Format a ToM scenario for evaluation.
        
        Args:
            scenario: Dict with 'story', 'question', 'options'
            
        Returns:
            FormattedPrompt ready for model
        """
        story = scenario.get("story", "")
        question = scenario.get("question", "")
        options = scenario.get("options", [])
        
        opt_str = " or ".join(options) if options else ""
        
        user_content = f"{story}\n\n{question}"
        if opt_str:
            user_content += f" (Answer with one word: {opt_str})"
        
        text = self.format_chat(self.SYSTEM_TOM, user_content)
        
        return FormattedPrompt(
            text=text,
            system=self.SYSTEM_TOM,
            user=user_content,
            expected_format="single_word"
        )
    
    def format_multi_agent(
        self,
        persona: str,
        context: str,
        task: str
    ) -> FormattedPrompt:
        """
        Format a multi-agent scenario.
        
        Args:
            persona: Role description (e.g., "You are Alice, negotiating...")
            context: Conversation history or situation
            task: What the agent should do
            
        Returns:
            FormattedPrompt
        """
        system = f"{persona}\n\n{self.SYSTEM_MULTI_AGENT}"
        user = f"{context}\n\n{task}"
        
        text = self.format_chat(system, user)
        
        return FormattedPrompt(
            text=text,
            system=system,
            user=user,
            expected_format="free"
        )
    
    def format_completion(self, prompt: str) -> str:
        """
        Simple completion format (no chat template).
        
        For backward compatibility with raw completion experiments.
        """
        return prompt
    
    def create_variations(
        self,
        scenario: Dict,
        n_variations: int = 3
    ) -> List[Dict]:
        """
        Create prompt variations of a scenario.
        
        Only varies the WORDING, not the content.
        """
        story = scenario.get("story", "")
        question = scenario.get("question", "")
        
        # Question variations
        question_templates = [
            question,  # Original
            question.replace("will", "would").replace("look", "search"),
            question.replace("Where", "In which location"),
        ]
        
        variations = []
        for i, q in enumerate(question_templates[:n_variations]):
            variations.append({
                **scenario,
                "question": q,
                "variation_id": i,
            })
        
        return variations


# Singleton for convenience
_formatter = PromptFormatter()

def format_tom(scenario: Dict) -> str:
    """Quick access to format ToM scenario."""
    return _formatter.format_tom_scenario(scenario).text

def format_chat(system: str, user: str) -> str:
    """Quick access to format chat."""
    return _formatter.format_chat(system, user)

