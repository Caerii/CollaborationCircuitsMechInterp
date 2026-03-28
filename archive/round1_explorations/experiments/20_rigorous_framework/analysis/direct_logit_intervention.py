"""
Direct Logit Intervention

Simpler alternative to activation patching: directly manipulate logits
at the answer position to see if we can flip the answer.

This bypasses the sequence length/position issues of activation patching.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogitInterventionResult:
    """Result from logit intervention experiment."""
    base_response: str
    intervened_response: str
    flipped: bool
    base_logits: Dict[str, float]
    intervened_logits: Dict[str, float]
    intervention_strength: float


class DirectLogitIntervention:
    """
    Directly manipulate logits at answer position.
    
    This is simpler than activation patching and avoids the sequence
    length issues. We just add/subtract logits for target tokens.
    
    Example:
        intervener = DirectLogitIntervention(model, tokenizer, chat_mode=True)
        
        result = intervener.intervene(
            prompt="Sally put the ball in the basket. Sally left...",
            answer_tokens=["basket", "box"],
            boost_token="box",
            strength=5.0
        )
    """
    
    CHAT_TEMPLATE = """<|im_start|>system
Think step by step in <think> tags. Then give ONE WORD answer.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 500,
        chat_mode: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.chat_mode = chat_mode
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for chat mode if enabled."""
        if not self.chat_mode:
            return prompt
        return self.CHAT_TEMPLATE.format(user_prompt=prompt)
    
    def find_answer_position(
        self,
        prompt: str,
        answer_tokens: List[str],
        max_tokens: int = 500
    ) -> Optional[int]:
        """
        Find token position where answer is actually generated.
        
        Returns:
            Token position (absolute) where answer token is generated, or None
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        import sys
        print(f"    Finding answer position (generating up to {max_tokens} tokens)...", end="", flush=True)
        sys.stdout.flush()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        print(" done!")
        sys.stdout.flush()
        
        full_ids = outputs[0]
        tokens = [self.tokenizer.decode([t]) for t in full_ids]
        
        # Find where answer tokens appear (after reasoning)
        think_end = None
        for i, tok in enumerate(tokens):
            if "</think>" in tok.lower():
                think_end = i
                break
        
        # Look for answer tokens after reasoning
        if think_end:
            for i in range(think_end + 1, len(tokens)):
                tok_text = tokens[i].lower()
                for ans_tok in answer_tokens:
                    if ans_tok.lower() in tok_text:
                        # Found answer token - return the position where it was generated
                        # (i.e., the position before it)
                        return prompt_len + (i - prompt_len) - 1
        
        # Fallback: return position after reasoning + a few tokens
        if think_end:
            return think_end + 5
        return prompt_len + 50
    
    def intervene(
        self,
        prompt: str,
        answer_tokens: List[str],
        boost_token: str,
        suppress_token: Optional[str] = None,
        strength: float = 5.0,
        position: Optional[int] = None
    ) -> LogitInterventionResult:
        """
        Intervene on logits at answer position.
        
        Args:
            prompt: Input prompt
            answer_tokens: All possible answer tokens
            boost_token: Token to boost (add strength to logit)
            suppress_token: Token to suppress (subtract strength from logit)
            strength: How much to add/subtract
            position: Specific position to intervene (None = auto-detect)
            
        Returns:
            LogitInterventionResult
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Find answer position if not provided
        if position is None:
            position = self.find_answer_position(prompt, answer_tokens)
            if position is None:
                position = prompt_len
        
        # Get token IDs
        boost_id = None
        suppress_id = None
        answer_ids = {}
        
        for tok in answer_tokens:
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                answer_ids[tok] = ids[0]
                if tok == boost_token:
                    boost_id = ids[0]
                if suppress_token and tok == suppress_token:
                    suppress_id = ids[0]
        
        if boost_id is None:
            raise ValueError(f"Could not encode boost_token: {boost_token}")
        
        # Get baseline
        import sys
        print("    Generating baseline...", end="", flush=True)
        sys.stdout.flush()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        print(" done!")
        sys.stdout.flush()
        
        base_response = self.tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True
        )
        
        # Get baseline logits
        base_logits = {}
        with torch.no_grad():
            # Run forward pass to get logits at answer position
            # This is approximate - we'll get logits during generation instead
            pass
        
        # Now intervene: custom generation with logit manipulation
        intervened_ids = inputs.input_ids.clone()
        intervened_logits = {}
        step_count = 0
        
        import sys
        print(f"    Generating with intervention (up to {self.max_new_tokens} tokens)...", end="", flush=True)
        sys.stdout.flush()
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                if step % 50 == 0 and step > 0:
                    print(f" {step}", end="", flush=True)
                    sys.stdout.flush()
                outputs = self.model(input_ids=intervened_ids)
                logits = outputs.logits
                
                # Apply intervention if at answer position
                if step_count == position - prompt_len:
                    # Store logits before intervention
                    for tok, tok_id in answer_ids.items():
                        intervened_logits[tok] = logits[0, -1, tok_id].item()
                    
                    # Intervene: boost target, suppress contrast
                    if boost_id is not None:
                        logits[0, -1, boost_id] += strength
                    if suppress_id is not None:
                        logits[0, -1, suppress_id] -= strength
                
                # Get next token
                next_token = logits[0, -1].argmax(dim=-1)
                next_token = next_token.unsqueeze(0).unsqueeze(0)  # (1, 1)
                intervened_ids = torch.cat([intervened_ids, next_token], dim=-1)
                step_count += 1
                
                if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                    break
                
                # Stop after answer position + some tokens
                if step_count > position - prompt_len + 20:
                    print(f"      Stopping after intervention (step {step_count})")
                    sys.stdout.flush()
                    break
        print(f"    Generation complete! Generated {step_count} tokens.")
        sys.stdout.flush()
        
        print(" done!")
        sys.stdout.flush()
        
        intervened_response = self.tokenizer.decode(
            intervened_ids[0][prompt_len:],
            skip_special_tokens=True
        )
        
        # Check if flipped
        base_lower = base_response.lower()
        intervened_lower = intervened_response.lower()
        
        # Simple flip check: did the answer change?
        base_has_boost = boost_token.lower() in base_lower
        intervened_has_boost = boost_token.lower() in intervened_lower
        
        flipped = base_has_boost != intervened_has_boost
        
        return LogitInterventionResult(
            base_response=base_response,
            intervened_response=intervened_response,
            flipped=flipped,
            base_logits={},  # Would need to capture during baseline
            intervened_logits=intervened_logits,
            intervention_strength=strength,
        )

