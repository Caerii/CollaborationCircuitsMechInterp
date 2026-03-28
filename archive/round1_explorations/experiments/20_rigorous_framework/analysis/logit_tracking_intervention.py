"""
Logit Tracking Intervention

The key insight: We need to track logits DURING generation to find when
the answer probability diverges, then intervene at that moment.

This is different from finding where the token appears in text - we need
to find where the DECISION is being made.
"""

import torch
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogitTrackingResult:
    """Result from logit tracking intervention."""
    base_response: str
    intervened_response: str
    flipped: bool
    intervention_positions: List[int]  # Where we intervened
    logit_trajectories: Dict[str, List[float]]  # Logit values over time
    decision_point: Optional[int]  # Where decision crystallized


class LogitTrackingIntervention:
    """
    Track logits during generation and intervene when answer probability diverges.
    
    Key difference from direct_logit_intervention:
    - We track logits DURING generation, not after
    - We find where probability diverges, not where token appears
    - We can intervene at multiple positions (distributed circuit)
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
    
    def track_and_intervene(
        self,
        prompt: str,
        answer_tokens: List[str],
        boost_token: str,
        suppress_token: Optional[str] = None,
        strength: float = 10.0,
        intervention_threshold: float = 0.1,  # Intervene when prob > threshold
        max_interventions: int = 5  # Max positions to intervene
    ) -> LogitTrackingResult:
        """
        Track logits during generation and intervene when answer probability spikes.
        
        This finds where the DECISION is being made, not where the token appears.
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Get token IDs
        answer_ids = {}
        boost_id = None
        suppress_id = None
        
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
        
        # Get baseline first
        print("    Getting baseline...", flush=True)
        sys.stdout.flush()
        with torch.no_grad():
            baseline_outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        base_response = self.tokenizer.decode(
            baseline_outputs[0][prompt_len:],
            skip_special_tokens=True
        )
        print("    Baseline complete!")
        sys.stdout.flush()
        
        # Now track logits during generation and intervene
        print("    Tracking logits and intervening during generation...", flush=True)
        sys.stdout.flush()
        
        intervened_ids = inputs.input_ids.clone()
        logit_trajectories = {tok: [] for tok in answer_ids.keys()}
        intervention_positions = []
        step_count = 0
        decision_point = None
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                if step % 50 == 0 and step > 0:
                    print(f"      Step {step}...", flush=True)
                    sys.stdout.flush()
                
                outputs = self.model(input_ids=intervened_ids)
                logits = outputs.logits
                probs = torch.softmax(logits[0, -1], dim=-1)
                
                # Track logits for answer tokens
                for tok in answer_tokens:
                    if tok in answer_ids:
                        logit_trajectories[tok].append(logits[0, -1, answer_ids[tok]].item())
                
                # Check if we should intervene
                # KEY INSIGHT: Intervene EARLY when logits first diverge, not later!
                should_intervene = False
                if boost_id is not None and suppress_id is not None:
                    boost_logit = logits[0, -1, boost_id].item()
                    suppress_logit = logits[0, -1, suppress_id].item()
                    logit_diff = boost_logit - suppress_logit
                    
                    # Strategy 1: Intervene early when logits are diverging (first 50 steps)
                    # This catches the decision as it's being formed
                    if step_count < 50 and abs(logit_diff) > 1.0:
                        should_intervene = True
                    
                    # Strategy 2: Intervene when suppress token is favored (we want to flip it)
                    # This catches when the "wrong" answer is being selected
                    elif suppress_logit > boost_logit and step_count < 200:
                        should_intervene = True
                    
                    # Strategy 3: Intervene at answer position (after reasoning)
                    boost_prob = probs[boost_id].item()
                    suppress_prob = probs[suppress_id].item()
                    if boost_prob > intervention_threshold and step_count > 100:
                        should_intervene = True
                
                # Apply intervention
                if should_intervene and len(intervention_positions) < max_interventions:
                    # Store decision point (first intervention)
                    if decision_point is None:
                        decision_point = step_count
                    
                    # Intervene
                    if boost_id is not None:
                        logits[0, -1, boost_id] += strength
                    if suppress_id is not None:
                        logits[0, -1, suppress_id] -= strength
                    
                    intervention_positions.append(step_count)
                    boost_logit = logits[0, -1, boost_id].item() if boost_id else 0
                    suppress_logit = logits[0, -1, suppress_id].item() if suppress_id else 0
                    print(f"      Intervened at step {step_count} (boost_logit={boost_logit:.2f}, suppress_logit={suppress_logit:.2f}, diff={boost_logit-suppress_logit:.2f})", flush=True)
                    sys.stdout.flush()
                
                # Get next token
                next_token = logits[0, -1].argmax(dim=-1)
                next_token = next_token.unsqueeze(0).unsqueeze(0)
                intervened_ids = torch.cat([intervened_ids, next_token], dim=-1)
                step_count += 1
                
                if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                    print(f"      EOS token generated at step {step_count}", flush=True)
                    sys.stdout.flush()
                    break
                
                # Don't stop early - let it generate fully to see the answer
                # Only stop if we've generated a lot and have the answer
                if step_count > 300:  # Generate enough to see full response
                    # Check if we have answer tokens in recent generation
                    recent_tokens = intervened_ids[0][-50:]
                    recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True).lower()
                    if any(tok.lower() in recent_text for tok in answer_tokens):
                        # We have an answer, can stop
                        break
        
        print(f"    Complete! Intervened at {len(intervention_positions)} positions.")
        sys.stdout.flush()
        
        intervened_response = self.tokenizer.decode(
            intervened_ids[0][prompt_len:],
            skip_special_tokens=True
        )
        
        # Check if flipped
        base_lower = base_response.lower()
        intervened_lower = intervened_response.lower()
        
        base_has_boost = boost_token.lower() in base_lower
        intervened_has_boost = boost_token.lower() in intervened_lower
        
        flipped = base_has_boost != intervened_has_boost
        
        return LogitTrackingResult(
            base_response=base_response,
            intervened_response=intervened_response,
            flipped=flipped,
            intervention_positions=intervention_positions,
            logit_trajectories=logit_trajectories,
            decision_point=decision_point,
        )

