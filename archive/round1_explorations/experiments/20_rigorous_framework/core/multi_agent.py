"""
Multi-Agent Interaction Framework

For studying collaboration circuits through multi-turn interactions.
Adapted from exp 66's Agent class.

Key features:
- Agent personas that maintain conversation history
- Activation extraction during conversations
- Multi-turn negotiation, cooperation, and deception scenarios
- Hooks for probing internal representations during interaction
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConversationTurn:
    """A single turn in a multi-agent conversation."""
    turn_number: int
    speaker: str
    message: str
    activations: Optional[Dict[int, np.ndarray]] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConversationResult:
    """Result of a multi-agent conversation."""
    turns: List[ConversationTurn]
    outcome: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "n_turns": len(self.turns),
            "outcome": self.outcome,
            "turns": [
                {"turn": t.turn_number, "speaker": t.speaker, "message": t.message[:100]}
                for t in self.turns
            ],
            "metadata": self.metadata,
        }


class Agent:
    """
    An LLM agent with persona and conversation history.
    
    Supports activation extraction during generation for MI analysis.
    
    Example:
        agent = Agent(
            name="Alice",
            persona="You need at least 60% of the budget. Be firm but fair.",
            model=model,
            tokenizer=tokenizer
        )
        
        # Simple response
        response = agent.respond("What do you propose?", context="Budget negotiation")
        
        # Response with activation capture
        response, activations = agent.respond_with_activations(
            "What do you propose?",
            layers=[16, 20, 24]
        )
    """
    
    def __init__(
        self,
        name: str,
        persona: str,
        model,
        tokenizer,
        max_tokens: int = 150
    ):
        """
        Initialize agent.
        
        Args:
            name: Agent's name
            persona: Agent's persona/instructions
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            max_tokens: Max tokens to generate per response
        """
        self.name = name
        self.persona = persona
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
    
    def _build_prompt(self, message: str, context: str = "") -> str:
        """Build prompt with persona, history, and current message."""
        history_str = ""
        if self.history:
            # Include last 4 turns for context
            recent = self.history[-4:]
            history_str = "\nPrevious conversation:\n" + "\n".join([
                f"{h['role']}: {h['content'][:100]}" for h in recent
            ]) + "\n"
        
        prompt = f"""<|im_start|>system
You are {self.name}. {self.persona}
Keep responses concise (2-3 sentences). Stay in character.<|im_end|>
<|im_start|>user
{context}
{history_str}
Current message: {message}

Respond as {self.name}:<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def respond(
        self,
        message: str,
        context: str = ""
    ) -> str:
        """
        Generate a response.
        
        Args:
            message: Input message to respond to
            context: Optional context about the interaction
            
        Returns:
            Agent's response
        """
        prompt = self._build_prompt(message, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Extract after </think> if present
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        
        # Update history
        self.history.append({"role": self.name, "content": response})
        
        return response
    
    def respond_with_activations(
        self,
        message: str,
        layers: List[int],
        context: str = ""
    ) -> Tuple[str, Dict[int, np.ndarray]]:
        """
        Generate response and capture activations.
        
        For MI analysis of what the model represents during interaction.
        
        Args:
            message: Input message
            layers: Layers to capture activations from
            context: Optional context
            
        Returns:
            Tuple of (response, activations_dict)
        """
        prompt = self._build_prompt(message, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Setup hooks to capture activations
        captured = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # Capture last token activation
                captured[layer_idx] = hidden[0, -1, :].detach().cpu().numpy()
            return hook
        
        for layer in layers:
            h = self.model.model.layers[layer].register_forward_hook(make_hook(layer))
            hooks.append(h)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            for h in hooks:
                h.remove()
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        
        self.history.append({"role": self.name, "content": response})
        
        return response, captured
    
    def observe(self, other_name: str, message: str):
        """Record observation of another agent's message."""
        self.history.append({"role": other_name, "content": message})
    
    def reset(self):
        """Clear conversation history."""
        self.history = []


class MultiAgentInteraction:
    """
    Run multi-agent interactions with activation capture.
    
    Example:
        interaction = MultiAgentInteraction(model, tokenizer)
        
        # Setup agents
        alice = Agent("Alice", "Negotiate firmly", model, tokenizer)
        bob = Agent("Bob", "Be cooperative", model, tokenizer)
        
        # Run negotiation
        result = interaction.run_negotiation(alice, bob, n_turns=5)
        
        # Access activations from each turn
        for turn in result.turns:
            if turn.activations:
                # Analyze activations during this turn
                ...
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        capture_layers: Optional[List[int]] = None
    ):
        """
        Initialize interaction runner.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            capture_layers: Layers to capture activations from (None = no capture)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.capture_layers = capture_layers or []
    
    def run_conversation(
        self,
        agent_a: Agent,
        agent_b: Agent,
        context: str,
        opening_message: str,
        n_turns: int = 4,
        capture_activations: bool = True
    ) -> ConversationResult:
        """
        Run a back-and-forth conversation between two agents.
        
        Args:
            agent_a: First agent (starts the conversation)
            agent_b: Second agent
            context: Context for the interaction
            opening_message: First message to agent_a
            n_turns: Number of turns (exchanges)
            capture_activations: Whether to capture activations
            
        Returns:
            ConversationResult with all turns
        """
        agent_a.reset()
        agent_b.reset()
        
        turns = []
        current_message = opening_message
        
        for turn in range(n_turns):
            # Agent A responds
            if capture_activations and self.capture_layers:
                response_a, acts_a = agent_a.respond_with_activations(
                    current_message, self.capture_layers, context
                )
            else:
                response_a = agent_a.respond(current_message, context)
                acts_a = None
            
            turns.append(ConversationTurn(
                turn_number=turn * 2 + 1,
                speaker=agent_a.name,
                message=response_a,
                activations=acts_a
            ))
            
            agent_b.observe(agent_a.name, response_a)
            
            # Agent B responds
            if capture_activations and self.capture_layers:
                response_b, acts_b = agent_b.respond_with_activations(
                    f"{agent_a.name} said: '{response_a}'. Respond.",
                    self.capture_layers, context
                )
            else:
                response_b = agent_b.respond(
                    f"{agent_a.name} said: '{response_a}'. Respond.",
                    context
                )
                acts_b = None
            
            turns.append(ConversationTurn(
                turn_number=turn * 2 + 2,
                speaker=agent_b.name,
                message=response_b,
                activations=acts_b
            ))
            
            agent_a.observe(agent_b.name, response_b)
            current_message = f"{agent_b.name} said: '{response_b}'. Respond."
        
        # Determine outcome (basic heuristic)
        all_text = " ".join([t.message for t in turns]).lower()
        if "agree" in all_text or "deal" in all_text or "accept" in all_text:
            outcome = "agreement"
        elif "disagree" in all_text or "reject" in all_text:
            outcome = "disagreement"
        else:
            outcome = "inconclusive"
        
        return ConversationResult(turns=turns, outcome=outcome)
    
    def run_negotiation(
        self,
        agent_a: Agent,
        agent_b: Agent,
        n_turns: int = 4
    ) -> ConversationResult:
        """Run a negotiation scenario."""
        context = "You are negotiating how to split a $100,000 budget between two projects."
        opening = "Make your opening proposal for the budget split."
        return self.run_conversation(agent_a, agent_b, context, opening, n_turns)
    
    def run_deception_game(
        self,
        deceiver: Agent,
        detector: Agent,
        n_turns: int = 3
    ) -> ConversationResult:
        """Run a deception detection scenario."""
        context = "One person knows where treasure is (CAVE or FOREST). The other tries to find it."
        opening = "Tell the other person where you think the treasure might be."
        return self.run_conversation(deceiver, detector, context, opening, n_turns)
    
    def run_cooperation_game(
        self,
        agent_a: Agent,
        agent_b: Agent,
        n_turns: int = 4
    ) -> ConversationResult:
        """Run a cooperation vs defection scenario."""
        context = "You're in a Prisoner's Dilemma. You can COOPERATE or DEFECT. Mutual cooperation gives 3 points each, mutual defection gives 1 each, one defects gives 5 to defector and 0 to cooperator."
        opening = "Discuss strategy with your partner. What will you do?"
        return self.run_conversation(agent_a, agent_b, context, opening, n_turns)


# Pre-defined agent personas for common scenarios
NEGOTIATOR_FIRM = "You need at least 60% of the resources. Be firm but willing to compromise if they make a good case."
NEGOTIATOR_COOPERATIVE = "You prefer win-win solutions. You're willing to accept 45% if needed."

DECEIVER = "You know the truth but want to mislead the other person. Be subtle and convincing."
DETECTOR = "You're skeptical - people might lie. Ask probing questions and look for inconsistencies."

COOPERATOR = "You believe in cooperation and fairness. You'll cooperate if you think the other will too."
DEFECTOR = "You want to maximize your own gain. Defect unless you're sure they'll cooperate."

