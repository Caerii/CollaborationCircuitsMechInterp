"""
Core infrastructure for rigorous MI experiments.

Provides:
    ActivationExtractor: Unified activation extraction with caching
    ChatExperimentRunner: Chat-based experiment running with proper token budget
    ResponseParser: Extract answers from <think> tag responses
    PromptFormatter: ALL prompt formatting in one place
    
Multi-Agent (for studying collaboration circuits):
    Agent: LLM agent with persona and history
    MultiAgentInteraction: Run multi-turn interactions with activation capture
"""

try:
    from .activation_extractor import ActivationExtractor
    from .chat_runner import ChatExperimentRunner
    from .response_parser import ResponseParser
    from .prompts import PromptFormatter, format_tom, format_chat
    from .multi_agent import (
        Agent,
        MultiAgentInteraction,
        ConversationTurn,
        ConversationResult,
        NEGOTIATOR_FIRM,
        NEGOTIATOR_COOPERATIVE,
        DECEIVER,
        DETECTOR,
        COOPERATOR,
        DEFECTOR,
    )
    from .cross_model import CrossModelTester, compare_models_on_scenarios, wilson_ci, cohens_h
except ImportError:
    # For direct imports when running as script
    from activation_extractor import ActivationExtractor
    from chat_runner import ChatExperimentRunner
    from response_parser import ResponseParser
    from prompts import PromptFormatter, format_tom, format_chat
    from multi_agent import (
        Agent,
        MultiAgentInteraction,
        ConversationTurn,
        ConversationResult,
        NEGOTIATOR_FIRM,
        NEGOTIATOR_COOPERATIVE,
        DECEIVER,
        DETECTOR,
        COOPERATOR,
        DEFECTOR,
    )
    from cross_model import CrossModelTester, compare_models_on_scenarios, wilson_ci, cohens_h

__all__ = [
    # Core
    "ActivationExtractor",
    "ChatExperimentRunner", 
    "ResponseParser",
    "PromptFormatter",
    "format_tom",
    "format_chat",
    # Multi-Agent
    "Agent",
    "MultiAgentInteraction",
    "ConversationTurn",
    "ConversationResult",
    # Pre-defined personas
    "NEGOTIATOR_FIRM",
    "NEGOTIATOR_COOPERATIVE",
    "DECEIVER",
    "DETECTOR",
    "COOPERATOR",
    "DEFECTOR",
    # Cross-model validation
    "CrossModelTester",
    "compare_models_on_scenarios",
    "wilson_ci",
    "cohens_h",
]

