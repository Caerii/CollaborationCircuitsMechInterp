"""
Circuit Analysis Tools for Chat Mode

Provides head ablation and circuit discovery specifically for chat-based models.
"""

from .ablation import HeadAblator, AblationResult
from .chat_circuit_analyzer import ChatModeCircuitAnalyzer

__all__ = [
    "HeadAblator",
    "AblationResult",
    "ChatModeCircuitAnalyzer",
]

