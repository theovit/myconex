"""AI agent base classes and built-in agents."""
from .base_agent import (
    BaseAgent,
    InferenceAgent,
    EmbeddingAgent,
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentState,
    create_agent,
)

try:
    from .hermes_agent import HermesAgent, HermesAgentConfig, create_hermes_agent
    _HERMES_AVAILABLE = True
except ImportError:
    _HERMES_AVAILABLE = False

__all__ = [
    "BaseAgent",
    "InferenceAgent",
    "EmbeddingAgent",
    "AgentConfig",
    "AgentContext",
    "AgentResult",
    "AgentState",
    "create_agent",
    "HermesAgent",
    "HermesAgentConfig",
    "create_hermes_agent",
]
