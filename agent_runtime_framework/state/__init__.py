"""
State management abstractions for journey-based agents.

Provides base classes for managing agent state with step-based state machines.
"""

from agent_runtime_framework.state.base import (
    BaseState,
    BaseJourneyState,
    StateSerializer,
)

__all__ = [
    "BaseState",
    "BaseJourneyState",
    "StateSerializer",
]
