"""
Agent Runtime Framework

A framework for building journey-based conversational agents.
"""

# Core components
from agent_runtime_framework.state import (
    BaseJourneyState,
    StateSerializer,
)
from agent_runtime_framework.tools import (
    BaseJourneyTools,
    ToolSchema,
    ToolSchemaBuilder,
    ToolParameter,
    ToolResult,
)
from agent_runtime_framework.agents import (
    JourneyAgent,
    JourneyConfig,
    AgentContext,
    AgentMessage,
    AgentResult,
)
from agent_runtime_framework.executor import (
    Executor,
    ExecutorHooks,
    CompositeHooks,
    ToolExecutor,
    CallableToolExecutor,
    MethodToolExecutor,
)
from agent_runtime_framework.memory import (
    MemoryStore,
    InMemoryStore,
    StateStore,
    ConversationStore,
    MemoryContext,
    MemoryManager,
    build_memory_context,
)
from agent_runtime_framework.prompts import (
    PromptTemplate,
    StepPromptMapping,
    PromptManager,
)
from agent_runtime_framework.router import (
    IntentRouter,
    IntentDetector,
    RouteDefinition,
)

__version__ = "0.1.0"

__all__ = [
    # State
    "BaseJourneyState",
    "StateSerializer",
    # Tools
    "BaseJourneyTools",
    "ToolSchema",
    "ToolSchemaBuilder",
    "ToolParameter",
    "ToolResult",
    # Agents
    "JourneyAgent",
    "JourneyConfig",
    "AgentContext",
    "AgentMessage",
    "AgentResult",
    # Executor
    "Executor",
    "ExecutorHooks",
    "CompositeHooks",
    "ToolExecutor",
    "CallableToolExecutor",
    "MethodToolExecutor",
    # Memory
    "MemoryStore",
    "InMemoryStore",
    "StateStore",
    "ConversationStore",
    "MemoryContext",
    "MemoryManager",
    "build_memory_context",
    # Prompts
    "PromptTemplate",
    "StepPromptMapping",
    "PromptManager",
    # Router
    "IntentRouter",
    "IntentDetector",
    "RouteDefinition",
]
