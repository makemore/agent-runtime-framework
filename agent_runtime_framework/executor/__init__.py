"""
Executor module for agent orchestration.

Provides the core execution loop for running agents with LLMs and tools.

Note: This LLMExecutor is for LLM+tool calling loops.
For multi-step workflows with checkpointing, use agent_runtime_core.steps.StepExecutor.
"""

from agent_runtime_framework.executor.loop import (
    LLMExecutor,
    LLMExecutorConfig,
    ExecutionResult,
    ToolExecutor,
    CallableToolExecutor,
    MethodToolExecutor,
    # Backwards compatibility
    Executor,
    ExecutorConfig,
)
from agent_runtime_framework.executor.hooks import (
    ExecutorHooks,
    LoggingHooks,
    CompositeHooks,
)

__all__ = [
    # New names
    "LLMExecutor",
    "LLMExecutorConfig",
    # Backwards compatibility
    "Executor",
    "ExecutorConfig",
    # Common
    "ExecutionResult",
    "ToolExecutor",
    "CallableToolExecutor",
    "MethodToolExecutor",
    "ExecutorHooks",
    "LoggingHooks",
    "CompositeHooks",
]
