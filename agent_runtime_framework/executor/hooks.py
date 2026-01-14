"""
Hooks for executor observability.

Hooks allow you to observe and log the execution process
without modifying the core executor logic.
"""

from abc import ABC
from typing import Any, Optional, Sequence
import logging


logger = logging.getLogger(__name__)


class ExecutorHooks(ABC):
    """
    Base class for executor hooks.
    
    Override methods to observe execution events.
    All methods are optional - default implementations do nothing.
    
    Example:
        class MyHooks(ExecutorHooks):
            async def on_tool_start(self, name: str, arguments: dict) -> None:
                print(f"Calling tool: {name}")
            
            async def on_tool_end(self, name: str, result: str) -> None:
                print(f"Tool {name} returned: {result[:100]}")
    """
    
    async def on_iteration_start(self, iteration: int) -> None:
        """Called at the start of each iteration."""
        pass
    
    async def on_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
    ) -> None:
        """Called before each LLM call."""
        pass
    
    async def on_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Called when LLM returns tool calls."""
        pass
    
    async def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Called before executing a tool."""
        pass
    
    async def on_tool_end(self, name: str, result: str) -> None:
        """Called after executing a tool."""
        pass
    
    async def on_complete(self, response: str, stopped_reason: str) -> None:
        """Called when execution completes."""
        pass


class LoggingHooks(ExecutorHooks):
    """
    Hooks that log execution events.
    
    Example:
        executor = Executor(
            llm_client=my_llm,
            hooks=LoggingHooks(level=logging.DEBUG),
        )
    """
    
    def __init__(
        self,
        level: int = logging.INFO,
        logger_name: Optional[str] = None,
    ):
        self.level = level
        self.logger = logging.getLogger(logger_name or __name__)
    
    async def on_iteration_start(self, iteration: int) -> None:
        self.logger.log(self.level, f"Starting iteration {iteration}")
    
    async def on_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
    ) -> None:
        tool_count = len(tools) if tools else 0
        self.logger.log(
            self.level,
            f"Calling LLM with {len(messages)} messages and {tool_count} tools",
        )
    
    async def on_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        names = [
            tc.get("function", {}).get("name", tc.get("name", "unknown"))
            for tc in tool_calls
        ]
        self.logger.log(self.level, f"LLM requested tools: {names}")
    
    async def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self.logger.log(self.level, f"Executing tool: {name}")
        self.logger.log(logging.DEBUG, f"Tool arguments: {arguments}")
    
    async def on_tool_end(self, name: str, result: str) -> None:
        preview = result[:200] + "..." if len(result) > 200 else result
        self.logger.log(self.level, f"Tool {name} completed: {preview}")
    
    async def on_complete(self, response: str, stopped_reason: str) -> None:
        self.logger.log(self.level, f"Execution complete: {stopped_reason}")


class CompositeHooks(ExecutorHooks):
    """
    Combine multiple hooks.
    
    Example:
        hooks = CompositeHooks([
            LoggingHooks(),
            MetricsHooks(),
            TracingHooks(),
        ])
    """
    
    def __init__(self, hooks: Sequence[ExecutorHooks]):
        self._hooks = list(hooks)
    
    def add(self, hook: ExecutorHooks) -> "CompositeHooks":
        """Add a hook."""
        self._hooks.append(hook)
        return self
    
    async def on_iteration_start(self, iteration: int) -> None:
        for hook in self._hooks:
            await hook.on_iteration_start(iteration)
    
    async def on_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
    ) -> None:
        for hook in self._hooks:
            await hook.on_llm_call(messages, tools)
    
    async def on_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        for hook in self._hooks:
            await hook.on_tool_calls(tool_calls)
    
    async def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        for hook in self._hooks:
            await hook.on_tool_start(name, arguments)
    
    async def on_tool_end(self, name: str, result: str) -> None:
        for hook in self._hooks:
            await hook.on_tool_end(name, result)
    
    async def on_complete(self, response: str, stopped_reason: str) -> None:
        for hook in self._hooks:
            await hook.on_complete(response, stopped_reason)
