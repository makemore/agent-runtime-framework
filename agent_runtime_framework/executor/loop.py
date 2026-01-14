"""
Core execution loop for agents.

The Executor handles the main agent loop:
1. Call LLM with messages and tools
2. If LLM returns tool calls, execute them
3. Add tool results to messages
4. Repeat until LLM returns text or max iterations reached
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence
import json
import logging

from agent_runtime_framework.agents.base import AgentMessage, LLMClient
from agent_runtime_framework.tools.schema import ToolSchema


logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """
    Configuration for the executor.
    
    Attributes:
        max_iterations: Maximum tool call iterations
        terminal_tools: Tool names that end the loop immediately
        require_tool_call: If True, force LLM to call a tool (tool_choice="required")
        stop_on_text: If True, stop when LLM returns text without tool calls
    """
    max_iterations: int = 10
    terminal_tools: set[str] = field(default_factory=set)
    require_tool_call: bool = False
    stop_on_text: bool = True


@dataclass
class ExecutionResult:
    """
    Result from an execution loop.
    
    Attributes:
        response: Final response content
        messages: All messages generated during execution
        tool_calls_made: Number of tool calls executed
        stopped_reason: Why the loop stopped
        usage: Aggregated token usage
    """
    response: str
    messages: list[AgentMessage] = field(default_factory=list)
    tool_calls_made: int = 0
    stopped_reason: str = "text_response"
    usage: dict[str, int] = field(default_factory=dict)


class ToolExecutor(Protocol):
    """
    Protocol for tool execution.
    
    Implement this to define how tools are executed.
    """
    
    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a tool and return the result."""
        ...


class CallableToolExecutor:
    """
    Tool executor that uses a callable mapping.
    
    Example:
        async def my_tool(arg1: str) -> str:
            return f"Result: {arg1}"
        
        executor = CallableToolExecutor({
            "my_tool": my_tool,
        })
    """
    
    def __init__(
        self,
        tools: dict[str, Callable[..., Any]],
    ):
        self._tools = tools
    
    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name."""
        if name not in self._tools:
            return f"Error: Unknown tool '{name}'"
        
        tool_fn = self._tools[name]
        try:
            result = tool_fn(**arguments)
            # Handle async functions
            if hasattr(result, "__await__"):
                result = await result
            return str(result)
        except Exception as e:
            logger.exception(f"Error executing tool {name}")
            return f"Error executing {name}: {str(e)}"


class MethodToolExecutor:
    """
    Tool executor that calls methods on an object.
    
    Example:
        class MyTools:
            async def search(self, query: str) -> str:
                return f"Results for: {query}"
        
        tools = MyTools()
        executor = MethodToolExecutor(tools)
    """
    
    def __init__(self, tools_instance: Any):
        self._tools = tools_instance
    
    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool method by name."""
        if not hasattr(self._tools, name):
            return f"Error: Unknown tool '{name}'"
        
        method = getattr(self._tools, name)
        try:
            result = method(**arguments)
            # Handle async methods
            if hasattr(result, "__await__"):
                result = await result
            return str(result)
        except Exception as e:
            logger.exception(f"Error executing tool {name}")
            return f"Error executing {name}: {str(e)}"


class Executor:
    """
    Core execution loop for agents.
    
    Handles the LLM interaction and tool execution loop.
    
    Example:
        executor = Executor(
            llm_client=my_llm,
            tool_executor=MethodToolExecutor(my_tools),
            config=ExecutorConfig(max_iterations=5),
        )
        
        result = await executor.run(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[my_tool_schema],
            system_prompt="You are a helpful assistant.",
        )
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: Optional[ToolExecutor] = None,
        config: Optional[ExecutorConfig] = None,
        hooks: Optional["ExecutorHooks"] = None,
    ):
        """
        Initialize the executor.
        
        Args:
            llm_client: LLM client for generating responses
            tool_executor: Executor for running tools
            config: Execution configuration
            hooks: Optional hooks for observability
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.config = config or ExecutorConfig()
        self.hooks = hooks
    
    async def run(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[ToolSchema]] = None,
        system_prompt: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Run the execution loop.
        
        Args:
            messages: Input messages (will be modified in place)
            tools: Available tool schemas
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Execution result with response and messages
        """
        # Prepare messages
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Convert tools to OpenAI format
        tool_dicts = [t.to_openai_format() for t in tools] if tools else None
        
        output_messages: list[AgentMessage] = []
        response_content = ""
        tool_calls_made = 0
        stopped_reason = "text_response"
        total_usage: dict[str, int] = {}
        
        for iteration in range(self.config.max_iterations):
            if self.hooks:
                await self.hooks.on_iteration_start(iteration)
            
            # Build LLM kwargs
            llm_kwargs: dict[str, Any] = {}
            if self.config.require_tool_call and tool_dicts:
                llm_kwargs["tool_choice"] = "required"
            
            # Call LLM
            if self.hooks:
                await self.hooks.on_llm_call(messages, tool_dicts)
            
            response = await self.llm_client.generate(
                messages=messages,
                tools=tool_dicts,
                **llm_kwargs,
            )
            
            # Aggregate usage
            if hasattr(response, "usage") and response.usage:
                for key, value in response.usage.items():
                    total_usage[key] = total_usage.get(key, 0) + value
            
            # Extract tool calls
            tool_calls = response.message.get("tool_calls", [])
            
            if tool_calls:
                if self.hooks:
                    await self.hooks.on_tool_calls(tool_calls)
                
                # Add assistant message with tool calls
                assistant_msg = AgentMessage(
                    role="assistant",
                    content=response.message.get("content"),
                    tool_calls=tool_calls,
                )
                output_messages.append(assistant_msg)
                messages.append(assistant_msg.to_dict())
                
                # Execute each tool call
                for tool_call in tool_calls:
                    name, arguments, tool_call_id = self._parse_tool_call(tool_call)
                    
                    if self.hooks:
                        await self.hooks.on_tool_start(name, arguments)
                    
                    # Execute tool
                    if self.tool_executor:
                        result = await self.tool_executor.execute(name, arguments)
                    else:
                        result = f"Error: No tool executor configured for '{name}'"
                    
                    if self.hooks:
                        await self.hooks.on_tool_end(name, result)
                    
                    # Add tool result message
                    tool_msg = AgentMessage(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                    output_messages.append(tool_msg)
                    messages.append(tool_msg.to_dict())
                    
                    response_content = result
                    tool_calls_made += 1
                
                # Check for terminal tool
                last_tool_name = self._parse_tool_call(tool_calls[-1])[0]
                if last_tool_name in self.config.terminal_tools:
                    stopped_reason = f"terminal_tool:{last_tool_name}"
                    break
            else:
                # No tool calls - text response
                response_content = response.message.get("content", "")
                output_messages.append(AgentMessage(
                    role="assistant",
                    content=response_content,
                ))
                stopped_reason = "text_response"
                
                if self.config.stop_on_text:
                    break
        else:
            stopped_reason = "max_iterations"
        
        if self.hooks:
            await self.hooks.on_complete(response_content, stopped_reason)
        
        return ExecutionResult(
            response=response_content,
            messages=output_messages,
            tool_calls_made=tool_calls_made,
            stopped_reason=stopped_reason,
            usage=total_usage,
        )
    
    def _parse_tool_call(
        self,
        tool_call: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        """Parse a tool call into (name, arguments, id)."""
        if "function" in tool_call:
            # OpenAI format
            tool_call_id = tool_call.get("id", f"call_{tool_call['function']['name']}")
            name = tool_call["function"]["name"]
            arguments = tool_call["function"].get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
        else:
            # Simplified format
            tool_call_id = tool_call.get("id", f"call_{tool_call['name']}")
            name = tool_call["name"]
            arguments = tool_call.get("arguments", {})
        
        return name, arguments, tool_call_id


# Import hooks here to avoid circular import
from agent_runtime_framework.executor.hooks import ExecutorHooks
