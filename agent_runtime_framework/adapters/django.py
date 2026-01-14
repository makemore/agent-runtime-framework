"""
Django adapter for agent_runtime_framework.

Bridges JourneyAgent to django_agent_runtime's AgentRuntime interface.
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Optional, Protocol, TypeVar
from uuid import UUID

from agent_runtime_framework.state import BaseJourneyState
from agent_runtime_framework.tools import BaseJourneyTools, ToolSchema
from agent_runtime_framework.agents import AgentContext, AgentMessage, AgentResult


# Type variables
StateT = TypeVar("StateT", bound=BaseJourneyState)
ToolsT = TypeVar("ToolsT", bound=BaseJourneyTools)
StepT = TypeVar("StepT", bound=Enum)


class DjangoRunContext(Protocol):
    """Protocol matching django_agent_runtime's RunContext."""
    
    @property
    def run_id(self) -> UUID: ...
    
    @property
    def conversation_id(self) -> Optional[UUID]: ...
    
    @property
    def input_messages(self) -> list[dict]: ...
    
    @property
    def params(self) -> dict: ...
    
    @property
    def metadata(self) -> dict: ...
    
    async def emit(self, event_type: Any, payload: dict) -> None: ...
    
    async def emit_user_message(self, content: str) -> None: ...
    
    async def checkpoint(self, state: dict) -> None: ...
    
    async def get_state(self) -> Optional[dict]: ...
    
    def cancelled(self) -> bool: ...


class DjangoRunResult(Protocol):
    """Protocol matching django_agent_runtime's RunResult."""
    final_output: dict
    final_messages: list[dict]
    usage: dict
    artifacts: dict


@dataclass
class RunResultImpl:
    """Implementation of RunResult for Django runtime."""
    final_output: dict
    final_messages: list[dict]
    usage: dict
    artifacts: dict


class DjangoStateStore:
    """
    State store backed by Django runtime checkpoints.
    
    Uses the RunContext's checkpoint/get_state methods for persistence.
    """
    
    def __init__(self, ctx: DjangoRunContext, agent_key: str):
        self._ctx = ctx
        self._agent_key = agent_key
        self._cached_state: Optional[dict] = None
    
    async def load(self) -> Optional[dict]:
        """Load state from Django checkpoint."""
        if self._cached_state is not None:
            return self._cached_state
        
        state = await self._ctx.get_state()
        if state:
            self._cached_state = state
        return state
    
    async def save(self, state: dict) -> None:
        """Save state to Django checkpoint."""
        self._cached_state = state
        await self._ctx.checkpoint(state)


class DjangoRuntimeAdapter(Generic[StateT, ToolsT, StepT]):
    """
    Adapter that wraps a JourneyAgent for use with django_agent_runtime.
    
    This adapter:
    - Converts Django's RunContext to framework's AgentContext
    - Uses Django's checkpoint system for state persistence
    - Emits events through Django's event bus
    - Returns results in Django's RunResult format
    
    Example:
        class QuoteRuntime(DjangoRuntimeAdapter[QuoteState, QuoteTools, QuoteStep]):
            @property
            def key(self) -> str:
                return "quote-agent"
            
            def get_initial_state(self) -> QuoteState:
                return QuoteState()
            
            def get_system_prompt(self, state: QuoteState) -> str:
                return PROMPTS[state.step]
            
            def get_tool_schemas(self, state: QuoteState) -> list[ToolSchema]:
                return STEP_TOOLS[state.step]
            
            def create_tools(
                self,
                state: QuoteState,
                ctx: DjangoRunContext,
                backend_client: Any,
            ) -> QuoteTools:
                return QuoteTools(
                    state=state,
                    backend_client=backend_client,
                    on_state_change=self._make_state_callback(ctx),
                )
            
            async def execute_tool(
                self,
                tools: QuoteTools,
                name: str,
                arguments: dict,
            ) -> str:
                method = getattr(tools, name)
                return await method(**arguments)
        
        # Register with Django runtime
        from django_agent_runtime.runtime.registry import register_runtime
        register_runtime(QuoteRuntime())
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        backend_client: Optional[Any] = None,
        max_tool_iterations: int = 10,
        terminal_tools: Optional[set[str]] = None,
    ):
        """
        Initialize the adapter.
        
        Args:
            llm_client: LLM client for generating responses
            backend_client: Backend client for API calls
            max_tool_iterations: Maximum tool call iterations
            terminal_tools: Tool names that end the loop
        """
        self._llm_client = llm_client
        self._backend_client = backend_client
        self._max_tool_iterations = max_tool_iterations
        self._terminal_tools = terminal_tools or set()
    
    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================
    
    @property
    @abstractmethod
    def key(self) -> str:
        """Unique identifier for this runtime."""
        ...
    
    @abstractmethod
    def get_initial_state(self) -> StateT:
        """Create initial state for a new conversation."""
        ...
    
    @abstractmethod
    def get_system_prompt(self, state: StateT) -> str:
        """Get the system prompt for the current state."""
        ...
    
    @abstractmethod
    def get_tool_schemas(self, state: StateT) -> list[ToolSchema]:
        """Get available tool schemas for the current state."""
        ...
    
    @abstractmethod
    def create_tools(
        self,
        state: StateT,
        ctx: DjangoRunContext,
        backend_client: Optional[Any],
    ) -> ToolsT:
        """
        Create a tools instance for the current state.
        
        Args:
            state: Current journey state
            ctx: Django run context
            backend_client: Backend client for API calls
        """
        ...
    
    @abstractmethod
    async def execute_tool(
        self,
        tools: ToolsT,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a tool call."""
        ...
    
    # =========================================================================
    # Optional overrides
    # =========================================================================
    
    def get_llm_client(self, ctx: DjangoRunContext) -> Any:
        """
        Get the LLM client to use.
        
        Override to customize LLM client selection.
        Default uses the client passed to __init__ or creates one from config.
        """
        if self._llm_client:
            return self._llm_client
        
        # Try to get from agent_runtime_core
        try:
            from agent_runtime_core.llm import get_llm_client
            return get_llm_client()
        except ImportError:
            raise ValueError(
                "No LLM client configured. Either pass llm_client to __init__ "
                "or install agent_runtime_core."
            )
    
    def get_backend_client(self, ctx: DjangoRunContext) -> Optional[Any]:
        """
        Get the backend client to use.
        
        Override to customize backend client selection.
        """
        return self._backend_client
    
    def is_terminal_state(self, state: StateT) -> bool:
        """Check if the journey is in a terminal state."""
        return state.is_terminal()
    
    def is_closing_message(self, message: str) -> bool:
        """Check if a message is a closing/thank you message."""
        closing_phrases = ["thank you", "thanks", "bye", "goodbye"]
        message_lower = message.lower().strip()
        return any(phrase in message_lower for phrase in closing_phrases)
    
    async def on_state_change(self, ctx: DjangoRunContext, state: StateT) -> None:
        """
        Called when state changes.
        
        Override for custom state change handling.
        Default saves checkpoint.
        """
        await ctx.checkpoint(state.to_dict())
    
    # =========================================================================
    # Django runtime interface
    # =========================================================================
    
    async def run(self, ctx: DjangoRunContext) -> DjangoRunResult:
        """
        Execute the agent run.
        
        This is the main entry point called by django_agent_runtime.
        """
        llm_client = self.get_llm_client(ctx)
        backend_client = self.get_backend_client(ctx)
        
        # Load or initialize state
        state = await self._load_or_init_state(ctx)
        
        # Check for closing message at terminal state
        if self.is_terminal_state(state):
            user_message = self._get_user_message(ctx)
            if user_message and self.is_closing_message(user_message):
                return RunResultImpl(
                    final_output={"response": ""},
                    final_messages=[],
                    usage={},
                    artifacts={},
                )
        
        # Build messages
        system_prompt = self.get_system_prompt(state)
        messages = self._build_messages(ctx, system_prompt)
        
        # Create tools with state change callback
        def make_callback():
            async def callback(s):
                await self.on_state_change(ctx, s)
            return callback
        
        tools = self.create_tools(state, ctx, backend_client)
        if hasattr(tools, 'on_state_change'):
            tools.on_state_change = make_callback()
        
        tool_schemas = self.get_tool_schemas(state)
        tool_dicts = [s.to_openai_format() for s in tool_schemas]
        
        # Run tool loop
        output_messages: list[dict] = []
        response_content = ""
        total_usage: dict = {}
        
        for iteration in range(self._max_tool_iterations):
            # Check for cancellation
            if ctx.cancelled():
                break
            
            # Call LLM
            response = await llm_client.generate(
                messages=messages,
                tools=tool_dicts if tool_dicts else None,
            )
            
            # Aggregate usage
            if hasattr(response, 'usage') and response.usage:
                for key, value in response.usage.items():
                    total_usage[key] = total_usage.get(key, 0) + value
            
            # Extract tool calls
            tool_calls = response.message.get("tool_calls", [])
            
            if tool_calls:
                # Add assistant message with tool calls
                assistant_msg = {
                    "role": "assistant",
                    "content": response.message.get("content"),
                    "tool_calls": tool_calls,
                }
                output_messages.append(assistant_msg)
                messages.append(assistant_msg)
                
                # Execute each tool call
                for tool_call in tool_calls:
                    name, arguments, tool_call_id = self._parse_tool_call(tool_call)
                    
                    # Emit tool call event
                    await ctx.emit("tool.call", {
                        "name": name,
                        "arguments": arguments,
                        "tool_call_id": tool_call_id,
                    })
                    
                    # Execute tool
                    result = await self.execute_tool(tools, name, arguments)
                    
                    # Emit tool result event
                    await ctx.emit("tool.result", {
                        "name": name,
                        "result": result[:500] if len(result) > 500 else result,
                        "tool_call_id": tool_call_id,
                    })
                    
                    # Add tool result message
                    tool_msg = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call_id,
                        "name": name,
                    }
                    output_messages.append(tool_msg)
                    messages.append(tool_msg)
                    response_content = result
                
                # Check for terminal tool
                last_tool = self._parse_tool_call(tool_calls[-1])[0]
                if last_tool in self._terminal_tools:
                    break
                
                # Update tools for new state (state may have changed)
                tools = self.create_tools(state, ctx, backend_client)
                if hasattr(tools, 'on_state_change'):
                    tools.on_state_change = make_callback()
                tool_schemas = self.get_tool_schemas(state)
                tool_dicts = [s.to_openai_format() for s in tool_schemas]
            else:
                # No tool calls - text response
                response_content = response.message.get("content", "")
                output_messages.append({
                    "role": "assistant",
                    "content": response_content,
                })
                
                # Emit assistant message
                await ctx.emit_user_message(response_content)
                break
        
        # Save final state
        await ctx.checkpoint(state.to_dict())
        
        return RunResultImpl(
            final_output={"response": response_content, "state": state.to_dict()},
            final_messages=output_messages,
            usage=total_usage,
            artifacts={},
        )
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    async def _load_or_init_state(self, ctx: DjangoRunContext) -> StateT:
        """Load state from checkpoint or create initial state."""
        state_dict = await ctx.get_state()
        if state_dict:
            return self._state_from_dict(state_dict)
        return self.get_initial_state()
    
    def _state_from_dict(self, data: dict) -> StateT:
        """
        Create state from dictionary.
        
        Override if your state class needs custom deserialization.
        Default calls get_initial_state() and updates from dict.
        """
        state = self.get_initial_state()
        return state.__class__.from_dict(data)
    
    def _get_user_message(self, ctx: DjangoRunContext) -> Optional[str]:
        """Get the last user message from input."""
        for msg in reversed(ctx.input_messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        return None
    
    def _build_messages(
        self,
        ctx: DjangoRunContext,
        system_prompt: str,
    ) -> list[dict]:
        """Build the message list for the LLM."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(ctx.input_messages)
        return messages
    
    def _parse_tool_call(
        self,
        tool_call: dict,
    ) -> tuple[str, dict, str]:
        """Parse a tool call into (name, arguments, id)."""
        import json
        
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
    
    def _make_state_callback(self, ctx: DjangoRunContext):
        """Create a state change callback that checkpoints."""
        async def callback(state: StateT) -> None:
            await self.on_state_change(ctx, state)
        return callback


def create_django_runtime(
    agent_class: type[DjangoRuntimeAdapter],
    llm_client: Optional[Any] = None,
    backend_client: Optional[Any] = None,
    **kwargs,
) -> DjangoRuntimeAdapter:
    """
    Factory function to create a Django runtime adapter.
    
    Args:
        agent_class: The adapter class to instantiate
        llm_client: LLM client to use
        backend_client: Backend client for API calls
        **kwargs: Additional arguments for the adapter
        
    Returns:
        Configured adapter instance
    """
    return agent_class(
        llm_client=llm_client,
        backend_client=backend_client,
        **kwargs,
    )
