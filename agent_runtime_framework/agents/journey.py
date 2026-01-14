"""
Journey-based agent implementation.

JourneyAgent provides a structured approach to building agents
that guide users through multi-step processes.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from agent_runtime_framework.agents.base import (
    BaseAgent,
    AgentContext,
    AgentResult,
    AgentMessage,
    LLMClient,
)
from agent_runtime_framework.state import BaseJourneyState
from agent_runtime_framework.tools import BaseJourneyTools, ToolSchema


# Type variables
StateT = TypeVar("StateT", bound=BaseJourneyState)
ToolsT = TypeVar("ToolsT", bound=BaseJourneyTools)
StepT = TypeVar("StepT", bound=Enum)


@dataclass
class JourneyConfig:
    """
    Configuration for a journey agent.
    
    Attributes:
        max_tool_iterations: Maximum tool call iterations per run
        terminal_tools: Tool names that end the tool loop
        closing_phrases: Phrases that indicate conversation end
    """
    max_tool_iterations: int = 10
    terminal_tools: set[str] = field(default_factory=set)
    closing_phrases: list[str] = field(default_factory=lambda: [
        "thank you", "thanks", "bye", "goodbye",
    ])


class JourneyAgent(BaseAgent, Generic[StateT, ToolsT, StepT]):
    """
    Base class for journey-based agents.
    
    A journey agent guides users through a multi-step process,
    with behavior driven by the current step in the journey.
    
    Subclasses must implement:
    - key: Unique agent identifier
    - get_initial_state: Create initial state for new conversations
    - get_system_prompt: Get prompt for current state
    - get_tool_schemas: Get available tools for current state
    - create_tools: Create tool instance for current state
    - execute_tool: Execute a tool call
    
    Optional overrides:
    - load_state: Load state from persistence
    - save_state: Save state to persistence
    - is_terminal_state: Check if journey is complete
    
    Example:
        class QuoteAgent(JourneyAgent[QuoteState, QuoteTools, QuoteStep]):
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
                self, state: QuoteState, ctx: AgentContext
            ) -> QuoteTools:
                return QuoteTools(state=state, ...)
            
            async def execute_tool(
                self, tools: QuoteTools, name: str, args: dict
            ) -> str:
                method = getattr(tools, name)
                return await method(**args)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[JourneyConfig] = None,
    ):
        """
        Initialize the journey agent.
        
        Args:
            llm_client: LLM client for generating responses
            config: Agent configuration
        """
        self.llm_client = llm_client
        self.config = config or JourneyConfig()
    
    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================
    
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
    def create_tools(self, state: StateT, ctx: AgentContext) -> ToolsT:
        """Create a tools instance for the current state."""
        ...
    
    @abstractmethod
    async def execute_tool(
        self,
        tools: ToolsT,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Execute a tool call.
        
        Args:
            tools: The tools instance
            name: Tool name to execute
            arguments: Tool arguments
            
        Returns:
            Tool result as a string
        """
        ...
    
    # =========================================================================
    # Optional overrides
    # =========================================================================
    
    async def load_state(self, ctx: AgentContext) -> Optional[StateT]:
        """
        Load state from persistence.
        
        Override to implement state persistence.
        Default returns None (no persistence).
        """
        return None
    
    async def save_state(self, ctx: AgentContext, state: StateT) -> None:
        """
        Save state to persistence.
        
        Override to implement state persistence.
        Default does nothing.
        """
        pass
    
    def is_terminal_state(self, state: StateT) -> bool:
        """
        Check if the journey is in a terminal state.
        
        Override for custom terminal state logic.
        Default delegates to state.is_terminal().
        """
        return state.is_terminal()
    
    def is_closing_message(self, message: str) -> bool:
        """Check if a message is a closing/thank you message."""
        message_lower = message.lower().strip()
        return any(
            phrase in message_lower
            for phrase in self.config.closing_phrases
        )
    
    # =========================================================================
    # Main run implementation
    # =========================================================================
    
    async def run(self, ctx: AgentContext) -> AgentResult:
        """
        Execute the journey agent.
        
        This is the main entry point that orchestrates:
        1. Loading/initializing state
        2. Building messages with system prompt
        3. Calling LLM with tools
        4. Executing tool calls
        5. Saving state
        6. Returning result
        """
        if self.llm_client is None:
            raise ValueError("LLM client is required to run the agent")
        
        # Load or initialize state
        state = await self.load_state(ctx) or self.get_initial_state()
        
        # Check for closing message at terminal state
        if self.is_terminal_state(state):
            user_message = ctx.get_user_message()
            if user_message and self.is_closing_message(user_message):
                return AgentResult(
                    response="",
                    state=state.to_dict(),
                )
        
        # Build messages
        system_prompt = self.get_system_prompt(state)
        messages = self._build_messages(ctx, system_prompt)
        
        # Create tools
        tools = self.create_tools(state, ctx)
        tool_schemas = self.get_tool_schemas(state)
        tool_dicts = [s.to_openai_format() for s in tool_schemas]
        
        # Run tool loop
        output_messages: list[AgentMessage] = []
        response_content = ""
        
        for _ in range(self.config.max_tool_iterations):
            # Call LLM
            response = await self.llm_client.generate(
                messages=messages,
                tools=tool_dicts if tool_dicts else None,
            )
            
            # Extract tool calls
            tool_calls = response.message.get("tool_calls", [])
            
            if tool_calls:
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
                    
                    # Execute tool
                    result = await self.execute_tool(tools, name, arguments)
                    
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
                
                # Check for terminal tool
                last_tool = self._parse_tool_call(tool_calls[-1])[0]
                if last_tool in self.config.terminal_tools:
                    break
                
                # Update tools for new state
                tools = self.create_tools(state, ctx)
                tool_schemas = self.get_tool_schemas(state)
                tool_dicts = [s.to_openai_format() for s in tool_schemas]
            else:
                # No tool calls - text response
                response_content = response.message.get("content", "")
                output_messages.append(AgentMessage(
                    role="assistant",
                    content=response_content,
                ))
                break
        
        # Save state
        await self.save_state(ctx, state)
        
        return AgentResult(
            response=response_content,
            messages=output_messages,
            state=state.to_dict(),
            usage=getattr(response, "usage", None),
        )
    
    def _build_messages(
        self,
        ctx: AgentContext,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Build the message list for the LLM."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in ctx.input_messages:
            messages.append(msg.to_dict())
        return messages
    
    def _parse_tool_call(
        self,
        tool_call: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
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
