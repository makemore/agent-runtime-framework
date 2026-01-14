"""
Base agent classes for conversational agents.

These abstractions define the core interface for agents that can
be used with any runtime (Django, FastAPI, CLI, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence
from uuid import UUID


@dataclass
class AgentMessage:
    """
    A message in an agent conversation.
    
    Compatible with OpenAI message format but framework-agnostic.
    
    Attributes:
        role: Message role (system, user, assistant, tool)
        content: Message content (string or structured)
        tool_calls: For assistant messages, any tool calls made
        tool_call_id: For tool messages, the ID of the call being responded to
        name: For tool messages, the name of the tool
        metadata: Additional message metadata
    """
    role: str
    content: Optional[str | dict | list] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentContext:
    """
    Context for an agent run.
    
    Contains all the information needed to execute an agent,
    including conversation history and metadata.
    
    Attributes:
        run_id: Unique ID for this run
        conversation_id: ID of the conversation
        input_messages: Messages to process
        metadata: Additional context (user info, settings, etc.)
    """
    run_id: UUID
    conversation_id: UUID
    input_messages: list[AgentMessage]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_user_message(self) -> Optional[str]:
        """Get the last user message content."""
        for msg in reversed(self.input_messages):
            if msg.role == "user" and isinstance(msg.content, str):
                return msg.content
        return None


@dataclass
class AgentResult:
    """
    Result from an agent run.
    
    Attributes:
        response: The final response content
        messages: All messages generated during the run
        state: Final agent state (if applicable)
        usage: Token usage information
        metadata: Additional result metadata
    """
    response: str
    messages: list[AgentMessage] = field(default_factory=list)
    state: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """
    Protocol for LLM clients.
    
    Any LLM client that implements this protocol can be used
    with the agent framework.
    """
    
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a response from the LLM."""
        ...


class BaseAgent(ABC):
    """
    Base class for all agents.
    
    Agents are the core abstraction for conversational AI. They:
    - Process user messages
    - Interact with LLMs
    - Execute tools
    - Manage state
    - Return responses
    
    Subclasses must implement the `run` method.
    
    Example:
        class MyAgent(BaseAgent):
            @property
            def key(self) -> str:
                return "my-agent"
            
            async def run(self, ctx: AgentContext) -> AgentResult:
                # Process the conversation
                response = await self._generate_response(ctx)
                return AgentResult(response=response)
    """
    
    @property
    @abstractmethod
    def key(self) -> str:
        """
        Unique identifier for this agent.
        
        Used for routing, persistence, and logging.
        """
        ...
    
    @abstractmethod
    async def run(self, ctx: AgentContext) -> AgentResult:
        """
        Execute the agent.
        
        Args:
            ctx: The agent context with conversation and metadata
            
        Returns:
            The result of the agent run
        """
        ...
