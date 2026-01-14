"""
Base tool classes for journey-based agents.

Tools are the primary way agents interact with the world. They:
- Operate on state (reading and modifying)
- Call external services (APIs, databases, etc.)
- Return results to the LLM for further processing
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

from agent_runtime_framework.state import BaseJourneyState


# Type variable for state
StateT = TypeVar("StateT", bound=BaseJourneyState)

# Type alias for state change callback
StateChangeCallback = Callable[[BaseJourneyState], Awaitable[None]]


@dataclass
class ToolResult:
    """
    Result from a tool execution.
    
    Attributes:
        content: The result content (string for LLM, or structured data)
        success: Whether the tool executed successfully
        error: Error message if success is False
        metadata: Additional metadata about the execution
    """
    content: str
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, content: str, **metadata: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(content=content, success=True, metadata=metadata)
    
    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "ToolResult":
        """Create a failed result."""
        return cls(content=error, success=False, error=error, metadata=metadata)


class BaseTool(ABC):
    """
    Base class for standalone tools.
    
    Standalone tools don't manage state - they just perform actions
    and return results. Use BaseJourneyTools for stateful tools.
    
    Example:
        class WeatherTool(BaseTool):
            async def get_weather(self, location: str) -> str:
                # Call weather API
                return f"Weather in {location}: Sunny, 22°C"
    """
    pass


class BaseJourneyTools(Generic[StateT]):
    """
    Base class for journey tools that operate on state.
    
    Journey tools:
    - Hold a reference to the journey state
    - Modify state as part of tool execution
    - Notify listeners when state changes
    - Optionally interact with a backend client
    
    The generic parameter StateT should be your journey state class.
    
    Example:
        class MyTools(BaseJourneyTools[MyJourneyState]):
            async def collect_name(self, name: str) -> str:
                self.state.name = name
                self.state.step = MyStep.NEXT_STEP
                await self._notify_state_change()
                return f"Got it, {name}! Now let's continue..."
    
    Attributes:
        state: The journey state object
        backend_client: Optional client for external API calls
        on_state_change: Callback invoked when state changes
    """
    
    def __init__(
        self,
        state: StateT,
        backend_client: Optional[Any] = None,
        on_state_change: Optional[StateChangeCallback] = None,
    ):
        """
        Initialize journey tools.
        
        Args:
            state: The journey state to operate on
            backend_client: Optional client for backend API calls
            on_state_change: Async callback invoked after state changes
        """
        self.state = state
        self.backend_client = backend_client
        self.on_state_change = on_state_change
    
    async def _notify_state_change(self) -> None:
        """
        Notify listeners that state has changed.
        
        Call this after modifying state to trigger persistence
        and any other state change handlers.
        """
        if self.on_state_change:
            await self.on_state_change(self.state)
    
    def _require_backend(self) -> Any:
        """
        Get the backend client, raising if not configured.
        
        Use this in tools that require a backend client.
        
        Returns:
            The backend client
            
        Raises:
            ValueError: If backend_client is not configured
        """
        if self.backend_client is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a backend_client "
                "but none was provided"
            )
        return self.backend_client
