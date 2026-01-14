"""
Base state classes for journey-based agents.

These abstractions provide a foundation for building stateful agents
with step-based state machines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar, Optional, Self
from uuid import UUID, uuid4


class StateSerializer:
    """
    Utility for serializing/deserializing state objects.
    
    Handles common types like Enum, UUID, datetime, and nested dataclasses.
    """
    
    @staticmethod
    def serialize(obj: Any) -> Any:
        """Convert an object to a JSON-serializable format."""
        if obj is None:
            return None
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (list, tuple)):
            return [StateSerializer.serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: StateSerializer.serialize(v) for k, v in obj.items()}
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dataclass_fields__"):
            return {k: StateSerializer.serialize(v) for k, v in asdict(obj).items()}
        return obj
    
    @staticmethod
    def deserialize_enum(value: Any, enum_class: type[Enum]) -> Enum:
        """Deserialize a value to an enum."""
        if isinstance(value, enum_class):
            return value
        return enum_class(value)


@dataclass
class BaseState(ABC):
    """
    Base class for all agent state objects.
    
    Provides serialization/deserialization and basic state management.
    
    Example:
        @dataclass
        class MyState(BaseState):
            name: str = ""
            count: int = 0
            
            def to_dict(self) -> dict[str, Any]:
                return {"name": self.name, "count": self.count}
            
            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> "MyState":
                return cls(name=data.get("name", ""), count=data.get("count", 0))
    """
    
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        ...
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create state from a dictionary."""
        ...
    
    def copy(self) -> Self:
        """Create a copy of this state."""
        return self.__class__.from_dict(self.to_dict())


# Type variable for step enum
StepT = TypeVar("StepT", bound=Enum)


@dataclass
class BaseJourneyState(BaseState, Generic[StepT]):
    """
    Base class for journey-based state with step tracking.
    
    A journey is a multi-step process where the agent guides the user
    through a series of steps (e.g., collecting information, processing,
    presenting results).
    
    Subclasses must define:
    - A step enum type
    - The initial step value
    - Any journey-specific state fields
    
    Example:
        class MyStep(str, Enum):
            WELCOME = "welcome"
            COLLECTING = "collecting"
            PROCESSING = "processing"
            COMPLETE = "complete"
        
        @dataclass
        class MyJourneyState(BaseJourneyState[MyStep]):
            step: MyStep = MyStep.WELCOME
            collected_data: str = ""
            
            def is_complete(self) -> bool:
                return self.step == MyStep.COMPLETE
            
            def to_dict(self) -> dict[str, Any]:
                return {
                    "step": self.step.value,
                    "collected_data": self.collected_data,
                }
            
            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> "MyJourneyState":
                return cls(
                    step=MyStep(data.get("step", "welcome")),
                    collected_data=data.get("collected_data", ""),
                )
    """
    
    # Subclasses should override with their step enum and default
    # step: StepT  # Can't have abstract field in dataclass, subclass must define
    
    def get_step(self) -> StepT:
        """Get the current step. Subclasses should have a 'step' field."""
        return getattr(self, "step")
    
    def set_step(self, step: StepT) -> None:
        """Set the current step."""
        setattr(self, "step", step)
    
    @abstractmethod
    def is_complete(self) -> bool:
        """Check if the journey is complete."""
        ...
    
    def is_terminal(self) -> bool:
        """
        Check if the journey is in a terminal state.
        
        Terminal states are states where no further action is expected
        (e.g., complete, cancelled, error). Override if your journey
        has multiple terminal states.
        """
        return self.is_complete()
