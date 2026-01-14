"""Pytest fixtures for agent_runtime_framework tests."""

import pytest
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from agent_runtime_framework.state import BaseJourneyState
from agent_runtime_framework.tools import BaseJourneyTools
from agent_runtime_framework.agents import AgentContext, AgentMessage


# Test step enum
class TestStep(str, Enum):
    WELCOME = "welcome"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    COMPLETE = "complete"


# Test state class
@dataclass
class TestJourneyState(BaseJourneyState[TestStep]):
    """Test journey state for testing."""
    step: TestStep = TestStep.WELCOME
    name: str = ""
    email: str = ""
    processed: bool = False
    
    def is_complete(self) -> bool:
        return self.step == TestStep.COMPLETE
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.value,
            "name": self.name,
            "email": self.email,
            "processed": self.processed,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestJourneyState":
        return cls(
            step=TestStep(data.get("step", "welcome")),
            name=data.get("name", ""),
            email=data.get("email", ""),
            processed=data.get("processed", False),
        )


# Test tools class
class TestJourneyTools(BaseJourneyTools[TestJourneyState]):
    """Test tools for testing."""
    
    async def collect_name(self, name: str) -> str:
        """Collect the user's name."""
        self.state.name = name
        self.state.step = TestStep.COLLECTING
        await self._notify_state_change()
        return f"Got it, {name}! What's your email?"
    
    async def collect_email(self, email: str) -> str:
        """Collect the user's email."""
        self.state.email = email
        self.state.step = TestStep.PROCESSING
        await self._notify_state_change()
        return f"Thanks! Processing your request..."
    
    async def complete_journey(self) -> str:
        """Complete the journey."""
        self.state.processed = True
        self.state.step = TestStep.COMPLETE
        await self._notify_state_change()
        return "All done! Your request has been processed."


@pytest.fixture
def test_state():
    """Create a fresh test state."""
    return TestJourneyState()


@pytest.fixture
def test_context():
    """Create a test agent context."""
    return AgentContext(
        run_id=uuid4(),
        conversation_id=uuid4(),
        input_messages=[
            AgentMessage(role="user", content="Hello, I need help"),
        ],
        metadata={"user_id": "test-user"},
    )


@pytest.fixture
def test_tools(test_state):
    """Create test tools with state."""
    return TestJourneyTools(state=test_state)


# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: Optional[list[dict]] = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict] = []
    
    def add_response(self, message: dict, usage: Optional[dict] = None):
        """Add a response to the queue."""
        self.responses.append({
            "message": message,
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
        })
    
    def add_text_response(self, content: str):
        """Add a simple text response."""
        self.add_response({"role": "assistant", "content": content})
    
    def add_tool_call_response(self, tool_name: str, arguments: dict, content: Optional[str] = None):
        """Add a tool call response."""
        self.add_response({
            "role": "assistant",
            "content": content,
            "tool_calls": [{
                "id": f"call_{tool_name}_{self.call_count}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": str(arguments) if not isinstance(arguments, str) else arguments,
                },
            }],
        })
    
    async def generate(self, messages: list, tools: Optional[list] = None, **kwargs):
        """Generate a mock response."""
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "kwargs": kwargs,
        })
        
        if self.call_count >= len(self.responses):
            # Default to empty text response
            return MockLLMResponse({"role": "assistant", "content": ""})
        
        response = self.responses[self.call_count]
        self.call_count += 1
        return MockLLMResponse(response["message"], response.get("usage", {}))


class MockLLMResponse:
    """Mock LLM response."""
    
    def __init__(self, message: dict, usage: Optional[dict] = None):
        self.message = message
        self.usage = usage or {}


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    return MockLLMClient()
