"""Tests for agents module."""

import pytest
from uuid import uuid4
from typing import Any

from agent_runtime_framework.agents import (
    BaseAgent,
    JourneyAgent,
    JourneyConfig,
    AgentContext,
    AgentResult,
    AgentMessage,
)
from agent_runtime_framework.tools import ToolSchema, ToolSchemaBuilder


class TestAgentMessage:
    """Tests for AgentMessage."""
    
    def test_to_dict_basic(self):
        msg = AgentMessage(role="user", content="Hello")
        
        result = msg.to_dict()
        
        assert result == {"role": "user", "content": "Hello"}
    
    def test_to_dict_with_tool_calls(self):
        msg = AgentMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call_1", "function": {"name": "test"}}],
        )
        
        result = msg.to_dict()
        
        assert result["role"] == "assistant"
        assert result["tool_calls"] == [{"id": "call_1", "function": {"name": "test"}}]
    
    def test_to_dict_tool_result(self):
        msg = AgentMessage(
            role="tool",
            content="Result",
            tool_call_id="call_1",
            name="test_tool",
        )
        
        result = msg.to_dict()
        
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert result["name"] == "test_tool"
    
    def test_from_dict(self):
        data = {
            "role": "assistant",
            "content": "Hello",
            "tool_calls": [{"id": "1"}],
        }
        
        msg = AgentMessage.from_dict(data)
        
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert msg.tool_calls == [{"id": "1"}]


class TestAgentContext:
    """Tests for AgentContext."""
    
    def test_get_user_message(self):
        ctx = AgentContext(
            run_id=uuid4(),
            conversation_id=uuid4(),
            input_messages=[
                AgentMessage(role="system", content="You are helpful"),
                AgentMessage(role="user", content="Hello"),
                AgentMessage(role="assistant", content="Hi!"),
                AgentMessage(role="user", content="How are you?"),
            ],
        )
        
        # Should get the last user message
        assert ctx.get_user_message() == "How are you?"
    
    def test_get_user_message_none(self):
        ctx = AgentContext(
            run_id=uuid4(),
            conversation_id=uuid4(),
            input_messages=[
                AgentMessage(role="system", content="System"),
            ],
        )
        
        assert ctx.get_user_message() is None


class TestJourneyConfig:
    """Tests for JourneyConfig."""
    
    def test_defaults(self):
        config = JourneyConfig()
        
        assert config.max_tool_iterations == 10
        assert config.terminal_tools == set()
        assert "thank you" in config.closing_phrases
    
    def test_custom_config(self):
        config = JourneyConfig(
            max_tool_iterations=5,
            terminal_tools={"complete", "cancel"},
            closing_phrases=["bye", "done"],
        )
        
        assert config.max_tool_iterations == 5
        assert "complete" in config.terminal_tools


class TestJourneyAgent:
    """Tests for JourneyAgent."""
    
    @pytest.mark.asyncio
    async def test_run_simple_response(self, mock_llm, test_context):
        from tests.conftest import TestJourneyState, TestJourneyTools, TestStep
        
        mock_llm.add_text_response("Welcome! What's your name?")
        
        class TestAgent(JourneyAgent[TestJourneyState, TestJourneyTools, TestStep]):
            @property
            def key(self) -> str:
                return "test-agent"
            
            def get_initial_state(self) -> TestJourneyState:
                return TestJourneyState()
            
            def get_system_prompt(self, state: TestJourneyState) -> str:
                return "You are a helpful assistant."
            
            def get_tool_schemas(self, state: TestJourneyState) -> list[ToolSchema]:
                return []
            
            def create_tools(self, state: TestJourneyState, ctx: AgentContext) -> TestJourneyTools:
                return TestJourneyTools(state=state)
            
            async def execute_tool(self, tools: TestJourneyTools, name: str, arguments: dict) -> str:
                method = getattr(tools, name)
                return await method(**arguments)
        
        agent = TestAgent(llm_client=mock_llm)
        result = await agent.run(test_context)
        
        assert result.response == "Welcome! What's your name?"
        assert result.state is not None
    
    @pytest.mark.asyncio
    async def test_run_with_tool_call(self, mock_llm, test_context):
        from tests.conftest import TestJourneyState, TestJourneyTools, TestStep
        
        # First: tool call
        mock_llm.add_response({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "collect_name",
                    "arguments": '{"name": "Alice"}',
                },
            }],
        })
        # Second: text response
        mock_llm.add_text_response("Great, Alice! What's your email?")
        
        class TestAgent(JourneyAgent[TestJourneyState, TestJourneyTools, TestStep]):
            @property
            def key(self) -> str:
                return "test-agent"
            
            def get_initial_state(self) -> TestJourneyState:
                return TestJourneyState()
            
            def get_system_prompt(self, state: TestJourneyState) -> str:
                return "Collect user info."
            
            def get_tool_schemas(self, state: TestJourneyState) -> list[ToolSchema]:
                return [
                    ToolSchemaBuilder("collect_name")
                    .description("Collect name")
                    .param("name", "string", required=True)
                    .build()
                ]
            
            def create_tools(self, state: TestJourneyState, ctx: AgentContext) -> TestJourneyTools:
                return TestJourneyTools(state=state)
            
            async def execute_tool(self, tools: TestJourneyTools, name: str, arguments: dict) -> str:
                method = getattr(tools, name)
                return await method(**arguments)
        
        agent = TestAgent(llm_client=mock_llm)
        result = await agent.run(test_context)
        
        # State should be updated
        assert result.state["name"] == "Alice"
        assert result.state["step"] == "collecting"
    
    @pytest.mark.asyncio
    async def test_closing_message_at_terminal(self, mock_llm, test_context):
        from tests.conftest import TestJourneyState, TestJourneyTools, TestStep
        
        # Set up context with closing message
        test_context.input_messages = [
            AgentMessage(role="user", content="Thank you!"),
        ]
        
        class TestAgent(JourneyAgent[TestJourneyState, TestJourneyTools, TestStep]):
            @property
            def key(self) -> str:
                return "test-agent"
            
            def get_initial_state(self) -> TestJourneyState:
                # Start in terminal state
                state = TestJourneyState()
                state.step = TestStep.COMPLETE
                return state
            
            def get_system_prompt(self, state: TestJourneyState) -> str:
                return "Done."
            
            def get_tool_schemas(self, state: TestJourneyState) -> list[ToolSchema]:
                return []
            
            def create_tools(self, state: TestJourneyState, ctx: AgentContext) -> TestJourneyTools:
                return TestJourneyTools(state=state)
            
            async def execute_tool(self, tools: TestJourneyTools, name: str, arguments: dict) -> str:
                return ""
        
        agent = TestAgent(llm_client=mock_llm)
        result = await agent.run(test_context)
        
        # Should return empty response without calling LLM
        assert result.response == ""
        assert mock_llm.call_count == 0
    
    def test_is_closing_message(self, mock_llm):
        from tests.conftest import TestJourneyState, TestJourneyTools, TestStep
        
        class TestAgent(JourneyAgent[TestJourneyState, TestJourneyTools, TestStep]):
            @property
            def key(self) -> str:
                return "test"
            
            def get_initial_state(self): return TestJourneyState()
            def get_system_prompt(self, state): return ""
            def get_tool_schemas(self, state): return []
            def create_tools(self, state, ctx): return TestJourneyTools(state=state)
            async def execute_tool(self, tools, name, args): return ""
        
        agent = TestAgent(llm_client=mock_llm)
        
        assert agent.is_closing_message("Thank you!") is True
        assert agent.is_closing_message("thanks") is True
        assert agent.is_closing_message("Goodbye") is True
        assert agent.is_closing_message("I have a question") is False
