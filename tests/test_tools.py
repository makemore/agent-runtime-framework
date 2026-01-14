"""Tests for tools module."""

import pytest
from typing import Any

from agent_runtime_framework.tools import (
    BaseTool,
    BaseJourneyTools,
    ToolResult,
    ToolSchema,
    ToolSchemaBuilder,
    ToolParameter,
)


class TestToolResult:
    """Tests for ToolResult."""
    
    def test_ok_result(self):
        result = ToolResult.ok("Success!", extra="data")
        assert result.content == "Success!"
        assert result.success is True
        assert result.error is None
        assert result.metadata == {"extra": "data"}
    
    def test_fail_result(self):
        result = ToolResult.fail("Something went wrong", code=500)
        assert result.content == "Something went wrong"
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.metadata == {"code": 500}


class TestToolParameter:
    """Tests for ToolParameter."""
    
    def test_basic_parameter(self):
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
        )
        
        schema = param.to_schema()
        assert schema == {
            "type": "string",
            "description": "Search query",
        }
    
    def test_enum_parameter(self):
        param = ToolParameter(
            name="priority",
            type="string",
            description="Priority level",
            enum=["low", "medium", "high"],
        )
        
        schema = param.to_schema()
        assert schema["enum"] == ["low", "medium", "high"]
    
    def test_array_parameter(self):
        param = ToolParameter(
            name="tags",
            type="array",
            description="List of tags",
            items={"type": "string"},
        )
        
        schema = param.to_schema()
        assert schema["items"] == {"type": "string"}
    
    def test_default_value(self):
        param = ToolParameter(
            name="limit",
            type="integer",
            default=10,
        )
        
        schema = param.to_schema()
        assert schema["default"] == 10


class TestToolSchema:
    """Tests for ToolSchema."""
    
    def test_to_openai_format(self):
        schema = ToolSchema(
            name="search",
            description="Search for items",
            parameters=[
                ToolParameter("query", "string", "Search query", required=True),
                ToolParameter("limit", "integer", "Max results"),
            ],
        )
        
        openai_format = schema.to_openai_format()
        
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "search"
        assert openai_format["function"]["description"] == "Search for items"
        assert openai_format["function"]["parameters"]["type"] == "object"
        assert "query" in openai_format["function"]["parameters"]["properties"]
        assert openai_format["function"]["parameters"]["required"] == ["query"]


class TestToolSchemaBuilder:
    """Tests for ToolSchemaBuilder."""
    
    def test_fluent_builder(self):
        schema = (
            ToolSchemaBuilder("create_quote")
            .description("Create an insurance quote")
            .param("property_id", "string", "Property ID", required=True)
            .param("coverage_type", "string", "Coverage type", enum=["basic", "standard", "premium"])
            .param("add_ons", "array", "Additional coverage", items={"type": "string"})
            .build()
        )
        
        assert schema.name == "create_quote"
        assert schema.description == "Create an insurance quote"
        assert len(schema.parameters) == 3
        assert schema.parameters[0].required is True
        assert schema.parameters[1].enum == ["basic", "standard", "premium"]
    
    def test_to_openai_format_shortcut(self):
        openai_format = (
            ToolSchemaBuilder("simple_tool")
            .description("A simple tool")
            .to_openai_format()
        )
        
        assert openai_format["function"]["name"] == "simple_tool"


class TestBaseJourneyTools:
    """Tests for BaseJourneyTools."""
    
    @pytest.mark.asyncio
    async def test_state_modification(self, test_tools, test_state):
        from tests.conftest import TestStep
        
        assert test_state.name == ""
        assert test_state.step == TestStep.WELCOME
        
        result = await test_tools.collect_name("Alice")
        
        assert "Alice" in result
        assert test_state.name == "Alice"
        assert test_state.step == TestStep.COLLECTING
    
    @pytest.mark.asyncio
    async def test_state_change_callback(self, test_state):
        from tests.conftest import TestJourneyTools
        
        callback_calls = []
        
        async def on_change(state):
            callback_calls.append(state.to_dict())
        
        tools = TestJourneyTools(state=test_state, on_state_change=on_change)
        
        await tools.collect_name("Bob")
        await tools.collect_email("bob@example.com")
        
        assert len(callback_calls) == 2
        assert callback_calls[0]["name"] == "Bob"
        assert callback_calls[1]["email"] == "bob@example.com"
    
    def test_require_backend_raises(self, test_tools):
        with pytest.raises(ValueError, match="requires a backend_client"):
            test_tools._require_backend()
    
    def test_require_backend_returns_client(self, test_state):
        from tests.conftest import TestJourneyTools
        
        mock_client = {"api": "mock"}
        tools = TestJourneyTools(state=test_state, backend_client=mock_client)
        
        assert tools._require_backend() == mock_client
