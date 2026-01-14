"""Tests for executor module."""

import pytest
from typing import Any, Optional

from agent_runtime_framework.executor import (
    Executor,
    ExecutorConfig,
    ExecutionResult,
    ExecutorHooks,
    LoggingHooks,
)
from agent_runtime_framework.executor.loop import (
    CallableToolExecutor,
    MethodToolExecutor,
)
from agent_runtime_framework.executor.hooks import CompositeHooks
from agent_runtime_framework.tools import ToolSchema, ToolSchemaBuilder


class TestCallableToolExecutor:
    """Tests for CallableToolExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        executor = CallableToolExecutor({"greet": greet})
        result = await executor.execute("greet", {"name": "Alice"})
        assert result == "Hello, Alice!"
    
    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        async def async_greet(name: str) -> str:
            return f"Hello, {name}!"
        
        executor = CallableToolExecutor({"greet": async_greet})
        result = await executor.execute("greet", {"name": "Bob"})
        assert result == "Hello, Bob!"
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        executor = CallableToolExecutor({})
        result = await executor.execute("unknown", {})
        assert "Unknown tool" in result
    
    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        def failing_tool():
            raise ValueError("Something went wrong")
        
        executor = CallableToolExecutor({"fail": failing_tool})
        result = await executor.execute("fail", {})
        assert "Error" in result
        assert "Something went wrong" in result


class TestMethodToolExecutor:
    """Tests for MethodToolExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_method(self):
        class Tools:
            async def search(self, query: str) -> str:
                return f"Results for: {query}"
        
        executor = MethodToolExecutor(Tools())
        result = await executor.execute("search", {"query": "test"})
        assert result == "Results for: test"
    
    @pytest.mark.asyncio
    async def test_execute_unknown_method(self):
        class Tools:
            pass
        
        executor = MethodToolExecutor(Tools())
        result = await executor.execute("unknown", {})
        assert "Unknown tool" in result


class TestExecutor:
    """Tests for Executor."""
    
    @pytest.mark.asyncio
    async def test_simple_text_response(self, mock_llm):
        mock_llm.add_text_response("Hello! How can I help?")
        
        executor = Executor(llm_client=mock_llm)
        result = await executor.run(
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        assert result.response == "Hello! How can I help?"
        assert result.stopped_reason == "text_response"
        assert result.tool_calls_made == 0
    
    @pytest.mark.asyncio
    async def test_tool_call_and_response(self, mock_llm):
        # First response: tool call
        mock_llm.add_response({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "London"}',
                },
            }],
        })
        # Second response: text after tool result
        mock_llm.add_text_response("The weather in London is sunny!")
        
        async def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 22°C"
        
        tool_executor = CallableToolExecutor({"get_weather": get_weather})
        
        executor = Executor(
            llm_client=mock_llm,
            tool_executor=tool_executor,
        )
        
        tools = [
            ToolSchemaBuilder("get_weather")
            .description("Get weather")
            .param("location", "string", required=True)
            .build()
        ]
        
        result = await executor.run(
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            tools=tools,
        )
        
        assert result.tool_calls_made == 1
        assert "sunny" in result.response.lower() or "Sunny" in result.response
    
    @pytest.mark.asyncio
    async def test_terminal_tool_stops_loop(self, mock_llm):
        mock_llm.add_response({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "complete_order",
                    "arguments": "{}",
                },
            }],
        })
        
        async def complete_order() -> str:
            return "Order completed!"
        
        tool_executor = CallableToolExecutor({"complete_order": complete_order})
        
        executor = Executor(
            llm_client=mock_llm,
            tool_executor=tool_executor,
            config=ExecutorConfig(terminal_tools={"complete_order"}),
        )
        
        result = await executor.run(
            messages=[{"role": "user", "content": "Complete my order"}],
        )
        
        assert result.stopped_reason == "terminal_tool:complete_order"
        assert result.response == "Order completed!"
    
    @pytest.mark.asyncio
    async def test_max_iterations(self, mock_llm):
        # Add many tool call responses
        for i in range(15):
            mock_llm.add_response({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "loop_tool",
                        "arguments": "{}",
                    },
                }],
            })
        
        async def loop_tool() -> str:
            return "Looping..."
        
        tool_executor = CallableToolExecutor({"loop_tool": loop_tool})
        
        executor = Executor(
            llm_client=mock_llm,
            tool_executor=tool_executor,
            config=ExecutorConfig(max_iterations=3),
        )
        
        result = await executor.run(
            messages=[{"role": "user", "content": "Loop"}],
        )
        
        assert result.stopped_reason == "max_iterations"
        assert result.tool_calls_made == 3
    
    @pytest.mark.asyncio
    async def test_system_prompt(self, mock_llm):
        mock_llm.add_text_response("Response")
        
        executor = Executor(llm_client=mock_llm)
        await executor.run(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are a helpful assistant.",
        )
        
        # Check that system prompt was prepended
        call = mock_llm.calls[0]
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][0]["content"] == "You are a helpful assistant."


class TestExecutorHooks:
    """Tests for ExecutorHooks."""
    
    @pytest.mark.asyncio
    async def test_hooks_called(self, mock_llm):
        mock_llm.add_response({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }],
        })
        mock_llm.add_text_response("Done")
        
        events = []
        
        class TrackingHooks(ExecutorHooks):
            async def on_iteration_start(self, iteration: int):
                events.append(("iteration_start", iteration))
            
            async def on_llm_call(self, messages, tools):
                events.append(("llm_call", len(messages)))
            
            async def on_tool_calls(self, tool_calls):
                events.append(("tool_calls", len(tool_calls)))
            
            async def on_tool_start(self, name, arguments):
                events.append(("tool_start", name))
            
            async def on_tool_end(self, name, result):
                events.append(("tool_end", name))
            
            async def on_complete(self, response, stopped_reason):
                events.append(("complete", stopped_reason))
        
        async def test_tool() -> str:
            return "Result"
        
        executor = Executor(
            llm_client=mock_llm,
            tool_executor=CallableToolExecutor({"test_tool": test_tool}),
            hooks=TrackingHooks(),
        )
        
        await executor.run(messages=[{"role": "user", "content": "Test"}])
        
        event_types = [e[0] for e in events]
        assert "iteration_start" in event_types
        assert "llm_call" in event_types
        assert "tool_calls" in event_types
        assert "tool_start" in event_types
        assert "tool_end" in event_types
        assert "complete" in event_types


class TestCompositeHooks:
    """Tests for CompositeHooks."""
    
    @pytest.mark.asyncio
    async def test_calls_all_hooks(self):
        calls = []
        
        class Hook1(ExecutorHooks):
            async def on_complete(self, response, reason):
                calls.append("hook1")
        
        class Hook2(ExecutorHooks):
            async def on_complete(self, response, reason):
                calls.append("hook2")
        
        composite = CompositeHooks([Hook1(), Hook2()])
        await composite.on_complete("response", "reason")
        
        assert calls == ["hook1", "hook2"]
