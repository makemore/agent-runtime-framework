"""Tests for memory module."""

import pytest
from uuid import uuid4

from agent_runtime_framework.memory import (
    MemoryStore,
    InMemoryStore,
    StateStore,
    ConversationStore,
    MemoryContext,
    build_memory_context,
)
from agent_runtime_framework.memory.store import StoredMessage
from agent_runtime_framework.memory.context import MemoryManager


class TestInMemoryStore:
    """Tests for InMemoryStore."""
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        store = InMemoryStore[dict]()
        
        await store.set("key1", {"value": 1})
        result = await store.get("key1")
        
        assert result == {"value": 1}
    
    @pytest.mark.asyncio
    async def test_get_missing(self):
        store = InMemoryStore[str]()
        result = await store.get("missing")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryStore[str]()
        
        await store.set("key", "value")
        await store.delete("key")
        
        result = await store.get("key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_exists(self):
        store = InMemoryStore[str]()
        
        assert await store.exists("key") is False
        await store.set("key", "value")
        assert await store.exists("key") is True
    
    def test_clear(self):
        store = InMemoryStore[str]()
        store._data["key1"] = "value1"
        store._data["key2"] = "value2"
        
        store.clear()
        
        assert len(store._data) == 0


class TestStateStore:
    """Tests for StateStore."""
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self):
        store = StateStore()
        conv_id = uuid4()
        agent_key = "test-agent"
        
        state = {"step": "welcome", "name": "Alice"}
        await store.save_state(conv_id, agent_key, state)
        
        loaded = await store.load_state(conv_id, agent_key)
        assert loaded == state
    
    @pytest.mark.asyncio
    async def test_load_missing_state(self):
        store = StateStore()
        conv_id = uuid4()
        
        loaded = await store.load_state(conv_id, "unknown-agent")
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_delete_state(self):
        store = StateStore()
        conv_id = uuid4()
        agent_key = "test-agent"
        
        await store.save_state(conv_id, agent_key, {"data": "test"})
        await store.delete_state(conv_id, agent_key)
        
        loaded = await store.load_state(conv_id, agent_key)
        assert loaded is None


class TestConversationStore:
    """Tests for ConversationStore."""
    
    @pytest.mark.asyncio
    async def test_add_and_get_messages(self):
        store = ConversationStore()
        conv_id = uuid4()
        
        await store.add_message(conv_id, StoredMessage("user", "Hello"))
        await store.add_message(conv_id, StoredMessage("assistant", "Hi there!"))
        
        messages = await store.get_messages(conv_id)
        
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
    
    @pytest.mark.asyncio
    async def test_add_messages_batch(self):
        store = ConversationStore()
        conv_id = uuid4()
        
        await store.add_messages(conv_id, [
            StoredMessage("user", "First"),
            StoredMessage("assistant", "Second"),
            StoredMessage("user", "Third"),
        ])
        
        messages = await store.get_messages(conv_id)
        assert len(messages) == 3
    
    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self):
        store = ConversationStore()
        conv_id = uuid4()
        
        for i in range(10):
            await store.add_message(conv_id, StoredMessage("user", f"Message {i}"))
        
        messages = await store.get_messages(conv_id, limit=3)
        
        assert len(messages) == 3
        assert messages[0].content == "Message 7"  # Last 3 messages
    
    @pytest.mark.asyncio
    async def test_clear_conversation(self):
        store = ConversationStore()
        conv_id = uuid4()
        
        await store.add_message(conv_id, StoredMessage("user", "Hello"))
        await store.clear_conversation(conv_id)
        
        messages = await store.get_messages(conv_id)
        assert messages == []


class TestMemoryContext:
    """Tests for MemoryContext."""
    
    def test_has_state(self):
        ctx = MemoryContext()
        assert ctx.has_state() is False
        
        ctx.state = {"step": "welcome"}
        assert ctx.has_state() is True
    
    def test_get_message_dicts(self):
        ctx = MemoryContext(
            messages=[
                StoredMessage("user", "Hello"),
                StoredMessage("assistant", "Hi!"),
            ]
        )
        
        dicts = ctx.get_message_dicts()
        
        assert len(dicts) == 2
        assert dicts[0] == {"role": "user", "content": "Hello"}


class TestBuildMemoryContext:
    """Tests for build_memory_context."""
    
    @pytest.mark.asyncio
    async def test_build_with_stores(self):
        state_store = StateStore()
        conv_store = ConversationStore()
        
        conv_id = uuid4()
        agent_key = "test-agent"
        
        # Set up data
        await state_store.save_state(conv_id, agent_key, {"step": "collecting"})
        await conv_store.add_message(conv_id, StoredMessage("user", "Hello"))
        
        # Build context
        ctx = await build_memory_context(
            conversation_id=conv_id,
            agent_key=agent_key,
            state_store=state_store,
            conversation_store=conv_store,
        )
        
        assert ctx.state == {"step": "collecting"}
        assert len(ctx.messages) == 1
    
    @pytest.mark.asyncio
    async def test_build_without_stores(self):
        ctx = await build_memory_context(
            conversation_id=uuid4(),
            agent_key="test",
        )
        
        assert ctx.state is None
        assert ctx.messages == []


class TestMemoryManager:
    """Tests for MemoryManager."""
    
    @pytest.mark.asyncio
    async def test_load_and_save(self):
        manager = MemoryManager()
        conv_id = uuid4()
        agent_key = "test-agent"
        
        # Save state
        await manager.save_state(conv_id, agent_key, {"step": "welcome"})
        
        # Save messages
        await manager.save_messages(conv_id, [
            StoredMessage("user", "Hello"),
        ])
        
        # Load context
        ctx = await manager.load_context(conv_id, agent_key)
        
        assert ctx.state == {"step": "welcome"}
        assert len(ctx.messages) == 1
    
    @pytest.mark.asyncio
    async def test_clear_and_delete(self):
        manager = MemoryManager()
        conv_id = uuid4()
        agent_key = "test-agent"
        
        await manager.save_state(conv_id, agent_key, {"data": "test"})
        await manager.save_messages(conv_id, [StoredMessage("user", "Hi")])
        
        await manager.clear_conversation(conv_id)
        await manager.delete_state(conv_id, agent_key)
        
        ctx = await manager.load_context(conv_id, agent_key)
        assert ctx.state is None
        assert ctx.messages == []
