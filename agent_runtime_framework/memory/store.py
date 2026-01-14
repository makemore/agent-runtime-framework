"""
Memory store abstractions.

These are framework-level abstractions that can be backed by
agent_runtime_core stores or other implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID


T = TypeVar("T")


class MemoryStore(ABC, Generic[T]):
    """
    Abstract base for memory stores.
    
    Memory stores provide persistence for agent data.
    Implementations can use in-memory storage, databases,
    or agent_runtime_core stores.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get a value by key."""
        ...
    
    @abstractmethod
    async def set(self, key: str, value: T) -> None:
        """Set a value by key."""
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a value by key."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return await self.get(key) is not None


class InMemoryStore(MemoryStore[T]):
    """
    Simple in-memory store for testing and development.
    
    Example:
        store = InMemoryStore[dict]()
        await store.set("user:123", {"name": "Alice"})
        user = await store.get("user:123")
    """
    
    def __init__(self):
        self._data: dict[str, T] = {}
    
    async def get(self, key: str) -> Optional[T]:
        return self._data.get(key)
    
    async def set(self, key: str, value: T) -> None:
        self._data[key] = value
    
    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
    
    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()


class StateStore(MemoryStore[dict[str, Any]]):
    """
    Store for agent state.
    
    Provides a typed interface for storing and retrieving
    agent state dictionaries.
    
    Example:
        store = StateStore(backend=my_redis_store)
        
        # Save state
        await store.save_state(conversation_id, agent_key, state.to_dict())
        
        # Load state
        state_dict = await store.load_state(conversation_id, agent_key)
    """
    
    def __init__(self, backend: Optional[MemoryStore[dict[str, Any]]] = None):
        self._backend = backend or InMemoryStore()
    
    def _make_key(self, conversation_id: UUID, agent_key: str) -> str:
        """Create a storage key."""
        return f"state:{conversation_id}:{agent_key}"
    
    async def get(self, key: str) -> Optional[dict[str, Any]]:
        return await self._backend.get(key)
    
    async def set(self, key: str, value: dict[str, Any]) -> None:
        await self._backend.set(key, value)
    
    async def delete(self, key: str) -> None:
        await self._backend.delete(key)
    
    async def save_state(
        self,
        conversation_id: UUID,
        agent_key: str,
        state: dict[str, Any],
    ) -> None:
        """Save agent state."""
        key = self._make_key(conversation_id, agent_key)
        await self.set(key, state)
    
    async def load_state(
        self,
        conversation_id: UUID,
        agent_key: str,
    ) -> Optional[dict[str, Any]]:
        """Load agent state."""
        key = self._make_key(conversation_id, agent_key)
        return await self.get(key)
    
    async def delete_state(
        self,
        conversation_id: UUID,
        agent_key: str,
    ) -> None:
        """Delete agent state."""
        key = self._make_key(conversation_id, agent_key)
        await self.delete(key)


@dataclass
class StoredMessage:
    """A message stored in conversation history."""
    role: str
    content: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class ConversationStore(MemoryStore[list[StoredMessage]]):
    """
    Store for conversation history.
    
    Provides methods for managing conversation message history.
    
    Example:
        store = ConversationStore()
        
        # Add messages
        await store.add_message(conv_id, StoredMessage("user", "Hello"))
        await store.add_message(conv_id, StoredMessage("assistant", "Hi!"))
        
        # Get history
        messages = await store.get_messages(conv_id)
    """
    
    def __init__(self, backend: Optional[MemoryStore[list[dict[str, Any]]]] = None):
        self._backend = backend or InMemoryStore()
    
    def _make_key(self, conversation_id: UUID) -> str:
        return f"conversation:{conversation_id}"
    
    async def get(self, key: str) -> Optional[list[StoredMessage]]:
        data = await self._backend.get(key)
        if data is None:
            return None
        return [
            StoredMessage(
                role=m["role"],
                content=m.get("content"),
                metadata=m.get("metadata", {}),
            )
            for m in data
        ]
    
    async def set(self, key: str, value: list[StoredMessage]) -> None:
        data = [m.to_dict() for m in value]
        await self._backend.set(key, data)
    
    async def delete(self, key: str) -> None:
        await self._backend.delete(key)
    
    async def get_messages(
        self,
        conversation_id: UUID,
        limit: Optional[int] = None,
    ) -> list[StoredMessage]:
        """Get messages for a conversation."""
        key = self._make_key(conversation_id)
        messages = await self.get(key) or []
        if limit:
            return messages[-limit:]
        return messages
    
    async def add_message(
        self,
        conversation_id: UUID,
        message: StoredMessage,
    ) -> None:
        """Add a message to conversation history."""
        key = self._make_key(conversation_id)
        messages = await self.get(key) or []
        messages.append(message)
        await self.set(key, messages)
    
    async def add_messages(
        self,
        conversation_id: UUID,
        messages: list[StoredMessage],
    ) -> None:
        """Add multiple messages to conversation history."""
        key = self._make_key(conversation_id)
        existing = await self.get(key) or []
        existing.extend(messages)
        await self.set(key, existing)
    
    async def clear_conversation(self, conversation_id: UUID) -> None:
        """Clear all messages for a conversation."""
        key = self._make_key(conversation_id)
        await self.delete(key)
