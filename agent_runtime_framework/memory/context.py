"""
Memory context utilities.

Helpers for building context from memory stores.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from agent_runtime_framework.memory.store import (
    StateStore,
    ConversationStore,
    StoredMessage,
)


@dataclass
class MemoryContext:
    """
    Context loaded from memory stores.
    
    Aggregates state and conversation history for an agent run.
    
    Attributes:
        state: Agent state dictionary
        messages: Conversation history
        metadata: Additional context metadata
    """
    state: Optional[dict[str, Any]] = None
    messages: list[StoredMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def has_state(self) -> bool:
        """Check if state was loaded."""
        return self.state is not None
    
    def get_message_dicts(self) -> list[dict[str, Any]]:
        """Get messages as dictionaries."""
        return [m.to_dict() for m in self.messages]


async def build_memory_context(
    conversation_id: UUID,
    agent_key: str,
    state_store: Optional[StateStore] = None,
    conversation_store: Optional[ConversationStore] = None,
    message_limit: Optional[int] = None,
) -> MemoryContext:
    """
    Build a memory context from stores.
    
    Loads state and conversation history for an agent run.
    
    Args:
        conversation_id: The conversation ID
        agent_key: The agent key for state lookup
        state_store: Optional state store
        conversation_store: Optional conversation store
        message_limit: Maximum messages to load
        
    Returns:
        MemoryContext with loaded data
    """
    context = MemoryContext()
    
    # Load state
    if state_store:
        context.state = await state_store.load_state(conversation_id, agent_key)
    
    # Load messages
    if conversation_store:
        context.messages = await conversation_store.get_messages(
            conversation_id,
            limit=message_limit,
        )
    
    return context


class MemoryManager:
    """
    High-level manager for agent memory.
    
    Coordinates state and conversation stores for agent runs.
    
    Example:
        manager = MemoryManager(
            state_store=StateStore(),
            conversation_store=ConversationStore(),
        )
        
        # Load context for a run
        context = await manager.load_context(conv_id, "my-agent")
        
        # Save after run
        await manager.save_state(conv_id, "my-agent", new_state)
        await manager.save_messages(conv_id, new_messages)
    """
    
    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        conversation_store: Optional[ConversationStore] = None,
    ):
        self.state_store = state_store or StateStore()
        self.conversation_store = conversation_store or ConversationStore()
    
    async def load_context(
        self,
        conversation_id: UUID,
        agent_key: str,
        message_limit: Optional[int] = None,
    ) -> MemoryContext:
        """Load memory context for an agent run."""
        return await build_memory_context(
            conversation_id=conversation_id,
            agent_key=agent_key,
            state_store=self.state_store,
            conversation_store=self.conversation_store,
            message_limit=message_limit,
        )
    
    async def save_state(
        self,
        conversation_id: UUID,
        agent_key: str,
        state: dict[str, Any],
    ) -> None:
        """Save agent state."""
        await self.state_store.save_state(conversation_id, agent_key, state)
    
    async def save_messages(
        self,
        conversation_id: UUID,
        messages: list[StoredMessage],
    ) -> None:
        """Save messages to conversation history."""
        await self.conversation_store.add_messages(conversation_id, messages)
    
    async def clear_conversation(self, conversation_id: UUID) -> None:
        """Clear conversation history."""
        await self.conversation_store.clear_conversation(conversation_id)
    
    async def delete_state(
        self,
        conversation_id: UUID,
        agent_key: str,
    ) -> None:
        """Delete agent state."""
        await self.state_store.delete_state(conversation_id, agent_key)
