"""Tests for state module."""

import pytest
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID
from datetime import datetime

from agent_runtime_framework.state import BaseState, BaseJourneyState, StateSerializer


class TestStateSerializer:
    """Tests for StateSerializer."""
    
    def test_serialize_none(self):
        assert StateSerializer.serialize(None) is None
    
    def test_serialize_enum(self):
        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"
        
        assert StateSerializer.serialize(Color.RED) == "red"
    
    def test_serialize_uuid(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        assert StateSerializer.serialize(uid) == "12345678-1234-5678-1234-567812345678"
    
    def test_serialize_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert StateSerializer.serialize(dt) == "2024-01-15T10:30:00"
    
    def test_serialize_list(self):
        class Status(str, Enum):
            ACTIVE = "active"
        
        result = StateSerializer.serialize([Status.ACTIVE, "hello", 123])
        assert result == ["active", "hello", 123]
    
    def test_serialize_dict(self):
        class Status(str, Enum):
            ACTIVE = "active"
        
        result = StateSerializer.serialize({"status": Status.ACTIVE, "count": 5})
        assert result == {"status": "active", "count": 5}
    
    def test_deserialize_enum(self):
        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"
        
        assert StateSerializer.deserialize_enum("red", Color) == Color.RED
        assert StateSerializer.deserialize_enum(Color.BLUE, Color) == Color.BLUE


class TestBaseJourneyState:
    """Tests for BaseJourneyState."""
    
    def test_get_step(self, test_state):
        from tests.conftest import TestStep
        assert test_state.get_step() == TestStep.WELCOME
    
    def test_set_step(self, test_state):
        from tests.conftest import TestStep
        test_state.set_step(TestStep.COLLECTING)
        assert test_state.step == TestStep.COLLECTING
    
    def test_is_complete(self, test_state):
        from tests.conftest import TestStep
        assert not test_state.is_complete()
        test_state.step = TestStep.COMPLETE
        assert test_state.is_complete()
    
    def test_is_terminal(self, test_state):
        from tests.conftest import TestStep
        assert not test_state.is_terminal()
        test_state.step = TestStep.COMPLETE
        assert test_state.is_terminal()
    
    def test_to_dict(self, test_state):
        test_state.name = "Alice"
        test_state.email = "alice@example.com"
        
        result = test_state.to_dict()
        assert result == {
            "step": "welcome",
            "name": "Alice",
            "email": "alice@example.com",
            "processed": False,
        }
    
    def test_from_dict(self):
        from tests.conftest import TestJourneyState, TestStep
        
        data = {
            "step": "collecting",
            "name": "Bob",
            "email": "bob@example.com",
            "processed": True,
        }
        
        state = TestJourneyState.from_dict(data)
        assert state.step == TestStep.COLLECTING
        assert state.name == "Bob"
        assert state.email == "bob@example.com"
        assert state.processed is True
    
    def test_copy(self, test_state):
        test_state.name = "Original"
        
        copied = test_state.copy()
        copied.name = "Copied"
        
        assert test_state.name == "Original"
        assert copied.name == "Copied"
