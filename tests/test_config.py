"""Tests for configuration module."""

import os
import pytest

from agent_runtime_framework.config import (
    FrameworkConfig,
    get_config,
    set_config,
    configure,
    is_debug,
    should_swallow_exceptions,
)


class TestFrameworkConfig:
    """Tests for FrameworkConfig."""
    
    def test_default_config(self):
        config = FrameworkConfig()
        assert config.debug is False
        assert config.swallow_tool_exceptions is True
        assert config.log_level == "INFO"
    
    def test_enable_debug(self):
        config = FrameworkConfig()
        config.enable_debug()
        
        assert config.debug is True
        assert config.swallow_tool_exceptions is False
        assert config.log_level == "DEBUG"
    
    def test_enable_production(self):
        config = FrameworkConfig(debug=True, swallow_tool_exceptions=False)
        config.enable_production()
        
        assert config.debug is False
        assert config.swallow_tool_exceptions is True
        assert config.log_level == "INFO"
    
    def test_from_env_debug_enabled(self, monkeypatch):
        monkeypatch.setenv("AGENT_RUNTIME_DEBUG", "1")
        config = FrameworkConfig.from_env()
        
        assert config.debug is True
        assert config.swallow_tool_exceptions is False
        assert config.log_level == "DEBUG"
    
    def test_from_env_debug_disabled(self, monkeypatch):
        monkeypatch.setenv("AGENT_RUNTIME_DEBUG", "0")
        config = FrameworkConfig.from_env()
        
        assert config.debug is False
        assert config.swallow_tool_exceptions is True
        assert config.log_level == "INFO"
    
    def test_from_env_custom_log_level(self, monkeypatch):
        monkeypatch.setenv("AGENT_RUNTIME_LOG_LEVEL", "WARNING")
        config = FrameworkConfig.from_env()
        
        assert config.log_level == "WARNING"


class TestGlobalConfig:
    """Tests for global config functions."""
    
    def setup_method(self):
        """Reset global config before each test."""
        # Reset to default
        set_config(FrameworkConfig())
    
    def test_get_config(self):
        config = get_config()
        assert isinstance(config, FrameworkConfig)
    
    def test_set_config(self):
        new_config = FrameworkConfig(debug=True)
        set_config(new_config)
        
        config = get_config()
        assert config.debug is True
    
    def test_configure_debug(self):
        configure(debug=True)
        
        assert is_debug() is True
        assert should_swallow_exceptions() is False
    
    def test_configure_production(self):
        configure(debug=False)
        
        assert is_debug() is False
        assert should_swallow_exceptions() is True
    
    def test_configure_custom_swallow(self):
        # Enable debug but explicitly keep swallowing exceptions
        configure(debug=True, swallow_tool_exceptions=True)
        
        assert is_debug() is True
        assert should_swallow_exceptions() is True
    
    def test_configure_log_level(self):
        configure(log_level="ERROR")
        
        config = get_config()
        assert config.log_level == "ERROR"
    
    def test_is_debug(self):
        configure(debug=False)
        assert is_debug() is False
        
        configure(debug=True)
        assert is_debug() is True
    
    def test_should_swallow_exceptions(self):
        configure(swallow_tool_exceptions=True)
        assert should_swallow_exceptions() is True
        
        configure(swallow_tool_exceptions=False)
        assert should_swallow_exceptions() is False

