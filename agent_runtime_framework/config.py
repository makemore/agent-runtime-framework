"""
Configuration for agent runtime framework.

Provides global configuration for debug/production modes.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameworkConfig:
    """
    Global configuration for the framework.
    
    Attributes:
        debug: Enable debug mode (exceptions propagate, verbose logging)
        swallow_tool_exceptions: Whether to catch tool exceptions (False in debug mode)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    debug: bool = False
    swallow_tool_exceptions: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "FrameworkConfig":
        """
        Create config from environment variables.
        
        Environment variables:
            AGENT_RUNTIME_DEBUG: Set to "1", "true", or "yes" to enable debug mode
            AGENT_RUNTIME_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR)
        """
        debug_env = os.getenv("AGENT_RUNTIME_DEBUG", "").lower()
        debug = debug_env in ("1", "true", "yes", "on")
        
        log_level = os.getenv("AGENT_RUNTIME_LOG_LEVEL", "DEBUG" if debug else "INFO")
        
        return cls(
            debug=debug,
            swallow_tool_exceptions=not debug,  # Don't swallow in debug mode
            log_level=log_level,
        )
    
    def enable_debug(self) -> None:
        """Enable debug mode."""
        self.debug = True
        self.swallow_tool_exceptions = False
        self.log_level = "DEBUG"
    
    def enable_production(self) -> None:
        """Enable production mode."""
        self.debug = False
        self.swallow_tool_exceptions = True
        self.log_level = "INFO"


# Global config instance
_config: Optional[FrameworkConfig] = None


def get_config() -> FrameworkConfig:
    """
    Get the global framework configuration.
    
    Returns:
        The current framework configuration
    """
    global _config
    if _config is None:
        _config = FrameworkConfig.from_env()
    return _config


def set_config(config: FrameworkConfig) -> None:
    """
    Set the global framework configuration.
    
    Args:
        config: The configuration to use
    """
    global _config
    _config = config


def configure(
    debug: Optional[bool] = None,
    swallow_tool_exceptions: Optional[bool] = None,
    log_level: Optional[str] = None,
) -> None:
    """
    Configure the framework.
    
    Args:
        debug: Enable debug mode
        swallow_tool_exceptions: Whether to catch tool exceptions
        log_level: Logging level
    
    Example:
        # Enable debug mode
        configure(debug=True)
        
        # Production mode with custom settings
        configure(debug=False, log_level="WARNING")
    """
    config = get_config()
    
    if debug is not None:
        config.debug = debug
        # Auto-adjust swallow_tool_exceptions if not explicitly set
        if swallow_tool_exceptions is None:
            config.swallow_tool_exceptions = not debug
    
    if swallow_tool_exceptions is not None:
        config.swallow_tool_exceptions = swallow_tool_exceptions
    
    if log_level is not None:
        config.log_level = log_level


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return get_config().debug


def should_swallow_exceptions() -> bool:
    """Check if exceptions should be swallowed."""
    return get_config().swallow_tool_exceptions

