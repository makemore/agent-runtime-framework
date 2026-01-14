"""
Prompt management utilities.

Provides a structured way to manage system prompts that vary
based on journey step and context.
"""

from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any, Callable, Generic, Optional, TypeVar


StepT = TypeVar("StepT", bound=Enum)


@dataclass
class PromptTemplate:
    """
    A prompt template with optional variable substitution.
    
    Supports Python string.Template syntax ($variable or ${variable}).
    
    Example:
        template = PromptTemplate(
            "Hello $name! You are at step ${step}.",
            defaults={"name": "there"},
        )
        prompt = template.render(name="Alice", step="welcome")
        # "Hello Alice! You are at step welcome."
    """
    template: str
    defaults: dict[str, Any] = field(default_factory=dict)
    
    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables."""
        variables = {**self.defaults, **kwargs}
        return Template(self.template).safe_substitute(variables)


@dataclass
class StepPromptMapping(Generic[StepT]):
    """
    Mapping of journey steps to prompts.
    
    Provides a structured way to define prompts for each step
    in a journey, with support for default prompts and dynamic
    prompt generation.
    
    Example:
        class MyStep(str, Enum):
            WELCOME = "welcome"
            COLLECTING = "collecting"
            COMPLETE = "complete"
        
        mapping = StepPromptMapping[MyStep](
            prompts={
                MyStep.WELCOME: "Welcome! How can I help?",
                MyStep.COLLECTING: "Please provide the information.",
            },
            default="I'm here to help.",
        )
        
        prompt = mapping.get(MyStep.WELCOME)  # "Welcome! How can I help?"
        prompt = mapping.get(MyStep.COMPLETE)  # "I'm here to help." (default)
    """
    prompts: dict[StepT, str | PromptTemplate] = field(default_factory=dict)
    default: str | PromptTemplate = ""
    
    def get(self, step: StepT, **kwargs: Any) -> str:
        """
        Get the prompt for a step.
        
        Args:
            step: The journey step
            **kwargs: Variables for template substitution
            
        Returns:
            The rendered prompt string
        """
        prompt = self.prompts.get(step, self.default)
        if isinstance(prompt, PromptTemplate):
            return prompt.render(**kwargs)
        return prompt
    
    def add(self, step: StepT, prompt: str | PromptTemplate) -> "StepPromptMapping[StepT]":
        """Add a prompt for a step (fluent interface)."""
        self.prompts[step] = prompt
        return self


class PromptManager(Generic[StepT]):
    """
    Manager for journey prompts with context enrichment.
    
    Provides a higher-level interface for managing prompts that
    can be enriched with context (user preferences, state, etc.).
    
    Example:
        manager = PromptManager[MyStep](
            step_prompts=StepPromptMapping(
                prompts={
                    MyStep.WELCOME: "Welcome to our service!",
                    MyStep.COLLECTING: "Please provide: $missing_fields",
                },
                default="How can I help?",
            ),
        )
        
        # Add context enricher
        manager.add_enricher(lambda prompt, ctx: (
            prompt + f"\\n\\nUser: {ctx.get('user_name', 'Guest')}"
        ))
        
        # Get enriched prompt
        prompt = manager.get_prompt(
            MyStep.WELCOME,
            context={"user_name": "Alice"},
        )
    """
    
    def __init__(
        self,
        step_prompts: Optional[StepPromptMapping[StepT]] = None,
    ):
        """
        Initialize the prompt manager.
        
        Args:
            step_prompts: Mapping of steps to prompts
        """
        self.step_prompts = step_prompts or StepPromptMapping()
        self._enrichers: list[Callable[[str, dict[str, Any]], str]] = []
    
    def add_enricher(
        self,
        enricher: Callable[[str, dict[str, Any]], str],
    ) -> "PromptManager[StepT]":
        """
        Add a prompt enricher.
        
        Enrichers are called in order to modify the prompt
        based on context.
        
        Args:
            enricher: Function that takes (prompt, context) and returns modified prompt
            
        Returns:
            Self for chaining
        """
        self._enrichers.append(enricher)
        return self
    
    def get_prompt(
        self,
        step: StepT,
        context: Optional[dict[str, Any]] = None,
        **template_vars: Any,
    ) -> str:
        """
        Get the prompt for a step with context enrichment.
        
        Args:
            step: The journey step
            context: Context for enrichers
            **template_vars: Variables for template substitution
            
        Returns:
            The final prompt string
        """
        context = context or {}
        
        # Get base prompt
        prompt = self.step_prompts.get(step, **template_vars)
        
        # Apply enrichers
        for enricher in self._enrichers:
            prompt = enricher(prompt, context)
        
        return prompt
    
    def set_prompt(
        self,
        step: StepT,
        prompt: str | PromptTemplate,
    ) -> "PromptManager[StepT]":
        """Set the prompt for a step."""
        self.step_prompts.add(step, prompt)
        return self
    
    def set_default(self, prompt: str | PromptTemplate) -> "PromptManager[StepT]":
        """Set the default prompt."""
        self.step_prompts.default = prompt
        return self


def preference_enricher(
    preference_key: str,
    format_string: str = "\n\nNote: User prefers {value}.",
) -> Callable[[str, dict[str, Any]], str]:
    """
    Create an enricher that adds user preference context.
    
    Example:
        manager.add_enricher(preference_enricher(
            "coverage_type",
            "\\n\\nUser previously selected {value} coverage.",
        ))
    """
    def enricher(prompt: str, context: dict[str, Any]) -> str:
        value = context.get(preference_key)
        if value:
            return prompt + format_string.format(value=value)
        return prompt
    return enricher
