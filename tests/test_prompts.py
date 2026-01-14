"""Tests for prompts module."""

import pytest
from enum import Enum

from agent_runtime_framework.prompts import (
    PromptManager,
    StepPromptMapping,
    PromptTemplate,
)
from agent_runtime_framework.prompts.manager import preference_enricher


class Step(str, Enum):
    WELCOME = "welcome"
    COLLECTING = "collecting"
    COMPLETE = "complete"


class TestPromptTemplate:
    """Tests for PromptTemplate."""
    
    def test_simple_render(self):
        template = PromptTemplate("Hello $name!")
        result = template.render(name="Alice")
        assert result == "Hello Alice!"
    
    def test_render_with_defaults(self):
        template = PromptTemplate(
            "Hello $name, you have $count items.",
            defaults={"count": 0},
        )
        result = template.render(name="Bob")
        assert result == "Hello Bob, you have 0 items."
    
    def test_render_override_defaults(self):
        template = PromptTemplate(
            "Status: $status",
            defaults={"status": "unknown"},
        )
        result = template.render(status="active")
        assert result == "Status: active"
    
    def test_safe_substitute_missing(self):
        template = PromptTemplate("Hello $name, your id is $id")
        result = template.render(name="Alice")
        assert result == "Hello Alice, your id is $id"


class TestStepPromptMapping:
    """Tests for StepPromptMapping."""
    
    def test_get_prompt(self):
        mapping = StepPromptMapping[Step](
            prompts={
                Step.WELCOME: "Welcome to our service!",
                Step.COLLECTING: "Please provide your information.",
            },
            default="How can I help?",
        )
        
        assert mapping.get(Step.WELCOME) == "Welcome to our service!"
        assert mapping.get(Step.COLLECTING) == "Please provide your information."
        assert mapping.get(Step.COMPLETE) == "How can I help?"  # Uses default
    
    def test_get_with_template(self):
        mapping = StepPromptMapping[Step](
            prompts={
                Step.COLLECTING: PromptTemplate("Please provide: $fields"),
            },
        )
        
        result = mapping.get(Step.COLLECTING, fields="name, email")
        assert result == "Please provide: name, email"
    
    def test_add_fluent(self):
        mapping = (
            StepPromptMapping[Step]()
            .add(Step.WELCOME, "Hello!")
            .add(Step.COMPLETE, "Goodbye!")
        )
        
        assert mapping.get(Step.WELCOME) == "Hello!"
        assert mapping.get(Step.COMPLETE) == "Goodbye!"


class TestPromptManager:
    """Tests for PromptManager."""
    
    def test_get_prompt(self):
        manager = PromptManager[Step](
            step_prompts=StepPromptMapping(
                prompts={Step.WELCOME: "Welcome!"},
                default="Default prompt",
            ),
        )
        
        assert manager.get_prompt(Step.WELCOME) == "Welcome!"
        assert manager.get_prompt(Step.COMPLETE) == "Default prompt"
    
    def test_enricher(self):
        manager = PromptManager[Step](
            step_prompts=StepPromptMapping(
                prompts={Step.WELCOME: "Welcome!"},
            ),
        )
        
        manager.add_enricher(
            lambda prompt, ctx: prompt + f"\n\nUser: {ctx.get('user_name', 'Guest')}"
        )
        
        result = manager.get_prompt(Step.WELCOME, context={"user_name": "Alice"})
        assert result == "Welcome!\n\nUser: Alice"
    
    def test_multiple_enrichers(self):
        manager = PromptManager[Step](
            step_prompts=StepPromptMapping(prompts={Step.WELCOME: "Base"}),
        )
        
        manager.add_enricher(lambda p, c: p + " [1]")
        manager.add_enricher(lambda p, c: p + " [2]")
        
        result = manager.get_prompt(Step.WELCOME)
        assert result == "Base [1] [2]"
    
    def test_set_prompt(self):
        manager = PromptManager[Step]()
        manager.set_prompt(Step.WELCOME, "New welcome!")
        
        assert manager.get_prompt(Step.WELCOME) == "New welcome!"
    
    def test_set_default(self):
        manager = PromptManager[Step]()
        manager.set_default("Default message")
        
        assert manager.get_prompt(Step.COMPLETE) == "Default message"


class TestPreferenceEnricher:
    """Tests for preference_enricher helper."""
    
    def test_adds_preference(self):
        enricher = preference_enricher("coverage_type", "\n\nPreferred: {value}")
        
        result = enricher("Base prompt", {"coverage_type": "premium"})
        assert result == "Base prompt\n\nPreferred: premium"
    
    def test_no_preference(self):
        enricher = preference_enricher("coverage_type")
        
        result = enricher("Base prompt", {})
        assert result == "Base prompt"
