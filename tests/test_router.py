"""Tests for router module."""

import pytest
from enum import Enum

from agent_runtime_framework.router import (
    IntentRouter,
    RouteDefinition,
    RouteResult,
    IntentDetector,
)


class Journey(str, Enum):
    QUOTE = "quote"
    CLAIM = "claim"
    SUPPORT = "support"


class TestRouteDefinition:
    """Tests for RouteDefinition."""
    
    def test_to_tool_schema(self):
        route = RouteDefinition(
            journey=Journey.QUOTE,
            name="route_to_quote",
            description="Get an insurance quote",
        )
        
        schema = route.to_tool_schema()
        assert schema.name == "route_to_quote"
        assert schema.description == "Get an insurance quote"
        assert schema.parameters == []


class TestIntentDetector:
    """Tests for IntentDetector."""
    
    def test_detect_single_keyword(self):
        detector = IntentDetector[Journey]()
        detector.add_keywords(Journey.QUOTE, ["quote", "price", "cost"])
        detector.add_keywords(Journey.CLAIM, ["claim", "damage", "incident"])
        
        result = detector.detect("I need a quote for my home")
        assert result is not None
        assert result.journey == Journey.QUOTE
    
    def test_detect_multiple_keywords(self):
        detector = IntentDetector[Journey]()
        detector.add_keywords(Journey.CLAIM, ["claim", "damage", "incident", "report"])
        
        result = detector.detect("I need to report damage and file a claim")
        assert result is not None
        assert result.journey == Journey.CLAIM
        assert result.confidence > 0.5
    
    def test_detect_no_match_with_default(self):
        detector = IntentDetector[Journey](default_journey=Journey.SUPPORT)
        detector.add_keywords(Journey.QUOTE, ["quote"])
        
        result = detector.detect("Hello, I have a question")
        assert result is not None
        assert result.journey == Journey.SUPPORT
        assert result.confidence == 0.5
    
    def test_detect_no_match_no_default(self):
        detector = IntentDetector[Journey]()
        detector.add_keywords(Journey.QUOTE, ["quote"])
        
        result = detector.detect("Hello, I have a question")
        assert result is None
    
    def test_fluent_add_keywords(self):
        detector = (
            IntentDetector[Journey]()
            .add_keywords(Journey.QUOTE, ["quote"])
            .add_keywords(Journey.CLAIM, ["claim"])
        )
        
        assert detector.detect("quote") is not None
        assert detector.detect("claim") is not None


class TestIntentRouter:
    """Tests for IntentRouter."""
    
    def test_add_route(self):
        router = IntentRouter[Journey]()
        router.add_route(RouteDefinition(
            journey=Journey.QUOTE,
            name="route_to_quote",
            description="Get a quote",
        ))
        
        assert "route_to_quote" in router.route_names
        assert Journey.QUOTE in router.journeys
    
    def test_add_routes(self):
        router = IntentRouter[Journey]()
        router.add_routes([
            RouteDefinition(Journey.QUOTE, "route_quote", "Quote"),
            RouteDefinition(Journey.CLAIM, "route_claim", "Claim"),
        ])
        
        assert len(router.route_names) == 2
    
    def test_get_tool_schemas(self):
        router = IntentRouter[Journey]()
        router.add_route(RouteDefinition(
            journey=Journey.QUOTE,
            name="route_to_quote",
            description="Get a quote",
        ))
        
        schemas = router.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "route_to_quote"
    
    def test_get_tool_schemas_openai(self):
        router = IntentRouter[Journey]()
        router.add_route(RouteDefinition(
            journey=Journey.QUOTE,
            name="route_to_quote",
            description="Get a quote",
        ))
        
        openai_schemas = router.get_tool_schemas_openai()
        assert len(openai_schemas) == 1
        assert openai_schemas[0]["function"]["name"] == "route_to_quote"
    
    def test_resolve_tool_call(self):
        router = IntentRouter[Journey]()
        router.add_route(RouteDefinition(
            journey=Journey.QUOTE,
            name="route_to_quote",
            description="Get a quote",
        ))
        
        journey = router.resolve_tool_call("route_to_quote")
        assert journey == Journey.QUOTE
        
        unknown = router.resolve_tool_call("unknown_tool")
        assert unknown is None
    
    def test_is_routing_tool(self):
        router = IntentRouter[Journey]()
        router.add_route(RouteDefinition(
            journey=Journey.QUOTE,
            name="route_to_quote",
            description="Get a quote",
        ))
        
        assert router.is_routing_tool("route_to_quote") is True
        assert router.is_routing_tool("other_tool") is False
    
    def test_fallback_detector(self):
        detector = IntentDetector[Journey](default_journey=Journey.SUPPORT)
        detector.add_keywords(Journey.QUOTE, ["quote"])
        
        router = IntentRouter[Journey](fallback_detector=detector)
        
        result = router.detect_fallback("I need a quote")
        assert result is not None
        assert result.journey == Journey.QUOTE
