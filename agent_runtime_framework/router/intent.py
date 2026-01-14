"""
Intent detection and routing for multi-journey agents.

The router pattern allows an agent to detect user intent and
route to the appropriate journey or sub-agent.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

from agent_runtime_framework.tools.schema import ToolSchema, ToolSchemaBuilder


JourneyT = TypeVar("JourneyT", bound=Enum)


@dataclass
class RouteDefinition(Generic[JourneyT]):
    """
    Definition of a route to a journey.
    
    Attributes:
        journey: The journey type to route to
        name: Tool name for this route (e.g., "route_to_quote")
        description: Description for the LLM to understand when to use this route
        keywords: Optional keywords that might indicate this intent
    """
    journey: JourneyT
    name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    
    def to_tool_schema(self) -> ToolSchema:
        """Convert to a tool schema for the LLM."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[],
        )


@dataclass
class RouteResult(Generic[JourneyT]):
    """
    Result of intent detection/routing.
    
    Attributes:
        journey: The detected journey type
        confidence: Confidence score (0-1) if available
        metadata: Additional routing metadata
    """
    journey: JourneyT
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class IntentDetector(Generic[JourneyT]):
    """
    Simple keyword-based intent detector.
    
    For more sophisticated detection, use the LLM-based router.
    This is useful for quick pre-filtering or fallback detection.
    
    Example:
        detector = IntentDetector[MyJourney]()
        detector.add_keywords(MyJourney.QUOTE, ["quote", "price", "cost", "insurance"])
        detector.add_keywords(MyJourney.CLAIM, ["claim", "damage", "incident", "report"])
        
        result = detector.detect("I need to file a claim for water damage")
        # RouteResult(journey=MyJourney.CLAIM, confidence=0.8)
    """
    
    def __init__(self, default_journey: Optional[JourneyT] = None):
        """
        Initialize the detector.
        
        Args:
            default_journey: Journey to return if no keywords match
        """
        self._keywords: dict[JourneyT, list[str]] = {}
        self._default = default_journey
    
    def add_keywords(
        self,
        journey: JourneyT,
        keywords: list[str],
    ) -> "IntentDetector[JourneyT]":
        """Add keywords for a journey."""
        if journey not in self._keywords:
            self._keywords[journey] = []
        self._keywords[journey].extend(keywords)
        return self
    
    def detect(self, message: str) -> Optional[RouteResult[JourneyT]]:
        """
        Detect intent from a message.
        
        Returns the journey with the most keyword matches,
        or the default journey if no matches.
        """
        message_lower = message.lower()
        scores: dict[JourneyT, int] = {}
        
        for journey, keywords in self._keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in message_lower)
            if score > 0:
                scores[journey] = score
        
        if not scores:
            if self._default:
                return RouteResult(journey=self._default, confidence=0.5)
            return None
        
        # Return journey with highest score
        best_journey = max(scores, key=lambda j: scores[j])
        max_possible = len(self._keywords.get(best_journey, []))
        confidence = scores[best_journey] / max_possible if max_possible > 0 else 0.5
        
        return RouteResult(journey=best_journey, confidence=min(confidence, 1.0))


class IntentRouter(Generic[JourneyT]):
    """
    LLM-based intent router for multi-journey agents.
    
    Uses tool calling to let the LLM decide which journey to route to.
    This is more flexible than keyword matching and can understand
    nuanced user requests.
    
    Example:
        class MyJourney(str, Enum):
            QUOTE = "quote"
            CLAIM = "claim"
            SUPPORT = "support"
        
        router = IntentRouter[MyJourney]()
        router.add_route(RouteDefinition(
            journey=MyJourney.QUOTE,
            name="route_to_quote",
            description="Route to get a new insurance quote",
        ))
        router.add_route(RouteDefinition(
            journey=MyJourney.CLAIM,
            name="route_to_claim", 
            description="Route to file or check on a claim",
        ))
        
        # Get tool schemas for LLM
        tools = router.get_tool_schemas()
        
        # After LLM calls a routing tool, resolve the journey
        journey = router.resolve_tool_call("route_to_claim")
        # MyJourney.CLAIM
    
    Attributes:
        routes: Mapping of tool names to route definitions
        fallback_detector: Optional keyword-based fallback
    """
    
    def __init__(
        self,
        fallback_detector: Optional[IntentDetector[JourneyT]] = None,
    ):
        """
        Initialize the router.
        
        Args:
            fallback_detector: Optional keyword detector for fallback
        """
        self._routes: dict[str, RouteDefinition[JourneyT]] = {}
        self._fallback = fallback_detector
    
    def add_route(
        self,
        route: RouteDefinition[JourneyT],
    ) -> "IntentRouter[JourneyT]":
        """Add a route definition."""
        self._routes[route.name] = route
        return self
    
    def add_routes(
        self,
        routes: list[RouteDefinition[JourneyT]],
    ) -> "IntentRouter[JourneyT]":
        """Add multiple route definitions."""
        for route in routes:
            self.add_route(route)
        return self
    
    def get_tool_schemas(self) -> list[ToolSchema]:
        """Get tool schemas for all routes."""
        return [route.to_tool_schema() for route in self._routes.values()]
    
    def get_tool_schemas_openai(self) -> list[dict[str, Any]]:
        """Get tool schemas in OpenAI format."""
        return [schema.to_openai_format() for schema in self.get_tool_schemas()]
    
    def resolve_tool_call(self, tool_name: str) -> Optional[JourneyT]:
        """
        Resolve a tool call to a journey type.
        
        Args:
            tool_name: The name of the routing tool that was called
            
        Returns:
            The journey type, or None if not a routing tool
        """
        route = self._routes.get(tool_name)
        return route.journey if route else None
    
    def is_routing_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a routing tool."""
        return tool_name in self._routes
    
    def detect_fallback(self, message: str) -> Optional[RouteResult[JourneyT]]:
        """
        Use fallback detector if available.
        
        Useful when LLM doesn't call a routing tool.
        """
        if self._fallback:
            return self._fallback.detect(message)
        return None
    
    @property
    def route_names(self) -> list[str]:
        """Get all route tool names."""
        return list(self._routes.keys())
    
    @property
    def journeys(self) -> list[JourneyT]:
        """Get all journey types that can be routed to."""
        return [route.journey for route in self._routes.values()]


def create_reroute_tool() -> ToolSchema:
    """
    Create a standard reroute tool schema.
    
    This tool allows the agent to reset routing when the user
    changes their mind about what they want to do.
    """
    return ToolSchema(
        name="reroute_conversation",
        description=(
            "Use when the customer wants to do something DIFFERENT from the "
            "current journey. For example, if they're getting a quote but ask "
            "about filing a claim. This resets the conversation to detect their "
            "new intent."
        ),
        parameters=[],
    )
