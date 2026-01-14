# Agent Runtime Framework

[![PyPI version](https://badge.fury.io/py/agent-runtime-framework.svg)](https://badge.fury.io/py/agent-runtime-framework)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python framework for building **journey-based conversational agents** with LLMs. Design multi-step conversational experiences where agents guide users through structured processes with state management, tool execution, and flexible routing.

## Recent Updates

| Version | Date | Changes |
|---------|------|---------|
| **0.1.0** | 2025-01-14 | Initial release - JourneyAgent, state management, Django adapter |

## 🎯 What is a Journey-Based Agent?

A journey-based agent guides users through a **multi-step process** (a "journey"), where:
- Each step has its own behavior, prompts, and available tools
- State is maintained throughout the conversation
- The agent transitions between steps based on user interactions
- Tools can modify state and trigger step transitions

**Perfect for:** Onboarding flows, data collection, quote generation, claim processing, multi-step forms, guided troubleshooting, and any structured conversational workflow.

## 🏗️ Architecture

The framework is built around several core abstractions:

```
┌─────────────────────────────────────────────────────────────┐
│                      JourneyAgent                           │
│  Orchestrates the conversation flow                         │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ State        │  │ Tools        │  │ Prompts      │
│ (Journey     │  │ (Actions     │  │ (Step-based  │
│  Progress)   │  │  & Logic)    │  │  Behavior)   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  ┌──────────────┐
                  │  Executor    │
                  │  (LLM Loop)  │
                  └──────────────┘
```

### Core Components

- **`JourneyAgent`**: Base class for building step-driven conversational agents
- **`BaseJourneyState`**: Manages journey state with step tracking and serialization
- **`BaseJourneyTools`**: Tools that can modify state and trigger transitions
- **`Executor`**: Handles the LLM interaction and tool execution loop
- **`IntentRouter`**: Routes user intents to different journeys
- **`MemoryStore`**: Persistence layer for state and conversation history
- **`PromptManager`**: Step-based prompt management with templating

## 🚀 Quick Start

### Installation

```bash
pip install agent_runtime_framework
```

### Basic Example

```python
from enum import Enum
from dataclasses import dataclass
from agent_runtime_framework import (
    JourneyAgent, BaseJourneyState, BaseJourneyTools,
    AgentContext, ToolSchema, ToolSchemaBuilder
)

# 1. Define your journey steps
class OnboardingStep(str, Enum):
    WELCOME = "welcome"
    COLLECT_NAME = "collect_name"
    COLLECT_EMAIL = "collect_email"
    COMPLETE = "complete"

# 2. Define your state
@dataclass
class OnboardingState(BaseJourneyState[OnboardingStep]):
    step: OnboardingStep = OnboardingStep.WELCOME
    name: str = ""
    email: str = ""
    
    def is_complete(self) -> bool:
        return self.step == OnboardingStep.COMPLETE
    
    def to_dict(self) -> dict:
        return {
            "step": self.step.value,
            "name": self.name,
            "email": self.email,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "OnboardingState":
        return cls(
            step=OnboardingStep(data.get("step", "welcome")),
            name=data.get("name", ""),
            email=data.get("email", ""),
        )

# 3. Define your tools
class OnboardingTools(BaseJourneyTools[OnboardingState]):
    async def save_name(self, name: str) -> str:
        self.state.name = name
        self.state.step = OnboardingStep.COLLECT_EMAIL
        await self._notify_state_change()
        return f"Great, {name}! Now, what's your email?"
    
    async def save_email(self, email: str) -> str:
        self.state.email = email
        self.state.step = OnboardingStep.COMPLETE
        await self._notify_state_change()
        return f"Perfect! You're all set, {self.state.name}!"

# 4. Define your agent
class OnboardingAgent(JourneyAgent[OnboardingState, OnboardingTools, OnboardingStep]):
    @property
    def key(self) -> str:
        return "onboarding-agent"
    
    def get_initial_state(self) -> OnboardingState:
        return OnboardingState()
    
    def get_system_prompt(self, state: OnboardingState) -> str:
        prompts = {
            OnboardingStep.WELCOME: "Welcome! Ask for the user's name.",
            OnboardingStep.COLLECT_NAME: "Collect the user's name using save_name tool.",
            OnboardingStep.COLLECT_EMAIL: "Collect the user's email using save_email tool.",
            OnboardingStep.COMPLETE: "Thank the user for completing onboarding.",
        }
        return prompts[state.step]
    
    def get_tool_schemas(self, state: OnboardingState) -> list[ToolSchema]:
        if state.step == OnboardingStep.COLLECT_NAME:
            return [
                ToolSchemaBuilder("save_name")
                .description("Save the user's name")
                .param("name", "string", "The user's name", required=True)
                .build()
            ]
        elif state.step == OnboardingStep.COLLECT_EMAIL:
            return [
                ToolSchemaBuilder("save_email")
                .description("Save the user's email")
                .param("email", "string", "The user's email", required=True)
                .build()
            ]
        return []
    
    def create_tools(self, state: OnboardingState, ctx: AgentContext) -> OnboardingTools:
        return OnboardingTools(state=state)
    
    async def execute_tool(self, tools: OnboardingTools, name: str, arguments: dict) -> str:
        method = getattr(tools, name)
        return await method(**arguments)

# 5. Run your agent
agent = OnboardingAgent(llm_client=my_llm_client)
result = await agent.run(context)
```

## 📚 Key Concepts

### Journey State

State is the heart of your agent. It tracks:
- **Current step** in the journey
- **Collected data** from the user
- **Progress indicators** and flags

State must be serializable (to/from dict) for persistence across conversation turns.

### Tools

Tools are the actions your agent can take. They:
- Execute business logic
- Modify state
- Trigger step transitions
- Return responses to the LLM

Tools inherit from `BaseJourneyTools` and have access to the current state.

### Step-Based Behavior

Each step in your journey can have:
- **Different system prompts** - Guide the LLM's behavior
- **Different available tools** - Control what actions are possible
- **Different validation logic** - Ensure data quality

This creates a structured, predictable conversation flow.

### Executor

The `Executor` handles the core LLM interaction loop:
1. Send messages + available tools to LLM
2. LLM responds with text or tool calls
3. Execute tool calls and add results to messages
4. Repeat until LLM returns text or max iterations reached

You typically don't use the Executor directly - `JourneyAgent` uses it internally.

## 🔧 Advanced Features

### Intent Routing

Route users to different journeys based on their intent:

```python
from agent_runtime_framework import IntentRouter, RouteDefinition

class Journey(str, Enum):
    QUOTE = "quote"
    CLAIM = "claim"
    SUPPORT = "support"

router = IntentRouter[Journey]()
router.add_route(RouteDefinition(
    journey=Journey.QUOTE,
    name="start_quote",
    description="Get a new insurance quote",
))
router.add_route(RouteDefinition(
    journey=Journey.CLAIM,
    name="file_claim",
    description="File or check on a claim",
))

# Get routing tools for LLM
tools = router.get_tool_schemas()

# After LLM calls a routing tool
journey = router.resolve_tool_call("start_quote")  # Journey.QUOTE
```

### Memory Management

Persist state and conversation history:

```python
from agent_runtime_framework import (
    StateStore, ConversationStore, MemoryManager
)

# Set up stores
state_store = StateStore()
conversation_store = ConversationStore()
manager = MemoryManager(state_store, conversation_store)

# Load context
context = await manager.load_context(conversation_id, "my-agent")

# Save after run
await manager.save_state(conversation_id, "my-agent", new_state)
await manager.save_messages(conversation_id, messages)
```

### Prompt Management

Organize prompts with templates and step mappings:

```python
from agent_runtime_framework import PromptTemplate, StepPromptMapping

# Simple mapping
prompts = StepPromptMapping[MyStep](
    prompts={
        MyStep.WELCOME: "Welcome! How can I help?",
        MyStep.COLLECTING: "Please provide your information.",
    },
    default="I'm here to assist you.",
)

# With templates
template = PromptTemplate(
    "Hello $name! You are at step ${step}.",
    defaults={"name": "there"},
)
prompts.add(MyStep.WELCOME, template)

# Render
prompt = prompts.get(MyStep.WELCOME, name="Alice", step="welcome")
```

### Execution Hooks

Observe and log execution events:

```python
from agent_runtime_framework import ExecutorHooks, LoggingHooks

class MyHooks(ExecutorHooks):
    async def on_tool_start(self, name: str, arguments: dict) -> None:
        print(f"🔧 Calling tool: {name}")

    async def on_tool_end(self, name: str, result: str) -> None:
        print(f"✅ Tool completed: {name}")

executor = Executor(
    llm_client=my_llm,
    hooks=MyHooks(),
)
```

## 🔌 Integration with agent_runtime_core

The framework is designed to work seamlessly with **`agent_runtime_core`**, a companion package that provides:
- **LLM client abstractions** - Unified interface for OpenAI, Anthropic, etc.
- **Production utilities** - Logging, monitoring, error handling
- **Configuration management** - Environment-based settings

When `agent_runtime_core` is installed, the framework can automatically use its LLM clients:

```python
from agent_runtime_core.llm import get_llm_client

# Framework automatically uses agent_runtime_core's LLM client
agent = MyAgent()  # No need to pass llm_client explicitly
```

The Django adapter also integrates with `agent_runtime_core` for production deployments.

## 🌐 Django Integration

Use the `DjangoRuntimeAdapter` to integrate with `django_agent_runtime`:

```python
from agent_runtime_framework.adapters import DjangoRuntimeAdapter

class MyDjangoAgent(DjangoRuntimeAdapter[MyState, MyTools, MyStep]):
    @property
    def key(self) -> str:
        return "my-agent"

    def get_initial_state(self) -> MyState:
        return MyState()

    def get_system_prompt(self, state: MyState) -> str:
        return PROMPTS[state.step]

    def get_tool_schemas(self, state: MyState) -> list[ToolSchema]:
        return TOOLS[state.step]

    def create_tools(self, state: MyState, ctx, backend_client) -> MyTools:
        return MyTools(state=state, backend_client=backend_client)

    async def execute_tool(self, tools: MyTools, name: str, args: dict) -> str:
        method = getattr(tools, name)
        return await method(**args)

# Register with Django
from django_agent_runtime.runtime.registry import register_runtime
register_runtime(MyDjangoAgent())
```

The adapter handles:
- Converting Django's `RunContext` to framework's `AgentContext`
- Using Django's checkpoint system for state persistence
- Emitting events through Django's event bus
- Returning results in Django's `RunResult` format

## 🧪 Testing

The framework includes comprehensive test utilities:

```bash
# Install dev dependencies
pip install agent_runtime_framework[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=agent_runtime_framework
```

Test fixtures are provided in `tests/conftest.py` for common testing scenarios.

## 📦 API Reference

### Core Classes

#### `JourneyAgent[StateT, ToolsT, StepT]`
Base class for journey-based agents.

**Must implement:**
- `key: str` - Unique agent identifier
- `get_initial_state() -> StateT` - Create initial state
- `get_system_prompt(state: StateT) -> str` - Get prompt for current step
- `get_tool_schemas(state: StateT) -> list[ToolSchema]` - Get available tools
- `create_tools(state: StateT, ctx: AgentContext) -> ToolsT` - Create tool instance
- `execute_tool(tools: ToolsT, name: str, args: dict) -> str` - Execute a tool

**Optional overrides:**
- `load_state(ctx: AgentContext) -> StateT | None` - Load persisted state
- `save_state(ctx: AgentContext, state: StateT) -> None` - Save state
- `is_terminal_state(state: StateT) -> bool` - Check if journey is complete

#### `BaseJourneyState[StepT]`
Base class for journey state with step tracking.

**Must implement:**
- `step: StepT` - Current step (as a field)
- `is_complete() -> bool` - Check if journey is complete
- `to_dict() -> dict` - Serialize to dictionary
- `from_dict(data: dict) -> Self` - Deserialize from dictionary

#### `BaseJourneyTools[StateT]`
Base class for journey tools that operate on state.

**Attributes:**
- `state: StateT` - The journey state
- `backend_client: Any` - Optional backend client
- `on_state_change: Callable` - Callback for state changes

**Methods:**
- `_notify_state_change()` - Call after modifying state

#### `ToolSchema`
Schema definition for LLM tools.

**Create with `ToolSchemaBuilder`:**
```python
schema = (
    ToolSchemaBuilder("my_tool")
    .description("What the tool does")
    .param("arg1", "string", "Description", required=True)
    .param("arg2", "number", "Description", required=False)
    .build()
)
```

#### `Executor`
Core execution loop for LLM + tool interactions.

```python
executor = Executor(
    llm_client=my_llm,
    tool_executor=MethodToolExecutor(tools),
    config=ExecutorConfig(max_iterations=10),
    hooks=MyHooks(),
)

result = await executor.run(
    messages=[{"role": "user", "content": "Hello"}],
    tools=[tool_schema],
    system_prompt="You are helpful.",
)
```

#### `IntentRouter[JourneyT]`
Routes user intents to different journeys.

```python
router = IntentRouter[MyJourney]()
router.add_route(RouteDefinition(
    journey=MyJourney.QUOTE,
    name="start_quote",
    description="Start a quote journey",
))

# Get tool schemas for LLM
tools = router.get_tool_schemas()

# Resolve tool call to journey
journey = router.resolve_tool_call("start_quote")
```

#### `MemoryStore[T]`
Abstract interface for persistence.

**Implementations:**
- `InMemoryStore[T]` - In-memory storage (for testing)
- `StateStore` - Specialized for agent state
- `ConversationStore` - Specialized for message history

#### `PromptManager[StepT]`
Manages step-based prompts with context enrichment.

```python
manager = PromptManager[MyStep](
    mapping=StepPromptMapping(...),
    context_enricher=lambda state: {"user_name": state.name},
)

prompt = manager.get_prompt(state)
```

## 🎨 Design Patterns

### Pattern 1: Linear Journey
Simple step-by-step flow (onboarding, data collection):
```
WELCOME → COLLECT_INFO → PROCESS → COMPLETE
```

### Pattern 2: Branching Journey
Different paths based on user input (quote with options):
```
WELCOME → CHOOSE_TYPE → [OPTION_A → ...] or [OPTION_B → ...]
```

### Pattern 3: Looping Journey
Repeat steps until condition met (multi-item cart):
```
START → ADD_ITEM → [MORE_ITEMS? → ADD_ITEM] → CHECKOUT
```

### Pattern 4: Error Recovery
Handle errors and retry:
```
STEP → [ERROR → RETRY → STEP] → NEXT_STEP
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Related Projects

- **`agent_runtime_core`** - Core utilities for production agent deployments
- **`django_agent_runtime`** - Django integration for agent runtimes

## 💡 Examples

Check out the `tests/` directory for complete working examples of:
- Basic journey agents
- State management
- Tool execution
- Memory persistence
- Intent routing
- Prompt management

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ for creating amazing conversational experiences**




