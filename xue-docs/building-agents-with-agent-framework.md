# Building Agents with Microsoft Agent Framework: A Deep Technical Guide

> This document provides an in-depth technical walkthrough of building, testing, deploying, and invoking AI agents using the [Microsoft Agent Framework](../README.md). It covers both **Python** and **.NET** implementations with real code snippets and links to source files.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Writing Agent Code](#2-writing-agent-code)
   - [Simple Chat Agent](#21-simple-chat-agent)
   - [Agent with Tools](#22-agent-with-tools)
   - [Agent with Memory](#23-agent-with-memory)
   - [Multi-Agent Workflows](#24-multi-agent-workflows)
3. [Deep Dive: Tools](#3-deep-dive-tools)
   - [Tool SDK and Definition](#31-tool-sdk-and-definition)
   - [MCP Protocol Integration](#32-mcp-protocol-integration)
   - [Tool Invocation Flow](#33-tool-invocation-flow-technical-internals)
   - [Authentication Between Agent and Tools](#34-authentication-between-agent-and-tools)
4. [Deep Dive: Memory](#4-deep-dive-memory)
   - [Short-Term Memory (Session State)](#41-short-term-memory-session-state)
   - [Conversation History Providers](#42-conversation-history-providers)
   - [Long-Term Semantic Memory (Mem0)](#43-long-term-semantic-memory-mem0)
   - [Persistent Storage (Redis)](#44-persistent-storage-redis)
   - [RAG / Semantic Search (Azure AI Search)](#45-rag--semantic-search-azure-ai-search)
   - [Cloud-Native Memory (Cosmos DB, Foundry Memory)](#46-cloud-native-memory-cosmos-db-foundry-memory)
   - [Holistic Multi-Memory Agent Example](#47-holistic-multi-memory-agent-example)
5. [Testing Agents](#5-testing-agents)
6. [Deploying Agents](#6-deploying-agents)
   - [Azure Functions (Python)](#61-azure-functions-python)
   - [OpenAI-Compatible HTTP Endpoints (.NET)](#62-openai-compatible-http-endpoints-net)
   - [A2A Protocol (Agent-to-Agent)](#63-a2a-protocol-agent-to-agent)
   - [AG-UI Protocol (Web UI Streaming)](#64-ag-ui-protocol-web-ui-streaming)
7. [Invoking Agents](#7-invoking-agents)
8. [Other Important Dimensions](#8-other-important-dimensions)
   - [Observability and Telemetry](#81-observability-and-telemetry)
   - [Middleware Pipeline](#82-middleware-pipeline)
   - [Declarative Agent Definitions](#83-declarative-agent-definitions)
   - [Multi-LLM Provider Support](#84-multi-llm-provider-support)
   - [Human-in-the-Loop](#85-human-in-the-loop)
   - [Checkpointing and Time-Travel](#86-checkpointing-and-time-travel)

---

## 1. Architecture Overview

The Microsoft Agent Framework is a **multi-language** (Python + .NET) framework organized around a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hosting / Deployment Layer                    │
│  Azure Functions │ ASP.NET Core │ A2A Protocol │ AG-UI / DevUI  │
├─────────────────────────────────────────────────────────────────┤
│                    Orchestration / Workflows                     │
│     Sequential │ Group Chat │ Handoff │ Concurrent │ Declarative │
├─────────────────────────────────────────────────────────────────┤
│                         Agent Layer                              │
│  Agent ─── ChatClient ─── Tools ─── Context Providers ─── Middleware │
├─────────────────────────────────────────────────────────────────┤
│                     Memory / State Layer                         │
│   Session State │ Mem0 │ Redis │ Azure AI Search │ Cosmos DB     │
├─────────────────────────────────────────────────────────────────┤
│                      Provider Layer                              │
│  Azure OpenAI │ OpenAI │ Anthropic │ Bedrock │ Ollama │ Foundry  │
└─────────────────────────────────────────────────────────────────┘
```

**Core Types (Python)**:
- [`Agent`](../python/packages/core/agent_framework/_agents.py) — The primary agent class. Wraps a chat client with tools, instructions, and context providers.
- [`FunctionTool`](../python/packages/core/agent_framework/_tools.py) — Represents a callable tool with auto-validated JSON schema.
- [`AgentSession`](../python/packages/core/agent_framework/_sessions.py) — Lightweight conversation state container.
- [`BaseContextProvider`](../python/packages/core/agent_framework/_sessions.py) — Hook for injecting context (memory, instructions, tools) before/after each agent invocation.
- [`Workflow`](../python/packages/core/agent_framework/_workflows/) — Graph-based orchestration engine.

**Core Types (.NET)**:
- [`AIAgent`](../dotnet/src/Microsoft.Agents.AI.Abstractions/AIAgent.cs) — Abstract base class for all agents.
- [`ChatClientAgent`](../dotnet/src/Microsoft.Agents.AI/ChatClient/ChatClientAgent.cs) — Agent implementation using `IChatClient`.
- [`AITool` / `AIFunction`](../dotnet/src/Microsoft.Agents.AI.Abstractions/) — Tool abstractions from `Microsoft.Extensions.AI`.
- [`AgentSession`](../dotnet/src/Microsoft.Agents.AI.Abstractions/AgentSession.cs) — Conversation state with serialization support.
- [`AIContextProvider`](../dotnet/src/Microsoft.Agents.AI.Abstractions/AIContextProvider.cs) — Context injection mechanism.

---

## 2. Writing Agent Code

### 2.1 Simple Chat Agent

The simplest agent wraps an LLM client with a name and instructions.

**Python** ([`01_hello_agent.py`](../python/samples/01-get-started/01_hello_agent.py)):

```python
import asyncio
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential

async def main():
    # Initialize the Azure OpenAI Responses client
    # Reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME,
    # AZURE_OPENAI_API_VERSION from environment variables
    client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    # Create an agent: wraps the client with identity + system prompt
    agent = client.as_agent(
        name="HaikuBot",
        instructions="You are an upbeat assistant that writes beautifully.",
    )

    # Single-turn invocation (returns AgentResponse)
    result = await agent.run("Write a haiku about Microsoft Agent Framework.")
    print(result)

    # Streaming invocation (yields AgentResponseUpdate chunks)
    async for chunk in agent.run("Tell me a fun fact.", stream=True):
        if chunk.text:
            print(chunk.text, end="", flush=True)

asyncio.run(main())
```

**.NET** (from [README quickstart](../README.md)):

```csharp
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Responses;

// Create an agent from an OpenAI Responses client
var agent = new OpenAIClient("<apikey>")
    .GetResponsesClient("gpt-4o-mini")
    .AsAIAgent(name: "HaikuBot", instructions: "You are an upbeat assistant.");

// Single-turn invocation
Console.WriteLine(await agent.RunAsync("Write a haiku about Agent Framework."));
```

**What happens under the hood:**
1. `as_agent()` / `AsAIAgent()` creates an `Agent` / `ChatClientAgent` that stores a reference to the underlying chat client, the system instructions, and an empty tool list.
2. `agent.run()` / `agent.RunAsync()` sends the user message plus the system instructions to the LLM via the chat client's `get_response()` / `GetResponseAsync()` method.
3. The LLM response is wrapped in an `AgentResponse` (Python) / `AgentResponse` (.NET) that contains the text, finish reason, and any tool calls.
4. When streaming, the client uses Server-Sent Events (SSE) to deliver incremental tokens.

---

### 2.2 Agent with Tools

Tools give agents the ability to **take actions** — call APIs, query databases, run code, etc.

**Python** ([`02_add_tools.py`](../python/samples/01-get-started/02_add_tools.py)):

```python
from agent_framework import tool
from typing import Annotated
from pydantic import Field
from random import randint

@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="The city and state, e.g. Seattle, WA")]
) -> str:
    """Get the current weather for a location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"Weather in {location}: {conditions[randint(0, 3)]}, {randint(50, 90)}°F"

agent = client.as_agent(
    name="WeatherAgent",
    instructions="You are a helpful weather assistant. Use the get_weather tool.",
    tools=[get_weather],  # Attach tools to the agent
)

result = await agent.run("What's the weather in Seattle?")
# The agent will: 1) decide to call get_weather("Seattle, WA")
#                 2) execute the function locally
#                 3) send the result back to the LLM
#                 4) return the LLM's final natural language response
```

**.NET** ([`Agent_Step03_UsingFunctionTools`](../dotnet/samples/GettingStarted/Agents/Agent_Step03_UsingFunctionTools/)):

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

// Define a tool as a static method with description attributes
[Description("Get current weather for a location")]
static string GetWeather(
    [Description("City and state")] string location)
{
    return $"Weather in {location}: sunny, 72°F";
}

// Create agent with tools
var agent = chatClient.AsAIAgent(
    name: "WeatherAgent",
    instructions: "Use the GetWeather tool when asked about weather.",
    tools: [AIFunctionFactory.Create(GetWeather)]
);

Console.WriteLine(await agent.RunAsync("What's the weather in Seattle?"));
```

**More tool examples**: See [`python/samples/02-agents/tools/`](../python/samples/02-agents/tools/) for advanced patterns including:
- [Approval workflows](../python/samples/02-agents/tools/function_tool_with_approval.py) — require human approval before tool execution
- [Session injection](../python/samples/02-agents/tools/function_tool_with_session_injection.py) — pass session state into tool functions
- [Error recovery](../python/samples/02-agents/tools/function_tool_recover_from_failures.py) — handle tool failures gracefully
- [Invocation limits](../python/samples/02-agents/tools/function_tool_with_max_invocations.py) — cap the number of tool calls per request

---

### 2.3 Agent with Memory

Memory lets agents retain information across conversation turns and even across sessions.

**Python** ([`04_memory.py`](../python/samples/01-get-started/04_memory.py)):

```python
from agent_framework import AgentSession, BaseContextProvider, SessionContext

class UserMemoryProvider(BaseContextProvider):
    """Remembers user's name across turns using session state."""
    DEFAULT_SOURCE_ID = "user_memory"

    async def before_run(self, *, agent, session, context, state):
        # 'state' is a per-provider dict persisted in session.state
        user_name = state.get("user_name")
        if user_name:
            context.extend_instructions(
                self.source_id,
                f"The user's name is {user_name}. Address them by name.",
            )

    async def after_run(self, *, agent, session, context, state):
        # Extract user info from the conversation
        for msg in context.input_messages:
            text = msg.text.lower() if msg.text else ""
            if "my name is" in text:
                name = text.split("my name is")[-1].strip().split()[0].title()
                state["user_name"] = name

# Create agent with memory provider
agent = client.as_agent(
    name="MemoryAgent",
    instructions="You are a friendly assistant.",
    context_providers=[UserMemoryProvider()],
)

# Multi-turn conversation with persistent session
session = agent.create_session()
await agent.run("My name is Alice", session=session)  # Stores name
await agent.run("What is 2 + 2?", session=session)    # Remembers "Alice"
```

**How session state works under the hood:**
1. `AgentSession` holds a `state: dict[str, Any]` that persists across `run()` calls.
2. Each `BaseContextProvider` gets a scoped slice of that state (keyed by `source_id`).
3. `before_run()` is called before the LLM invocation — providers inject context messages, instructions, or tools.
4. `after_run()` is called after — providers can extract and store information.
5. The session is serializable via `session.to_dict()` / `AgentSession.from_dict()` for persistence.

---

### 2.4 Multi-Agent Workflows

The framework provides a **graph-based workflow engine** for orchestrating multiple agents.

**Handoff Pattern** — agents dynamically route to each other ([`handoff_simple.py`](../python/samples/03-workflows/orchestrations/handoff_simple.py)):

```python
from agent_framework.orchestrations import HandoffBuilder

# Define specialist agents
triage = client.as_agent(name="triage_agent", instructions="Route to the right specialist.")
refund_agent = client.as_agent(name="refund_agent", instructions="Process refunds.", tools=[process_refund])
order_agent = client.as_agent(name="order_agent", instructions="Handle order inquiries.", tools=[check_order])

# Build a handoff workflow — agents call "handoff_to_<name>" tools to transfer
workflow = (
    HandoffBuilder(
        name="customer_support",
        participants=[triage, refund_agent, order_agent],
    )
    .with_start_agent(triage)
    .build()
)

events = await workflow.run("I received a damaged item and want a refund")
# Flow: triage → handoff_to_refund_agent → refund_agent processes → output
```

**Group Chat Pattern** — a central orchestrator selects the next speaker ([`group_chat_simple_selector.py`](../python/samples/03-workflows/orchestrations/group_chat_simple_selector.py)):

```python
from agent_framework.orchestrations import GroupChatBuilder

analyst = client.as_agent(name="analyst", instructions="Analyze data and provide insights.")
writer = client.as_agent(name="writer", instructions="Write polished reports.")

workflow = (
    GroupChatBuilder(
        name="report_team",
        participants=[analyst, writer],
        orchestrator=client,  # LLM decides who speaks next
    )
    .build()
)

events = await workflow.run("Create a quarterly report on AI trends")
```

**More workflow patterns**: See [`python/samples/03-workflows/`](../python/samples/03-workflows/) for:
- [Sequential agents](../python/samples/03-workflows/orchestrations/sequential_agents.py)
- [Concurrent fan-out/fan-in](../python/samples/03-workflows/orchestrations/concurrent_agents.py)
- [Checkpoint & resume](../python/samples/03-workflows/checkpoint/)
- [Human-in-the-loop](../python/samples/03-workflows/human-in-the-loop/)
- [Declarative YAML workflows](../python/samples/03-workflows/declarative/)

---

## 3. Deep Dive: Tools

### 3.1 Tool SDK and Definition

**Python SDK**: Tools are defined using the `@tool` decorator or the `FunctionTool` class from [`agent_framework._tools`](../python/packages/core/agent_framework/_tools.py).

```python
from agent_framework import tool, FunctionTool
from typing import Annotated
from pydantic import Field

# Method 1: @tool decorator (recommended)
@tool(approval_mode="never_require")
def search_database(
    query: Annotated[str, Field(description="SQL-like query string")],
    limit: Annotated[int, Field(description="Max results", default=10)]
) -> str:
    """Search the product database."""
    return f"Found {limit} results for: {query}"

# Method 2: FunctionTool constructor (for dynamic tools)
dynamic_tool = FunctionTool(
    func=my_function,
    name="my_tool",
    description="Does something useful",
    input_model={"type": "object", "properties": {"x": {"type": "string"}}},
)

# Method 3: Declaration-only tool (no implementation — used for hosted/remote tools)
declaration_only = FunctionTool(
    func=None,  # No local implementation
    name="remote_api_call",
    description="Calls a remote API",
    input_model=MyInputSchema,
)
```

**.NET SDK**: Tools use `AIFunction` from `Microsoft.Extensions.AI`:

```csharp
using Microsoft.Extensions.AI;

// Method 1: Factory from static method
AITool weatherTool = AIFunctionFactory.Create(GetWeather);

// Method 2: From delegate
AITool mathTool = AIFunctionFactory.Create(
    (int a, int b) => a + b,
    name: "add",
    description: "Add two numbers"
);
```

**JSON Schema Generation**: When you attach a tool to an agent, the framework automatically generates a [JSON Schema](https://json-schema.org/) from the function signature. This schema is sent to the LLM as part of the chat request so the model knows what tools are available and what arguments they expect.

For example, the `get_weather` Python tool above generates:
```json
{
  "name": "get_weather",
  "description": "Get the current weather for a location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g. Seattle, WA"
      }
    },
    "required": ["location"]
  }
}
```

The schema inference engine (in [`_tools.py`](../python/packages/core/agent_framework/_tools.py)) handles:
- Python type annotations → JSON Schema types
- `Annotated[type, Field(...)]` → descriptions, defaults, enums
- `Literal["a", "b"]` → enum values
- Pydantic `BaseModel` → nested object schemas
- Optional/default values → non-required fields

### 3.2 MCP Protocol Integration

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard for connecting AI agents to external tools and data sources. The framework integrates MCP as a **first-class tool type**.

**Python MCP Integration** ([`_mcp.py`](../python/packages/core/agent_framework/_mcp.py)):

```python
from agent_framework import Agent

# Connect to a remote MCP server (e.g., GitHub)
github_mcp = client.get_mcp_tool(
    name="GitHub",
    url="https://api.githubcopilot.com/mcp/",
    headers={"Authorization": f"Bearer {github_pat}"},
    approval_mode="never_require",
)

agent = client.as_agent(
    name="GitHubAgent",
    instructions="Help with GitHub operations.",
    tools=[github_mcp],
)

result = await agent.run("List my repositories")
```

See [`mcp_github_pat.py`](../python/samples/02-agents/mcp/mcp_github_pat.py) and [`mcp_api_key_auth.py`](../python/samples/02-agents/mcp/mcp_api_key_auth.py) for complete examples.

**.NET MCP Integration** ([`ModelContextProtocol/`](../dotnet/samples/GettingStarted/ModelContextProtocol/)):

```csharp
AITool mcpTool = new HostedMcpServerTool(
    serverName: "microsoft_learn",
    serverAddress: "https://learn.microsoft.com/api/mcp"
)
{
    AllowedTools = ["microsoft_docs_search"],
    ApprovalMode = HostedMcpServerToolApprovalMode.NeverRequire
};

AIAgent agent = chatClient.CreateAIAgent(
    instructions: "Answer by searching Microsoft Learn.",
    tools: [mcpTool]
);
```

**How MCP works technically:**

1. **Connection**: The framework creates an MCP `ClientSession` using the [MCP SDK](https://github.com/modelcontextprotocol/python-sdk). The transport can be HTTP/SSE (streamable HTTP) or stdio for local servers.

2. **Tool Discovery**: On initialization, `MCPTool.load_tools()` calls `session.list_tools()` over the MCP protocol. Each remote tool is converted into a local `FunctionTool` with:
   - The MCP tool's `inputSchema` mapped to the local JSON Schema format
   - A `partial(self.call_tool, tool_name)` wrapper as the execution function
   - The configured `approval_mode` applied per-tool

3. **Tool Invocation**: When the LLM requests a tool call:
   ```
   Agent → LLM: "tools: [github_list_repos, github_create_issue, ...]"
   LLM → Agent: "call github_list_repos(owner='user')"
   Agent → MCP Server: session.call_tool("github_list_repos", arguments={"owner": "user"})
   MCP Server → Agent: CallToolResult(content=[TextContent("repo1, repo2, ...")])
   Agent → LLM: "tool_result: repo1, repo2, ..."
   LLM → Agent: "You have the following repositories: ..."
   ```

4. **Reconnection**: The MCP client handles `ClosedResourceError` by transparently reconnecting and retrying the operation once.

5. **Distributed Tracing**: OpenTelemetry trace context is injected into the MCP `_meta` field via `_inject_otel_into_mcp_meta()`, enabling end-to-end tracing across agent and MCP server.

**Exposing an agent AS an MCP server** ([`agent_as_mcp_server.py`](../python/samples/02-agents/mcp/agent_as_mcp_server.py)):

```python
agent = client.as_agent(
    name="RestaurantAgent",
    tools=[get_specials, get_item_price],
)

# Convert the agent into an MCP server
server = agent.as_mcp_server()
# The agent's tools become MCP tools accessible by other agents
```

### 3.3 Tool Invocation Flow (Technical Internals)

Here is the complete flow from user query to tool execution to final response:

```
┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────┐    ┌────────────┐
│  User     │───▶│  Agent    │───▶│  Chat Client │───▶│   LLM    │───▶│  Response   │
│  Message  │    │  .run()   │    │  .get_resp() │    │ (OpenAI) │    │ w/ tool_call│
└──────────┘    └───────────┘    └──────────────┘    └──────────┘    └─────┬──────┘
                                                                           │
                     ┌─────────────────────────────────────────────────────┘
                     ▼
              ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
              │ Parse tool   │───▶│ _auto_invoke_     │───▶│ FunctionTool │
              │ call content │    │ function()        │    │  .invoke()   │
              └──────────────┘    │ • validate args   │    │ • run func   │
                                  │ • check approval  │    │ • parse result│
                                  │ • run middleware   │    └──────┬───────┘
                                  └───────────────────┘           │
                     ┌────────────────────────────────────────────┘
                     ▼
              ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
              │ Content.from │───▶│ Append to message │───▶│ Send back to │
              │ _function_   │    │ history & re-call  │    │ LLM for next │
              │ result()     │    │ the LLM           │    │ response     │
              └──────────────┘    └───────────────────┘    └──────────────┘
```

**Step-by-step** (referencing [`_tools.py`](../python/packages/core/agent_framework/_tools.py) and [`_agents.py`](../python/packages/core/agent_framework/_agents.py)):

1. **User calls `agent.run("What's the weather?")`**
2. **Context providers run** (`before_run`): inject history, instructions, extra tools
3. **Chat client sends** messages + tool schemas to LLM
4. **LLM responds** with a `function_call` content (tool name + JSON arguments)
5. **Auto-invocation loop** (`_auto_invoke_function()`):
   - Looks up the `FunctionTool` in the tool map by name
   - Validates arguments against JSON Schema (`_validate_arguments_against_schema()`)
   - Checks approval mode — if `always_require`, returns a `function_approval_request` and pauses
   - Runs through the middleware pipeline if configured (`FunctionInvocationContext`)
   - Calls `tool.invoke(arguments=parsed_args)` which executes the Python/MCP function
   - Wraps the result in `Content.from_function_result(call_id=..., result=...)`
6. **Result appended to message history**, LLM is called again
7. **Loop repeats** until LLM responds with text (no more tool calls) or `max_iterations` (default: 40) is reached
8. **Context providers run** (`after_run`): store conversation, update memory
9. **Return `AgentResponse`** to caller

**Configuration** ([`FunctionInvocationConfiguration`](../python/packages/core/agent_framework/_tools.py)):

```python
agent = client.as_agent(
    name="MyAgent",
    tools=[my_tool],
    function_invocation_configuration={
        "enabled": True,
        "max_iterations": 20,           # Max LLM↔tool round-trips
        "max_consecutive_errors_per_request": 3,  # Fail-fast on repeated errors
        "terminate_on_unknown_calls": True,        # Error on unknown tool names
        "include_detailed_errors": True,           # Send stack traces to LLM
    },
)
```

### 3.4 Authentication Between Agent and Tools

**Local tools** (Python functions, .NET methods): No authentication needed — they execute in the same process.

**MCP tools** (remote servers): Authentication is handled via HTTP headers passed to the MCP client:

```python
# API key auth
mcp_tool = client.get_mcp_tool(
    url="https://my-api.example.com/mcp/",
    headers={"X-API-Key": os.environ["MY_API_KEY"]},
)

# OAuth/Bearer token auth
mcp_tool = client.get_mcp_tool(
    url="https://api.githubcopilot.com/mcp/",
    headers={"Authorization": f"Bearer {token}"},
)
```

See [`mcp_api_key_auth.py`](../python/samples/02-agents/mcp/mcp_api_key_auth.py) for a complete example.

In .NET, MCP auth headers are similarly configured:

```csharp
new HostedMcpServerTool(
    serverName: "my_server",
    serverAddress: "https://example.com/mcp",
    headers: new Dictionary<string, string> { ["Authorization"] = $"Bearer {token}" }
);
```

**Underlying transport**: The MCP SDK uses **HTTP with Server-Sent Events (SSE)** for streamable transport. The `ClientSession` maintains a persistent HTTP connection. Auth headers are included in every HTTP request made by the MCP client. For local MCP servers (e.g., running a subprocess), **stdio** transport is used — no auth needed since it's a local process.

> **Hypothesis**: For hosted MCP servers with OAuth flows (e.g., needing user consent), the framework likely supports an interceptor or callback pattern where the MCP session can trigger an OAuth redirect flow. The `.NET` `HostedMcpServerTool` with `ApprovalMode` settings suggests a model where tool calls requiring elevated permissions can be paused pending user authorization. The exact OAuth integration pattern is not fully documented in the codebase, but the architecture supports it through the approval flow mechanism.

---

## 4. Deep Dive: Memory

The framework supports a **layered memory architecture** through the `BaseContextProvider` pattern. Each memory type hooks into the agent lifecycle via `before_run()` (retrieve) and `after_run()` (store).

```
┌─────────────────────────────────────────────────────────────────┐
│                       Agent.run() Lifecycle                      │
├────────────┬────────────────────────────────────────┬───────────┤
│ before_run │          LLM Invocation                │ after_run │
│            │                                        │           │
│ ┌────────┐ │                                        │ ┌────────┐│
│ │Session │ │  Messages + Context + Tools  ──▶ LLM   │ │Session ││
│ │State   │ │                                        │ │State   ││
│ ├────────┤ │                                        │ ├────────┤│
│ │History │ │  ◀── Response + Tool Calls             │ │History ││
│ ├────────┤ │                                        │ ├────────┤│
│ │Mem0    │ │                                        │ │Mem0    ││
│ ├────────┤ │                                        │ ├────────┤│
│ │Redis   │ │                                        │ │Redis   ││
│ ├────────┤ │                                        │ ├────────┤│
│ │RAG     │ │                                        │ │  N/A   ││
│ └────────┘ │                                        │ └────────┘│
└────────────┴────────────────────────────────────────┴───────────┘
```

### 4.1 Short-Term Memory (Session State)

**What**: A mutable Python dict stored on `AgentSession.state` that persists across `run()` calls within the same session.

**Source**: [`_sessions.py`](../python/packages/core/agent_framework/_sessions.py)

```python
session = agent.create_session()
# session.state is a dict[str, Any]

# Context providers get a scoped view:
class MyProvider(BaseContextProvider):
    async def before_run(self, *, state, **kwargs):
        # 'state' is session.state[self.source_id] — isolated per provider
        count = state.get("call_count", 0)
        state["call_count"] = count + 1
```

**Underlying technology**: Pure in-memory Python dict. Serializable via `session.to_dict()` which uses a **type registry** pattern — Pydantic models and custom classes implementing `SerializationMixin` are serialized with a `"type"` discriminator so they can be restored via `AgentSession.from_dict()`.

**When to use**: Lightweight per-session state (user preferences, turn counters, extracted entities).

### 4.2 Conversation History Providers

**What**: Stores and retrieves the full conversation history for multi-turn interactions.

**Python** — `InMemoryHistoryProvider` (built-in default):

```python
from agent_framework import BaseHistoryProvider

# The built-in InMemoryHistoryProvider stores messages in session.state["messages"]
# It's automatically used when you create a session and call agent.run() multiple times

session = agent.create_session()
await agent.run("Hello", session=session)       # Turn 1 stored
await agent.run("Tell me more", session=session) # Turn 2 includes Turn 1 history
```

**.NET** — `ChatHistoryMemoryProvider` ([`Memory/ChatHistoryMemoryProvider.cs`](../dotnet/src/Microsoft.Agents.AI/Memory/ChatHistoryMemoryProvider.cs)):

```csharp
// Built-in: stores chat history in AgentSession for multi-turn conversations
var agent = chatClient.AsAIAgent(
    name: "ChatBot",
    contextProviders: [new ChatHistoryMemoryProvider()]
);
```

**Cosmos DB History Provider** ([`CosmosChatHistoryProvider`](../dotnet/src/Microsoft.Agents.AI.CosmosNoSql/)):

```csharp
// Persists conversation history to Cosmos DB with TTL and partitioning
var historyProvider = new CosmosChatHistoryProvider(cosmosClient, "mydb", "conversations")
{
    PartitionKeyPath = "/conversationId",
    TimeToLive = TimeSpan.FromHours(24),
    MaxMessages = 100,
};
```

**Underlying technology**: Cosmos DB uses hierarchical partition keys (TenantId/UserId/ConversationId) for multi-tenant isolation. Messages are stored as JSON documents with transactional batch operations for atomicity. Automatic batch splitting handles the 2MB Cosmos DB limit.

### 4.3 Long-Term Semantic Memory (Mem0)

**What**: [Mem0](https://mem0.ai/) provides persistent, semantically searchable memory that survives across sessions. It automatically extracts, indexes, and retrieves relevant memories.

**Python** ([`mem0/`](../python/packages/mem0/) package, sample: [`mem0/`](../python/samples/02-agents/context_providers/mem0/)):

```python
from agent_framework_mem0 import Mem0ContextProvider
from mem0 import AsyncMemoryClient

mem0_client = AsyncMemoryClient(api_key=os.environ["MEM0_API_KEY"])

mem0_provider = Mem0ContextProvider(
    client=mem0_client,
    user_id="user-123",           # Scope memories to this user
    agent_id="support-agent",     # Scope memories to this agent
    application_id="my-app",      # Scope memories to this app
)

agent = client.as_agent(
    name="SupportAgent",
    instructions="You are a helpful support agent.",
    context_providers=[mem0_provider],
)

# Session 1: User tells the agent their preferences
session1 = agent.create_session()
await agent.run("I prefer dark mode and use Python.", session=session1)
# Mem0's after_run() stores: user prefers dark mode, uses Python

# Session 2 (hours/days later): Agent remembers!
session2 = agent.create_session()
result = await agent.run("Help me set up my environment.", session=session2)
# Mem0's before_run() retrieves: "User prefers dark mode, uses Python"
# Agent responds: "I'll help set up your Python environment with dark mode..."
```

**.NET** ([`Mem0Provider`](../dotnet/src/Microsoft.Agents.AI.Mem0/)):

```csharp
var mem0Provider = new Mem0Provider(
    httpClient,
    stateInitializer: _ => new(new Mem0ProviderScope
    {
        ApplicationId = "my-app",
        UserId = "user-123"
    })
);
```

**How Mem0 works under the hood:**

1. **`before_run()`**: Extracts the user's input text, calls `mem0_client.search(query=input_text, filters={user_id, agent_id})` to find relevant memories. Results are prepended to the context as a system message: `"## Memories\nConsider the following memories: ..."`.

2. **`after_run()`**: Collects input + response messages, filters to user/assistant/system roles, calls `mem0_client.add(messages=[...], user_id=..., agent_id=...)`.

3. **Mem0's internal processing** (hypothesis based on Mem0 documentation): Mem0 uses an **embedding model** (e.g., OpenAI `text-embedding-3-small`) to convert messages into vector embeddings. These are stored in a **vector database** (Qdrant, Pinecone, or Mem0's hosted service). Retrieval uses **cosine similarity** between the query embedding and stored memory embeddings. Mem0 also performs **automatic deduplication and conflict resolution** — if a user says "I like Python" then later "I actually prefer Go", Mem0 updates the memory rather than storing both.

4. **Indexing delay**: Memories are indexed asynchronously — there's an empirical ~12-second delay before newly stored memories become searchable.

### 4.4 Persistent Storage (Redis)

**What**: Redis provides both **full-text search** and **hybrid (text + vector) search** for conversation context, plus simple **list-based history** storage.

**Python** ([`redis/`](../python/packages/redis/) package, sample: [`redis/`](../python/samples/02-agents/context_providers/redis/)):

```python
from agent_framework_redis import RedisContextProvider, RedisHistoryProvider
from redisvl.utils.vectorize import OpenAITextVectorizer

# Option A: Context provider with hybrid search (text + vector)
redis_context = RedisContextProvider(
    redis_url="redis://localhost:6379",
    application_id="my-app",
    user_id="user-123",
    vectorizer=OpenAITextVectorizer(model="text-embedding-3-small"),
    vector_field_name="embedding",
)

# Option B: Simple history provider (Redis Lists)
redis_history = RedisHistoryProvider(
    redis_url="redis://localhost:6379",
    max_messages=50,
)

agent = client.as_agent(
    name="MemoryAgent",
    context_providers=[redis_context, redis_history],
)
```

**How Redis storage works under the hood:**

1. **RedisContextProvider** uses [RediSearch](https://redis.io/docs/stack/search/) (Redis Stack module):
   - **Schema**: Hash-based storage with fields: `role`, `content`, `conversation_id`, `message_id`, plus partition tags (`application_id`, `agent_id`, `user_id`, `thread_id`)
   - **Text search**: BM25 full-text search on the `content` field
   - **Vector search**: If a vectorizer is provided, messages are embedded on write and stored as `float32` byte vectors. Retrieval uses hybrid queries combining BM25 text scores with cosine vector similarity
   - **Multi-tenant isolation**: Partition tags enable prefix-based key separation

2. **RedisHistoryProvider** uses Redis **Lists** (`LPUSH` / `LRANGE`):
   - Messages serialized as JSON strings via `Message.to_dict()`
   - Key pattern: `{key_prefix}:{session_id}`
   - `LTRIM` enforces `max_messages` cap
   - Chronological order, no search/ranking

### 4.5 RAG / Semantic Search (Azure AI Search)

**What**: [Azure AI Search](https://learn.microsoft.com/azure/search/) provides enterprise-grade semantic search for grounding agent responses in document corpora.

**Python** ([`azure-ai-search/`](../python/packages/azure-ai-search/) package, sample: [`azure_ai_search/`](../python/samples/02-agents/context_providers/azure_ai_search/)):

```python
from agent_framework_azure_ai_search import AzureAISearchContextProvider

search_provider = AzureAISearchContextProvider(
    endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
    index_name="product-docs",
    credential=AzureCliCredential(),
    # Semantic mode: fast hybrid search
    semantic_configuration_name="default",
    top_k=5,
)

agent = client.as_agent(
    name="DocSearchAgent",
    instructions="Answer questions using the product documentation.",
    context_providers=[search_provider],
)

result = await agent.run("How do I configure SSO?")
# before_run() retrieves top 5 relevant document chunks
# Agent synthesizes answer from retrieved context
```

**How Azure AI Search works under the hood:**

Two retrieval modes are supported:

1. **Semantic Mode** (fast, recommended):
   - **Hybrid query**: Combines BM25 keyword matching with vector similarity (if vector field exists)
   - **Semantic ranking**: Microsoft's cross-encoder re-ranks results by semantic relevance
   - **Extractive captions**: Highlights the most relevant passages
   - The provider auto-discovers vector fields from the index schema and uses server-side vectorization if configured

2. **Agentic Mode** (deeper reasoning, slower):
   - Uses Azure AI Search's `KnowledgeBaseRetrievalClient` for multi-hop reasoning
   - The LLM plans retrieval queries, evaluates results, and iterates
   - Configurable `retrieval_reasoning_effort`: `"minimal"`, `"low"`, `"medium"`
   - This is hypothesis-based on the codebase's `_agentic_retrieval_search.py`: the client creates a Knowledge Base over the search index, then sends the user query for AI-planned retrieval

### 4.6 Cloud-Native Memory (Cosmos DB, Foundry Memory)

**.NET Cosmos DB** ([`Microsoft.Agents.AI.CosmosNoSql`](../dotnet/src/Microsoft.Agents.AI.CosmosNoSql/)):
- Hierarchical partition keys (TenantId → UserId → ConversationId) for multi-tenant apps
- Transactional batch operations for atomic message writes
- TTL (time-to-live) for automatic cleanup
- Automatic batch splitting on 2MB size limits

**.NET Foundry Memory** ([`Microsoft.Agents.AI.FoundryMemory`](../dotnet/src/Microsoft.Agents.AI.FoundryMemory/)):

```csharp
var foundryMemory = new FoundryMemoryProvider(
    projectClient,
    "memory-store-name",
    stateInitializer: _ => new(new FoundryMemoryProviderScope("user-123"))
);

// Creates a managed memory store with embedding + chat models
// Async memory updates with polling for completion
// Default: max 5 memories retrieved per query
```

> **Hypothesis on Foundry Memory internals**: Azure AI Foundry Memory likely uses a managed vector store under the hood. When `StoreAIContextAsync()` is called, it sends messages to the Foundry service which: (1) extracts key facts, (2) generates embeddings, (3) stores in a managed vector database. Retrieval uses semantic search with configurable max results. The async update pattern with polling suggests a queue-based architecture where memory updates are processed asynchronously.

### 4.7 Holistic Multi-Memory Agent Example

Here is a complete example combining **all memory types** into a single agent:

```python
"""
A customer support agent with layered memory:
1. Session state     — tracks current conversation context (turn count, active ticket)
2. History provider  — maintains full conversation history within a session
3. Mem0              — remembers user preferences across sessions (long-term)
4. Redis             — stores and searches recent support interactions (medium-term)
5. Azure AI Search   — retrieves knowledge base articles (RAG)
"""
import asyncio
import os
from agent_framework import Agent, AgentSession, BaseContextProvider, tool
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework_mem0 import Mem0ContextProvider
from agent_framework_redis import RedisContextProvider, RedisHistoryProvider
from agent_framework_azure_ai_search import AzureAISearchContextProvider
from azure.identity import AzureCliCredential
from mem0 import AsyncMemoryClient


# ── 1. Session State Provider: Tracks in-conversation context ──────────────
class TicketTracker(BaseContextProvider):
    """Tracks the active support ticket within a session."""
    DEFAULT_SOURCE_ID = "ticket_tracker"

    async def before_run(self, *, agent, session, context, state):
        state.setdefault("turn_count", 0)
        state["turn_count"] += 1

        ticket_id = state.get("active_ticket")
        if ticket_id:
            context.extend_instructions(
                self.source_id,
                f"Active support ticket: {ticket_id} (turn {state['turn_count']}). "
                "Continue working on this ticket unless the user starts a new topic.",
            )

    async def after_run(self, *, agent, session, context, state):
        # Extract ticket ID if the agent created one
        for msg in context.get_messages(include_response=True):
            if msg.text and "TICKET-" in msg.text:
                import re
                match = re.search(r"TICKET-\d+", msg.text)
                if match:
                    state["active_ticket"] = match.group()

    # Underlying technology: Pure Python dict on AgentSession.state.
    # Survives across run() calls in the same session.
    # Serializable via session.to_dict() for persistence to any store.


# ── 2. Long-term Memory: Mem0 for cross-session preferences ───────────────
# Underlying technology: Mem0 embeds messages with an embedding model
# (e.g., text-embedding-3-small), stores in a vector DB (Qdrant/Pinecone),
# and retrieves via cosine similarity search.
# Automatic deduplication and conflict resolution.
mem0_provider = Mem0ContextProvider(
    client=AsyncMemoryClient(api_key=os.environ["MEM0_API_KEY"]),
    user_id="customer-456",
    agent_id="support-agent",
    application_id="support-app",
)

# ── 3. Redis: Medium-term searchable interaction history ───────────────────
# Underlying technology: Redis Stack with RediSearch module.
# Stores messages as Redis Hashes with full-text (BM25) + vector indices.
# Hybrid search combines keyword matching with embedding similarity.
redis_context = RedisContextProvider(
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
    application_id="support-app",
    user_id="customer-456",
)

# ── 4. Azure AI Search: RAG over knowledge base articles ──────────────────
# Underlying technology: Azure AI Search with hybrid retrieval.
# BM25 keyword matching + vector similarity + semantic re-ranking.
# Extractive captions highlight relevant passages.
search_provider = AzureAISearchContextProvider(
    endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
    index_name="support-knowledge-base",
    credential=AzureCliCredential(),
    semantic_configuration_name="default",
    top_k=3,
)

# ── 5. Tools ──────────────────────────────────────────────────────────────
@tool(approval_mode="never_require")
def create_support_ticket(
    title: str,
    description: str,
    priority: str = "medium",
) -> str:
    """Create a new support ticket."""
    import random
    ticket_id = f"TICKET-{random.randint(10000, 99999)}"
    return f"Created {ticket_id}: {title} (priority: {priority})"

@tool(approval_mode="always_require")
def escalate_ticket(ticket_id: str, reason: str) -> str:
    """Escalate a ticket to a senior agent. Requires approval."""
    return f"Escalated {ticket_id}: {reason}"


# ── Assemble the agent ────────────────────────────────────────────────────
async def main():
    credential = AzureCliCredential()
    llm_client = AzureOpenAIResponsesClient(credential=credential)

    agent = llm_client.as_agent(
        name="SupportAgent",
        instructions=(
            "You are a customer support agent. "
            "Use the knowledge base to answer product questions. "
            "Create tickets for issues. Escalate when appropriate. "
            "Be personal — use what you know about the customer."
        ),
        tools=[create_support_ticket, escalate_ticket],
        context_providers=[
            TicketTracker(),        # 1. Session state
            # InMemoryHistoryProvider is auto-included for conversation history
            mem0_provider,          # 2. Long-term memory (Mem0)
            redis_context,          # 3. Medium-term search (Redis)
            search_provider,        # 4. RAG (Azure AI Search)
        ],
    )

    # ── Multi-turn session ────────────────────────────────────────────
    session = agent.create_session()

    # Turn 1: User provides context
    print("Turn 1:", await agent.run(
        "Hi, I'm having trouble with SSO. I use SAML and my IdP is Okta.",
        session=session,
    ))
    # Memory flow:
    #   before_run: Mem0 searches for customer-456 memories → might find past preferences
    #               Azure AI Search finds SSO/SAML/Okta articles in knowledge base
    #               Redis searches recent interactions for this customer
    #   LLM generates response using all context
    #   after_run:  Mem0 stores "customer uses SAML with Okta IdP"
    #               Redis stores this interaction for future search
    #               TicketTracker records turn_count=1

    # Turn 2: Follow-up
    print("Turn 2:", await agent.run(
        "It was working yesterday. Can you create a ticket?",
        session=session,
    ))
    # Memory flow:
    #   before_run: TicketTracker injects turn_count=2
    #               History provider loads Turn 1 conversation
    #               Mem0 retrieves "uses SAML with Okta"
    #   LLM calls create_support_ticket tool
    #   after_run:  TicketTracker extracts TICKET-12345 from response

    # ── New session (simulating next day) ─────────────────────────────
    new_session = agent.create_session()

    print("New session:", await agent.run(
        "Any update on my SSO issue?",
        session=new_session,
    ))
    # Memory flow:
    #   Session state is fresh (no turn history)
    #   BUT: Mem0 retrieves "uses SAML with Okta, had SSO issue, ticket TICKET-12345"
    #   AND: Redis finds yesterday's interaction via semantic search
    #   Agent can reference past context even in a new session!


if __name__ == "__main__":
    asyncio.run(main())
```

**Memory type comparison:**

| Memory Type | Scope | Lifetime | Retrieval | Technology | Use Case |
|---|---|---|---|---|---|
| **Session State** | Per-session, per-provider | Session duration | Key-value lookup | Python dict | Turn counters, active entities |
| **History Provider** | Per-session | Session duration | Chronological | In-memory list or Redis List | Multi-turn conversation |
| **Mem0** | Per-user/agent/app | Permanent | Semantic vector search | Embedding model + vector DB | User preferences, learned facts |
| **Redis** | Per-user/agent/app | Configurable TTL | BM25 + vector hybrid | RediSearch module | Recent interactions, searchable logs |
| **Azure AI Search** | Global (index) | Permanent | Hybrid + semantic ranking | BM25 + vector + cross-encoder | Knowledge base articles, documentation |
| **Cosmos DB** (.NET) | Per-tenant/user/conversation | Configurable TTL | SQL-like queries | Document DB with partition keys | Multi-tenant chat history |
| **Foundry Memory** (.NET) | Per-user (string scope) | Permanent | Semantic search | Managed vector store | Cross-session memories (Azure-native) |

---

## 5. Testing Agents

### Python Testing

The framework uses **pytest** with async support. Tests live alongside packages in `tests/` directories.

```bash
cd python/

# Run all tests
uv run poe test

# Run tests for a specific package
uv run poe test --package core

# Run a specific test file
uv run pytest packages/core/tests/test_tools.py -v
```

**Test structure** (from [`python/packages/core/tests/`](../python/packages/core/tests/)):

```python
import pytest
from agent_framework import Agent, FunctionTool, tool, AgentSession

@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test that an agent correctly invokes tools."""
    # Arrange: Create a mock tool
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Create agent with a mock/test chat client
    agent = test_client.as_agent(
        name="TestAgent",
        tools=[add],
    )

    # Act: Run the agent
    result = await agent.run("What is 2 + 3?")

    # Assert: Verify the response
    assert "5" in result.text
```

### .NET Testing

The .NET codebase uses **xUnit** with **Moq** for mocking:

```bash
cd dotnet/

# Run all tests
dotnet test

# Run specific test project
dotnet test tests/Microsoft.Agents.AI.Tests/
```

**Test patterns** (from [`dotnet/tests/`](../dotnet/tests/)):

```csharp
[Fact]
public async Task Agent_WithTools_InvokesCorrectly()
{
    // Arrange
    var mockClient = new Mock<IChatClient>();
    var agent = mockClient.Object.AsAIAgent(
        name: "TestAgent",
        tools: [AIFunctionFactory.Create((int a, int b) => a + b, "add")]
    );

    // Act
    var response = await agent.RunAsync("What is 2 + 3?");

    // Assert
    Assert.Contains("5", response.Text);
}
```

### Testing with DevUI

The framework includes a [DevUI](../python/packages/devui/) for interactive testing and debugging:

```python
from agent_framework_devui import DevUI

# Launch interactive UI for testing your agent
dev_ui = DevUI(agent=my_agent)
dev_ui.run()  # Opens browser at http://localhost:8000
```

See the [DevUI package](../python/packages/devui/) and [.NET DevUI samples](../dotnet/samples/GettingStarted/DevUI/).

---

## 6. Deploying Agents

### 6.1 Azure Functions (Python)

The framework provides turnkey Azure Functions hosting via [`agent_framework_azurefunctions`](../python/packages/azurefunctions/).

**Sample**: [`06_host_your_agent.py`](../python/samples/01-get-started/06_host_your_agent.py)

```python
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework_azurefunctions import AgentFunctionApp
from azure.identity import AzureCliCredential

# Create your agent
agent = AzureOpenAIChatClient(credential=AzureCliCredential()).as_agent(
    name="HostedAgent",
    instructions="You are a helpful assistant hosted in Azure Functions.",
)

# Register with AgentFunctionApp — this generates HTTP endpoints automatically
app = AgentFunctionApp(
    agents=[agent],
    enable_health_check=True,
    max_poll_retries=50,
)
# Endpoint: POST /api/agents/HostedAgent/run
```

**How it works under the hood:**
- Uses **Azure Durable Functions** with **Durable Entities** for state persistence
- Each agent session is a durable entity that survives across function invocations
- Long-running operations return `202 Accepted` with a status URL for polling
- The `AgentFunctionApp` auto-generates HTTP trigger functions for each registered agent

**Deployment**:
```bash
# Deploy to Azure (requires Azure Functions Core Tools)
func azure functionapp publish <your-function-app-name>
```

See more hosting samples at [`python/samples/04-hosting/`](../python/samples/04-hosting/).

### 6.2 OpenAI-Compatible HTTP Endpoints (.NET)

The .NET framework can expose agents as OpenAI-compatible REST endpoints, making them drop-in replacements for OpenAI APIs.

**Source**: [`Microsoft.Agents.AI.Hosting.OpenAI`](../dotnet/src/Microsoft.Agents.AI.Hosting.OpenAI/)

```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

// Map OpenAI-compatible chat completions endpoint
app.MapOpenAIChatCompletions(agentBuilder, path: "/v1/chat/completions");

// Map OpenAI-compatible responses endpoint
app.MapOpenAIResponses(agentBuilder, path: "/v1/responses");

app.Run();
```

**Endpoints exposed:**
- `POST /v1/chat/completions` — Standard chat completions (non-streaming: JSON, streaming: SSE)
- `POST /v1/responses` — OpenAI Responses API format
- `POST /v1/conversations` — Conversation management with session persistence

**Streaming uses Server-Sent Events (SSE):**
```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

See hosted agent samples at [`dotnet/samples/HostedAgents/`](../dotnet/samples/HostedAgents/).

### 6.3 A2A Protocol (Agent-to-Agent)

The [A2A protocol](https://google.github.io/A2A/) enables agents to communicate over HTTP using JSON-RPC.

**Python** ([`a2a/`](../python/packages/a2a/) package):

```python
from agent_framework_a2a import A2AAgent

# Connect to a remote A2A agent
remote_agent = A2AAgent(url="https://remote-agent.example.com/a2a")

# Use it like any other agent
result = await remote_agent.run("Process this data")

# Or use it as a tool in another agent
local_agent = client.as_agent(
    name="Orchestrator",
    tools=[remote_agent.as_tool()],
)
```

**.NET** ([`Microsoft.Agents.AI.Hosting.A2A`](../dotnet/src/Microsoft.Agents.AI.Hosting.A2A/)):

```csharp
// Expose an agent as an A2A endpoint
app.MapA2A(agentBuilder);

// Connect to a remote A2A agent
var remoteAgent = new A2AAgent("https://remote-agent.example.com/a2a");
```

**A2A protocol details:**
- Transport: HTTP with JSON-RPC 2.0 message format
- Long-running tasks return continuation tokens for polling
- Supports task states: submitted → working → completed/failed/canceled
- Content types: text, files (URI or bytes), data parts, artifacts

See samples at [`dotnet/samples/A2AClientServer/`](../dotnet/samples/A2AClientServer/) and [`dotnet/samples/GettingStarted/A2A/`](../dotnet/samples/GettingStarted/A2A/).

### 6.4 AG-UI Protocol (Web UI Streaming)

The [AG-UI protocol](https://docs.ag-ui.com/) provides a standardized streaming interface for web UIs.

**Python** ([`ag-ui/`](../python/packages/ag-ui/) package):

```python
from fastapi import FastAPI
from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint

app = FastAPI()

add_agent_framework_fastapi_endpoint(
    app=app,
    agent=my_agent,
    path="/agent",
    # dependencies=[Depends(verify_token)],  # Optional auth
)

# Client connects via: POST /agent → Server-Sent Events stream
```

**AG-UI features**: Chat, backend tools, human-in-the-loop, generative UI, shared state, predictive updates.

See samples at [`dotnet/samples/AGUIClientServer/`](../dotnet/samples/AGUIClientServer/).

---

## 7. Invoking Agents

Once deployed, agents can be invoked in several ways:

**Direct invocation (in-process):**
```python
result = await agent.run("Hello!")
```

**HTTP REST (OpenAI-compatible):**
```bash
curl -X POST https://my-agent.azurewebsites.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

**A2A Protocol (JSON-RPC):**
```python
from agent_framework_a2a import A2AAgent
remote = A2AAgent(url="https://my-agent.example.com/a2a")
result = await remote.run("Hello!")
```

**Azure Functions:**
```bash
curl -X POST https://my-func.azurewebsites.net/api/agents/MyAgent/run \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

**As a tool in another agent:**
```python
sub_agent = client.as_agent(name="SubAgent", instructions="...")
main_agent = client.as_agent(
    name="MainAgent",
    tools=[sub_agent.as_tool()],  # Agent becomes a callable tool
)
```

---

## 8. Other Important Dimensions

### 8.1 Observability and Telemetry

The framework has built-in [OpenTelemetry](https://opentelemetry.io/) support for distributed tracing.

**Python** ([observability samples](../python/samples/02-agents/observability/)):

```python
from agent_framework.observability import configure_otel_providers, get_tracer
from opentelemetry.trace import SpanKind

# Configure OpenTelemetry exporters (OTLP, Azure Monitor, etc.)
configure_otel_providers(enable_sensitive_data=True)

with get_tracer().start_as_current_span("MyScenario", kind=SpanKind.CLIENT) as span:
    result = await agent.run("Hello!")
    # All agent operations, tool calls, and LLM requests are traced
```

**.NET** ([`AgentOpenTelemetry/`](../dotnet/samples/GettingStarted/AgentOpenTelemetry/)):

```csharp
// OpenTelemetry is built into the agent middleware pipeline
var agent = chatClient
    .AsAIAgentBuilder()
    .Use(new OpenTelemetryAgent())   // Adds tracing spans
    .Use(new LoggingAgent())          // Adds structured logging
    .Build();
```

**What gets traced**: Agent runs, LLM calls (with token counts), tool invocations (with arguments and results), MCP calls (with distributed trace context propagation), context provider execution.

### 8.2 Middleware Pipeline

Agents support a composable middleware pipeline for cross-cutting concerns.

**Python** ([middleware samples](../python/samples/02-agents/middleware/)):

```python
from agent_framework import AgentMiddleware

class AuditMiddleware(AgentMiddleware):
    async def invoke(self, context, next_handler):
        print(f"[AUDIT] Agent called with: {context.messages}")
        result = await next_handler(context)
        print(f"[AUDIT] Agent responded: {result.text}")
        return result

agent = client.as_agent(
    name="AuditedAgent",
    middleware=[AuditMiddleware()],
)
```

**.NET** — uses the decorator pattern:

```csharp
var agent = chatClient
    .AsAIAgentBuilder()
    .Use(next => new LoggingAgent(next))           // Logging middleware
    .Use(next => new OpenTelemetryAgent(next))     // Telemetry middleware
    .Use(contextProviders)                          // Context injection
    .Build();
```

**Middleware types**: `AgentMiddleware` (wraps agent calls), `ChatMiddleware` (wraps LLM calls), `FunctionMiddleware` (wraps tool calls).

### 8.3 Declarative Agent Definitions

Agents and workflows can be defined in YAML/JSON without code.

**YAML agent definition** (from [`agent-samples/`](../agent-samples/)):

```yaml
kind: Prompt
name: CustomerSupportBot
instructions: "You are a helpful customer support agent..."
model:
    id: gpt-4.1-mini
    provider: OpenAI
    apiType: Chat
    connection:
        kind: ApiKey
        key: =Env.OPENAI_API_KEY   # PowerFx expression for env var
tools:
    - name: search_knowledge_base
      description: "Search the support knowledge base"
      parameters:
          query: { type: string, required: true }
outputSchema:
    properties:
        response: { type: string }
        sentiment: { type: string, enum: [positive, neutral, negative] }
```

**Python declarative loading** ([`declarative/`](../python/packages/declarative/)):

```python
from agent_framework_declarative import load_agent

agent = load_agent("path/to/agent.yaml")
result = await agent.run("Help me with my order")
```

See [`python/samples/03-workflows/declarative/`](../python/samples/03-workflows/declarative/) and [`dotnet/samples/GettingStarted/DeclarativeAgents/`](../dotnet/samples/GettingStarted/DeclarativeAgents/).

### 8.4 Multi-LLM Provider Support

The framework supports multiple LLM providers through a pluggable client architecture:

| Provider | Python Package | .NET Package | Sample |
|---|---|---|---|
| Azure OpenAI | Built-in (`agent_framework.azure`) | `Microsoft.Agents.AI.OpenAI` | [Python](../python/samples/02-agents/providers/) / [.NET](../dotnet/samples/GettingStarted/AgentProviders/) |
| OpenAI | Built-in (`agent_framework.openai`) | `Microsoft.Agents.AI.OpenAI` | [Python](../python/samples/02-agents/providers/) / [.NET](../dotnet/samples/GettingStarted/AgentWithOpenAI/) |
| Anthropic Claude | `agent-framework-anthropic` | `Microsoft.Agents.AI.Anthropic` | [Python](../python/packages/anthropic/) / [.NET](../dotnet/samples/GettingStarted/AgentWithAnthropic/) |
| AWS Bedrock | `agent-framework-bedrock` | — | [Python](../python/packages/bedrock/) |
| Ollama (local) | `agent-framework-ollama` | — | [Python](../python/packages/ollama/) |
| Azure AI Foundry | `agent-framework-azure-ai` | `Microsoft.Agents.AI.AzureAI` | [Python](../python/packages/azure-ai/) / [.NET](../dotnet/samples/GettingStarted/FoundryAgents/) |
| Foundry Local | `agent-framework-foundry-local` | — | [Python](../python/packages/foundry_local/) |

### 8.5 Human-in-the-Loop

The framework supports pausing execution for human input at both the tool and workflow levels.

**Tool-level approval** ([`function_tool_with_approval.py`](../python/samples/02-agents/tools/function_tool_with_approval.py)):

```python
@tool(approval_mode="always_require")
def delete_account(user_id: str) -> str:
    """Delete a user account. Requires human approval."""
    return f"Account {user_id} deleted."

# When the LLM calls this tool, the agent pauses and returns
# a function_approval_request in the response.
# The caller must provide approval before the tool executes.
result = await agent.run("Delete my account")
# result.user_input_requests contains the approval request
```

**Workflow-level HITL** ([`human-in-the-loop/`](../python/samples/03-workflows/human-in-the-loop/)):
- Workflows can emit `request_info` events that pause execution
- Human provides input, workflow resumes from checkpoint
- Supports review/approval cycles between agents

### 8.6 Checkpointing and Time-Travel

Workflows support **checkpointing** for fault tolerance and **time-travel debugging**.

```python
from agent_framework import WorkflowBuilder, InMemoryCheckpointStorage

storage = InMemoryCheckpointStorage()

workflow = WorkflowBuilder(start_executor=my_executor).build(
    checkpoint_storage=storage
)

# Run workflow — checkpoints are created at each superstep
events = await workflow.run("process this data")

# Resume from a checkpoint (e.g., after a failure)
events = await workflow.run(
    "process this data",
    checkpoint_id=last_checkpoint_id,
)
```

**How checkpointing works:**
- The workflow engine uses a **Pregel-style superstep model** where all ready executors run concurrently
- At each superstep boundary, a `WorkflowCheckpoint` is created containing: workflow state, per-executor message queues, iteration count, and a `graph_signature_hash` for topology validation
- State writes go to a **pending buffer** during execution; `state.commit()` is called at superstep boundaries
- Checkpoints can be stored in memory, on disk, or in any custom backend
- Time-travel: reload any checkpoint to replay from that point

See [`python/samples/03-workflows/checkpoint/`](../python/samples/03-workflows/checkpoint/) and [.NET checkpointing](../dotnet/src/Microsoft.Agents.AI.Workflows/).

---

## Summary: Agent Powerfulness Spectrum

| Level | Capabilities | Example |
|---|---|---|
| **L1: Simple Chat** | Single LLM call with instructions | [01_hello_agent.py](../python/samples/01-get-started/01_hello_agent.py) |
| **L2: Tool-Using** | LLM + local/remote tools (functions, MCP, APIs) | [02_add_tools.py](../python/samples/01-get-started/02_add_tools.py) |
| **L3: Memory-Aware** | L2 + session state + long-term memory (Mem0/Redis/RAG) | [04_memory.py](../python/samples/01-get-started/04_memory.py) |
| **L4: Multi-Agent** | L3 + workflow orchestration (handoff, group chat) | [handoff_simple.py](../python/samples/03-workflows/orchestrations/handoff_simple.py) |
| **L5: Production** | L4 + hosting + observability + HITL + checkpointing | [06_host_your_agent.py](../python/samples/01-get-started/06_host_your_agent.py) |
| **L6: Declarative** | L5 defined in YAML/JSON, no code required | [agent-samples/](../agent-samples/) |
