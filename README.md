# HybridAgent

A multi-model agent abstraction built on LangGraph.

- **Router (Jev):** uses LangChain's official `TypeSafeClassifier` and a categorical `Choice` to select the next tool or finish.
- **Executor (GPT/Gemini/Claude):** creates arguments for the selected tool using native tool calling, then generates the final answer.
- **LangGraph:** manages state, routing, tool execution, loops, streaming and checkpoint integration.

Status: early development (`0.1.0`). The LangChain TypeSafe integration is evolving; pin versions and test before production.

## Install

```bash
pip install -e ".[dev,jev,openai]"
```

Set `TYPESAFE_API_KEY` and `OPENAI_API_KEY` in your environment.

## Architecture

```text
START
  |
Jev Router (Choice: tool A / tool B / finish)
  |                         |
  | call_tool               | finish
  v                         v
Executor creates args     Executor creates final answer
  |
Tool execution
  |
Jev Router (loop)
```

Jev is a decision model, not a text-generation model. It chooses *which* tool should run. The executor model creates the tool arguments and writes the final response.

## Quick start

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_typesafe import TypeSafeClassifier

from hybrid_agent import create_hybrid_agent
from hybrid_agent.routing import JevRouter

@tool
def lookup_employee(employee_id: int) -> str:
    """Look up an employee by their ID."""
    return f"Employee {employee_id}: Priya Sharma, Engineering"

router = JevRouter(classifier=TypeSafeClassifier())
executor = ChatOpenAI(model="gpt-4.1")

agent = create_hybrid_agent(
    router=router,
    executor=executor,
    tools=[lookup_employee],
    max_iterations=6,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Find employee 101"}]
})
print(result["final_answer"])
```

## Design guarantees

- Jev returns a categorical tool choice; it does not generate tool arguments.
- The executor is bound to only the selected tool when generating arguments.
- The tool name is checked against the registered allowlist.
- Tool arguments are validated by the LangChain tool schema at invocation.
- A maximum iteration count prevents unbounded tool loops.
- The graph accepts a LangGraph checkpointer.
- Tool execution is sequential in v0.1.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Security

Treat model output and tool arguments as untrusted. Enforce authorization inside each tool, use timeouts for external calls, avoid arbitrary code execution tools, and do not log credentials or sensitive tool results.

## Roadmap

- [x] LangGraph graph factory
- [x] Jev router using official LangChain TypeSafe integration
- [x] Separate executor model
- [x] Tool allowlist and iteration guard
- [ ] Adapter contract tests against live TypeSafe API
- [ ] Async invocation and streaming facade
- [ ] Middleware, retries and tracing
- [ ] Persistent checkpoint examples
