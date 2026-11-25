from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from .routing.base import Router
from .schemas import RouteDecision
from .state import HybridAgentState


DEFAULT_EXECUTOR_PROMPT = (
    "You are the execution model for a multi-model agent. When given a selected "
    "tool, call that tool with appropriate arguments. When asked for a final "
    "response, answer using the conversation and tool results. Do not invent "
    "facts or tool results."
)


def create_hybrid_agent(
    *,
    router: Router,
    executor: BaseChatModel,
    tools: Sequence[BaseTool],
    max_iterations: int = 8,
    executor_system_prompt: str = DEFAULT_EXECUTOR_PROMPT,
    checkpointer: Any = None,
):
    """Build and compile a LangGraph agent with Jev routing and LLM execution."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    registry = {tool.name: tool for tool in tools}
    if len(registry) != len(tools):
        raise ValueError("Tool names must be unique")

    tool_descriptions = {tool.name: tool.description for tool in tools}

    def router_node(state: HybridAgentState) -> dict:
        iteration = state.get("iteration", 0)
        if iteration >= max_iterations:
            decision = RouteDecision(
                action="finish",
                rationale="Maximum tool-routing iterations reached.",
            )
        else:
            decision = router.route(
                messages=state.get("messages", []),
                tools=tool_descriptions,
                tool_results=state.get("tool_results", []),
                iteration=iteration,
                max_iterations=max_iterations,
            )
            if (
                decision.action == "call_tool"
                and decision.tool_name not in registry
            ):
                raise ValueError(
                    f"Router selected unregistered tool: {decision.tool_name!r}"
                )
        return {"route_decision": decision.model_dump()}

    def route_after_router(state: HybridAgentState) -> str:
        return (
            "execute_tool"
            if state.get("route_decision", {}).get("action") == "call_tool"
            else "finalize"
        )

    def execute_tool_node(state: HybridAgentState) -> dict:
        decision = state["route_decision"]
        tool_name = decision["tool_name"]
        selected_tool = registry[tool_name]

        # Bind only the selected tool. The executor supplies arguments, not
        # the tool choice; Jev owns that decision.
        tool_call_prompt = [
            SystemMessage(
                content=(
                    f"{executor_system_prompt}\n"
                    f"Jev selected exactly one tool: {tool_name}. "
                    "Call this tool once with arguments matching its schema. "
                    "Do not select a different tool."
                )
            ),
            *state.get("messages", []),
        ]
        response = executor.bind_tools([selected_tool]).invoke(tool_call_prompt)
        calls = getattr(response, "tool_calls", None) or []
        matching_calls = [call for call in calls if call.get("name") == tool_name]

        if len(matching_calls) != 1:
            error_text = (
                f"Executor did not return exactly one call to {tool_name!r}; "
                f"received {len(matching_calls)} matching calls."
            )
            call_id = f"hybrid-tool-{state.get('iteration', 0) + 1}"
            tool_message = ToolMessage(
                content=error_text,
                tool_call_id=call_id,
                name=tool_name,
                status="error",
            )
            record = {
                "tool_name": tool_name,
                "arguments": {},
                "result": error_text,
                "error": True,
            }
            return {
                "messages": [response, tool_message],
                "tool_results": [*state.get("tool_results", []), record],
                "iteration": state.get("iteration", 0) + 1,
            }

        call = matching_calls[0]
        arguments = call.get("args") or {}
        call_id = call.get("id") or f"hybrid-tool-{state.get('iteration', 0) + 1}"

        try:
            result = selected_tool.invoke(arguments)
            result_text = (
                result if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False, default=str)
            )
            is_error = False
        except Exception as exc:
            result_text = f"Tool execution failed ({type(exc).__name__}): {exc}"
            is_error = True

        tool_message = ToolMessage(
            content=result_text,
            tool_call_id=call_id,
            name=tool_name,
            status="error" if is_error else "success",
        )
        record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result_text,
            "error": is_error,
        }
        return {
            "messages": [response, tool_message],
            "tool_results": [*state.get("tool_results", []), record],
            "iteration": state.get("iteration", 0) + 1,
        }

    def finalize_node(state: HybridAgentState) -> dict:
        history = state.get("tool_results", [])
        context = (
            "Tool execution history (JSON):\n"
            + json.dumps(history, ensure_ascii=False, default=str)
        )
        response = executor.invoke(
            [
                SystemMessage(content=executor_system_prompt),
                SystemMessage(content=context),
                *state.get("messages", []),
            ]
        )
        content = response.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        return {
            "messages": [AIMessage(content=content)],
            "final_answer": content,
        }

    graph = StateGraph(HybridAgentState)
    graph.add_node("router", router_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"execute_tool": "execute_tool", "finalize": "finalize"},
    )
    graph.add_edge("execute_tool", "router")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
