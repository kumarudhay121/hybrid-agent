from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class HybridAgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: list[dict[str, Any]]
    iteration: int
    route_decision: dict[str, Any]
    final_answer: str
