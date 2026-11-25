from typing import Protocol, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from ..schemas import RouteDecision


class Router(Protocol):
    def route(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: dict[str, str],
        tool_results: list[dict],
        iteration: int,
        max_iterations: int,
        config: RunnableConfig | None = None,
    ) -> RouteDecision:
        ...
