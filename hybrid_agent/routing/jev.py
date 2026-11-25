from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_typesafe import Choice, TypeSafeClassifier

from ..schemas import RouteDecision


class JevRouter:
    """Select the next tool with LangChain's official TypeSafeClassifier.

    Jev makes a bounded categorical decision. It does not generate tool
    arguments; the executor model is responsible for those.
    """

    def __init__(
        self,
        classifier: TypeSafeClassifier | None = None,
        *,
        question_name: str = "next_action",
        instructions: str = (
            "Which single tool, if any, should be called next to make progress "
            "on the user's request? Choose finish when no more tool calls are "
            "needed or the available tools cannot help."
        ),
    ) -> None:
        self.classifier = classifier or TypeSafeClassifier()
        self.question_name = question_name
        self.instructions = instructions

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
        criteria = dict(tools)
        criteria["finish"] = (
            "Do not call another tool. The task is complete, or no registered "
            "tool is appropriate."
        )

        state = {
            "messages": list(messages),
            "tool_results": tool_results,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "available_tools": criteria,
        }
        response = self.classifier.invoke(
            {
                "state": state,
                "questions": {
                    self.question_name: Choice(
                        instructions=self.instructions,
                        criteria=criteria,
                    )
                },
            },
            config=config,
        )

        selected = response.choices[self.question_name].choice
        if selected == "finish":
            return RouteDecision(
                action="finish",
                rationale="Jev selected finish.",
            )
        if selected not in tools:
            raise ValueError(
                f"Jev selected unknown tool {selected!r}; "
                f"available choices: {list(tools)}"
            )
        return RouteDecision(
            action="call_tool",
            tool_name=selected,
            rationale="Selected by Jev TypeSafeClassifier.",
        )
