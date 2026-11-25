from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class RouteDecision(BaseModel):
    """Normalized routing decision used by the LangGraph controller."""

    action: Literal["call_tool", "finish"] = Field(
        description="Select an allowed tool or finish the tool phase."
    )
    tool_name: str | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "call_tool" and not self.tool_name:
            raise ValueError("tool_name is required for call_tool")
        if self.action == "finish" and self.tool_name is not None:
            raise ValueError("tool_name must be omitted for finish")
        return self
