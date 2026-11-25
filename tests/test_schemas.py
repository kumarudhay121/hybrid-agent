import pytest
from pydantic import ValidationError

from hybrid_agent.schemas import RouteDecision


def test_call_tool_requires_name():
    with pytest.raises(ValidationError):
        RouteDecision(action="call_tool")


def test_finish_rejects_tool_name():
    with pytest.raises(ValidationError):
        RouteDecision(action="finish", tool_name="lookup")


def test_valid_tool_decision():
    decision = RouteDecision(action="call_tool", tool_name="lookup")
    assert decision.tool_name == "lookup"


def test_valid_finish_decision():
    assert RouteDecision(action="finish").action == "finish"
