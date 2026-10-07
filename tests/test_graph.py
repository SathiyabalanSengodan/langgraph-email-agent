"""Graph tests using a fake LLM, so they run offline with no API key."""

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from email_agent import nodes
from email_agent.graph import build_graph


class FakeLLM:
    """Returns a fixed classification and draft instead of calling a model."""

    def __init__(self, classification: dict):
        self.classification = classification

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        if "classify" in prompt:
            return self.classification
        return AIMessage(content="Thanks for reaching out - we're on it.")


@pytest.fixture
def run(monkeypatch):
    """Build a fresh graph whose nodes use a fake LLM with the given classification."""

    def _run(intent: str, urgency: str):
        fake = FakeLLM({"intent": intent, "urgency": urgency, "topic": "t", "summary": "s"})
        monkeypatch.setattr(nodes, "get_llm", lambda: fake)
        app = build_graph(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "test"}}
        state = {"email_content": "hi", "sender_email": "a@b.c", "email_id": "e1"}
        return app, config, app.invoke(state, config)

    return _run


def test_routine_email_is_sent_without_review(run):
    _, _, result = run("question", "low")
    assert "__interrupt__" not in result
    assert result["draft_response"]
    assert result["ticket_id"].startswith("BUG-")
    assert result["search_results"]


@pytest.mark.parametrize(("intent", "urgency"), [("billing", "high"), ("complex", "low")])
def test_urgent_or_complex_email_pauses_for_review(run, intent, urgency):
    _, _, result = run(intent, urgency)
    assert result["__interrupt__"][0].value["intent"] == intent


def test_approved_with_edit_sends_edited_reply(run):
    app, config, _ = run("billing", "critical")
    final = app.invoke(Command(resume={"approved": True, "edited_response": "Edited"}), config)
    assert final["draft_response"] == "Edited"
    assert app.get_state(config).next == ()


def test_rejected_draft_ends_without_sending(run, capsys):
    app, config, _ = run("billing", "critical")
    app.invoke(Command(resume={"approved": False}), config)
    assert app.get_state(config).next == ()
    assert "Sending reply" not in capsys.readouterr().out


def test_needs_human_review_rules():
    assert nodes.needs_human_review({"urgency": "high", "intent": "question"})
    assert nodes.needs_human_review({"urgency": "low", "intent": "complex"})
    assert not nodes.needs_human_review({"urgency": "medium", "intent": "billing"})
