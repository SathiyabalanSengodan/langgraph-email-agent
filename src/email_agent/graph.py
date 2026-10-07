"""Graph wiring.

START → read_email → classify_intent ─┬─ search_documentation ─┬─ write_response
                                      └─ bug_tracking ─────────┘
write_response → send_reply → END                (routine emails)
write_response → human_review → send_reply | END (urgent or complex emails)
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from email_agent.nodes import (
    bug_tracking,
    classify_intent,
    human_review,
    read_email,
    search_documentation,
    send_reply,
    write_response,
)
from email_agent.state import EmailAgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the email agent graph.

    A checkpointer is required for human review (interrupts) when running locally.
    LangGraph Studio / Server supply their own, so pass none there.
    """
    builder = StateGraph(EmailAgentState)

    builder.add_node("read_email", read_email)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("search_documentation", search_documentation)
    builder.add_node("bug_tracking", bug_tracking)
    builder.add_node("write_response", write_response)
    builder.add_node("human_review", human_review)
    builder.add_node("send_reply", send_reply)

    builder.add_edge(START, "read_email")
    builder.add_edge("read_email", "classify_intent")
    # Fan out: search docs and create a ticket in parallel (same superstep)
    builder.add_edge("classify_intent", "search_documentation")
    builder.add_edge("classify_intent", "bug_tracking")
    # Fan in: write_response waits for both
    builder.add_edge(["search_documentation", "bug_tracking"], "write_response")
    # write_response and human_review route themselves with Command(goto=...)
    builder.add_edge("send_reply", END)

    return builder.compile(checkpointer=checkpointer)


# Entry point for LangGraph Studio (see langgraph.json)
graph = build_graph()
