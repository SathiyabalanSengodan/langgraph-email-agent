"""State schemas shared by every node in the graph."""

from typing import Literal, TypedDict


class EmailClassification(TypedDict):
    """Structured output the LLM returns when classifying an email."""

    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict, total=False):
    """Data passed between nodes. Each node returns only the keys it changes."""

    # Raw email data (provided as input)
    email_content: str
    sender_email: str
    email_id: str

    # Classification result
    classification: EmailClassification

    # Bug tracking
    ticket_id: str

    # Context gathered for the reply
    search_results: list[str]
    customer_history: dict

    # Generated content
    draft_response: str
