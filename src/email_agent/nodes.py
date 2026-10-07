"""Graph nodes. Each takes the current state and returns the keys it updates."""

import uuid
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from email_agent.llm import get_llm
from email_agent.state import EmailAgentState, EmailClassification


def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content.

    Placeholder: in production this would fetch from an inbox or API.
    """
    return {}


def classify_intent(state: EmailAgentState) -> dict:
    """Use the LLM to classify the email's intent and urgency."""
    structured_llm = get_llm().with_structured_output(EmailClassification)

    classification = structured_llm.invoke(
        f"""Analyze this customer email and classify it:

Email: {state["email_content"]}
From: {state["sender_email"]}

Provide classification, including intent, urgency, topic, and summary."""
    )
    return {"classification": classification}


def search_documentation(state: EmailAgentState) -> dict:
    """Search the knowledge base for relevant information.

    Placeholder: replace with a real search (vector store, docs API, ...).
    """
    classification = state.get("classification", {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}".strip()
    return {
        "search_results": [
            f"[placeholder doc 1 for '{query}']",
            f"[placeholder doc 2 for '{query}']",
        ]
    }


def bug_tracking(state: EmailAgentState) -> dict:
    """Create a bug-tracking ticket.

    Placeholder: replace with a call to Jira, Linear, GitHub Issues, ...
    """
    return {"ticket_id": f"BUG-{uuid.uuid4().hex[:8]}"}


def needs_human_review(classification: EmailClassification | dict) -> bool:
    """Decide whether a draft must be approved by a person before sending."""
    return (
        classification.get("urgency") in ("high", "critical")
        or classification.get("intent") == "complex"
    )


def write_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Draft a reply from the gathered context, then route on urgency and intent."""
    classification = state.get("classification", {})

    context_sections = []
    if state.get("search_results"):
        docs = "\n".join(f"- {doc}" for doc in state["search_results"])
        context_sections.append(f"Relevant documentation:\n{docs}")
    if state.get("customer_history"):
        tier = state["customer_history"].get("tier", "standard")
        context_sections.append(f"Customer tier: {tier}")
    context = "\n\n".join(context_sections)

    response = get_llm().invoke(
        f"""Draft a response to this customer email:
{state["email_content"]}

Email intent: {classification.get("intent", "unknown")}
Urgency level: {classification.get("urgency", "medium")}

{context}

Guidelines:
- Be professional and helpful
- Address their specific concern
- Use the provided documentation when relevant
- Be brief"""
    )

    return Command(
        update={"draft_response": response.content},
        goto="human_review" if needs_human_review(classification) else "send_reply",
    )


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", "__end__"]]:
    """Pause for a person to approve, edit, or reject the draft.

    Resume with Command(resume={"approved": True}) or
    Command(resume={"approved": True, "edited_response": "..."}) or
    Command(resume={"approved": False}).
    """
    classification = state.get("classification", {})

    # interrupt() must come first: code before it re-runs when the graph resumes
    decision = interrupt(
        {
            "email_id": state.get("email_id"),
            "original_email": state["email_content"],
            "draft_response": state.get("draft_response", ""),
            "urgency": classification.get("urgency"),
            "intent": classification.get("intent"),
            "action": "Please review and approve/edit this response",
        }
    )

    if isinstance(decision, dict) and decision.get("approved"):
        edited = decision.get("edited_response") or state.get("draft_response", "")
        return Command(update={"draft_response": edited}, goto="send_reply")

    # Rejected: a person will handle this email directly
    return Command(goto=END)


def send_reply(state: EmailAgentState) -> dict:
    """Send the reply.

    Placeholder: replace with an email service (SMTP, SendGrid, Gmail API, ...).
    """
    print(f"Sending reply to {state.get('sender_email')}: {state['draft_response'][:60]}...")
    return {}
