"""Command-line demo: run emails through the agent and review drafts in the terminal.

uv run email-agent                    # interactive: type an email
uv run email-agent --demo             # run the built-in sample emails
uv run email-agent --demo --auto-approve
"""

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from email_agent.graph import build_graph

SAMPLE_EMAILS = [
    "I was charged twice for my subscription! This is urgent!",
    "I was wondering if this was available in blue?",
    "Can you tell me how long the sale is on?",
    "The app crashes every time I try to upload a file.",
    "My subscription is going to end in a few months, what is the new rate?",
]


def ask_for_decision(review: dict) -> dict:
    """Show the draft and ask the reviewer to approve, edit, or reject it."""
    print("\n" + "=" * 70)
    print(f"REVIEW NEEDED  (intent: {review['intent']}, urgency: {review['urgency']})")
    print("-" * 70)
    print(f"Customer wrote:\n  {review['original_email']}\n")
    print(f"Draft reply:\n{review['draft_response']}")
    print("=" * 70)

    while True:
        choice = input("[a]pprove, [e]dit, or [r]eject? ").strip().lower()
        if choice in ("a", "approve"):
            return {"approved": True}
        if choice in ("e", "edit"):
            return {"approved": True, "edited_response": input("Your reply: ").strip()}
        if choice in ("r", "reject"):
            return {"approved": False}


def process_email(app, content: str, sender: str, auto_approve: bool) -> None:
    """Run one email through the graph, handling the human-review pause."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    email_id = f"email_{uuid.uuid4().hex[:6]}"

    result = app.invoke(
        {"email_content": content, "sender_email": sender, "email_id": email_id},
        config,
    )

    classification = result.get("classification", {})
    print(
        f"\n[{email_id}] intent={classification.get('intent')} "
        f"urgency={classification.get('urgency')} ticket={result.get('ticket_id')}"
    )

    # The graph paused at human_review: collect a decision and resume
    if "__interrupt__" in result:
        review = result["__interrupt__"][0].value
        decision = {"approved": True} if auto_approve else ask_for_decision(review)
        app.invoke(Command(resume=decision), config)
        if not decision["approved"]:
            print("Rejected - left for a person to handle.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph customer-support email agent")
    parser.add_argument("--demo", action="store_true", help="run the built-in sample emails")
    parser.add_argument("--auto-approve", action="store_true", help="approve every draft")
    parser.add_argument("--sender", default="customer@example.com")
    args = parser.parse_args()

    load_dotenv()
    app = build_graph(checkpointer=InMemorySaver())

    if args.demo:
        for content in SAMPLE_EMAILS:
            print(f"\n>>> {content}")
            process_email(app, content, args.sender, args.auto_approve)
        return

    print("Type a customer email and press Enter (empty line to quit).")
    while content := input("\nEmail: ").strip():
        process_email(app, content, args.sender, args.auto_approve)


if __name__ == "__main__":
    main()
