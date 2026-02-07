#!/usr/bin/env python3
"""Interactive demo for job search system.

Run with: uv run python demo.py

Commands:
  /reset  - Start a new search session
  /budget - Show token usage and remaining budget
  /quit   - Exit the program
"""

import os
import sys

from chatbot import JobSearchChatbot
from search import format_results
from token_tracker import get_tracker


def check_api_key() -> bool:
    """Check if OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    return True


def main() -> None:
    """Run interactive job search demo."""
    if not check_api_key():
        sys.exit(1)

    print("Job Search Demo")
    print("=" * 60)
    print("Loading job data... (this may take a minute)")

    # Use a subset for faster loading (set DEMO_MAX_JOBS env var to override)
    max_jobs = int(os.environ.get("DEMO_MAX_JOBS", "100000"))

    # Initialize chatbot
    chatbot = JobSearchChatbot("jobs.jsonl", max_jobs=max_jobs)

    # Get token tracker
    tracker = get_tracker()

    print("\n" + "=" * 60)
    print("Job Search Ready!")
    print("=" * 60)
    print("\nEnter natural language queries to search for jobs.")
    print("Refine results by adding context in follow-up messages.")
    print("\nExample flow:")
    print("  > data science jobs")
    print("  > at nonprofits or social good companies")
    print("  > make it remote")
    print("\nCommands:")
    print("  /reset  - Start a new search session")
    print("  /budget - Show token usage ($10 budget)")
    print("  /quit   - Exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except EOFError, KeyboardInterrupt:
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            break

        if user_input.lower() == "/budget":
            print(tracker.get_summary())
            continue

        if user_input.lower() == "/reset":
            chatbot.reset()
            print("Search context reset. Start a new search!")
            continue

        # Run search
        try:
            response = chatbot.chat(user_input, top_k=10)
            print(format_results(response))
        except Exception as e:
            print(f"Error: {e}")

    # Show final usage on exit
    print("\n" + tracker.get_summary())
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
