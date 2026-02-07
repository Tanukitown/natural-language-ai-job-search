#!/usr/bin/env python3
"""Interactive demo for job search system.

Run with: uv run python demo.py

Commands:
  /back   - Undo last refinement
  /reset  - Start a new search session
  /budget - Show token usage and remaining budget
  /quit   - Exit the program

Pagination:
  /next       - Show next page of results
  /prev       - Show previous page of results
  /first      - Return to the first page of results
  /details N  - Show full details for job number N
"""

import os
import sys
import warnings
from collections.abc import Generator

# Suppress Pydantic V1 compatibility warning from langchain
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from chatbot import JobSearchChatbot, format_job_details, format_single_result  # noqa: E402
from models import SearchResult  # noqa: E402
from token_tracker import get_tracker  # noqa: E402

PAGE_SIZE = 10


def clear_screen() -> None:
    """Clear terminal screen, scrollback buffer, and move cursor to top."""
    # ANSI escape codes: clear screen + clear scrollback + move cursor to home
    print("\033[2J\033[3J\033[H", end="", flush=True)


def check_api_key() -> bool:
    """Check if OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    return True


def print_commands(context_depth: int) -> None:
    """Print available commands."""
    back_hint = f" ({context_depth} deep)" if context_depth > 0 else " (at start)"
    print(f"Commands: /back{back_hint} /reset /budget /quit")


def print_pagination(current_page: int, total_fetched: int, has_more: bool) -> None:
    """Print pagination controls on a separate line."""
    start = current_page * PAGE_SIZE + 1
    end = min((current_page + 1) * PAGE_SIZE, total_fetched)

    first_hint = "/first" if current_page > 1 else ""
    prev_hint = "/prev" if current_page > 0 else ""
    next_hint = "/next" if has_more or end < total_fetched else ""

    controls = " | ".join(filter(None, [first_hint, prev_hint, next_hint]))
    page_info = f"Page {current_page + 1} (showing {start}-{end} of {total_fetched}+)"
    if controls:
        print(f"{page_info}: {controls} | /details N")
    else:
        print(f"{page_info}: /details N")


def show_page(
    results: list[SearchResult], page: int, generator_exhausted: bool
) -> None:
    """Display a specific page of results."""
    start_idx = page * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_results = results[start_idx:end_idx]

    print("-" * 40)
    for result in page_results:
        print(format_single_result(result))
        print()

    total = len(results)
    showing_end = min(end_idx, total)

    if generator_exhausted:
        print(f"Showing {start_idx + 1}-{showing_end} of {total} results")
    else:
        print(f"Showing {start_idx + 1}-{showing_end} of {total}+ results")


def fetch_results_until(
    results: list[SearchResult],
    generator: Generator[SearchResult, None, None] | None,
    target_count: int,
) -> tuple[list[SearchResult], Generator[SearchResult, None, None] | None]:
    """Fetch results from generator until we have at least target_count.

    Returns updated results list and generator (None if exhausted).
    """
    if generator is None:
        return results, None

    while len(results) < target_count:
        try:
            result = next(generator)
            results.append(result)
        except StopIteration:
            return results, None

    return results, generator


def stream_first_page(
    generator: Generator[SearchResult, None, None],
) -> tuple[list[SearchResult], Generator[SearchResult, None, None] | None]:
    """Stream and display the first page of results as they arrive.

    Returns (fetched results, remaining generator or None if exhausted).
    """
    results: list[SearchResult] = []
    print("-" * 40, flush=True)

    for result in generator:
        results.append(result)
        print(format_single_result(result), flush=True)
        print(flush=True)
        if len(results) >= PAGE_SIZE:
            # More results available
            print(f"Showing 1-{len(results)} of {len(results)}+ results")
            return results, generator

    # Generator exhausted
    if results:
        print(f"Showing 1-{len(results)} of {len(results)} results")
    return results, None


def main() -> None:
    """Run interactive job search demo."""
    if not check_api_key():
        sys.exit(1)

    clear_screen()
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
    print("=" * 60)

    # Results pagination state
    fetched_results: list[SearchResult] = []
    results_generator: Generator[SearchResult, None, None] | None = None
    current_page = 0
    viewing_details: int | None = None  # Rank of job being viewed, or None

    print_commands(chatbot.get_context_depth())

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
            print_commands(chatbot.get_context_depth())
            if fetched_results:
                has_more = results_generator is not None or (
                    current_page + 1
                ) * PAGE_SIZE < len(fetched_results)
                print_pagination(current_page, len(fetched_results), has_more)
            continue

        if user_input.lower() == "/next":
            if not fetched_results:
                print("No search results. Run a search first.")
                print_commands(chatbot.get_context_depth())
                continue

            next_page_start = (current_page + 1) * PAGE_SIZE
            # Fetch more results if needed
            fetched_results, results_generator = fetch_results_until(
                fetched_results, results_generator, next_page_start + PAGE_SIZE
            )

            if next_page_start >= len(fetched_results):
                print("No more results available.")
            else:
                current_page += 1
                show_page(fetched_results, current_page, results_generator is None)

            print_commands(chatbot.get_context_depth())
            has_more = results_generator is not None or (
                current_page + 1
            ) * PAGE_SIZE < len(fetched_results)
            print_pagination(current_page, len(fetched_results), has_more)
            continue

        if user_input.lower() == "/prev":
            if not fetched_results:
                print("No search results. Run a search first.")
                print_commands(chatbot.get_context_depth())
                continue

            if current_page == 0:
                print("Already on the first page.")
            else:
                current_page -= 1
                show_page(fetched_results, current_page, results_generator is None)

            print_commands(chatbot.get_context_depth())
            has_more = results_generator is not None or (
                current_page + 1
            ) * PAGE_SIZE < len(fetched_results)
            print_pagination(current_page, len(fetched_results), has_more)
            continue

        if user_input.lower() == "/first":
            if not fetched_results:
                print("No search results. Run a search first.")
                print_commands(chatbot.get_context_depth())
                continue

            if current_page == 0:
                print("Already on the first page.")
            else:
                current_page = 0
                show_page(fetched_results, current_page, results_generator is None)

            print_commands(chatbot.get_context_depth())
            has_more = results_generator is not None or (
                current_page + 1
            ) * PAGE_SIZE < len(fetched_results)
            print_pagination(current_page, len(fetched_results), has_more)
            continue

        if user_input.lower() == "/return":
            if viewing_details is None:
                print("Not viewing job details. Nothing to return from.")
            else:
                viewing_details = None
                show_page(fetched_results, current_page, results_generator is None)

            print_commands(chatbot.get_context_depth())
            if fetched_results:
                has_more = results_generator is not None or (
                    current_page + 1
                ) * PAGE_SIZE < len(fetched_results)
                print_pagination(current_page, len(fetched_results), has_more)
            continue

        if user_input.lower().startswith("/details"):
            parts = user_input.split()
            if len(parts) != 2:
                print("Usage: /details <number>")
                print_commands(chatbot.get_context_depth())
                continue

            try:
                job_rank = int(parts[1])
            except ValueError:
                print("Invalid number. Usage: /details <number>")
                print_commands(chatbot.get_context_depth())
                continue

            if not fetched_results:
                print("No search results. Run a search first.")
                print_commands(chatbot.get_context_depth())
                continue

            # Find the job by rank (fetch more if needed)
            fetched_results, results_generator = fetch_results_until(
                fetched_results, results_generator, job_rank
            )

            found_result = None
            for result in fetched_results:
                if result.rank == job_rank:
                    found_result = result
                    break

            if found_result is None:
                print(f"Job #{job_rank} not found in results.")
                print_commands(chatbot.get_context_depth())
                has_more = results_generator is not None or (
                    current_page + 1
                ) * PAGE_SIZE < len(fetched_results)
                print_pagination(current_page, len(fetched_results), has_more)
                continue

            viewing_details = job_rank
            clear_screen()
            print(format_job_details(found_result))
            print("Commands: /return")
            continue

        if user_input.lower() == "/back":
            if chatbot.go_back():
                depth = chatbot.get_context_depth()
                if depth == 0:
                    print("Back to start. Enter a new search.")
                else:
                    print(f"Went back. Context depth: {depth}")
                # Clear results on back
                fetched_results = []
                results_generator = None
                current_page = 0
            else:
                print("Already at the start. Nothing to go back to.")
            print_commands(chatbot.get_context_depth())
            continue

        if user_input.lower() == "/reset":
            chatbot.reset()
            fetched_results = []
            results_generator = None
            current_page = 0
            print("Search context reset. Start a new search!")
            print_commands(chatbot.get_context_depth())
            continue

        # Run search
        try:
            clear_screen()
            print("Parsing query...", flush=True)
            query, results_generator = chatbot.chat_stream(user_input)
            segmented = chatbot.get_segmented_query()
            print(f"Searching for: '{segmented}'", flush=True)

            # Reset pagination state for new search
            fetched_results = []
            current_page = 0

            # Stream first page of results (displays as they arrive)
            fetched_results, results_generator = stream_first_page(results_generator)

            if not fetched_results:
                print("No results found.")

        except Exception as e:
            print(f"Error: {e}")
            fetched_results = []
            results_generator = None

        print_commands(chatbot.get_context_depth())
        if fetched_results:
            has_more = results_generator is not None or (
                current_page + 1
            ) * PAGE_SIZE < len(fetched_results)
            print_pagination(current_page, len(fetched_results), has_more)

    # Show final usage on exit
    print("\n" + tracker.get_summary())
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
