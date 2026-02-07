"""Conversational job search chatbot with refinement capabilities."""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from models import ConversationState, SearchContext, SearchResponse
from search import JobSearchEngine, format_results


class ParsedIntent(BaseModel):
    """Parsed intent from user message."""

    search_query: str = Field(description="The core search query to embed")
    workplace_type: str | None = Field(
        default=None, description="Remote, Onsite, or Hybrid filter"
    )
    company_focus: str | None = Field(
        default=None,
        description="Company type/mission filter (e.g., nonprofit, social)",
    )
    is_new_search: bool = Field(
        default=False, description="True if this is a completely new search"
    )


SYSTEM_PROMPT = """You are a job search assistant that helps parse user queries.

Given the conversation history and new user message, extract the search intent.

Rules:
1. If user is refining, combine previous context with new filters
2. Identify if they want Remote, Onsite, or Hybrid work
3. Identify company characteristics (nonprofit, social good, tech, etc.)
4. Build a comprehensive search query that captures all context

Respond with JSON only:
{
    "search_query": "complete query for embedding search",
    "workplace_type": "Remote" | "Onsite" | "Hybrid" | null,
    "company_focus": "specific company type or null",
    "is_new_search": true/false
}"""


class JobSearchChatbot:
    """Conversational chatbot for job search with refinement."""

    def __init__(
        self,
        jobs_path: str | None = None,
        max_jobs: int | None = None,
    ) -> None:
        """Initialize the chatbot.

        Args:
            jobs_path: Path to jobs.jsonl file.
            max_jobs: Maximum jobs to load (None for all).
        """
        self.engine = JobSearchEngine(jobs_path)
        self.state = ConversationState()
        self.max_jobs = max_jobs
        self._engine_loaded = False

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
        )

    def _ensure_engine_loaded(self) -> None:
        """Load the search engine if not already loaded."""
        if not self._engine_loaded:
            self.engine.load_data(self.max_jobs)
            self._engine_loaded = True

    def _parse_intent(self, user_message: str) -> ParsedIntent:
        """Parse user intent using LLM.

        Args:
            user_message: The user's input message.

        Returns:
            ParsedIntent with extracted search parameters.
        """
        # Build context from previous searches
        context_parts = []
        for ctx in self.state.search_contexts:
            context_parts.append(f"Previous query: {ctx.raw_query}")
            if ctx.parsed_intent:
                context_parts.append(f"Parsed as: {ctx.parsed_intent}")

        context_str = "\n".join(context_parts) if context_parts else "No prior context"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Conversation context:\n{context_str}\n\n"
                    f"New user message: {user_message}\n\n"
                    "Extract the search intent as JSON:"
                )
            ),
        ]

        response = self.llm.invoke(messages)

        # Track token usage
        from token_tracker import track_chat

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            track_chat(
                input_tokens=response.usage_metadata.get("input_tokens", 0),
                output_tokens=response.usage_metadata.get("output_tokens", 0),
            )

        content = str(response.content)

        # Parse JSON response
        import json

        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            data = json.loads(content.strip())
            return ParsedIntent(**data)
        except json.JSONDecodeError, ValueError:
            # Fallback: use message as direct search
            return ParsedIntent(search_query=user_message, is_new_search=True)

    def chat(self, user_message: str, top_k: int = 10) -> SearchResponse:
        """Process a user message and return search results.

        Args:
            user_message: The user's input message.
            top_k: Number of results to return.

        Returns:
            SearchResponse with ranked job results.
        """
        self._ensure_engine_loaded()

        # Add user message to history
        self.state.add_user_message(user_message)

        # Parse intent
        intent = self._parse_intent(user_message)

        # Store search context
        self.state.search_contexts.append(
            SearchContext(
                raw_query=user_message,
                parsed_intent=intent.search_query,
                filters={
                    "workplace_type": intent.workplace_type,
                    "company_focus": intent.company_focus,
                },
            )
        )

        # Determine weights based on context
        if intent.company_focus:
            # Boost company embedding when filtering by company characteristics
            response = self.engine.search(
                query=intent.search_query,
                top_k=top_k * 3,
                weight_explicit=0.35,
                weight_inferred=0.25,
                weight_company=0.40,
            )
        else:
            response = self.engine.search(
                query=intent.search_query,
                top_k=top_k * 3,
            )

        # Apply post-search filters
        filtered_results = []
        for result in response.results:
            # Workplace type filter
            if intent.workplace_type:
                job_wt = result.job.get_workplace_type().lower()
                if job_wt != intent.workplace_type.lower():
                    continue

            # Company focus filter (search in company profile/activities)
            if intent.company_focus:
                v7 = result.job.v7_processed_job_data
                if v7 and v7.company_profile:
                    cp = v7.company_profile
                    searchable = " ".join(
                        [
                            cp.name or "",
                            cp.industry or "",
                            cp.tagline or "",
                            " ".join(cp.organization_types),
                            " ".join(cp.activities),
                        ]
                    ).lower()
                    focus_terms = intent.company_focus.lower().split()
                    if not any(term in searchable for term in focus_terms):
                        continue

            filtered_results.append(result)
            if len(filtered_results) >= top_k:
                break

        # Re-rank
        for i, r in enumerate(filtered_results, 1):
            r.rank = i

        final_response = SearchResponse(
            query=intent.search_query,
            results=filtered_results,
            total_found=len(filtered_results),
        )

        # Add assistant response to history
        summary = f"Found {len(filtered_results)} jobs for '{intent.search_query}'"
        self.state.add_assistant_message(summary)

        return final_response

    def reset(self) -> None:
        """Reset conversation state for a new search session."""
        self.state = ConversationState()


def run_interactive() -> None:
    """Run an interactive chat session."""
    from token_tracker import get_tracker

    print("Job Search Chatbot")
    print("=" * 40)
    print("Type your search queries. Commands:")
    print("  /reset  - Start a new search")
    print("  /budget - Show token usage")
    print("  /quit   - Exit")
    print("=" * 40)

    chatbot = JobSearchChatbot()
    tracker = get_tracker()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print(tracker.get_summary())
            print("Goodbye!")
            break

        if user_input.lower() == "/budget":
            print(tracker.get_summary())
            continue

        if user_input.lower() == "/reset":
            chatbot.reset()
            print("Search context reset. Start a new search!")
            continue

        response = chatbot.chat(user_input)
        print(format_results(response))


if __name__ == "__main__":
    run_interactive()
