"""Conversational job search chatbot with refinement capabilities."""

import os
import textwrap
from typing import Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from geocoding import geocode_with_fallback
from models import ConversationState, SearchContext, SearchResponse, SearchResult
from search import JobSearchEngine


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
    min_salary: float | None = Field(
        default=None, description="Minimum salary in USD (annual)"
    )
    max_salary: float | None = Field(
        default=None, description="Maximum salary in USD (annual)"
    )
    location_city: str | None = Field(default=None, description="City filter")
    location_state: str | None = Field(default=None, description="State filter")
    center_lat: float | None = Field(
        default=None, description="Center latitude for radius search"
    )
    center_lon: float | None = Field(
        default=None, description="Center longitude for radius search"
    )
    radius_miles: float | None = Field(
        default=None, description="Radius in miles for location search"
    )
    seniority_level: str | None = Field(
        default=None, description="Entry, Mid, Senior, Director, etc."
    )
    exclude_terms: list[str] = Field(
        default_factory=list,
        description="Terms to exclude from results (from negation)",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Important keywords for exact matching"
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
4. Extract salary requirements (convert to annual USD)
5. For ANY location-based search, use radius search:
   - "jobs in Cleveland" -> Cleveland with 30 mile radius
   - "jobs near Austin" -> Austin with 30 mile radius  
   - "within 50 miles of Denver" -> Denver with 50 mile radius
6. Identify seniority level (Entry, Mid, Senior, Director, VP, etc.)
7. Extract important keywords for exact matching
8. Handle NEGATION: Extract terms to exclude when user says "not", "no", "without", "exclude", "except"
   - "not entry level" -> exclude_terms: ["entry"]
   - "no startups" -> exclude_terms: ["startup"]
   - "without relocation" -> exclude_terms: ["relocation"]
9. Build a comprehensive search query (WITHOUT the negated terms)

IMPORTANT: When a user mentions a city/location:
- ALWAYS set location_city and location_state (for geocoding fallback)
- Try to set center_lat and center_lon if you know the coordinates
- Default radius is 30 miles unless user specifies otherwise

Common US city coordinates:
- New York: 40.7128, -74.0060
- Los Angeles: 34.0522, -118.2437
- Chicago: 41.8781, -87.6298
- San Francisco: 37.7749, -122.4194
- Seattle: 47.6062, -122.3321
- Austin: 30.2672, -97.7431
- Denver: 39.7392, -104.9903
- Boston: 42.3601, -71.0589
- Atlanta: 33.7490, -84.3880
- Miami: 25.7617, -80.1918
- Cleveland: 41.4993, -81.6944
- Cincinnati: 39.1031, -84.5120

For cities not listed, set location_city/location_state and leave coordinates null.
The system will look up coordinates automatically.

Respond with JSON only:
{
    "search_query": "complete query for embedding search (exclude negated terms)",
    "workplace_type": "Remote" | "Onsite" | "Hybrid" | null,
    "company_focus": "specific company type or null",
    "min_salary": number or null,
    "max_salary": number or null,
    "location_city": "city name for geocoding (REQUIRED if location mentioned)",
    "location_state": "state name for geocoding",
    "center_lat": latitude or null (will be geocoded if missing),
    "center_lon": longitude or null (will be geocoded if missing),
    "radius_miles": search radius in miles (default 30 if location mentioned),
    "seniority_level": "Entry" | "Mid" | "Senior" | "Director" | "VP" | null,
    "exclude_terms": ["terms", "to", "filter", "out"],
    "keywords": ["important", "exact", "match", "terms"],
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
            intent = ParsedIntent(**data)

            # Geocode if location mentioned but no coordinates
            if intent.location_city and (
                intent.center_lat is None or intent.center_lon is None
            ):
                lat, lon = geocode_with_fallback(
                    intent.location_city,
                    intent.location_state,
                    intent.center_lat,
                    intent.center_lon,
                )
                intent.center_lat = lat
                intent.center_lon = lon
                # Set default radius if not specified
                if intent.radius_miles is None and lat is not None:
                    intent.radius_miles = 30.0

            return intent
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
                    "min_salary": intent.min_salary,
                    "max_salary": intent.max_salary,
                    "location_city": intent.location_city,
                    "location_state": intent.location_state,
                    "seniority_level": intent.seniority_level,
                    "center_lat": intent.center_lat,
                    "center_lon": intent.center_lon,
                    "radius_miles": intent.radius_miles,
                },
            )
        )

        # Use hybrid search with keywords if provided
        keywords = intent.keywords if intent.keywords else None

        # Determine weights based on context
        if intent.company_focus:
            # Boost company embedding when filtering by company characteristics
            response = self.engine.hybrid_search(
                query=intent.search_query,
                keywords=keywords,
                top_k=top_k * 5,
                keyword_boost=0.1,
            )
            # Adjust scores by re-searching with company weight
            response = self.engine.search(
                query=intent.search_query,
                top_k=top_k * 5,
                weight_explicit=0.35,
                weight_inferred=0.25,
                weight_company=0.40,
            )
        else:
            response = self.engine.hybrid_search(
                query=intent.search_query,
                keywords=keywords,
                top_k=top_k * 5,
            )

        # Apply structured filters
        filtered_results = self.engine.filter_results(
            response.results,
            workplace_type=intent.workplace_type,
            min_salary=intent.min_salary,
            max_salary=intent.max_salary,
            location_city=intent.location_city,
            location_state=intent.location_state,
            seniority_level=intent.seniority_level,
            center_lat=intent.center_lat,
            center_lon=intent.center_lon,
            radius_miles=intent.radius_miles,
            exclude_terms=intent.exclude_terms if intent.exclude_terms else None,
        )

        # Company focus filter (search in company profile/activities)
        if intent.company_focus:
            company_filtered = []
            for result in filtered_results:
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
                    if any(term in searchable for term in focus_terms):
                        company_filtered.append(result)
            filtered_results = company_filtered

        # Limit to top_k
        filtered_results = filtered_results[:top_k]

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

    def go_back(self) -> bool:
        """Remove the last search context to undo the last refinement.

        Returns:
            True if there was a context to remove, False if already at start.
        """
        if len(self.state.search_contexts) > 0:
            # Remove last search context
            self.state.search_contexts.pop()
            # Remove last user and assistant messages (2 messages per interaction)
            if len(self.state.messages) >= 2:
                self.state.messages.pop()  # assistant
                self.state.messages.pop()  # user
            elif len(self.state.messages) >= 1:
                self.state.messages.pop()
            return True
        return False

    def get_context_depth(self) -> int:
        """Get the number of search refinements in current context."""
        return len(self.state.search_contexts)

    def get_segmented_query(self) -> str:
        """Get the accumulated query with separators showing each refinement.

        Returns:
            Query parts joined with ' | ' to show /back boundaries.
        """
        if not self.state.search_contexts:
            return ""
        return " | ".join(ctx.raw_query for ctx in self.state.search_contexts)

    def chat_stream(
        self, user_message: str, top_k: int | None = None, add_to_context: bool = True
    ) -> tuple[str, Generator[SearchResult, None, None]]:
        """Process a user message and return a stream of results.

        Args:
            user_message: The user's input message.
            top_k: Number of results to return. None for unlimited.
            add_to_context: Whether to add this query to the context (default True).

        Returns:
            Tuple of (search query, generator of SearchResult objects).
        """
        self._ensure_engine_loaded()

        if add_to_context:
            self.state.add_user_message(user_message)

        # Parse intent
        intent = self._parse_intent(user_message)

        if add_to_context:
            self.state.search_contexts.append(
                SearchContext(
                    raw_query=user_message,
                    parsed_intent=intent.search_query,
                    filters={
                        "workplace_type": intent.workplace_type,
                        "company_focus": intent.company_focus,
                        "min_salary": intent.min_salary,
                        "max_salary": intent.max_salary,
                        "location_city": intent.location_city,
                        "location_state": intent.location_state,
                        "seniority_level": intent.seniority_level,
                        "center_lat": intent.center_lat,
                        "center_lon": intent.center_lon,
                        "radius_miles": intent.radius_miles,
                    },
                )
            )

        # Use hybrid search with keywords if provided
        keywords = intent.keywords if intent.keywords else None

        # Search candidates: get many candidates for filtering
        search_k = (top_k * 5) if top_k else 5000

        # Determine weights based on context
        if intent.company_focus:
            response = self.engine.search(
                query=intent.search_query,
                top_k=search_k,
                weight_explicit=0.35,
                weight_inferred=0.25,
                weight_company=0.40,
            )
        else:
            response = self.engine.hybrid_search(
                query=intent.search_query,
                keywords=keywords,
                top_k=search_k,
            )

        # Return streaming generator
        def result_stream() -> Generator[SearchResult, None, None]:
            count = 0
            for result in self.engine.filter_results_stream(
                response.results,
                workplace_type=intent.workplace_type,
                min_salary=intent.min_salary,
                max_salary=intent.max_salary,
                location_city=intent.location_city,
                location_state=intent.location_state,
                seniority_level=intent.seniority_level,
                center_lat=intent.center_lat,
                center_lon=intent.center_lon,
                radius_miles=intent.radius_miles,
                exclude_terms=intent.exclude_terms if intent.exclude_terms else None,
                max_results=top_k,
            ):
                # Apply company focus filter if needed
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
                yield result
                count += 1
            # Update conversation history
            self.state.add_assistant_message(
                f"Found {count} jobs for '{intent.search_query}'"
            )

        return intent.search_query, result_stream()


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width, preserving paragraph breaks.

    Args:
        text: Text to wrap.
        width: Maximum line width.

    Returns:
        Wrapped text with preserved paragraph structure.
    """
    # Split on double newlines to preserve paragraphs
    paragraphs = text.split("\n\n")
    wrapped_paragraphs = []

    for para in paragraphs:
        # Replace single newlines with spaces (they're likely just line breaks)
        para = para.replace("\n", " ")
        # Collapse multiple spaces
        para = " ".join(para.split())
        if para:
            wrapped = textwrap.fill(para, width=width)
            wrapped_paragraphs.append(wrapped)

    return "\n\n".join(wrapped_paragraphs)


def format_single_result(result: SearchResult) -> str:
    """Format a single search result for display.

    Args:
        result: SearchResult to format.

    Returns:
        Formatted string representation.
    """
    job = result.job
    workplace_type = job.get_workplace_type() or "Not Specified"
    lines = [
        f"{result.rank}. {job.get_title()}",
        f"   Company: {job.get_company_name()}",
        f"   Location: {job.get_location()} ({workplace_type})",
        f"   Salary: {job.get_salary_display()}",
        f"   Score: {result.score:.3f}",
    ]
    if job.apply_url:
        lines.append(f"   Apply: {job.apply_url}")
    return "\n".join(lines)


def format_job_details(result: SearchResult) -> str:
    """Format full job details for display.

    Args:
        result: SearchResult to format.

    Returns:
        Formatted string with comprehensive job details.
    """
    job = result.job
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append(f"Job #{result.rank}: {job.get_title()}")
    lines.append("=" * 60)

    # Company section
    lines.append("\n--- Company ---")
    lines.append(f"Name: {job.get_company_name()}")

    if job.v7_processed_job_data and job.v7_processed_job_data.company_profile:
        profile = job.v7_processed_job_data.company_profile
        if profile.industry:
            lines.append(f"Industry: {profile.industry}")
        if profile.tagline:
            lines.append(f"Tagline: {profile.tagline}")
        if profile.website:
            lines.append(f"Website: {profile.website}")
        if profile.organization_types:
            lines.append(f"Type: {', '.join(profile.organization_types)}")

    # Work arrangement section
    lines.append("\n--- Work Arrangement ---")
    lines.append(f"Location: {job.get_location()}")
    workplace_type = job.get_workplace_type()
    if workplace_type:
        lines.append(f"Workplace Type: {workplace_type}")

    if job.v7_processed_job_data and job.v7_processed_job_data.work_arrangement:
        wa = job.v7_processed_job_data.work_arrangement
        if wa.commitment:
            lines.append(f"Commitment: {', '.join(wa.commitment)}")

    # Compensation section
    lines.append("\n--- Compensation ---")
    lines.append(f"Salary: {job.get_salary_display()}")

    if (
        job.v7_processed_job_data
        and job.v7_processed_job_data.compensation_and_benefits
    ):
        cb = job.v7_processed_job_data.compensation_and_benefits
        if cb.benefits:
            enabled_benefits = [k for k, v in cb.benefits.items() if v]
            if enabled_benefits:
                lines.append(f"Benefits: {', '.join(enabled_benefits)}")

    # Experience section
    if job.v7_processed_job_data and job.v7_processed_job_data.experience_requirements:
        exp = job.v7_processed_job_data.experience_requirements
        lines.append("\n--- Experience ---")
        if exp.seniority_level:
            lines.append(f"Seniority: {exp.seniority_level}")
        if exp.requirements_summary:
            lines.append(f"Requirements: {exp.requirements_summary}")

    # Description section
    if job.job_information and job.job_information.stripped_description:
        lines.append("\n--- Description ---")
        desc = job.job_information.stripped_description
        # Format description with proper line wrapping
        lines.append(wrap_text(desc, width=80))

    # Apply section
    if job.apply_url:
        lines.append("\n--- Apply ---")
        lines.append(job.apply_url)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def run_interactive() -> None:
    """Run an interactive chat session."""
    from token_tracker import get_tracker

    def print_commands(depth: int) -> None:
        """Print available commands."""
        back_hint = f" ({depth} deep)" if depth > 0 else " (at start)"
        print(f"Commands: /back{back_hint} /reset /budget /quit")

    print("Job Search Chatbot")
    print("=" * 40)
    print("Type your search queries to find jobs.")
    print("=" * 40)

    chatbot = JobSearchChatbot()
    tracker = get_tracker()
    print_commands(chatbot.get_context_depth())

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
            print_commands(chatbot.get_context_depth())
            continue

        if user_input.lower() == "/back":
            if chatbot.go_back():
                depth = chatbot.get_context_depth()
                if depth == 0:
                    print("Back to start. Enter a new search.")
                else:
                    print(f"Went back. Context depth: {depth}")
            else:
                print("Already at the start. Nothing to go back to.")
            print_commands(chatbot.get_context_depth())
            continue

        if user_input.lower() == "/reset":
            chatbot.reset()
            print("Search context reset. Start a new search!")
            print_commands(chatbot.get_context_depth())
            continue

        query, results_stream = chatbot.chat_stream(user_input)
        print(f"\nSearching for: '{query}'")
        print("-" * 40)
        count = 0
        for result in results_stream:
            print(format_single_result(result))
            print()
            count += 1
        print(f"Total: {count} results")

        print_commands(chatbot.get_context_depth())


if __name__ == "__main__":
    run_interactive()
