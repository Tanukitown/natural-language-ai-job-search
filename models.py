"""Pydantic models for job search system."""

from typing import Any

from pydantic import BaseModel, Field


class GeoLocation(BaseModel):
    """Geographic coordinates."""

    lat: float
    lon: float


class WorkplaceLocation(BaseModel):
    """Location details for a job."""

    continent: str | None = None
    country_code: str | None = None
    state: str | None = None
    city: str | None = None
    county: str | None = None


class WorkArrangement(BaseModel):
    """Work arrangement details."""

    workplace_type: str | None = None  # Onsite, Remote, Hybrid
    commitment: list[str] = Field(default_factory=list)  # Full Time, Part Time
    workplace_locations: list[WorkplaceLocation] = Field(default_factory=list)


class Salary(BaseModel):
    """Salary information."""

    low: float | None = None
    high: float | None = None
    currency: str | None = None
    frequency: str | None = None


class CompensationBenefits(BaseModel):
    """Compensation and benefits details."""

    salary: Salary | None = None
    benefits: dict[str, bool] = Field(default_factory=dict)


class CompanyProfile(BaseModel):
    """Company information."""

    name: str | None = None
    website: str | None = None
    industry: str | None = None
    tagline: str | None = None
    organization_types: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)


class ExperienceRequirements(BaseModel):
    """Experience requirements."""

    seniority_level: str | None = None
    requirements_summary: str | None = None


class JobTitles(BaseModel):
    """Job title information."""

    explicit: dict[str, Any] | None = None
    inferred: list[dict[str, Any]] = Field(default_factory=list)


class V7ProcessedJobData(BaseModel):
    """Processed job data with embeddings and structured fields."""

    job_titles: JobTitles | None = None
    work_arrangement: WorkArrangement | None = None
    compensation_and_benefits: CompensationBenefits | None = None
    company_profile: CompanyProfile | None = None
    experience_requirements: ExperienceRequirements | None = None
    embedding_text_explicit: str | None = None
    embedding_text_inferred: str | None = None
    embedding_text_company: str | None = None
    embedding_explicit_vector: list[float] | None = None
    embedding_inferred_vector: list[float] | None = None
    embedding_company_vector: list[float] | None = None


class JobInformation(BaseModel):
    """Raw job information."""

    title: str | None = None
    description: str | None = None
    stripped_description: str | None = None


class Job(BaseModel):
    """Complete job posting."""

    id: str
    apply_url: str | None = None
    job_information: JobInformation | None = None
    v7_processed_job_data: V7ProcessedJobData | None = None
    geoloc: list[GeoLocation] = Field(default_factory=list, alias="_geoloc")

    def get_title(self) -> str:
        """Get the best available job title."""
        if self.v7_processed_job_data and self.v7_processed_job_data.job_titles:
            explicit = self.v7_processed_job_data.job_titles.explicit
            if explicit and isinstance(explicit, dict):
                return str(explicit.get("value", ""))
        if self.job_information and self.job_information.title:
            return self.job_information.title
        return "Unknown Title"

    def get_company_name(self) -> str:
        """Get the company name."""
        if self.v7_processed_job_data and self.v7_processed_job_data.company_profile:
            name = self.v7_processed_job_data.company_profile.name
            if name:
                return name
        return "Unknown Company"

    def get_location(self) -> str:
        """Get a formatted location string."""
        if self.v7_processed_job_data and self.v7_processed_job_data.work_arrangement:
            wa = self.v7_processed_job_data.work_arrangement
            if wa.workplace_type == "Remote":
                return "Remote"
            if wa.workplace_locations:
                loc = wa.workplace_locations[0]
                parts = [loc.city, loc.state, loc.country_code]
                return ", ".join(p for p in parts if p)
        return "Location not specified"

    def get_workplace_type(self) -> str:
        """Get the workplace type (Remote, Onsite, Hybrid)."""
        if self.v7_processed_job_data and self.v7_processed_job_data.work_arrangement:
            return self.v7_processed_job_data.work_arrangement.workplace_type or ""
        return ""


class SearchResult(BaseModel):
    """A single search result with relevance score."""

    job: Job
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Response from a search query."""

    query: str
    results: list[SearchResult]
    total_found: int


class ConversationMessage(BaseModel):
    """A message in the conversation history."""

    role: str  # "user" or "assistant"
    content: str


class SearchContext(BaseModel):
    """Context for refining searches."""

    raw_query: str
    parsed_intent: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    """State of the conversation for refinement."""

    messages: list[ConversationMessage] = Field(default_factory=list)
    search_contexts: list[SearchContext] = Field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(ConversationMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(ConversationMessage(role="assistant", content=content))

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context."""
        summaries = []
        for ctx in self.search_contexts:
            summaries.append(f"Query: {ctx.raw_query}")
            if ctx.parsed_intent:
                summaries.append(f"Intent: {ctx.parsed_intent}")
        return "\n".join(summaries)
