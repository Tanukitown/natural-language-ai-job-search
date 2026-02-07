"""Job search engine using FAISS and OpenAI embeddings."""

import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Generator

import faiss
import numpy as np
from openai import OpenAI

from models import (
    Job,
    SearchResponse,
    SearchResult,
)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in miles.
    """
    # Earth's radius in miles
    R = 3959.0

    # Convert to radians
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return R * c


# Embedding weights for different search aspects
WEIGHT_EXPLICIT = 0.5  # Job title, skills, requirements
WEIGHT_INFERRED = 0.3  # Related/implied qualifications
WEIGHT_COMPANY = 0.2  # Company characteristics

EMBEDDING_DIM = 1536
CACHE_DIR = Path(".cache")


class JobSearchEngine:
    """Search engine for job postings using vector similarity."""

    def __init__(self, jobs_path: str | Path | None = None) -> None:
        """Initialize the search engine.

        Args:
            jobs_path: Path to the jobs.jsonl file. If None, uses default.
        """
        self.jobs_path = Path(jobs_path) if jobs_path else Path("jobs.jsonl")
        self.jobs: list[Job] = []
        self.job_id_to_index: dict[str, int] = {}  # Fast lookup for hybrid search
        self.openai_client: OpenAI | None = None

        # FAISS indices for each embedding type
        self.index_explicit: faiss.IndexFlatIP | None = None
        self.index_inferred: faiss.IndexFlatIP | None = None
        self.index_company: faiss.IndexFlatIP | None = None

        # Keyword search index (job_idx -> searchable text)
        self.keyword_index: dict[int, str] = {}

        self._loaded = False

    def _get_cache_path(self, max_jobs: int | None) -> Path:
        """Get cache file path based on job count."""
        suffix = f"_{max_jobs}" if max_jobs else "_all"
        return CACHE_DIR / f"search_cache{suffix}.pkl"

    def _save_cache(self, max_jobs: int | None) -> None:
        """Save indices and jobs to disk cache."""
        CACHE_DIR.mkdir(exist_ok=True)
        cache_path = self._get_cache_path(max_jobs)

        cache_data = {
            "jobs": self.jobs,
            "keyword_index": self.keyword_index,
            "index_explicit": faiss.serialize_index(self.index_explicit),
            "index_inferred": faiss.serialize_index(self.index_inferred),
            "index_company": faiss.serialize_index(self.index_company),
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved to {cache_path}")

    def _load_cache(self, max_jobs: int | None) -> bool:
        """Load indices and jobs from disk cache.

        Returns:
            True if cache was loaded successfully.
        """
        cache_path = self._get_cache_path(max_jobs)
        if not cache_path.exists():
            return False

        try:
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            self.jobs = cache_data["jobs"]
            self.keyword_index = cache_data["keyword_index"]
            self.index_explicit = faiss.deserialize_index(cache_data["index_explicit"])
            self.index_inferred = faiss.deserialize_index(cache_data["index_inferred"])
            self.index_company = faiss.deserialize_index(cache_data["index_company"])

            # Rebuild job ID to index mapping
            self.job_id_to_index = {job.id: i for i, job in enumerate(self.jobs)}

            print(f"Loaded {len(self.jobs)} jobs from cache.")
            return True
        except Exception as e:
            print(f"Cache load failed: {e}")
            return False

    def _build_keyword_text(self, job: Job) -> str:
        """Build searchable text for keyword matching."""
        parts = [job.get_title(), job.get_company_name()]

        v7 = job.v7_processed_job_data
        if v7:
            if v7.embedding_text_explicit:
                parts.append(v7.embedding_text_explicit)
            if v7.company_profile:
                if v7.company_profile.industry:
                    parts.append(v7.company_profile.industry)
                parts.extend(v7.company_profile.activities)

        return " ".join(parts).lower()

    def load_data(self, max_jobs: int | None = None, use_cache: bool = True) -> None:
        """Load jobs from JSONL file and build FAISS indices.

        Args:
            max_jobs: Maximum number of jobs to load. None for all.
            use_cache: Whether to use disk cache for faster loading.
        """
        if self._loaded:
            return

        # Try loading from cache first
        if use_cache and self._load_cache(max_jobs):
            self._loaded = True
            return

        print(f"Loading jobs from {self.jobs_path}...")

        embeddings_explicit: list[list[float]] = []
        embeddings_inferred: list[list[float]] = []
        embeddings_company: list[list[float]] = []

        with open(self.jobs_path, "r") as f:
            for i, line in enumerate(f):
                if max_jobs and i >= max_jobs:
                    break

                data = json.loads(line)
                job = Job.model_validate(data)
                self.jobs.append(job)
                self.job_id_to_index[job.id] = i  # Fast lookup for hybrid search

                # Extract embeddings
                v7 = job.v7_processed_job_data
                if v7:
                    explicit = v7.embedding_explicit_vector or [0.0] * EMBEDDING_DIM
                    inferred = v7.embedding_inferred_vector or [0.0] * EMBEDDING_DIM
                    company = v7.embedding_company_vector or [0.0] * EMBEDDING_DIM
                else:
                    explicit = [0.0] * EMBEDDING_DIM
                    inferred = [0.0] * EMBEDDING_DIM
                    company = [0.0] * EMBEDDING_DIM

                embeddings_explicit.append(explicit)
                embeddings_inferred.append(inferred)
                embeddings_company.append(company)

                # Build keyword index
                self.keyword_index[i] = self._build_keyword_text(job)

                if (i + 1) % 10000 == 0:
                    print(f"  Loaded {i + 1} jobs...")

        print(f"Loaded {len(self.jobs)} jobs. Building indices...")

        # Convert to numpy and normalize for cosine similarity
        explicit_np = self._normalize(np.array(embeddings_explicit, dtype=np.float32))
        inferred_np = self._normalize(np.array(embeddings_inferred, dtype=np.float32))
        company_np = self._normalize(np.array(embeddings_company, dtype=np.float32))

        # Build FAISS indices (using inner product on normalized vectors = cosine)
        self.index_explicit = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index_inferred = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index_company = faiss.IndexFlatIP(EMBEDDING_DIM)

        self.index_explicit.add(explicit_np)  # pyrefly: ignore
        self.index_inferred.add(inferred_np)  # pyrefly: ignore
        self.index_company.add(company_np)  # pyrefly: ignore

        self._loaded = True
        print("Indices built successfully.")

        # Save cache for faster future loading
        if use_cache:
            self._save_cache(max_jobs)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a query text using OpenAI."""
        if self.openai_client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.openai_client = OpenAI(api_key=api_key)

        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )

        # Track token usage
        from token_tracker import track_embedding

        if response.usage:
            track_embedding(response.usage.total_tokens)

        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return self._normalize(embedding.reshape(1, -1))

    def search(
        self,
        query: str,
        top_k: int = 10,
        weight_explicit: float = WEIGHT_EXPLICIT,
        weight_inferred: float = WEIGHT_INFERRED,
        weight_company: float = WEIGHT_COMPANY,
    ) -> SearchResponse:
        """Search for jobs matching the query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            weight_explicit: Weight for explicit job data matching.
            weight_inferred: Weight for inferred/related matching.
            weight_company: Weight for company matching.

        Returns:
            SearchResponse with ranked results.
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Type narrowing for indices
        assert self.index_explicit is not None
        assert self.index_inferred is not None
        assert self.index_company is not None

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Search each index - get more results to combine
        search_k = min(top_k * 3, len(self.jobs))

        scores_explicit, indices_explicit = (
            self.index_explicit.search(  # pyrefly: ignore
                query_embedding, search_k
            )
        )
        scores_inferred, indices_inferred = (
            self.index_inferred.search(  # pyrefly: ignore
                query_embedding, search_k
            )
        )
        scores_company, indices_company = self.index_company.search(  # pyrefly: ignore
            query_embedding, search_k
        )

        # Combine scores using weights
        combined_scores: dict[int, float] = {}

        for idx, score in zip(indices_explicit[0], scores_explicit[0]):
            combined_scores[int(idx)] = combined_scores.get(int(idx), 0) + (
                float(score) * weight_explicit
            )

        for idx, score in zip(indices_inferred[0], scores_inferred[0]):
            combined_scores[int(idx)] = combined_scores.get(int(idx), 0) + (
                float(score) * weight_inferred
            )

        for idx, score in zip(indices_company[0], scores_company[0]):
            combined_scores[int(idx)] = combined_scores.get(int(idx), 0) + (
                float(score) * weight_company
            )

        # Sort by combined score and get top_k
        sorted_indices = sorted(
            combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True
        )[:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            results.append(
                SearchResult(
                    job=self.jobs[idx],
                    score=combined_scores[idx],
                    rank=rank,
                )
            )

        return SearchResponse(
            query=query,
            results=results,
            total_found=len(combined_scores),
        )

    def keyword_search(self, keywords: list[str], top_k: int = 100) -> set[int]:
        """Find job indices matching keywords.

        Args:
            keywords: List of keywords to search for.
            top_k: Maximum number of results.

        Returns:
            Set of job indices matching any keyword.
        """
        matches: set[int] = set()
        keywords_lower = [k.lower() for k in keywords]

        for idx, text in self.keyword_index.items():
            if any(kw in text for kw in keywords_lower):
                matches.add(idx)
                if len(matches) >= top_k:
                    break

        return matches

    def hybrid_search(
        self,
        query: str,
        keywords: list[str] | None = None,
        top_k: int = 10,
        keyword_boost: float = 0.1,
    ) -> SearchResponse:
        """Combine embedding search with keyword matching.

        Args:
            query: Natural language search query.
            keywords: Optional explicit keywords to boost.
            top_k: Number of results to return.
            keyword_boost: Score boost for keyword matches.

        Returns:
            SearchResponse with ranked results.
        """
        # Get embedding search results
        response = self.search(query, top_k=top_k * 3)

        # Extract keywords from query if not provided
        if keywords is None:
            # Simple keyword extraction: words > 3 chars
            keywords = [w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) > 3]

        keyword_matches = self.keyword_search(keywords, top_k=1000)

        # Boost scores for keyword matches (using fast ID lookup)
        boosted_results = []
        for result in response.results:
            job_idx = self.job_id_to_index.get(result.job.id)
            if job_idx is not None and job_idx in keyword_matches:
                result.score += keyword_boost
            boosted_results.append(result)

        # Re-sort by boosted score
        boosted_results.sort(key=lambda r: r.score, reverse=True)

        # Re-rank
        for i, result in enumerate(boosted_results[:top_k], 1):
            result.rank = i

        return SearchResponse(
            query=query,
            results=boosted_results[:top_k],
            total_found=len(boosted_results),
        )

    def _build_searchable_text(self, job: Job) -> str:
        """Build a searchable text string from job for exclusion filtering.

        Args:
            job: Job to build searchable text from.

        Returns:
            Lowercase string containing title, description, company, seniority.
        """
        parts = [job.get_title(), job.get_company_name()]

        v7 = job.v7_processed_job_data
        if v7:
            if (
                v7.experience_requirements
                and v7.experience_requirements.seniority_level
            ):
                parts.append(v7.experience_requirements.seniority_level)
            if v7.company_profile:
                if v7.company_profile.industry:
                    parts.append(v7.company_profile.industry)
                parts.extend(v7.company_profile.organization_types)

        if job.job_information and job.job_information.stripped_description:
            parts.append(job.job_information.stripped_description[:500])

        return " ".join(p for p in parts if p).lower()

    def filter_results(
        self,
        results: list[SearchResult],
        workplace_type: str | None = None,
        min_salary: float | None = None,
        max_salary: float | None = None,
        location_city: str | None = None,
        location_state: str | None = None,
        seniority_level: str | None = None,
        center_lat: float | None = None,
        center_lon: float | None = None,
        radius_miles: float | None = None,
        exclude_terms: list[str] | None = None,
    ) -> list[SearchResult]:
        """Apply structured filters to search results.

        Args:
            results: List of search results to filter.
            workplace_type: Remote, Onsite, or Hybrid.
            min_salary: Minimum salary filter.
            max_salary: Maximum salary filter.
            location_city: Filter by city name.
            location_state: Filter by state.
            seniority_level: Filter by seniority (Entry, Mid, Senior, etc.)
            center_lat: Center latitude for radius search.
            center_lon: Center longitude for radius search.
            radius_miles: Search radius in miles.
            exclude_terms: Terms to exclude from results.

        Returns:
            Filtered list of results.
        """
        filtered = []

        for result in results:
            job = result.job
            v7 = job.v7_processed_job_data

            # Workplace type filter
            if workplace_type:
                job_wt = job.get_workplace_type().lower()
                if job_wt != workplace_type.lower():
                    continue

            # Salary filter
            if min_salary or max_salary:
                has_salary = (
                    v7
                    and v7.compensation_and_benefits
                    and v7.compensation_and_benefits.salary
                    and (
                        v7.compensation_and_benefits.salary.low is not None
                        or v7.compensation_and_benefits.salary.high is not None
                    )
                )
                if not has_salary:
                    # Skip jobs without salary data when salary filter is active
                    continue
                sal = v7.compensation_and_benefits.salary  # pyrefly: ignore
                if min_salary and sal.high and sal.high < min_salary:
                    continue
                if max_salary and sal.low and sal.low > max_salary:
                    continue

            # Radius filter (requires center coordinates and radius)
            if center_lat is not None and center_lon is not None and radius_miles:
                # Check geoloc field first
                if job.geoloc:
                    # Find minimum distance to any job location
                    min_dist = min(
                        haversine_distance(center_lat, center_lon, geo.lat, geo.lon)
                        for geo in job.geoloc
                    )
                    if min_dist > radius_miles:
                        continue
                elif job.get_workplace_type().lower() == "remote":
                    # Remote jobs pass radius filter (work from anywhere)
                    pass
                else:
                    # No location data and not remote - skip
                    continue
            # Location filter (city/state text match)
            elif location_city or location_state:
                if (
                    v7
                    and v7.work_arrangement
                    and v7.work_arrangement.workplace_locations
                ):
                    loc = v7.work_arrangement.workplace_locations[0]
                    if location_city and loc.city:
                        if location_city.lower() not in loc.city.lower():
                            continue
                    if location_state and loc.state:
                        if location_state.lower() not in loc.state.lower():
                            continue
                elif job.get_workplace_type().lower() != "remote":
                    continue  # Skip if no location data and not remote

            # Seniority filter
            if seniority_level and v7 and v7.experience_requirements:
                job_seniority = v7.experience_requirements.seniority_level or ""
                if seniority_level.lower() not in job_seniority.lower():
                    continue

            # Exclusion filter (negation handling)
            if exclude_terms:
                searchable = self._build_searchable_text(job)
                if any(term.lower() in searchable for term in exclude_terms):
                    continue

            filtered.append(result)

        return filtered

    def search_with_filters(
        self,
        query: str,
        top_k: int = 10,
        workplace_type: str | None = None,
        company_types: list[str] | None = None,
    ) -> SearchResponse:
        """Search with additional filters.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            workplace_type: Filter by Remote, Onsite, or Hybrid.
            company_types: Filter by company organization types.

        Returns:
            SearchResponse with filtered and ranked results.
        """
        # Get more results to filter from
        response = self.search(query, top_k=top_k * 5)

        filtered_results = []
        for result in response.results:
            # Apply workplace filter
            if workplace_type:
                job_workplace = result.job.get_workplace_type()
                if job_workplace.lower() != workplace_type.lower():
                    continue

            # Apply company type filter
            if company_types:
                v7 = result.job.v7_processed_job_data
                if v7 and v7.company_profile and v7.company_profile.organization_types:
                    org_types = [
                        t.lower() for t in v7.company_profile.organization_types
                    ]
                    if not any(
                        ct.lower() in " ".join(org_types) for ct in company_types
                    ):
                        continue
                else:
                    continue

            filtered_results.append(result)
            if len(filtered_results) >= top_k:
                break

        # Re-rank
        for i, result in enumerate(filtered_results, 1):
            result.rank = i

        return SearchResponse(
            query=query,
            results=filtered_results,
            total_found=len(filtered_results),
        )

    def filter_results_stream(
        self,
        results: list[SearchResult],
        workplace_type: str | None = None,
        min_salary: float | None = None,
        max_salary: float | None = None,
        location_city: str | None = None,
        location_state: str | None = None,
        seniority_level: str | None = None,
        center_lat: float | None = None,
        center_lon: float | None = None,
        radius_miles: float | None = None,
        exclude_terms: list[str] | None = None,
        max_results: int | None = None,
    ) -> Generator[SearchResult, None, None]:
        """Stream search results through filters, yielding matches as found.

        Args:
            results: List of search results to filter.
            workplace_type: Remote, Onsite, or Hybrid.
            min_salary: Minimum salary filter.
            max_salary: Maximum salary filter.
            location_city: Filter by city name.
            location_state: Filter by state.
            seniority_level: Filter by seniority (Entry, Mid, Senior, etc.)
            center_lat: Center latitude for radius search.
            center_lon: Center longitude for radius search.
            radius_miles: Search radius in miles.
            exclude_terms: Terms to exclude from results.
            max_results: Maximum results to yield (None for unlimited).

        Yields:
            SearchResult objects that pass all filters.
        """
        count = 0
        rank = 0

        for result in results:
            if max_results is not None and count >= max_results:
                return

            job = result.job
            v7 = job.v7_processed_job_data

            # Workplace type filter
            if workplace_type:
                job_wt = job.get_workplace_type().lower()
                if job_wt != workplace_type.lower():
                    continue

            # Salary filter
            if min_salary or max_salary:
                has_salary = (
                    v7
                    and v7.compensation_and_benefits
                    and v7.compensation_and_benefits.salary
                    and (
                        v7.compensation_and_benefits.salary.low is not None
                        or v7.compensation_and_benefits.salary.high is not None
                    )
                )
                if not has_salary:
                    # Skip jobs without salary data when salary filter is active
                    continue
                sal = v7.compensation_and_benefits.salary  # pyrefly: ignore
                if min_salary and sal.high and sal.high < min_salary:
                    continue
                if max_salary and sal.low and sal.low > max_salary:
                    continue

            # Radius filter
            if center_lat is not None and center_lon is not None and radius_miles:
                if job.geoloc:
                    min_dist = min(
                        haversine_distance(center_lat, center_lon, geo.lat, geo.lon)
                        for geo in job.geoloc
                    )
                    if min_dist > radius_miles:
                        continue
                elif job.get_workplace_type().lower() == "remote":
                    pass  # Remote jobs pass
                else:
                    continue
            elif location_city or location_state:
                if (
                    v7
                    and v7.work_arrangement
                    and v7.work_arrangement.workplace_locations
                ):
                    loc = v7.work_arrangement.workplace_locations[0]
                    if location_city and loc.city:
                        if location_city.lower() not in loc.city.lower():
                            continue
                    if location_state and loc.state:
                        if location_state.lower() not in loc.state.lower():
                            continue
                elif job.get_workplace_type().lower() != "remote":
                    continue

            # Seniority filter
            if seniority_level and v7 and v7.experience_requirements:
                job_seniority = v7.experience_requirements.seniority_level or ""
                if seniority_level.lower() not in job_seniority.lower():
                    continue

            # Exclusion filter (negation handling)
            if exclude_terms:
                searchable = self._build_searchable_text(job)
                if any(term.lower() in searchable for term in exclude_terms):
                    continue

            # Update rank and yield
            rank += 1
            result.rank = rank
            count += 1
            yield result


def format_results(response: SearchResponse) -> str:
    """Format search results for display.

    Args:
        response: SearchResponse to format.

    Returns:
        Formatted string representation.
    """
    lines = [
        f"\nSearch: '{response.query}'",
        f"Found: {response.total_found} matches\n",
    ]

    for result in response.results:
        job = result.job
        workplace_type = job.get_workplace_type() or "Not Specified"
        lines.append(f"{result.rank}. {job.get_title()}")
        lines.append(f"   Company: {job.get_company_name()}")
        lines.append(f"   Location: {job.get_location()} ({workplace_type})")
        lines.append(f"   Salary: {job.get_salary_display()}")
        lines.append(f"   Score: {result.score:.3f}")
        if job.apply_url:
            lines.append(f"   Apply: {job.apply_url}")
        lines.append("")

    return "\n".join(lines)
