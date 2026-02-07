"""Job search engine using FAISS and OpenAI embeddings."""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from models import (
    Job,
    SearchResponse,
    SearchResult,
)

# Embedding weights for different search aspects
WEIGHT_EXPLICIT = 0.5  # Job title, skills, requirements
WEIGHT_INFERRED = 0.3  # Related/implied qualifications
WEIGHT_COMPANY = 0.2  # Company characteristics

EMBEDDING_DIM = 1536


class JobSearchEngine:
    """Search engine for job postings using vector similarity."""

    def __init__(self, jobs_path: str | Path | None = None) -> None:
        """Initialize the search engine.

        Args:
            jobs_path: Path to the jobs.jsonl file. If None, uses default.
        """
        self.jobs_path = Path(jobs_path) if jobs_path else Path("jobs.jsonl")
        self.jobs: list[Job] = []
        self.openai_client: OpenAI | None = None

        # FAISS indices for each embedding type
        self.index_explicit: faiss.IndexFlatIP | None = None
        self.index_inferred: faiss.IndexFlatIP | None = None
        self.index_company: faiss.IndexFlatIP | None = None

        self._loaded = False

    def load_data(self, max_jobs: int | None = None) -> None:
        """Load jobs from JSONL file and build FAISS indices.

        Args:
            max_jobs: Maximum number of jobs to load. None for all.
        """
        if self._loaded:
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
        lines.append(f"{result.rank}. {job.get_title()}")
        lines.append(f"   Company: {job.get_company_name()}")
        lines.append(f"   Location: {job.get_location()}")
        lines.append(f"   Score: {result.score:.3f}")
        if job.apply_url:
            lines.append(f"   Apply: {job.apply_url}")
        lines.append("")

    return "\n".join(lines)
