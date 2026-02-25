# Conversational Job Search Demo

A semantic job search system with conversational refinement capabilities. This repository serves as an example project to showcase a simple conversational chatbot built on top of vector search and LLM intent parsing.

## Requirements

- Python 3.11+
- OpenAI API key (for embeddings and chat)
- `jobs.jsonl` in the project root (dataset not included)
- ~2GB RAM for full dataset (100K+ jobs)

## Quick Start

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run the demo (installs dependencies automatically)
./run.sh
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY = 'your-key-here'
.\run.ps1
```

The scripts will install [uv](https://github.com/astral-sh/uv) if needed, sync dependencies, and launch the demo.

> **Note**: Place `jobs.jsonl` in the project root before running the demo; the dataset itself is not included in this repository.

## Demo Commands

| Command      | Description                                                |
| ------------ | ---------------------------------------------------------- |
| `/back`      | Undo last refinement (shows context depth)                 |
| `/next`      | Show next page of results                                  |
| `/prev`      | Show previous page of results                              |
| `/first`     | Return to the first page of results                        |
| `/details N` | Show full details for job number N (from the current list) |
| `/return`    | Return to the list page after viewing details              |
| `/reset`     | Clear all context and start fresh                          |
| `/budget`    | Show token usage and remaining budget                      |
| `/quit`      | Exit the demo                                              |

## Data File

This project requires `jobs.jsonl` in the project root. The file contains job postings used for the demo (not included in the repository due to size).

Each line is a JSON object with:

- `id`: Unique job identifier
- `apply_url`: Link to apply
- `job_information`: Raw job data (title, description)
- `v7_processed_job_data`: Structured data with 3 embeddings (1536-dim each)

## Features

### 1. Search

Natural language search across job postings using semantic embeddings.

```python
from search import JobSearchEngine

engine = JobSearchEngine("jobs.jsonl")
engine.load_data()

response = engine.search("data science machine learning", top_k=10)
```

### 2. Refine

Conversational refinement using context accumulation.

```python
from chatbot import JobSearchChatbot

chat = JobSearchChatbot("jobs.jsonl")

# Initial search
chat.chat("data science jobs")

# Refine by company type
chat.chat("at companies that care about social good")

# Refine by work arrangement
chat.chat("make it remote")
```

## How It Works

### Data Representation

Each job has 3 pre-computed embeddings (OpenAI `text-embedding-3-small`, 1536 dims):

| Embedding                   | Purpose                       | Source                             |
| --------------------------- | ----------------------------- | ---------------------------------- |
| `embedding_explicit_vector` | Match job requirements        | Title, skills, certifications      |
| `embedding_inferred_vector` | Match related qualifications  | Related titles, inferred skills    |
| `embedding_company_vector`  | Match company characteristics | Company name, industry, activities |

### Search Algorithm

1. **Query Embedding**: User query is embedded using the same OpenAI model
2. **Multi-Index Search**: Query is compared against all 3 FAISS indices
3. **Hybrid Search**: Keyword matching boosts exact term matches
4. **Weighted Combination**: Scores are combined with configurable weights:
   - Explicit: 50% (what the job explicitly states)
   - Inferred: 30% (related/implied qualifications)
   - Company: 20% (company characteristics)
5. **Structured Filtering**: Salary, location, seniority, workplace type filters
6. **Caching**: FAISS indices cached to disk for fast subsequent loads

### Relevance Ranking

Final score = `(0.5 × explicit_similarity) + (0.3 × inferred_similarity) + (0.2 × company_similarity) + keyword_boost`

When searching for company-specific traits (nonprofits, social good), the company weight is boosted to 40%.

### Refinement Flow

1. User sends a message
2. LLM (GPT-4o-mini) parses intent considering conversation history
3. Extracted: search query, workplace type, company focus, salary range, location, seniority level, radius, keywords
4. Hybrid search runs with keyword boosting
5. Structured filters applied (salary, location, seniority, radius)
6. Results streamed as they pass filters
7. Context saved for next refinement

### Geocoding

Location-based searches use coordinate-based radius filtering with the Haversine distance formula:

1. **LLM Extraction**: GPT-4o-mini extracts city/state and attempts to provide coordinates for common cities
2. **Geocoding Fallback**: If coordinates are missing, the system calls OpenStreetMap Nominatim API
3. **Caching**: Geocoding results are cached to `.cache/geocode_cache.json` to avoid repeated API calls
4. **Radius Filter**: Jobs within the specified radius (default 30 miles) pass; remote jobs always pass

Example: "Jobs near Akron, Ohio" → geocodes to (41.08, -81.52) → 30-mile radius filter

## Project Structure

```
├── run.sh         # Single-command launcher (bash)
├── run.ps1        # Single-command launcher (PowerShell)
├── models.py      # Pydantic data models
├── search.py      # FAISS-based search engine
├── chatbot.py     # Conversational interface with LangChain
├── geocoding.py   # Location geocoding with caching
├── demo.py        # Interactive demo script
└── jobs.jsonl     # job postings dataset (not tracked)
```

## Trade-offs

| Decision                  | Benefit                      | Cost                    |
| ------------------------- | ---------------------------- | ----------------------- |
| FAISS in-memory           | Fast search (<100ms)         | ~2GB RAM for 100K jobs  |
| Pre-computed embeddings   | No embedding latency per job | Data size, freshness    |
| GPT-4o-mini for parsing   | Cost-effective, fast         | Less nuanced than GPT-4 |
| Post-search filtering     | Simple, flexible             | May reduce result count |
| Weighted embedding fusion | Balances multiple signals    | Weights are heuristic   |

## What Works Well

- **Job titles and skills**: "python developer", "data scientist", "product manager"
- **Company characteristics**: Embeddings capture industry, org type effectively
- **Compound queries**: "remote machine learning engineer" combines multiple aspects
- **Refinement flow**: Context accumulation works for progressive narrowing
- **Structured filters**: "jobs paying over $100k in Seattle" works
- **Radius search**: "jobs within 25 miles of San Francisco" uses Haversine distance
- **Streaming**: Results display as they're found
- **Pagination**: Lazy loading with `/next` and `/prev` navigation, fetches results on demand
- **Fast loading**: Cached indices load in seconds after first run
- **Negation handling**: "not entry level" extracts exclude terms and filters results post-search
- **Smart salary display**: Detects hourly vs annual, formats appropriately ($23.73/hr vs $150,000/yr)
- **Salary filter precision**: Jobs without salary data excluded when salary filter is active

## What's Tricky

- **Rare combinations**: Limited training data for niche intersections

## Future Improvements

1. **Re-ranking Model**: Train a cross-encoder for more accurate relevance
2. **Evaluation**: Build test set with relevance judgments for tuning weights

## Token Usage and Cost

Each query consumes tokens in two main ways:

1. **LLM Intent Parsing**: Each user query is sent to GPT-4o-mini to extract search intent and filters. This typically uses **100–300 tokens per query** (prompt + response).
2. **Embedding API Calls**: Each unique search query is embedded using OpenAI's `text-embedding-3-small` model. This usually consumes **10–50 tokens per query** (depending on query length).

- **Refinements** (using `/back`, `/reset`, or follow-up queries) also trigger LLM calls and new embeddings.
- **No tokens are consumed for streaming, filtering, or pagination**—these are handled locally.
- The `/budget` command shows your current token usage and remaining budget.

**Example:**

- 10 queries with refinements ≈ 1,000–3,500 tokens total (LLM + embeddings)
- At $0.50 per 1M tokens (embedding) and $5 per 1M tokens (GPT-4o-mini), most users stay well under $1 for typical usage.
