# HiringCafe Job Search

A semantic job search system with conversational refinement capabilities.

## Quick Start

```bash
# Install dependencies
uv sync

# Download the jobs data file (not included in repo due to size)
# Place jobs.jsonl (8.4GB, 100K job postings) in the project root

# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run the interactive demo
uv run python demo.py
```

## Data File

This project requires `jobs.jsonl` in the project root. The file contains 100,000 job postings from HiringCafe (8.4GB) and is not included in the repository due to its size.

Each line is a JSON object with:
- `id`: Unique job identifier
- `apply_url`: Link to apply
- `job_information`: Raw job data (title, description)
- `v7_processed_job_data`: Structured data with 3 embeddings (1536-dim each)

## Features

### 1. Search
Natural language search across 100K job postings using semantic embeddings.

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

| Embedding | Purpose | Source |
|-----------|---------|--------|
| `embedding_explicit_vector` | Match job requirements | Title, skills, certifications |
| `embedding_inferred_vector` | Match related qualifications | Related titles, inferred skills |
| `embedding_company_vector` | Match company characteristics | Company name, industry, activities |

### Search Algorithm

1. **Query Embedding**: User query is embedded using the same OpenAI model
2. **Multi-Index Search**: Query is compared against all 3 FAISS indices
3. **Weighted Combination**: Scores are combined with configurable weights:
   - Explicit: 50% (what the job explicitly states)
   - Inferred: 30% (related/implied qualifications)  
   - Company: 20% (company characteristics)
4. **Post-Filtering**: Filters applied for workplace type, company focus, etc.

### Relevance Ranking

Final score = `(0.5 × explicit_similarity) + (0.3 × inferred_similarity) + (0.2 × company_similarity)`

When searching for company-specific traits (nonprofits, social good), the company weight is boosted to 40%.

### Refinement Flow

1. User sends a message
2. LLM (GPT-4o-mini) parses intent considering conversation history
3. Extracted: search query, workplace type filter, company focus
4. Search runs with adjusted weights based on intent
5. Results filtered by extracted constraints
6. Context saved for next refinement

## Project Structure

```
├── models.py      # Pydantic data models
├── search.py      # FAISS-based search engine
├── chatbot.py     # Conversational interface with LangChain
├── demo.py        # Runnable demo script
└── jobs.jsonl     # 100K job postings dataset
```

## Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| FAISS in-memory | Fast search (<100ms) | ~2GB RAM for 100K jobs |
| Pre-computed embeddings | No embedding latency per job | Data size, freshness |
| GPT-4o-mini for parsing | Cost-effective, fast | Less nuanced than GPT-4 |
| Post-search filtering | Simple, flexible | May reduce result count |
| Weighted embedding fusion | Balances multiple signals | Weights are heuristic |

## What Works Well

- **Job titles and skills**: "python developer", "data scientist", "product manager"
- **Company characteristics**: Embeddings capture industry, org type effectively
- **Compound queries**: "remote machine learning engineer" combines multiple aspects
- **Refinement flow**: Context accumulation works for progressive narrowing

## What's Tricky

- **Salary filters**: Requires structured field parsing, not embedding-based
- **Location specificity**: "jobs in San Francisco" needs geo-filtering
- **Negation**: "not entry level" doesn't work well with embeddings
- **Rare combinations**: Limited training data for niche intersections

## Improvements with More Time

1. **Hybrid Search**: Combine embedding search with keyword/BM25 for exact matches
2. **Structured Filters**: Parse salary ranges, experience levels from conversation
3. **Re-ranking Model**: Train a cross-encoder for more accurate relevance
4. **Caching**: Cache embeddings and indices to disk for faster startup
5. **Streaming**: Stream results as they're found for better UX
6. **Evaluation**: Build test set with relevance judgments for tuning weights
7. **Geo-Search**: Use `_geoloc` field for location-based filtering

## Requirements

- Python 3.11+
- OpenAI API key (for embeddings and chat)
- ~2GB RAM for full dataset
