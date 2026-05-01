# Fashion_Bot

Fashion_Bot is a fashion assistant with two main capabilities:

1. `Fashion News / QA RAG`
   Answer fashion questions using retrieved articles, guides, and trend content.
2. `VLM + Personal Recommendation`
   Understand clothing items and generate personalized outfit recommendations.

The project uses `Vertex AI` as the model backend for:
- text generation
- embeddings
- later VLM-based image understanding

The goal is to keep both workstreams separate enough that two people can build in parallel with minimal overlap.

## Local Setup

Use a local virtual environment and a local `.env` file.

Why:
- each teammate gets an isolated Python environment
- both teammates install the same project dependencies
- secrets and machine-specific settings stay out of git

### `.env.example`

`.env.example` is a template file that shows which environment variables the project expects.

Workflow:
1. copy `.env.example` to `.env`
2. fill in your real local values

Example:

```powershell
Copy-Item .env.example .env
```

Important:
- commit `.env.example`
- do not commit `.env`

### Virtual Environment

Each teammate should create their own `.venv` locally.
Do not copy the `.venv` folder from one machine to another.

Recommended setup:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Recommended project standard:
- Python `3.11`
- one `.venv` per teammate machine
- install from the shared `requirements.txt`

### Team Setup Checklist

Every teammate should run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then update `.env` with the right local values for:
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEX_MODEL_TEXT`
- `VERTEX_MODEL_VISION`
- `EMBEDDING_MODEL`
- `DATABASE_URL`

## Team Split

### Track A: Fashion News / QA RAG

Main responsibilities:
- collect fashion articles and style guides
- clean and chunk documents
- generate embeddings
- retrieve relevant passages
- answer questions with citations

Deliverable:
- a QA module or API that returns:
  - `answer`
  - `citations`
  - `sources`

### Track B: VLM + Personal Recommendation

Main responsibilities:
- ingest H&M catalog metadata from `articles.csv`
- ingest optional user wardrobe items later
- normalize item metadata into one shared item schema
- build recommendation retrieval and ranking
- use Vertex AI VLM later to enrich missing attributes

Deliverable:
- a recommendation module or API that returns:
  - `parsed_constraints`
  - `outfits`
  - `explanations`
  - `missing_items`

## Why This Split Works

This split has minimal overlap because:
- the RAG system works on `fashion text documents`
- the recommendation system works on `catalog + wardrobe item metadata`
- the shared boundary is only the schema and API contract

## Proposed Backend

### Application Layer
- `FastAPI` for the backend API

### Model Layer
- `Vertex AI Gemini` for text generation
- `Vertex AI Embeddings` for RAG retrieval
- `Vertex AI Vision / multimodal model` for later clothing attribute enrichment

### Data Layer
- `articles.csv` as the first catalog metadata source
- local files or cloud storage for clothing images
- vector database for article retrieval
- relational table for item metadata

## Development Plan

### Phase 1: Metadata-First Recommender
Use `articles.csv` as the main source of clothing information.

This means you can already:
- filter clothing categories
- normalize colors and patterns
- retrieve matching items
- build simple outfit recommendations

No VLM is required yet.

### Phase 2: Add Recommendation Logic
Build the actual outfit pipeline on top of structured metadata:
- parse user request
- turn request into constraints
- retrieve candidate items
- compose outfits
- rank combinations
- return top outfit suggestions

This is still mostly metadata-based.

### Phase 3: Add Vertex AI VLM Enrichment
Use the VLM only as an upgrade layer to infer attributes that are not cleanly available in `articles.csv`, such as:
- `style`
- `occasion`
- `formality`
- `season`

This keeps the VLM from becoming a blocker.

### Phase 4: Connect Both Tracks
Expose both systems through one API or app:
- `/qa` for fashion RAG
- `/recommend` for outfit generation

Later, recommendation explanations can optionally call the QA system for extra styling context.

## Proposed File Structure

```text
Fashion_Bot/
├── README.md
├── .gitignore
├── articles.csv
├── pyproject.toml
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   │   ├── hm/
│   │   │   ├── articles.csv
│   │   │   └── images/
│   │   ├── fashion_articles/
│   │   └── user_wardrobe/
│   ├── processed/
│   │   ├── catalog_items/
│   │   ├── article_chunks/
│   │   └── wardrobe_items/
│   └── sample/
│       ├── sample_articles.csv
│       ├── sample_questions.json
│       └── sample_user_queries.json
├── notebooks/
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── qa.py
│   │   └── recommend.py
│   └── dependencies.py
├── src/
│   ├── shared/
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── constants.py
│   │   └── utils.py
│   ├── integrations/
│   │   ├── vertex_ai.py
│   │   ├── embeddings.py
│   │   └── storage.py
│   ├── rag/
│   │   ├── ingest_articles.py
│   │   ├── chunk_articles.py
│   │   ├── embed_articles.py
│   │   ├── retrieve.py
│   │   └── answer.py
│   ├── recommender/
│   │   ├── ingest_catalog.py
│   │   ├── normalize_catalog.py
│   │   ├── vlm_enrichment.py
│   │   ├── query_parser.py
│   │   ├── retrieval.py
│   │   ├── ranker.py
│   │   └── outfits.py
│   └── database/
│       ├── item_store.py
│       ├── vector_store.py
│       └── migrations/
├── tests/
│   ├── test_rag.py
│   ├── test_recommender.py
│   ├── test_catalog_preprocessing.py
│   └── test_api.py
└── scripts/
    ├── run_api.py
    ├── build_catalog.py
    └── build_rag_index.py
```

## What Each Folder Is For

### `app/`
FastAPI entrypoint and API routes.

### `src/shared/`
Code used by both teammates:
- config
- shared schemas
- constants
- utility helpers

### `src/integrations/`
Wrappers for external services, especially `Vertex AI`.

### `src/rag/`
Owned mainly by the Fashion News / QA track.

Contains:
- article ingestion
- chunking
- embedding
- retrieval
- grounded answer generation

### `src/recommender/`
Owned mainly by the recommendation track.

Contains:
- H&M ingestion
- metadata normalization
- VLM enrichment
- user query parsing
- recommendation retrieval
- outfit ranking

### `src/database/`
Shared persistence layer for:
- item metadata
- vector retrieval
- future migrations

### `scripts/`
Simple scripts for running pipelines locally.

## Proposed API Surface

### `POST /qa`
Input:
- user fashion question

Output:
- answer
- citations
- sources

### `POST /recommend`
Input:
- user request
- optional `use_owned_only`
- optional user wardrobe context

Output:
- parsed constraints
- recommended outfits
- explanations
- missing items

## Current Recommender Data Status

The current recommender preprocessing output is:
- `data/processed/catalog_items/catalog_items_mvp.csv`

This file is:
- filtered to `adult-only` items: `women` and `men`
- limited to the 4 MVP recommendation roles:
  - `top`
  - `bottom`
  - `outerwear`
  - `shoes`
- cleaned and normalized for recommendation logic

The current build intentionally excludes noisy categories such as:
- `bodysuit`
- `other_shoe`
- `slippers`

## Recommendation Track Notes

For the recommender system:
- start with `articles.csv`
- use metadata-first retrieval and outfit composition
- add user wardrobe support after the catalog-only version works
- add VLM enrichment later

This is the safer build order because it gives you a working system earlier.

## Immediate Next Steps

1. Create and activate `.venv`.
2. Copy `.env.example` to `.env`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Let the QA track continue in `src/rag/`.
5. Continue recommendation logic in `src/recommender/query_parser.py`, `retrieval.py`, `ranker.py`, and `outfits.py`.
