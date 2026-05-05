# Fashion_Bot

Fashion_Bot is a fashion assistant with two main capabilities:

1. `Fashion News / QA RAG`
   Answer fashion questions using retrieved articles, guides, and trend content.
2. `LLM + Personal Recommendation`
   Understand user outfit requests and generate personalized clothing recommendations.

The project currently uses `OpenAI` as the LLM backend for:
- query parsing
- final outfit reranking
- future recommendation explanation upgrades

The recommendation core is still metadata-first and deterministic, so the system can keep working even if the LLM is unavailable.

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

### Team Setup Checklist

Every teammate should run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then update `.env` with:
- `OPENAI_API_KEY`
- `OPENAI_MODEL_QUERY_PARSER`
- `OPENAI_MODEL_RERANKER`
- `OPENAI_EMBEDDING_MODEL`
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
```text
QA/
├── data/
│   ├── processed_articles/
│   │   ├── fashion_qa_articles_clean.jsonl
│   │   └── fashion_qa_chunks.jsonl
│   ├── raw_articles/
│   │   └── ... (raw article .txt files)
│   ├── evaluation_qa_results.json
│   └── url_list.json
├── index/
│   └── ... (vector index files)
└── scripts/
    ├── web_scraping.py
    ├── process_for_rag.py
    ├── build_db.py
    ├── query_answer.py
    └── evaluation_qa.py
```

#### `web_scraping.py`
- Reads URL groups from `data/url_list.json`.
- Downloads article pages and extracts the main text.
- Cleans obvious boilerplate/paywall-like content.
- Saves cleaned raw text files into `data/raw_articles/` with URL, scope, and title metadata.

#### `process_for_rag.py`
- Reads raw article `.txt` files from `data/raw_articles/`.
- Parses metadata and cleans duplicate/noisy text.
- Splits each article into overlapping chunks for retrieval.
- Writes processed outputs to `data/processed_articles/` (clean articles + chunk records).

#### `build_db.py`
- Loads processed chunk records.
- Converts chunks into embeddings.
- Stores embeddings + metadata in Chroma vector DB under `index/fashion_chroma_db`.

#### `query_answer.py`
- Classifies a question into likely fashion scopes.
- Retrieves relevant chunks from the vector DB (scope-aware, with global fallback).
- Builds a grounded prompt and generates an answer with cited sources.
- Main callable flow is `qa_answer(question)`.

#### `evaluation_qa.py`
- Runs a fixed set of test questions.
- Evaluates retrieval coverage by measuring recall@k (did we retrieve expected scope content).
- Uses an LLM judge to score whether answers are supported by retrieved evidence.
- Prints per-case results and an overall summary.


### Track B: LLM + Personal Recommendation

Main responsibilities:
- ingest H&M catalog metadata from `articles.csv`
- ingest optional user wardrobe items later
- normalize item metadata into one shared item schema
- build recommendation retrieval and ranking
- use OpenAI later to improve parsing and reranking

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
- `OpenAI Responses API` for text generation
- `OpenAI embeddings` for retrieval support
- `OpenAI models` for query parsing and final outfit reranking

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

No LLM is required yet.

### Phase 2: Add Recommendation Logic
Build the actual outfit pipeline on top of structured metadata:
- parse user request
- turn request into constraints
- retrieve candidate items
- compose outfits
- rank combinations
- return top outfit suggestions

This is still mostly metadata-based.

### Phase 3: Add OpenAI LLM Support
Use the LLM as an upgrade layer for:
- query parsing
- final outfit reranking
- better recommendation explanations

This keeps the retrieval core stable and testable.

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
├── notebooks/
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── qa.py
│   │   └── recommend.py
│   └── dependencies.py
├── src/
│   ├── shared/
│   ├── integrations/
│   │   ├── openai_client.py
│   │   ├── embeddings.py
│   │   └── storage.py
│   ├── rag/
│   ├── recommender/
│   │   ├── ingest_catalog.py
│   │   ├── normalize_catalog.py
│   │   ├── query_parser.py
│   │   ├── retrieval.py
│   │   ├── ranker.py
│   │   ├── outfits.py
│   │   └── vlm_enrichment.py
│   └── database/
├── eval/
│   ├── benchmark_queries.json
│   ├── rubric.md
│   ├── run_benchmark.py
│   └── compare_outputs.py
├── tests/
└── scripts/
```

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

## Evaluation

The repo includes a simple benchmark harness under `eval/`.

Run the current benchmark:

```powershell
python eval\run_benchmark.py
```

Compare two saved runs:

```powershell
python eval\compare_outputs.py eval\results\deterministic_baseline.json eval\results\benchmark_results_YYYYMMDDTHHMMSSZ.json
```

## Immediate Next Steps

1. Activate `.venv`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Paste your OpenAI key into `.env`.
4. Re-run the benchmark.
5. Continue improving parsing and reranking quality.
