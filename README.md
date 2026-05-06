# Fashion_Bot

Fashion_Bot is a fashion assistant with two main capabilities:

1. **Fashion News / QA RAG**  
   Answer fashion questions using retrieved articles, guides, and trend content.

2. **LLM + Personal Recommendation**  
   Understand user outfit requests and generate personalized clothing recommendations.

The project uses **OpenAI** as the main LLM/embedding backend for the recommendation track:

- Query parsing (optional refinement on top of deterministic rules)
- Grounded **combo composition** (picker chooses only `item_id`s from retrieval pools)
- Outfit reranking and explanation text (when enabled)
- Embedding API for **dense retrieval** / catalog vector build (when enabled)

The QA/RAG track may also use OpenAI or other models via LangChain in `src/qa/scripts/` (see Track A).

**The recommendation core remains metadata-first with deterministic fallbacks**, so it keeps working if the API key is missing or specific flags are turned off.

Step-by-step **parse → retrieve → compose → rerank → explain** reference for Track B lives in **`RECOMMENDER_PIPELINE.md`** (recommended when you revisit the codebase after a break).

---

## Why local `.env` and `requirements.txt`

- Each teammate gets an isolated Python environment.
- Everyone installs the same **root** dependencies for the shared API and Track B.
- Secrets and machine-specific settings stay out of git.

---

## Local setup

### `.env.example`

`.env.example` is a template that lists environment variables the project expects.

1. Copy `.env.example` to `.env`
2. Fill in your real local values

```powershell
Copy-Item .env.example .env
```

Important:

- Commit `.env.example`
- Do **not** commit `.env`

### Virtual environment

Each teammate should create their own `.venv` locally. Do not copy `.venv` between machines.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Root `requirements.txt`

The file at the repo root installs **FastAPI**, **recommender stack** (pandas, numpy, SQLAlchemy, psycopg, pgvector), **OpenAI SDK**, and **pytest**.  
**Track A** scripts under `src/qa/scripts/` may require **additional** packages (e.g. `requests`, `beautifulsoup4`, LangChain, Chroma)—install those when you work on that track (see imports in those scripts).

### Team setup checklist

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then update `.env`:

- Always set `OPENAI_API_KEY` if you want LLM parsing/reranking/composition or to build embeddings.
- Leave the default model values alone unless you intentionally want different OpenAI models.
- Leave `CATALOG_ITEMS_CSV=data/processed/catalog_items/catalog_items_demo.csv` if you want the included demo catalog.
- Leave `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/fashion_bot` if you use the provided local Docker Postgres.
- Leave `ENABLE_DENSE_RETRIEVAL_RERANK=false` at first. Turn it on only after the embeddings table has been built successfully.

`.env.example` shows the recommended local defaults for the full Track B setup.

---

## How to run the system (Track B — recommendation)

### 0. Copy `.env.example` to `.env`

```powershell
Copy-Item .env.example .env
```

For the normal local setup, these values are already good by default in both files:

- `OPENAI_MODEL_QUERY_PARSER=gpt-4o-mini`
- `OPENAI_MODEL_RERANKER=gpt-4o-mini`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-large`
- `CATALOG_ITEMS_CSV=data/processed/catalog_items/catalog_items_demo.csv`
- `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/fashion_bot`
- `ENABLE_DENSE_RETRIEVAL_RERANK=false`

Only change `.env` when your setup is different:

- Set `OPENAI_API_KEY` to your real key if you want OpenAI-powered parsing/reranking/composition or need to build embeddings.
- Change `CATALOG_ITEMS_CSV` only if you want to use a different processed catalog file.
- Change `DATABASE_URL` only if your Postgres host, port, database, username, or password differ from the local Docker defaults.
- Change model names or retrieval flags only if you are intentionally testing a different configuration.

### 1. Catalog data

By default, Track B uses `data/processed/catalog_items/catalog_items_demo.csv`, so you do not need to change `CATALOG_ITEMS_CSV` for the demo path.  
Rebuild or swap the catalog only if you want different inventory, using `scripts/build_catalog.py` and modules under `src/recommender/`.

### 2. Optional: Postgres + pgvector (dense retrieval / embeddings)

```powershell
docker compose up -d
```

If you are using the provided local Docker setup, the default `.env` / `.env.example` value already matches this:

- `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/fashion_bot`

Build or refresh item embeddings (OpenAI API; run when catalog or copy changes):

```powershell
python scripts\build_catalog_embeddings.py
```

Check extension/table/counts:

```powershell
python scripts\check_catalog_embeddings_db.py
```

After embeddings exist, change `ENABLE_DENSE_RETRIEVAL_RERANK` from `false` to `true` in `.env` if you want dense reranking enabled.

### 3. Run the API

```powershell
python scripts\run_api.py
```

- UI / root: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- Recommend: `POST /recommend` with JSON body `{"user_query": "..."}` (see `RecommendationRequest` in `src/shared/schemas.py`)

The `/qa` route exists but returns a **placeholder** until Track A connects the RAG pipeline.

### 4. CLI smoke test (no server)

```powershell
python scripts\test_recommender.py
```

---

## Recommendation pipeline (current code)

High-level order inside `build_outfits`:

1. **Parse query** — deterministic base, optionally merged with **`llm_parse_query`**
2. **Retrieve per-role pools** — sparse scoring; optional **per-role dense rerank** if enabled and DB is populated
3. **Compose outfits** — **`llm_compose_outfits`** (grounded, disjoint item_ids across three looks when possible) **or** deterministic **`rank_outfits`**
4. **Rerank** — optional LLM reorder of a shortlist (skipped when OpenAI composer already returns three outfits)
5. **Pick three** — diversity helper or use the three from the composer
6. **Explanations** — LLM and/or template **`_build_explanation`**

Responses include **`llm_status`** (`query_parser`, `combo_builder`, `reranker`) describing which path ran.

---

## Team split

### Track A: Fashion News / QA RAG

Main responsibilities:

- Collect fashion articles and style guides
- Clean and chunk documents
- Generate embeddings
- Retrieve relevant passages
- Answer questions with citations

Deliverable:

- A QA module or API that returns:

  - `answer`
  - `citations`
  - `sources`

```text
data/qa/
├── processed_articles/
│   ├── fashion_qa_articles_clean.jsonl
│   └── fashion_qa_chunks.jsonl
├── raw_articles/
│   └── ... (raw article .txt files)
├── url_list.json
└── index/
    └── ... (vector index files)

src/qa/scripts/
├── web_scraping.py
├── process_for_rag.py
├── build_db.py
└── query_answer.py

scripts/
├── qa_build_db.py
└── qa_answer.py
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
- Stores embeddings + metadata in Chroma vector DB under `data/qa/index/fashion_chroma_db`.

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

---

### Track B: LLM + Personal Recommendation

Main responsibilities:

- Ingest H&M catalog metadata from `articles.csv`
- Ingest optional user wardrobe items later
- Normalize item metadata into one shared item schema
- Build recommendation retrieval and ranking
- Use OpenAI for parsing, grounded combo selection, reranking, explanations, and optional embeddings

Deliverable:

- A recommendation module or API that returns:

  - `parsed_constraints`
  - `outfits`
  - explanations (per outfit)
  - `missing_items`
  - `llm_status` (which layers ran)

Implementation lives under `src/recommender/`, `src/integrations/`, and `app/routes/recommend.py`.

---

## Why this split works

This split has minimal overlap because:

- The RAG system works on **fashion text documents**.
- The recommendation system works on **catalog + wardrobe item metadata**.
- The shared boundary is mainly the **API contract** and optional **shared OpenAI configuration**.

---

## Proposed backend

### Application layer

- **FastAPI** for the backend API (`app/main.py`).

### Model layer

- **OpenAI Responses API**-style calls for JSON outputs (`src/integrations/openai_client.py`)
- **OpenAI embeddings** for dense retrieval and building catalog vectors
- Configurable models for query parsing, reranking, judging, and combo composition

### Data layer

- `articles.csv` / processed **catalog CSV** as the first clothing metadata source
- Local files for clothing images (`data/processed/...`)
- **Vector DB / pgvector** for article or catalog embeddings (depending on track)
- Relational metadata for items when using Postgres

### HTTP routes (current)

- `GET /` — static UI
- `GET /health`
- `POST /recommend` — outfit recommendations
- `POST /qa` — stub until Track A integrates the pipeline

---

## Development plan

### Phase 1: Metadata-first recommender

Use processed catalog CSV as the main source of clothing information so you can:

- Filter clothing categories
- Normalize colors and patterns
- Retrieve matching items
- Build simple outfit recommendations

No LLM is required for a baseline.

### Phase 2: Add recommendation logic

Build the outfit pipeline on top of structured metadata:

- Parse user request
- Turn request into constraints
- Retrieve candidate items
- Compose outfits (deterministic and/or LLM-grounded)
- Rank combinations
- Return top outfit suggestions

### Phase 3: Add OpenAI support

Use the LLM as an upgrade layer for:

- Query parsing refinement
- Grounded combo composition from pools
- Reranking and explanations
- Embeddings for dense retrieval

Keeps retrieval testable when flags are off.

### Phase 4: Connect both tracks

Expose both systems through one API or app:

- `/qa` for fashion RAG
- `/recommend` for outfit generation

Later, recommendation explanations can optionally call the QA system for extra styling context.

---

## Proposed file structure

```text
Fashion_Bot/
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
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
│   │   └── pgvector_store.py
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
│   ├── run_retrieval_eval.py
│   ├── judge_retrieval_run.py
│   ├── queries_eval_10_mens.txt
│   └── artifacts/
├── tests/
├── scripts/
└── data/
```

(Not every folder may exist in your clone; treat this as the target layout.)

---

## Current recommender data status

The recommender preprocessing output includes files such as:

- `data/processed/catalog_items/catalog_items_mvp.csv`

Typical properties:

- Filtered to **adult** items (`women` and `men` where applicable)
- Limited to MVP recommendation roles: `top`, `bottom`, `outerwear`, `shoes`
- Cleaned and normalized for recommendation logic

Noisy categories may be excluded (e.g. `bodysuit`, `other_shoe`, `slippers`), depending on the build.

For demos, `catalog_items_demo.csv` is also used—set **`CATALOG_ITEMS_CSV`** in `.env` to the file you actually keep in the repo.

---

## Wardrobe (user-owned clothes)

The recommender supports a per-user wardrobe stored in Postgres. Users can upload photos, the backend tags them with a vision model, stores metadata in `wardrobe_items`, and then **mixes wardrobe items with the public catalog** during `/recommend`.

Key idea:

- `user_id` controls **which wardrobe you write to** (upload) and **which wardrobe you read from** (recommend).
- The public catalog is still used unless you intentionally build a wardrobe-only mode.

### API endpoints

- **Upload one wardrobe image**: `POST /wardrobe/upload`
  - multipart form fields: `user_id` (string), `image` (file)
  - returns `wardrobe_item_id` (content-hash based)
- **Clear wardrobe** (DB rows for that user): `POST /wardrobe/clear`
  - form field: `user_id`
  - returns `deleted` count

### Image IDs and separating from catalog items

Wardrobe `item_id` is generated from the uploaded image bytes:

- `content_hash = sha256(file_bytes)`
- `wardrobe_item_id = content_hash[:16]`

This keeps wardrobe IDs stable and avoids collisions with H&M/catalog numeric IDs.

### Serving wardrobe images

Wardrobe images are staged under:

- `data/user_wardrobe/<user_id>/uploads/<wardrobe_item_id>.<ext>`

FastAPI serves that folder at:

- `/user_wardrobe/...`

### Frontend demo

The demo UI at `/` includes a small panel to:

- upload a wardrobe image for a `user_id`
- clear the wardrobe for a `user_id`

The UI also includes `user_id` in `/recommend` requests so you can see wardrobe items appear in results.

## Evaluation

### Recommender / retrieval (current scripts)

Artifacts: `eval/artifacts/` (raw runs and judged JSON; many files are gitignored).

```powershell
python eval\run_retrieval_eval.py --file eval\queries_eval_10_mens.txt
python eval\judge_retrieval_run.py
```

See `eval/rubric.md` for the judge rubric.

### Older / generic benchmark harness

If your branch still contains:

```powershell
python eval\run_benchmark.py
python eval\compare_outputs.py eval\results\deterministic_baseline.json eval\results\benchmark_results_YYYYMMDDTHHMMSSZ.json
```

use those as documented in older notes; otherwise prefer **`run_retrieval_eval.py` / `judge_retrieval_run.py`** above.

---

## Immediate next steps

1. Activate `.venv`.
2. `pip install -r requirements.txt`.
3. Copy `.env.example` → `.env` and set `OPENAI_API_KEY` (and `DATABASE_URL` if using Docker Postgres).
4. Run `python scripts\run_api.py` and exercise `POST /recommend`.
5. Optional: open `http://127.0.0.1:8000/` and use the Wardrobe panel to upload an image for `demo_user`, then run recommendations to see wardrobe items appear.
6. (Optional) `docker compose up -d`, `python scripts\build_catalog_embeddings.py`, then enable dense retrieval in `.env`.
7. Continue improving parsing, retrieval, and outfit quality using `eval/` runs.
