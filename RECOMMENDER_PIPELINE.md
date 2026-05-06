# Track B recommender pipeline (reference)

Companion to **`README.md`**. Use this file to remember **how `/recommend` works** inside the code—**parse → retrieve → compose → rerank → pick 3 → explain**.

Main orchestrator: **`build_outfits`** in `src/recommender/outfits.py`.

---

## High-level order

1. **Parse query** — deterministic rules, optionally merged with LLM (`llm_parse_query`); `semantic_query` filled for dense embedding when useful.
2. **Retrieve per-role pools** — sparse scoring on CSV catalog; optional per-role dense rerank (Postgres + embeddings) when enabled.
3. **Compose outfits** — **`llm_compose_outfits`** (grounded `item_id`s only) **or** deterministic **`rank_outfits`** (Cartesian product of small per-role trims).
4. **Rerank** — optional LLM reorder over a shortlist (**skipped** if the OpenAI composer already produced three valid outfits).
5. **Pick three** — composer: first three; deterministic path: **`_select_top_diverse_outfits`** (disjoint `item_id`s when possible).
6. **Explanations** — LLM via rerank endpoint for selected outfits if needed; else **`_build_explanation`** template.

Returned payload includes **`llm_status`** (`query_parser`, `combo_builder`, `reranker`) and **`candidate_pools`** preview (five items per role).

---

## 1. Parse query (`parse_user_query`)

**File:** `src/recommender/query_parser.py`

| Step | What happens |
|------|----------------|
| Deterministic | `_deterministic_parse_user_query`: detects target group (defaults men for demo), colors, garment keywords → `preferred_categories`, expands `required_roles` with **outerwear** when triggered, formality / occasion, `search_terms`, `raw_query`. Intent booleans default **false**. |
| LLM (optional) | If `OPENAI_API_KEY` and `ENABLE_OPENAI_QUERY_PARSER`: `llm_parse_query` returns JSON patches aligned with deterministic base. |
| Merge | `_merge_llm_constraints`: overlays whitelisted keys; `target_group` forced to **men** for current catalog; `semantic_query` merged when non-empty string. |
| Embedding line | `_ensure_semantic_query`: LLM rewrite wins; else compact line from categories / colors / formality / intents (`_deterministic_semantic_query`); if still empty, key omitted → dense uses **`raw_query`**. |

**Output:** One **`constraints`** dict for retrieval + compose + prompts. **`parser_source`:** `"deterministic"` or `"openai"`.

---

## 2. Retrieve per-role pools (`retrieve_candidates_by_role`)

**File:** `src/recommender/retrieval.py`

| Step | What happens |
|------|----------------|
| Load catalog | `load_catalog_items()` from `CATALOG_ITEMS_CSV` (process-local cache). |
| Per role | Filter rows: `target_group`, `recommendation_role`. |
| Sparse score | `candidate_score` per row (`_score_item`: colors, categories, term overlap, formality / occasion proxies, **`_score_guardrails`** / **`_detect_query_intents`**). |
| Shortlist | Sort by score; take head — size from **`ROLE_LIMITS`**, enlarged to **`dense_shortlist_k_per_role`** when dense rerank enabled. |
| Dense rerank (optional) | If `ENABLE_DENSE_RETRIEVAL_RERANK` and OpenAI configured: embed **one** query string (**`semantic_query` or `raw_query`**), fetch vectors from DB, cosine reorder; tie-break on sparse score. Requires Docker Postgres + built embeddings table. |
| Diversity | **`_select_diverse_role_candidates`**: caps items per **`normalized_category`** so pools aren’t dominated by one category. |

**Output:** **`candidates_by_role`**: `dict[role, list[item dict]]`.

---

## 3. Compose outfits

### Path A — OpenAI composer

**Files:** `outfits.py` (`_try_llm_compose_outfits`, `_outfits_from_llm_compose`), `openai_client.py` (`llm_compose_outfits`)

Requires: `ENABLE_OPENAI_COMBO_COMPOSER`, API key, every required role nonempty.

Pools trimmed per role (**`combo_composer_max_items_per_role_for_llm`**, minimum 4). Model returns exactly **three** outfits: each **`required_role` → valid `item_id` from pools**. **`_outfits_from_llm_compose`** rejects invalid IDs or **any repeated `item_id` across all three outfits**. Outfit **`score`** = **`score_outfit_items`** (sum item scores + color cohesion + dressy-vs-sport penalty).

### Path B — Deterministic

**File:** `src/recommender/ranker.py` — **`rank_outfits`**

Uses top **4** items per role (**3** for outerwear), full **Cartesian product**, score each combo like above, sort, take top 30 → **`_select_diverse_outfits`** → keeps **10** diverse candidates ordered by score.

**Output:** **`ranked_outfits`** — either **3** (LLM) or **≤10** (deterministic ladder before final pick).

---

## 4. Rerank (LLM, optional)

**Files:** `outfits.py` (`_apply_llm_reranking`), `openai_client.py` (`llm_rerank_outfits`)

Runs only if **composer was not OpenAI** and **`ENABLE_OPENAI_RERANKER`** and API configured.

Sends **up to 8** outfits with simplified item fields; model returns **`ranked_outfit_ids`** + **`explanations`**. Prompt asks for **disjoint item_id sets** in the top three when possible.

If composer succeeded earlier, **this rerank step is skipped** and **`reranker`** becomes **`skipped`**.

---

## 5. Pick three

| Source | Logic |
|--------|--------|
| OpenAI composer | **`ranked_outfits[:3]`** — fixed order from model. |
| Deterministic (+ optional rerank) | **`_select_top_diverse_outfits(..., limit=3)`** in `outfits.py` — greedy diversity, prefers **pairwise disjoint** `item_id` sets vs prior picks when possible; penalizes duplicate category signatures. |

---

## 6. Explanations

**File:** `outfits.py`

If all three outfits from composer already carry **`llm_explanation`**, no extra LLM call.

Otherwise, with reranker enabled: **`_apply_llm_explanations_to_selected_outfits`** calls **`llm_rerank_outfits`** again with **`selected_outfit_*`** IDs.

Final API field **`explanation`**: **`llm_explanation`** or **`_build_explanation`** (template using categories, colors, formality).

---

## Response shape highlights

| Field | Meaning |
|-------|--------|
| `parsed_constraints` | Final merged constraints (roles, intents, optional `semantic_query`, etc.). |
| `llm_status.query_parser` | `deterministic` or `openai`. |
| `llm_status.combo_builder` | `deterministic` or `openai`. |
| `llm_status.reranker` | `deterministic`, `openai`, or **`skipped`** (composer path). |
| `candidate_pools` | First **five** summarized items **per role** (debug / UI teaser). |
| `outfits` | Up to three: `score`, `items` (summaries + `image_url`), `explanation`. |
| `missing_items` | Roles in **`required_roles`** with empty pools. |

---

## `semantic_query` and dense retrieval

**Retriever** (`_dense_rerank_role_pool`) prefers **`constraints["semantic_query"]`** when embedding the query; otherwise **`raw_query`**.

The parser is responsible for **setting** `semantic_query` (LLM + deterministic fallback via `_ensure_semantic_query`). Without it, dense still runs but embeds **raw user text** only.

---

## Key modules (quick map)

| Module | Role |
|--------|------|
| `src/recommender/query_parser.py` | Text → `constraints`; merge LLM; `semantic_query`. |
| `src/recommender/retrieval.py` | Catalog load; sparse + optional dense rerank per role; pool diversity. |
| `src/recommender/ranker.py` | Deterministic combo enumeration + cohesion scoring. |
| `src/recommender/outfits.py` | **`build_outfits`**; LLM compose/validate; diversity pick three; explanations. |
| `src/integrations/openai_client.py` | **`llm_parse_query`**, **`llm_compose_outfits`**, **`llm_rerank_outfits`**, embeddings helpers. |
| `src/integrations/pgvector_store.py` / embeddings scripts | Postgres + vectors for dense path. |

---

## Eval (sanity-check after changes)

```powershell
python eval\run_retrieval_eval.py --file eval\queries_eval_10_mens.txt
```

Artifacts: `eval/artifacts/raw/run_<timestamp>.json`. Compare runs when tuning parser, retrieval flags, or prompts.
