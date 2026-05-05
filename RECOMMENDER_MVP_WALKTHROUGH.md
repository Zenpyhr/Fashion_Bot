# Recommender MVP Walkthrough

This note explains the current MVP version of the outfit recommendation system. It is meant as a quick personal reference for revisiting how the retrieval pipeline works and where each part lives in the code.

The current recommender is not database-based yet. It loads the processed demo catalog from:

```text
data/processed/catalog_items/catalog_items_demo.csv
```

The system uses structured metadata first, then uses OpenAI only after the candidate outfits have already been narrowed down.

## Overall Flow

```text
user query
-> query parser
-> retrieve individual clothing candidates
-> build outfit combinations
-> score full outfits
-> diversify the final options
-> LLM rerank/explain
-> frontend displays outfit cards with images
```

Main entry point:

```text
src/recommender/outfits.py
build_outfits(...)
```

## 1. Query Understanding

Code to check:

```text
src/recommender/query_parser.py
parse_user_query(...)
_deterministic_parse_user_query(...)
```

Example user query:

```text
Give me a casual men's outfit in blue with sneakers
```

The parser turns the sentence into structured constraints:

```python
{
    "target_group": "men",
    "required_roles": ["top", "bottom", "shoes"],
    "preferred_colors": ["blue"],
    "preferred_categories": ["sneakers"],
    "formality": "casual",
    "occasion": "casual",
}
```

Intuition:

```text
The parser turns human language into filters and preferences.
```

There are two parser layers:

```text
deterministic keyword parser
optional OpenAI parser refinement
```

OpenAI parser helper:

```text
src/integrations/openai_client.py
llm_parse_query(...)
```

## 2. Catalog Loading

Code to check:

```text
src/recommender/retrieval.py
load_catalog_items(...)
```

The catalog path comes from:

```text
src/shared/config.py
settings.catalog_items_csv
```

For the MVP demo, this points to:

```text
data/processed/catalog_items/catalog_items_demo.csv
```

The CSV is loaded into a pandas dataframe and cached with `@lru_cache`, so it is loaded once per backend process.

Intuition:

```text
Catalog loading opens the clothing table and keeps it ready for retrieval.
```

## 3. Role-Based Retrieval

Code to check:

```text
src/recommender/retrieval.py
retrieve_candidates_by_role(...)
```

The system retrieves separately for each outfit role:

```text
tops compete with tops
bottoms compete with bottoms
shoes compete with shoes
outerwear competes with outerwear
```

For a normal outfit, required roles are:

```python
["top", "bottom", "shoes"]
```

Intuition:

```text
Role retrieval fills each outfit slot separately.
```

## 4. Single-Item Scoring

Code to check:

```text
src/recommender/retrieval.py
_score_item(...)
```

Each item starts with a base score:

```python
score = 10
```

Then the system adds or subtracts points.

Color match:

```text
_score_color_preferences(...)
```

Category match:

```text
_score_category_preferences(...)
```

Query term overlap:

```text
_score_query_term_overlap(...)
```

Formality proxy:

```text
_score_formality_proxy(...)
FORMALITY_CATEGORY_WEIGHTS
```

Occasion proxy:

```text
_score_occasion_proxy(...)
OCCASION_HINTS
```

Intuition:

```text
Single-item scoring asks how well one clothing piece matches the request.
```

## 5. Candidate Pool Diversity

Code to check:

```text
src/recommender/retrieval.py
_select_diverse_role_candidates(...)
ROLE_LIMITS
ROLE_CATEGORY_LIMITS
```

This prevents one category from dominating a role pool. For example, if the highest-scoring tops are all hoodies, the system still tries to include other top categories.

Current role pool sizes:

```python
ROLE_LIMITS = {
    "top": 20,
    "bottom": 15,
    "shoes": 12,
    "outerwear": 10,
}
```

Current category limits:

```python
ROLE_CATEGORY_LIMITS = {
    "top": 3,
    "bottom": 3,
    "shoes": 2,
    "outerwear": 2,
}
```

Intuition:

```text
Candidate diversity stops one clothing type from taking over the shortlist.
```

## 6. Outfit Combination

Code to check:

```text
src/recommender/ranker.py
rank_outfits(...)
```

After individual item retrieval, the system builds full outfits by combining the best candidates from each role:

```text
top A + bottom A + shoes A
top A + bottom A + shoes B
top A + bottom B + shoes A
...
```

It does not combine all catalog rows. It only combines the strongest narrowed candidates.

Intuition:

```text
Outfit combination tries small outfit recipes from the best pieces.
```

## 7. Whole-Outfit Scoring

Code to check:

```text
src/recommender/ranker.py
rank_outfits(...)
_color_cohesion_score(...)
_section_theme_penalty(...)
```

Each outfit score starts as:

```text
sum of item candidate scores
```

Then it adds outfit-level logic:

```text
color cohesion bonus
formal/sport mismatch penalty
```

Intuition:

```text
Outfit scoring asks whether the pieces work together as a full look.
```

## 8. Outfit Shortlist Diversity

Code to check:

```text
src/recommender/ranker.py
_select_diverse_outfits(...)
_outfit_similarity(...)
```

This prevents the shortlist from being several tiny variations of the same outfit.

It compares outfits by:

```text
shared item ids
same categories
same colors
```

Intuition:

```text
Shortlist diversity gives the LLM a more interesting set of options to judge.
```

## 9. LLM Reranking

Code to check:

```text
src/recommender/outfits.py
_apply_llm_reranking(...)
```

OpenAI prompt/helper:

```text
src/integrations/openai_client.py
llm_rerank_outfits(...)
```

The LLM does not search the full catalog. It only sees a small shortlist of already-generated outfits.

The LLM is used as a stylist reviewer:

```text
Which outfit best matches the query?
Which options are too repetitive?
How should the final recommendations be explained?
```

Intuition:

```text
LLM reranking adds styling judgment after deterministic retrieval has narrowed the space.
```

## 10. Final Top-3 Diversity

Code to check:

```text
src/recommender/outfits.py
_select_top_diverse_outfits(...)
```

This is the final anti-repetition step before returning results.

It tries to avoid:

```text
hoodie + shorts + sneakers
hoodie + shorts + sneakers
hoodie + shorts + sneakers
```

and prefer more distinct structures when available:

```text
tshirt + shorts + sneakers
hoodie + trousers + sneakers
shirt + shorts + boots
```

Intuition:

```text
Final diversity makes the top 3 feel like real options instead of duplicates.
```

## 11. Final LLM Explanations

Code to check:

```text
src/recommender/outfits.py
_apply_llm_explanations_to_selected_outfits(...)
```

This gives every final outfit a natural explanation.

Intuition:

```text
Final explanations make the output readable and demo-friendly.
```

## 12. Image Display

The metadata stores a relative path:

```text
data/processed/demo_images/050/0504658004.jpg
```

The backend converts it into a browser URL:

```text
/demo_images/050/0504658004.jpg
```

Code to check:

```text
src/recommender/outfits.py
_image_url_from_path(...)
```

FastAPI serves the image folder:

```text
app/main.py
app.mount("/demo_images", StaticFiles(...))
```

The frontend renders the image:

```text
app/static/app.js
```

Intuition:

```text
The CSV stores portable paths. FastAPI turns those paths into URLs the browser can display.
```

## Current MVP Summary

The current MVP recommender is:

```text
CSV-based
metadata-first
role-aware
rule-scored
diversity-aware
LLM-reranked
image-enabled
```

It is not yet:

```text
database-backed
user-wardrobe-backed
VLM-enriched
personalized with persistent feedback
```

One-sentence version:

```text
The recommender finds good individual pieces, combines them into possible outfits, scores and diversifies those outfits, then asks the LLM to pick and explain the strongest final options.
```
