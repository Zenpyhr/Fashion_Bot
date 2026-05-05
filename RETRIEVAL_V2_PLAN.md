# Retrieval V2 Plan

This plan describes the next improvement to the MVP recommendation system: richer LLM query parsing plus description-aware retrieval. The goal is to improve retrieval quality without changing the core H&M item schema yet.

This V2 is designed as a **linear retrieval funnel**:

```text
hard sparse filters (musts / must-nots)
-> sparse scoring shortlist (metadata + optional keyword guardrails)
-> dense embedding rerank (semantic matching inside safe shortlist)
-> outfit assembly
-> final LLM reranking + explanation
```

This keeps constraints reliable (roles/target group/avoids), keeps compute cheap, and makes debugging easier than running dense+sparse retrieval in parallel and trying to fuse them.

## Current MVP

The current recommender works like this:

```text
user query
-> parser extracts simple constraints
-> retrieval scores individual items from CSV metadata
-> ranker builds outfit combinations
-> final diversity filter
-> OpenAI reranks and explains final outfits
```

Current LLM usage:

```text
query parser refinement
final outfit reranking
final explanation generation
```

Current deterministic retrieval uses:

```text
target_group
recommendation_role
normalized_category
normalized_color
normalized_pattern
section_theme
description through simple query-term overlap
```

## Problem

The current parser understands simple requests well:

```text
blue
sneakers
men
casual
work
```

But it is weaker on richer style language:

```text
relaxed but polished
warm jacket
not sporty
minimal
structured
lightweight
cozy
clean office outfit
```

The H&M descriptions often contain useful details like:

```text
padded
hood
zip
cotton
denim
relaxed
slim
ribbed
drawstring
long sleeves
```

We should use those details more directly.

## Key Idea

Do not ask the LLM to score thousands of items.

Instead:

```text
LLM parses the user request into a richer style brief
-> deterministic retrieval applies that style brief to existing metadata
-> LLM reranks only the final small outfit shortlist
```

This keeps the system:

```text
fast
cheap
explainable
easy to debug
flexible
```

## Query Schema vs Item Schema

This change mostly affects the query schema, not the item schema.

The item schema can stay the same for now:

```text
item_id
source_type
image_path
display_name
description
target_group
recommendation_role
normalized_category
product_family
normalized_color
color_detail
color_tone
normalized_pattern
section_theme
```

The LLM parser will produce richer retrieval instructions:

```text
style_direction
positive_terms
negative_terms
avoid_categories
preferred_materials
preferred_details
semantic_query
```

These fields are not item columns. They are query-side instructions used by retrieval.

## Output Normalization + Guardrails (Important)

To make these fields usable by deterministic scoring, normalize and validate them.

Text lists (`style_direction`, `positive_terms`, `negative_terms`, `preferred_materials`, `preferred_details`):

```text
lowercase
strip punctuation at ends
collapse whitespace
allow multiword phrases ("fine knit", "smart casual")
limit list sizes (example: <= 12 terms per list)
drop very short tokens (example: length < 3) unless known (ex: "zip")
```

Matching should be token/phrase-aware (not naive substring). Treat hyphens as spaces so:

```text
fleece-lined ~= fleece lined
fine-knit ~= fine knit
```

`avoid_categories` must map to real catalog values. If the LLM returns an unknown category, drop it (or map it) rather than silently doing nothing.

## Parser V2 Output

Add these optional fields to the parsed constraints:

```python
{
    "style_direction": ["relaxed", "polished"],
    "positive_terms": ["structured", "button", "fine-knit"],
    "negative_terms": ["sport", "gym", "running"],
    "avoid_categories": ["shorts", "tank_top"],
    "preferred_materials": ["cotton", "denim"],
    "preferred_details": ["hood", "zip", "collar"],
    "semantic_query": "polished smart casual men's outfit with structured relaxed pieces"
}
```

Meaning:

```text
style_direction
High-level words describing the intended style.

positive_terms
Words or phrases that should boost matching items.

negative_terms
Words or phrases that should penalize matching items.

avoid_categories
Categories that should be strongly penalized or filtered.

preferred_materials
Material hints to look for in description.

preferred_details
Construction/detail hints to look for in description.

semantic_query
A clean text phrase that can later be used for dense embedding retrieval.
```

## Example

User query:

```text
I want a warm casual jacket, not sporty
```

Parser V2 could output:

```python
{
    "target_group": "men",
    "required_roles": ["top", "bottom", "shoes", "outerwear"],
    "formality": "casual",
    "occasion": "casual",
    "positive_terms": ["warm", "padded", "quilted", "fleece", "hood"],
    "negative_terms": ["sport", "running", "gym"],
    "preferred_details": ["hood", "zip"],
    "semantic_query": "warm casual outfit with a padded or hooded jacket, not sporty"
}
```

Retrieval can then boost descriptions like:

```text
Short padded jacket with a jersey-lined hood...
```

and penalize descriptions like:

```text
Lightweight running jacket in fast-drying fabric...
```

## Retrieval Scoring V2

Add a new description-aware score (sparse keyword/phrase hints). Optionally add a dense rerank stage inside a shortlist.

Code area:

```text
src/recommender/retrieval.py
```

New helper idea:

```python
def _score_text_hints(row: pd.Series, constraints: dict) -> int:
    ...
```

It should build a searchable text field from:

```text
display_name
description
normalized_category
normalized_pattern
section_theme
```

Then apply:

```text
+3 for each positive term match
+2 for each preferred material match
+2 for each preferred detail match
-4 for each negative term match
-10 for avoid category match
```

These weights should be modest so the new text score improves retrieval without overpowering role/category/color.

Two important safety rules:

```text
1) Match whole words/phrases where possible (avoid accidental substring hits).
2) Bound the impact of text hints (cap/clamp), so it acts like a booster/guardrail.
   Example: clamp text_hint_score to [-15, +15] per item.
```

### Retrieval Funnel (Sparse -> Dense)

We want a single pipeline (not parallel retrieval + fusion):

```text
Step A: hard sparse filters per role pool
Step B: sparse scoring to get a shortlist
Step C: dense embedding rerank within the shortlist
```

Step A (hard filters) should enforce:

```text
target_group
recommendation_role (tops/bottoms/shoes/outerwear)
explicit category constraints (ex: sneakers/jacket) when present
explicit color constraints when present
avoid_categories (as filter or very strong penalty)
```

Step B (sparse scoring shortlist) uses existing metadata scoring plus `_score_text_hints`.
Choose a shortlist size per role pool (example):

```text
top_k_sparse_per_role = 200
```

Step C (dense rerank) computes cosine similarity between:

```text
embedding(semantic_query or user_query)
and
embedding(item_text)
```

but only for the shortlist. Dense rerank size (example):

```text
top_n_dense_per_role = 50
```

Dense retrieval is best used for positive semantics (cozy ~= fleece-lined, polished ~= tailored). Negations like "not sporty" should primarily be handled in the sparse guardrails/filters.

## Final LLM Reranker

The final reranker already receives:

```text
user_query
constraints
outfit_candidates
```

So if Parser V2 adds richer fields to `constraints`, the final LLM reranker automatically gets them.

Code area:

```text
src/recommender/outfits.py
_apply_llm_reranking(...)
```

Prompt area:

```text
src/integrations/openai_client.py
llm_rerank_outfits(...)
```

Potential prompt update:

```text
Use the query constraints, especially positive_terms, negative_terms,
style_direction, and avoid_categories, when judging the final outfits.
```

## Dense + Sparse Retrieval Later

Keyword matching is useful but limited.

Example:

```text
User says cozy
Description says fleece-lined
```

Sparse keyword matching may miss that relationship.

Dense retrieval can understand that:

```text
cozy ~= fleece-lined ~= warm ~= soft
```

Longer-term hybrid retrieval:

```text
structured metadata score
+ sparse keyword score
+ dense description similarity score
+ outfit-level score
+ LLM reranking
```

Dense retrieval should still happen inside role pools:

```text
dense search among tops
dense search among bottoms
dense search among shoes
```

not blindly across the full catalog.

## Why Not Do Dense Only / First?

Dense retrieval is powerful, but it needs more infrastructure and guardrails:

```text
embedding generation
embedding storage
similarity search
cache/versioning
possibly pgvector later
```

Also, embeddings can be unreliable for:

```text
hard constraints (roles/categories/colors)
negation ("not sporty")
```

So V2 uses a funnel: sparse filters/guards keep constraints correct, then dense rerank improves semantic match quality within a safe shortlist.

Parser V2 and text hint scoring are easier to add now, and they prepare the system for dense retrieval by creating `semantic_query`.

## Implementation Steps

1. Extend parser merge logic.

Code:

```text
src/recommender/query_parser.py
_merge_llm_constraints(...)
```

Add keys:

```text
style_direction
positive_terms
negative_terms
avoid_categories
preferred_materials
preferred_details
semantic_query
```

2. Update OpenAI parser prompt.

Code:

```text
src/integrations/openai_client.py
llm_parse_query(...)
```

Tell it to return richer retrieval hints with allowed values where appropriate.

3. Add deterministic defaults.

Code:

```text
src/recommender/query_parser.py
_deterministic_parse_user_query(...)
```

Add empty defaults:

```python
"style_direction": [],
"positive_terms": [],
"negative_terms": [],
"avoid_categories": [],
"preferred_materials": [],
"preferred_details": [],
"semantic_query": user_query,
```

4. Add text hint scoring.

Code:

```text
src/recommender/retrieval.py
_score_text_hints(...)
_score_item(...)
```

Apply positive and negative hint scoring to the combined item text.

5. Update reranker prompt.

Code:

```text
src/integrations/openai_client.py
llm_rerank_outfits(...)
```

Tell the reranker to use the richer constraints when judging final outfits.

6. (Optional but recommended) Add dense embedding rerank stage.

Code areas (example structure; adapt to your repo layout):

```text
scripts/build_rag_index.py  (build embeddings for item_text)
src/recommender/retrieval.py (dense rerank within shortlisted items)
```

Notes:

```text
Store embeddings keyed by item_id.
Build item_text exactly like `_score_text_hints` uses (same normalization).
Only run dense similarity for top_k_sparse_per_role candidates.
```

7. Update test script display.

Code:

```text
scripts/test_recommender.py
```

Show the new parser fields so we can inspect whether the LLM parser is producing useful hints.

8. Run targeted tests.

Use queries like:

```text
I want a warm casual jacket, not sporty
Give me a relaxed but polished dinner outfit
Build a clean office outfit, not too formal
I want a lightweight summer outfit with sneakers
Give me something cozy with a zip hoodie
```

9. Compare quality.

Use:

```text
scripts/test_recommender.py
eval/run_benchmark.py
eval/compare_outputs.py
```

Human judgment is still needed for style quality, but automatic checks can confirm that:

```text
required roles are present
requested colors appear
avoided categories are reduced
top 3 are diverse
image URLs work
```

Add a small fixed evaluation set (10-30 queries). Track:

```text
role coverage rate
avoid-category violation rate
basic attribute compliance (target_group, explicit color/category)
top-1 / top-3 acceptability (quick human label)
```

## Impact On VLM Work

This plan does not require your teammate to change the basic VLM target schema.

VLM wardrobe items should still match the current item fields first:

```text
display_name
description
target_group
recommendation_role
normalized_category
product_family
normalized_color
color_detail
color_tone
normalized_pattern
section_theme
image_path
```

Optional VLM enrichment fields can help later:

```text
style_tags
occasion_tags
material
fit
season
formality
vlm_confidence
```

For Retrieval V2, the most important VLM field is still:

```text
description
```

A good VLM description makes both sparse and dense retrieval better.

## Recommended Order

Do this first:

```text
Parser V2 + description-aware sparse scoring
```

Do this later:

```text
dense embeddings over item text
hybrid dense + sparse retrieval
database or pgvector-backed storage
user wardrobe persistence
```

## One-Sentence Summary

Parser V2 turns the user query into a richer style brief, retrieval applies that brief to existing item metadata and descriptions, and the final LLM reranker uses the same brief to choose and explain better outfit options.
