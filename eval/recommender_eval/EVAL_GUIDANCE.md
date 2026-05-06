# Recommender Evaluation Guidance

This document defines the recommended evaluation setup for the current wardrobe-aware recommender.

It replaces the older "5 my clothes + 5 random H&M clothes" idea with a cleaner setup:

- use the real uploaded wardrobe items as the personalization test set
- use the H&M catalog as the completion source
- build queries around the uploaded wardrobe
- compare sparse vs dense, with and without wardrobe

## 1. Goal

The current system has two jobs:

1. return a good outfit
2. use the user's uploaded clothes when they are relevant, without forcing them when they are not

The evaluation should therefore answer three questions:

1. Does the final outfit still look good?
2. Does the wardrobe get used when it should?
3. Does dense retrieval improve wardrobe use instead of hurting constraint fit?

## 2. Current Wardrobe Test Set

As of the current demo database, `demo_user` has 5 uploaded wardrobe items and all 5 have embeddings.

Current wardrobe items:

- `3eeeb70dbf204f35`: navy Columbia sweatshirt, role `top`, category `shirt`, color `blue`
- `d334b01e7453f847`: grey chunky sneakers, role `shoes`, category `sneakers`, color `gray`
- `4591986de1616e18`: black tailored trousers, role `bottom`, category `trousers`, color `black`
- `98fb635141451387`: blue washed denim overshirt, role `top`, category `jacket`, color `blue`
- `e810bb3085818e63`: beige half-zip sweatshirt, role `top`, category `sweater`, color `beige`

Important constraint:

- the wardrobe is top-heavy: 3 tops, 1 bottom, 1 pair of shoes
- the benchmark should not expect wardrobe-only outfits
- success means the system can anchor on wardrobe items and use H&M items to complete the rest of the outfit

## 3. Query Set Design

Use one wardrobe benchmark with 12 total queries, split into 3 buckets.

### 3.1 Buckets

- `positive`
  These queries should clearly use one of the uploaded wardrobe items.

- `mix`
  These queries should still use wardrobe, but the wardrobe item is only one piece of a fuller look and H&M should complete the outfit.

- `negative_control`
  These queries should not force wardrobe into the result when the uploaded items are a poor match.

### 3.2 Exact Query Count

- 5 `positive`
- 4 `mix`
- 3 `negative_control`

This is small enough for manual review and large enough to show whether wardrobe dense retrieval helps.

### 3.3 Query Writing Rules

For every wardrobe-based query:

- anchor the query to a real uploaded item by color, category, or obvious function
- keep wording natural, as if written by a user
- do not mention internal ids
- do not create impossible expectations, such as wardrobe-only outerwear if the user did not upload outerwear
- when possible, include one clear intent dimension: casual, office, neutral, weekend, layered, smart casual

For negative controls:

- choose intents that do not naturally fit the uploaded wardrobe
- use them to test whether the system avoids irrelevant wardrobe insertion

### 3.4 Recommended 12-Query Benchmark

#### Positive Queries

- `W01`: "Use my grey sneakers in a casual everyday outfit"
- `W02`: "Build a smart casual outfit with black trousers"
- `W03`: "Style a casual look around a blue overshirt"
- `W04`: "Make a neutral outfit using a beige half-zip sweatshirt"
- `W05`: "Use the navy sweatshirt in a relaxed weekend outfit"

#### Mix Queries

- `W06`: "Build a clean office outfit using black trousers"
- `W07`: "Give me a relaxed weekend outfit that uses grey sneakers"
- `W08`: "Style the blue overshirt with trousers and sneakers"
- `W09`: "Create a layered neutral outfit using the beige half-zip"

#### Negative Controls

- `W10`: "Formal dinner outfit with dark dress shoes"
- `W11`: "Rainy day outfit with a hooded jacket"
- `W12`: "All black outfit with white sneakers"

### 3.5 Query Metadata To Store

Store the benchmark as structured rows, not only plain text.

Each query row should include:

- `id`
- `query`
- `bucket`: `positive | mix | negative_control`
- `expected_wardrobe_use`: `yes | no`
- `anchor_category`: optional, such as `sneakers`, `trousers`, `jacket`, `sweater`, `shirt`
- `anchor_color`: optional, such as `gray`, `black`, `blue`, `beige`

Recommended default:

- `expected_wardrobe_use=yes` for `positive` and `mix`
- `expected_wardrobe_use=no` for `negative_control`

## 4. System Variants To Compare

Run the same benchmark across 4 variants.

- `catalog_sparse`
  No wardrobe, no dense retrieval.

- `catalog_dense`
  No wardrobe, dense retrieval enabled.

- `wardrobe_sparse`
  Wardrobe enabled, dense retrieval disabled.

- `wardrobe_dense`
  Wardrobe enabled, dense retrieval enabled.

Interpretation:

- `catalog_sparse -> catalog_dense` measures dense retrieval without personalization
- `catalog_dense -> wardrobe_dense` measures the effect of turning wardrobe on
- `wardrobe_sparse -> wardrobe_dense` is the main comparison for whether dense retrieval improves wardrobe use

## 5. Primary Metrics

Use a small metric set. These are the primary numbers to report.

### 5.1 `role_complete_top1_rate`

Definition:

- fraction of queries where the top-1 outfit contains all required roles

Why keep it:

- basic sanity check
- if this is low, the system is failing before personalization matters

Where to report:

- all 4 variants
- all query buckets combined

### 5.2 `category_any_hit_top1_rate`

Definition:

- fraction of queries where top-1 contains at least one explicitly requested category

Examples:

- "with black trousers" should hit `trousers`
- "with grey sneakers" should hit `sneakers`
- "hooded jacket" should hit `jacket` or a close outerwear category if that mapping is intentionally allowed

Why keep it:

- strongest simple deterministic check for whether explicit item intent survives retrieval

Where to report:

- all 4 variants
- all query buckets combined

### 5.3 `judge_overall`

Definition:

- mean LLM-as-judge overall score for the benchmark

Judge dimensions already used:

- relevance
- constraint fit
- coherence

Why keep it:

- highest-signal end-to-end quality number

Where to report:

- all 4 variants
- overall and by bucket if possible

### 5.4 `wardrobe_hit_at_1_rate`

Definition:

- fraction of queries where top-1 contains at least one wardrobe item

Important scope rule:

- only treat this as a success metric on `positive` and `mix` queries
- do not use it as a success metric on `negative_control`

Why keep it:

- simplest signal that personalization is actually showing up in the winning outfit

Where to report:

- `wardrobe_sparse` and `wardrobe_dense`
- on `positive + mix` only

### 5.5 `wardrobe_constraint_override_rate`

Definition:

- among queries where top-1 uses wardrobe, fraction where wardrobe use causes a miss on an important explicit constraint

Typical examples:

- wardrobe item causes the wrong category to anchor the outfit
- wardrobe item breaks an explicit palette request
- wardrobe item pulls the outfit away from the requested style

Why keep it:

- this is the main risk metric for the new design
- it tells us whether personalization is helping or forcing bad matches

Where to report:

- `wardrobe_sparse` and `wardrobe_dense`
- on `positive + mix`

### 5.6 `negative_control_wardrobe_intrusion_rate`

Definition:

- fraction of `negative_control` queries where top-1 contains a wardrobe item even though the benchmark expects no wardrobe use

Why keep it:

- negative controls are the cleanest way to measure over-insertion

Where to report:

- `wardrobe_sparse` and `wardrobe_dense`
- on `negative_control` only

### 5.7 `dense_lift`

This is not a raw per-query metric. It is the main comparison headline.

Definition:

- `wardrobe_dense - wardrobe_sparse` on:
  - `wardrobe_hit_at_1_rate`
  - `judge_overall`
  - `wardrobe_constraint_override_rate`
  - `negative_control_wardrobe_intrusion_rate`

Desired pattern:

- higher `wardrobe_hit_at_1_rate`
- same or higher `judge_overall`
- same or lower `wardrobe_constraint_override_rate`
- same or lower `negative_control_wardrobe_intrusion_rate`

## 6. Metrics We Are Not Using As Primary Numbers

These may still exist in old scripts, but they are not headline metrics for the new benchmark.

- `wardrobe_hit_at_3_rate`
- `color_any_hit_top1_rate`
- `top3_duplicate_signature_rate`
- full retrieval ranking metrics like MRR or NDCG

Reason:

- they add noise or redundancy for phase 1
- they do not directly answer the current architecture question as well as the smaller primary metric set

## 7. Evaluation Workflow

### Step 1. Confirm Wardrobe Data

Before every benchmark run, confirm:

- the expected wardrobe rows exist in `wardrobe_items`
- the same wardrobe ids exist in `wardrobe_item_embeddings`

Minimum requirement:

- row count in `wardrobe_items` matches the intended uploaded test set
- row count in `wardrobe_item_embeddings` matches the same set

### Step 2. Freeze The Benchmark

Do not change the 12 benchmark queries once measurement starts.

If you need to revise the benchmark later:

- create a new versioned query file
- keep the old one for comparison

### Step 3. Run The 4 Variants

Run the same query set across:

- `catalog_sparse`
- `catalog_dense`
- `wardrobe_sparse`
- `wardrobe_dense`

The benchmark runner should save:

- raw outputs
- deterministic metric rows
- metric summaries
- optional judge outputs

### Step 4. Compute Metrics By Scope

Report metrics with the correct scope:

- `role_complete_top1_rate`: all queries
- `category_any_hit_top1_rate`: all queries
- `judge_overall`: all queries, and optionally by bucket
- `wardrobe_hit_at_1_rate`: `positive + mix` only
- `wardrobe_constraint_override_rate`: `positive + mix` only
- `negative_control_wardrobe_intrusion_rate`: `negative_control` only

### Step 5. Write The Comparison

The main conclusion should focus on:

1. whether wardrobe dense retrieval increases relevant wardrobe usage
2. whether it improves or harms end-to-end outfit quality
3. whether it over-inserts wardrobe on negative controls

## 8. Manual Review Guidance

For a small benchmark like this, every run should include manual review of all 12 top-1 outfits.

For each query, inspect:

- top-1 items
- whether any wardrobe item appears
- whether the wardrobe item is the right one
- whether the outfit still fits the query

Use the judge output as supporting evidence, not ground truth.

When a failure happens, label it with one main reason:

- wrong wardrobe item
- wardrobe missing when expected
- wardrobe forced when not expected
- explicit category miss
- style mismatch
- coherence mismatch

## 9. Recommended Artifact Layout

Keep all benchmark artifacts under `eval/recommender_eval/artifacts/`.

Recommended additions:

- one structured query file for the wardrobe benchmark
- one suite result per run
- one short manual review report for the latest run

Suggested query filename:

- `queries_wardrobe_eval_demo_user.json`

Suggested fields for that file:

```json
{
  "id": "W01",
  "query": "Use my grey sneakers in a casual everyday outfit",
  "bucket": "positive",
  "expected_wardrobe_use": "yes",
  "anchor_category": "sneakers",
  "anchor_color": "gray"
}
```

## 10. Decision Summary

This is the evaluation design to use unless the benchmark is intentionally revised later.

- use the 5 real uploaded wardrobe items as the personalization test set
- use 12 total benchmark queries
- split queries into `positive`, `mix`, and `negative_control`
- compare `catalog_sparse`, `catalog_dense`, `wardrobe_sparse`, and `wardrobe_dense`
- report 6 primary numbers:
  - `role_complete_top1_rate`
  - `category_any_hit_top1_rate`
  - `judge_overall`
  - `wardrobe_hit_at_1_rate`
  - `wardrobe_constraint_override_rate`
  - `negative_control_wardrobe_intrusion_rate`
- use `dense_lift` as the comparison headline rather than as a separate raw metric

This setup is intentionally small, interpretable, and aligned with the current system design.
