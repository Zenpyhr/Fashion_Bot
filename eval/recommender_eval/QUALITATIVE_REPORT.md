# Qualitative Evaluation Report (Track B — Clothes Recommendation)

This qualitative evaluation is **artifact-driven**: we reuse the saved outputs from the quantitative suite and write human-readable case studies + error analysis.

## Artifacts used

Suite folder:

- `eval/recommender_eval/artifacts/suites/suite_20260506_154651/`

Compared variants:

- **Baseline**: `baseline_det`
- **Best non-wardrobe**: `parser_plus_dense`
- **Wardrobe-enabled**: `wardrobe_on` (same config as `parser_plus_dense`, with `user_id="demo_user"`)

Primary evidence sources:

- `*/run.json` for outputs (top outfits + parsed constraints)
- `*/judge.json` for judge scores and natural-language reasons (used as supporting evidence, not ground truth)

---

## Method

We conducted a structured manual review over the stable 10-query set (`eval/recommender_eval/queries_eval_10_mens.txt`).

For each query, we inspected:

- parsed constraints (e.g. roles, formality/occasion, category hints, color intent)
- the top outfit(s) returned by each variant
- judge “reasons” where helpful for identifying mismatches

We then wrote case studies for representative queries and compiled an error taxonomy describing the most common failure modes.

---

## Case studies (selected)

### Case 1 — Negation + outerwear: “Warm casual jacket, not sporty”

**Expected**: a warm, casual outfit with a jacket/outerwear emphasis; avoid sporty cues.

**Baseline (`baseline_det`)**:
- Judge noted the jacket request is matched, but **bottoms and shoes can drift** (e.g. shorts + sneakers) which weakens “warm” and “not sporty”.

**Best non-wardrobe (`parser_plus_dense`)**:
- Score dropped mainly on **constraint fit** in this case (overall 8; constraint_fit 1).
- Judge reasons highlight two issues:
  - the “jacket” intent can be satisfied by an item **categorized as hoodie** (taxonomy mismatch: hoodie vs jacket)
  - **shorts** appearing conflicts with “warm” and can read sporty

**Wardrobe (`wardrobe_on`)**:
- Improved constraint fit here (overall 13; constraint_fit 4) but still shows an occasional **shorts-in-warm-context** mismatch in some top-3 options.

**Takeaway**: negation and weather/warmth are partially handled via heuristics; wardrobe helps but we need stronger “warmth” and “not sporty” guardrails for bottoms.

---

### Case 2 — Style nuance: “Relaxed but polished dinner outfit”

**Expected**: smart casual, dinner-appropriate, not overly formal.

**Best non-wardrobe (`parser_plus_dense`)**:
- Perfect judge outcome (overall 15) with a coherent “polished but relaxed” combination (e.g. linen shirt + tailored trousers + dressier shoes).

**Wardrobe (`wardrobe_on`)**:
- Also strong (overall 14), showing that wardrobe integration does not inherently break dinner styling.

**Takeaway**: the system performs very well when constraints map cleanly to canonical categories (shirt/trousers/shoes) and formality cues are consistent.

---

### Case 3 — Work constraint: “Clean office outfit, not too formal”

**Expected**: office-appropriate but not a full formal suit.

**Best non-wardrobe (`parser_plus_dense`)**:
- Perfect judge outcome (overall 15). Judge reasons specifically call out role correctness and a coherent smart-casual office look (shirt + trousers + boots/sneakers).

**Wardrobe (`wardrobe_on`)**:
- Also perfect (overall 15), indicating wardrobe insertion can still respect office constraints.

**Takeaway**: office constraints are handled reliably; this is a strong demo query.

---

### Case 4 — Explicit color constraint: “All black outfit with white sneakers”

**Expected**: top and bottom black; shoes white sneakers.

**Best non-wardrobe (`parser_plus_dense`)**:
- Perfect judge outcome (overall 15) and explicitly credits the system for meeting the “all black + white sneakers” constraint.

**Wardrobe (`wardrobe_on`)**:
- Judge shows a constraint-fit drop (overall 13; constraint_fit 3) because **some retrieved items were not black**, even though intent was mostly respected.

**Takeaway**: wardrobe boosting can override strict palette constraints; “wardrobe relevance” should be conditioned on the query (or wardrobe items need better color-aware filtering).

---

### Case 5 — Weather function: “Rainy day outfit with a hooded jacket”

**Expected**: hooded jacket present; outfit reads weather-appropriate.

**Best non-wardrobe (`parser_plus_dense`)**:
- Very strong (overall 14) with the hooded jacket constraint satisfied; judge reasons emphasize the coherence of hoodie/outdoor trousers/boots for rain.

**Wardrobe (`wardrobe_on`)**:
- Perfect (overall 15) in this run.

**Takeaway**: weather-function queries are a strong category, especially when “hooded jacket” is treated as a hard category/role hint.

---

### Case 6 — Minimalism + palette restriction: “Minimal clean outfit, neutral colors only”

**Expected**: minimal styling; strictly neutral palette.

**Best non-wardrobe (`parser_plus_dense`)**:
- Perfect judge outcome (overall 15). Judge reasons state it matches neutral palette and minimal look.

**Wardrobe (`wardrobe_on`)**:
- Major regression (overall 8). Judge reasons identify **non-neutral colors** (e.g. blue) violating “neutral only”.

**Takeaway**: wardrobe boost can harm strict constraints (especially color). This is a prime example to motivate “conditional wardrobe prioritization” as future work.

---

## Error analysis (taxonomy)

We group observed qualitative issues into the following buckets:

1) **Constraint miss (color/category)**  
2) **Negation / intent miss** (e.g., “not sporty”)  
3) **Weather/context mismatch** (e.g., shorts in “warm/rainy” contexts)  
4) **Wardrobe tradeoff** (personalization overrides strict constraints)  
5) **Repetition in top‑3** (top‑3 too similar; supported quantitatively by duplicate-signature rate)

### Observed failure patterns in this suite

- **Negation + warmth nuance**: “Warm casual jacket, not sporty” shows shorts appearing in some top outfits, conflicting with “warm” and sometimes with “not sporty”.  
- **Wardrobe tradeoff on strict constraints**:
  - “All black outfit with white sneakers” and “Minimal clean outfit, neutral colors only” show that wardrobe-on can violate strict color constraints even when relevance remains high.

These qualitative findings match the quantitative signals:

- `wardrobe_on` achieves wardrobe hit@1/hit@3 = 100% but has lower `Category any@1` and lower judge overall than `parser_plus_dense`.

---

## Limitations and future work

- **Wardrobe relevance gating**: wardrobe items are currently boosted unconditionally once a user_id is present. Future work: boost only when the query implies “use my wardrobe” or when wardrobe items match key constraints (e.g., requested color/category).
- **Strict constraint enforcement**: implement hard filters for “neutral only” / “all black” when those constraints are explicit.
- **Negation + weather semantics**: improve deterministic intent detection (e.g., “warm” / “rainy”) so bottoms (shorts) are penalized when inappropriate.
- **Judge reliability**: LLM judge can be subjective; future work: retry on null responses, multi-judge, and more deterministic checks (role completeness, color hit, diversity).
- **Small evaluation set**: 10-query set is useful for regression but limited for broad claims. Future work: expand tagged benchmark and report by-tag breakdowns.
- **Wardrobe embeddings (Option 3)**: wardrobe items do not yet participate in dense embedding retrieval. Adding embeddings for wardrobe could improve relevance and reduce constraint violations by better semantic matching.

