# Qualitative Evaluation Report (Track B — Clothes Recommendation)

This qualitative evaluation is **artifact-driven**: it is grounded in outputs from `run_quant_suite.py` (`run.json`, deterministic metrics, and optional LLM judge). Narrative interpretations are ours; numerical summaries match the cited suite tables.

---

## Artifacts used

Suite folder:

- `eval/recommender_eval/artifacts/suites/suite_20260506_234438/`

Query file:

- `eval/recommender_eval/queries_wardrobe_eval_demo_user.json` (12 queries: 5 `positive`, 4 `mix`, 3 `negative_control`)

Compared variants:

- **`catalog_sparse`**: wardrobe off, sparse retrieval only (dense rerank disabled).
- **`catalog_dense`**: wardrobe off, sparse + dense rerank on the catalog.
- **`wardrobe_sparse`**: `user_id="demo_user"`, sparse only (dense rerank disabled).
- **`wardrobe_dense`**: `user_id="demo_user"`, sparse + unified dense rerank on catalog ∪ wardrobe pools.

Evidence:

- Variant folders: `*/run.json`, `*/metrics_summary.json`, `*/judge.json`
- Suite roll-up: `report.md`

We follow `EVAL_GUIDANCE.md`: treat **`wardrobe_hit_at_1` and `wardrobe_constraint_override`** on **positive + mix** only; treat **`negative_control_wardrobe_intrusion`** on **negative_control** only. Judge outputs are **supporting evidence**, not ground truth (`EVAL_GUIDANCE.md` §8).

---

## Snapshot (quantitative, this suite)

From `suite_20260506_234438/report.md`:

| Variant | Role complete@1 | Category hit@1 | Wardrobe hit@1 | Wardrobe override | Neg-control intrusion |
|---------|----------------:|---------------:|---------------:|------------------:|----------------------:|
| catalog_sparse | 100% | 58.3% | — | — | — |
| catalog_dense | 100% | 66.7% | — | — | — |
| wardrobe_sparse | 100% | 83.3% | 77.8% | 28.6% | 100.0% |
| wardrobe_dense | 100% | **100%** | **100%** | **0.0%** | **100.0%** |

Dense lift (`wardrobe_dense` minus `wardrobe_sparse`): wardrobe hit +22.2 pp; wardrobe override −28.6 pp; negative-control intrusion unchanged at 100%; mean judge overall down slightly (suite reports −16.7% on that delta—on a **0–15** overall scale this is roughly a **−0.17** point shift, easy to misread as a large drop).

**Headline.** Dense retrieval plus wardrobe materially improves anchoring on explicit garment types and brings wardrobe pieces into **every** positive/mix top-1 in this run, without the “override when wardrobe is present” spike seen under wardrobe_sparse. The main systemic problem remains **over-insertion of wardrobe on negative_control queries**.

---

## Method

Over each of the 12 benchmark rows we inspected:

1. Parsed constraints (`required_roles`, `preferred_categories`, colors, semantic query).

2. **Top-1 outfit** (`top_outfits[0]`): per-item `normalized_category`, `normalized_color`, `recommendation_role` / `role`, and **`source_type` (catalog vs wardrobe)**.

3. Where useful, **`judge.json`** `reasons` for mismatches (noting inconsistencies when judge text contradicts item fields).

Benchmark intent (`EVAL_GUIDANCE.md`): success is anchoring **real uploads** (`demo_user`’s wardrobe) and completing the look from H&M—not wardrobe-only outfits (the demo wardrobe is top-heavy).

---

## Case studies

Each case compares **catalog** variants (personalization absent) versus **wardrobe_sparse** versus **wardrobe_dense** unless noted.

### Case 1 — Overshirt anchor (positive): “Style a casual look around a blue overshirt” (W03)

**Expected**: Casual look centered on the uploaded **blue overshirt** (benchmark anchor: category `jacket`, color blue).

**`catalog_sparse`** (no wardrobe):

- Top-1 resolves to coherent blue outfit pieces but **never uses the wardrobe overshirt**; parser often prefers `shirt`-shaped hints over `jacket`/overshirt (see `preferred_categories`: `shirt` in `run.json`). Deterministic “category_any_hit_top1” is **false** for this anchor in catalog runs.

**`wardrobe_dense`**:

- Top-1 can place the wardrobe item **Blue washed denim overshirt** as top (category `jacket`, blue), with catalog shorts + sneakers—all blue—which matches the anchored intent visually and by category.

**`wardrobe_sparse`** (failure mode):

- Top-1 can still pick another **blue wardrobe shirt** that is **not** the overshirt, trading one wardrobe item for another and failing the **`jacket` anchor** metric (deterministic override flagged in metrics rows).

**Takeaway**: Dense pooling + embeddings help **surface the intended wardrobe piece**, not merely “something blue from the closet.” Sparse-only wardrobe mode still competes wrongly across tops.

---

### Case 2 — Mix: trousers + sneakers + overshirt (W08): “Style the blue overshirt with trousers and sneakers”

**Expected**: Blue overshirt + **trousers** + **sneakers**; wardrobe should contribute the overshirt.

**`wardrobe_sparse`**:

- Top-ranked outfit can still be dominated by triple-blue scoring (e.g. **shirt-style top + blue trousers + blue sneakers**) where the top wardrobe slot is occupied by ** another blue top**, missing the overshirt-specific anchor (`jacket`), again triggering deterministic override metrics.

**`wardrobe_dense`**:

- Top-1 aligns with **`jacket + trousers + sneakers`** combinations that include **Blue washed denim overshirt**, blue trousers, blue sneakers—in line with wording and anchor metadata.

**Takeaway**: The **dense rerank step** materially reduces wrong-wardrobe-top competition when the query is multi-constraint (garment phrase + bottoms + footwear).

---

### Case 3 — Shoes-forward positive (W01): “Use my grey sneakers in a casual everyday outfit”

**Expected**: Outfit includes uploaded **grey sneakers** (anchors: `sneakers`, gray).

**`wardrobe_sparse`**:

- In this suite run, grey sneakers sometimes **did not appear in top-1** (wardrobe_miss under sparse), even though catalogue could supply grey-coded sneakers—the model preferred a different maximizing outfit.

**`wardrobe_dense`**:

- Wardrobe sneakers appear in winning outfits; metrics show wardrobe hit and no constraint override.

**Takeaway**: Dense retrieval is doing the expected job of **lifting the user’s footwear** into the top combination when sparse scoring alone underweights it.

---

### Case 4 — Mix weekend + grey sneakers (W07): “Give me a relaxed weekend outfit that uses grey sneakers”

Same structural pattern as W01 under wardrobe_sparse (**miss** wardrobe at top in this run) versus wardrobe_dense (**hit**).

**Takeaway**: Repeats that **dense lift on wardrobe_hit@1** is not anecdotal—it shows up consistently on sneakers-forward prompts where the embedding text and query semantics align.

---

### Case 5 — Office completion with anchored trousers (W06): “Build a clean office outfit using black trousers”

**Expected**: Office-appropriate; black tailored trousers anchored (often wardrobe bottom).

Across catalog and wardrobe variants, top-1 outfits in artifacts remain role-complete (`top`/`bottom`/`shoes`). Wardrobe_dense places **Black tailored trousers** as bottom with coherent upper and footwear (e.g. black shirt/boots combos in summarized outputs).

**Takeaway**: **Office + explicit trouser mentions** behave as a stabilizing bracket for the funnel; personalization often correctly reuses trousers while letting the composer pick formal-adjacent catalog tops/shoes.

---

### Case 6 — Negative control: wardrobe should stay out—but does not

Benchmark labels `negative_control` with `expected_wardrobe_use: "no"`. Quantitative intrusion is **100%** under both **`wardrobe_sparse`** and **`wardrobe_dense`** in this suite.

Representative **`wardrobe_dense` top-1** patterns from `run.json`:

- **W10** (“Formal dinner outfit with dark dress shoes”) — wardrobe **black trousers** still appear beside formal catalog shirt and dressier shoes (intrusion counts any wardrobe slice present).

- **W11** (“Rainy day outfit with a hooded jacket”) — e.g. **grey wardrobe sneakers** in a rain-oriented outerwear-heavy outfit alongside catalog hoodie/pants/jacket.

- **W12** (“All black outfit with white sneakers”) — top-1 can be **all black sneakers** (`normalized_color`: black), **violating explicit white sneakers** preference, combined with wardrobe trousers; judge text for this prompt is **internally contradictory** on whether white sneakers are satisfied—illustrating why we do not defer to judge verbatim.

**Takeaway**: Retrieval and scoring still **prioritize plausible-looking outfits composed with strong wardrobe overlaps** rather than enforcing “no wardrobe unless asked.” Fixing this belongs in policy (intent detection or post-filter), not embeddings alone.

---

## Error analysis (taxonomy)

We label observed failures primarily from **deterministic cues** (`category_any_hit`, wardrobe flags) and secondarily from **human review** of outfits.

| Bucket | Frequency / signal (this suite) | Example IDs |
|--------|----------------------------------|-------------|
| **Wrong wardrobe garment** — query asks one upload; another replaces it under sparse/boost | Seen under `wardrobe_sparse` on overshirt-style queries | W03, W08 |
| **Negative-control intrusion** — wardrobe in top when benchmark says absent | Universal (100% neg-control) under both wardrobe variants | W10–W12 |
| **Strict attribute miss** — text asks white sneakers; shoes are black catalog | Seen in artifact review | W12 top-1 under wardrobe_dense |
| **Parser/category skew** — `overshirt` parsed toward `shirt` | Contributes catalog-side misses vs `jacket` anchor | W03 in catalog_sparse |
| **Judge mismatch** — high scores paired with contradictory `reasons` | Undermines naive “judge-as-oracle” | W12 judge.json |

Older failure modes cited in legacy reports (**repetition in top‑3**, duplicate signatures) are **no longer headline metrics** in `EVAL_GUIDANCE.md`; cite them only if recomputed separately.

---

## Limitations and future work

1. **Wardrobe gating**: When queries do not reference “my / use my / wardrobe” concepts, personalization should downgrade or omit wardrobe items (`negative_control` is systematically violated today).

2. **Conditional boost**: A fixed wardrobe boost yields strong positive/mix coverage but reinforces intrusion; consider score-modulation from query intent and **semantic similarity to query embedding**.

3. **Hard constraints after retrieval**: Palette-level rules (“white sneakers,” “neutral only”) need **explicit filters** once parsed, regardless of retrieval rank.

4. **Judge calibration**: Means cluster near ceilings (relevance ~5/5 overall run); aggregate judge shifts are noisy. Prefer **paired read-through** plus deterministic checks (`report.md`) for claims.

5. **Benchmark size**: 12 queries suffice for directional claims and regressions—not for broad demographic or style coverage.

6. **Operational coupling**: Metrics assume the frozen **`demo_user`** wardrobe and Postgres embeddings documented in `EVAL_GUIDANCE.md`; different uploads invalidate cross-run comparison without a new freeze.

---

## User study

We did not run a structured user study for this milestone; qualitative conclusions rest on structured case review and the benchmarks above (`QUALITATIVE_EVAL_GUIDE.md` encourages stating this plainly when absent).
