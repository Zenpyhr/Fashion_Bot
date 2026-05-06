# Qualitative evaluation guide (Track B — recommender)

This guide shows how to write the **Qualitative Analysis** section using the artifacts you already produced with:

`eval/recommender_eval/run_quant_suite.py`

It is designed to be **human-written**, but **artifact-driven** (no guesswork).

---

## 0) Pick the suite artifacts to use

Use the suite folder that includes the judge table in `report.md` and has `judge.json` files.

Example (your latest judged suite):

- `eval/recommender_eval/artifacts/suites/suite_20260506_154651/`

Inside you will use:

- `report.md` (high-level numbers; helps pick cases)
- `baseline_det/run.json` (baseline outputs)
- `parser_plus_dense/run.json` (best non-wardrobe outputs)
- `wardrobe_on/run.json` (wardrobe outputs)
- `*/judge.json` (optional: judge reasons, helpful for explaining failures)

---

## 1) What you are writing (rubric mapping)

Your professor’s “Qualitative Analysis” typically expects these subsections:

1. **Case studies** (6–8 examples): before/after comparisons with short explanation.
2. **Error analysis**: bucket failures, count them, give representative examples.
3. **Limitations and future work**: honest constraints + next steps tied to observed failures.
4. (Optional) **User study**: if you run one, summarize it; if not, state why.

---

## 2) Case studies (how to do them with existing outputs)

### 2.1 Choose 6–8 queries

Start from the 10 stable queries in:

- `eval/recommender_eval/queries_eval_10_mens.txt`

Recommended coverage set (good mix of constraints):

- Warm casual jacket, not sporty
- Clean office outfit, not too formal
- Rainy day outfit with a hooded jacket
- All black outfit with white sneakers
- Minimal clean outfit, neutral colors only
- Outfit with boots for a casual weekend
- Something cozy with a zip hoodie (optional)
- Smart casual date night outfit, not flashy (optional)

### 2.2 Compare variants for each case

For each query, compare:

- **Baseline**: `baseline_det`
- **Improved**: `parser_plus_dense`
- **Wardrobe (only for 1–2 cases)**: `wardrobe_on` (uses `demo_user`)

### 2.3 What to extract from `run.json`

For the query entry, copy:

- `parsed_constraints` (only key fields: `required_roles`, `preferred_colors`, `preferred_categories`, `formality`, `occasion`, `semantic_query`)
- `top_outfits[0]` (top-1) **items** list: `normalized_category`, `normalized_color`, `role`, and whether `source_type=="wardrobe"`
- optional: `top_outfits[0].signature` (quick shorthand)
- optional: `judge.json` reason if it clearly explains a mismatch (e.g., “shorts don’t match warm jacket”)

### 2.4 Case-study template (copy/paste)

Use this exact structure for consistency:

**Query**: `<text>`

**Expected** (1 sentence):
- `<what the user asked for, in plain terms>`

**Baseline (`baseline_det`)**:
- Parsed constraints: `<roles/colors/formality/occasion/category hints>`
- Top-1 outfit: `<category(color) per item, 1 line>`
- What went wrong (if any): `<1–2 bullets>`

**Improved (`parser_plus_dense`)**:
- Parsed constraints: `<same fields>`
- Top-1 outfit: `<category(color) per item>`
- Why it’s better: `<1–2 bullets>`

**Wardrobe (`wardrobe_on`, optional)**:
- Top-1 outfit wardrobe usage: `<which item(s) are from wardrobe>`
- Tradeoff: `<did it violate any category/color constraint?>`

**Takeaway** (1 sentence):
- `<what this example demonstrates about your design>`

---

## 3) Error analysis (how to implement)

### 3.1 Define buckets (recommended)

Use 5–7 buckets (don’t overcomplicate):

1. **Constraint miss** (color/category/occasion/formality)
2. **Role incomplete** (missing required roles)
3. **Negation miss** (“not sporty” violated)
4. **Repetition** (top‑3 too similar)
5. **Wardrobe tradeoff** (wardrobe helps personalization but hurts constraint fit)
6. **Coherence mismatch** (items don’t belong together)

### 3.2 Label each query once

For each of the 10 queries, assign **one primary bucket** for:

- baseline_det
- parser_plus_dense
- wardrobe_on (only if you evaluated it)

Then report:

- “Most frequent failures for baseline: …”
- “Most frequent failures after improvements: …”

### 3.3 Include representative examples

For the top 2 buckets:

- Include 1 concrete query example
- Include 1 short hypothesis for why it happens
- Include 1 fix idea

---

## 4) Limitations + future work (tie to what you observed)

Write 6–10 bullets. Examples:

- Judge variance / occasional null responses (mitigation: retries, multi-judge).
- Wardrobe boost can override constraint fit (mitigation: conditional boost; wardrobe embeddings).
- Small stable query set (mitigation: expand tagged benchmark).
- Negation handled heuristically (mitigation: contradiction-aware parsing).
- No sizing/fit personalization (mitigation: user profile fields).
- LLM latency/cost (mitigation: caching, cheaper models, early-exit).

---

## 5) Optional user study (lightweight)

If you have time, run a small internal study:

- 2–3 people (classmates, teammates, friends)
- 5 prompts each (include 1 wardrobe prompt if you demo wardrobe)
- rate 1–5: relevance, coherence, “would wear”, perceived personalization

### 5.1 Study format (what to write in a report)

Use a standard structure:

- **Participants**: N participants, brief demographics (e.g., “CS students, 20–30, mixed familiarity with fashion”).
- **Procedure**:
  - Each participant enters 5 queries (or you provide 5 fixed queries).
  - The system returns 3 outfits; participant reviews top‑1 and optionally top‑3.
  - Participant rates 1–5 on the rubric below and leaves free‑text feedback.
- **Measures (1–5 Likert)**:
  - Relevance: “Does it match what you asked for?”
  - Coherence: “Do pieces go together?”
  - Would‑wear: “Would you wear outfit #1?”
  - (Optional) Personalization: “Does it feel like it uses *my* wardrobe?” (only for wardrobe demo)
- **Results**: mean scores + 2–4 representative quotes.
- **Limitations**: small N, non-representative participants, no long‑term usage.

### 5.2 Free‑text feedback prompts (copy/paste)

Ask participants:

- “What did you like about the top outfit?”
- “What was missing or wrong?”
- “What would you change?”
- (Wardrobe) “Did you notice any of your uploaded items being used? Was that good or distracting?”

### 5.3 Example write-up (SIMULATED placeholder)

If you have not run a real user study yet, you can include a clearly labeled placeholder
in drafts. **Do not present simulated feedback as real.**

Below is an example of how this section typically looks in a report.

#### User study (simulated, for format only)

- **Participants**: N=2 simulated participants (“P1”, “P2”).
- **Protocol**: Each participant reviewed the system on a fixed 10‑query set (the same prompts used in qualitative case studies).

**Aggregate ratings (1–5, simulated)**

- Relevance: mean 4.6
- Coherence: mean 4.5
- Would‑wear: mean 4.0
- Personalization (wardrobe): mean 4.2

**Selected feedback quotes (simulated)**

- **P1 on “Warm casual jacket, not sporty”**: “The jacket choice is on point, but I don’t want shorts with it—swap for trousers or jeans. Sneakers are fine but boots would feel warmer.”
- **P2 on “Clean office outfit, not too formal”**: “This feels wearable and not overdressed. I like that it avoids a full suit vibe. I’d prefer less ‘sport’ looking shoes.”
- **P1 on “All black outfit with white sneakers”**: “This matches the ask exactly. The white sneaker constraint is respected and the rest stays neutral.”
- **P2 on “Rainy day outfit with a hooded jacket”**: “Good functional direction. I want clearer emphasis on water‑resistant outerwear; the hood helps.”
- **P1 on “Minimal clean outfit, neutral colors only”**: “The palette is clean. The top‑3 options were a bit similar; I’d like one more adventurous neutral option.”
- **P2 on wardrobe usage (demo_user)**: “It’s cool that my uploaded jacket shows up, but sometimes it overrides what I asked for (e.g., boots). It should prioritize my items only when relevant.”

**Limitations (example text)**

- This simulated subsection is for demonstrating the reporting format only.
- A real study should recruit actual users and collect ratings/feedback under consistent conditions.

If not, explicitly say:

> We did not run a user study due to time; we relied on structured case studies + error taxonomy.

