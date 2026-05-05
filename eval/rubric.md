# Recommender Evaluation Rubric

Use this rubric to compare two system versions on the same benchmark queries.

## How To Use

1. Run the benchmark script to generate outputs for all queries.
2. Compare the new outputs against the previous baseline.
3. For each query, score the top outfit and the top-3 set.
4. Record notes about obvious wins, regressions, and repetitive behavior.

## Scoring Scale

Use `1` to `5` for each category:

- `1` = very poor
- `2` = weak
- `3` = acceptable
- `4` = good
- `5` = excellent

## Categories

### Query Match
Does the top outfit actually match what the user asked for?

Look for:
- correct target group
- correct vibe like casual, work, dinner
- obvious requested category hints like blazer, jacket, sneakers

### Role Completeness
Does the returned top outfit include all needed clothing roles?

For the current MVP that usually means:
- `top`
- `bottom`
- `shoes`
- optional `outerwear` when the query suggests it

### Color Fit
Does the outfit follow the requested color direction?

Look for:
- requested colors appearing in the outfit
- neutral requests producing a restrained palette
- no obvious mismatches with explicit color asks

### Style / Formality Fit
Does the outfit feel casual, smart casual, business, or sporty in a believable way?

This is especially important for:
- work
- office
- dinner
- polished
- sporty

### Outfit Coherence
Do the pieces feel like they belong together?

Look for:
- categories that make sense together
- no obviously awkward combinations
- reasonable overall palette

### Top-3 Diversity
Are the top 3 outfits meaningfully different from each other?

Low diversity means:
- same top with nearly identical bottoms
- same full look repeated in different item ids

High diversity means:
- different but still relevant outfit options

## Suggested Review Template

For each query, write:

```text
Query ID:
Winning system:
Query match: /5
Role completeness: /5
Color fit: /5
Style/formality fit: /5
Outfit coherence: /5
Top-3 diversity: /5
Notes:
```

## Automatic Checks To Track

Alongside human scoring, keep an eye on:
- missing required roles
- wrong target group
- requested colors not represented
- repetitive top-3 outputs
- noisy categories reappearing
