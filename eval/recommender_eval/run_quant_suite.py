"""Run a quantitative evaluation suite over multiple system variants.

Writes a clean folder under:
  eval/recommender_eval/artifacts/suites/<timestamp>/

Variants (minimal ablation):
  - baseline_det
  - parser_only
  - parser_plus_dense
  - wardrobe_on (parser_plus_dense + user_id=demo_user)

This script produces:
  - run.json (raw outputs)
  - metrics.json + metrics_summary.json (deterministic metrics)
  - judge.json (optional; LLM judge)
  - report.md (suite-level summary)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.openai_client import llm_score_retrieval, openai_is_configured
from src.recommender.outfits import build_outfits
from src.shared.config import settings


def _load_queries_txt(path: Path) -> list[str]:
    queries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    return queries


def _load_queries_benchmark_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("benchmark_queries.json must be a list of {id, query, tags}.")
    out: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        q = str(row.get("query") or "").strip()
        if not q:
            continue
        out.append({"id": row.get("id"), "query": q, "tags": row.get("tags") or []})
    return out


def _summarize_api_result(result: dict) -> dict:
    outfits = result.get("outfits", []) or []
    parsed = result.get("parsed_constraints") or {}
    return {
        "llm_status": result.get("llm_status"),
        "parsed_constraints": parsed,
        "candidate_pools": result.get("candidate_pools", {}) or {},
        "top_outfits": [
            {
                "score": outfit.get("score"),
                "signature": " + ".join(str(item.get("normalized_category")) for item in outfit.get("items", [])),
                "colors": [item.get("normalized_color") for item in outfit.get("items", [])],
                "items": [
                    {
                        "item_id": item.get("item_id"),
                        "source_type": item.get("source_type"),
                        "display_name": item.get("display_name"),
                        "role": item.get("recommendation_role"),
                        "normalized_category": item.get("normalized_category"),
                        "normalized_color": item.get("normalized_color"),
                        "section_theme": item.get("section_theme"),
                        "score": item.get("score"),
                    }
                    for item in outfit.get("items", [])
                ],
            }
            for outfit in outfits
        ],
    }


def _safe_list(x) -> list:
    return x if isinstance(x, list) else []


def _roles_present(outfit: dict) -> set[str]:
    roles = set()
    for item in _safe_list(outfit.get("items")):
        r = str(item.get("role") or "").strip()
        if r:
            roles.add(r)
    return roles


def _colors_present(outfit: dict) -> list[str]:
    colors: list[str] = []
    for item in _safe_list(outfit.get("items")):
        c = item.get("normalized_color")
        if c:
            colors.append(str(c).lower())
    return colors


def _categories_present(outfit: dict) -> set[str]:
    cats = set()
    for item in _safe_list(outfit.get("items")):
        c = item.get("normalized_category")
        if c:
            cats.add(str(c).lower())
    return cats


def _wardrobe_count(outfit: dict) -> int:
    count = 0
    for item in _safe_list(outfit.get("items")):
        if str(item.get("source_type") or "").lower() == "wardrobe":
            count += 1
    return count


def _shared_item_ids(outfits: list[dict]) -> int:
    seen = set()
    dupes = 0
    for outfit in outfits:
        for item in _safe_list(outfit.get("items")):
            item_id = item.get("item_id")
            if item_id is None:
                continue
            item_id = str(item_id)
            if item_id in seen:
                dupes += 1
            else:
                seen.add(item_id)
    return dupes


def _compute_metrics_for_query(query: str, summary: dict, *, user_id: str | None) -> dict[str, Any]:
    parsed = summary.get("parsed_constraints") or {}
    required_roles = [str(r) for r in _safe_list(parsed.get("required_roles"))]
    preferred_colors = [str(c).lower() for c in _safe_list(parsed.get("preferred_colors")) if str(c).strip()]
    preferred_categories = [str(c).lower() for c in _safe_list(parsed.get("preferred_categories")) if str(c).strip()]

    outfits = _safe_list(summary.get("top_outfits"))
    top1 = outfits[0] if outfits else {}
    top3 = outfits[:3] if outfits else []

    roles_top1 = _roles_present(top1)
    missing_roles_top1 = [r for r in required_roles if r not in roles_top1]

    colors_top1 = _colors_present(top1)
    categories_top1 = _categories_present(top1)

    color_any_hit_top1 = None
    color_majority_hit_top1 = None
    if preferred_colors:
        color_any_hit_top1 = any(c in preferred_colors for c in colors_top1)
        if colors_top1:
            hits = sum(1 for c in colors_top1 if c in preferred_colors)
            color_majority_hit_top1 = hits >= ((len(colors_top1) // 2) + 1)
        else:
            color_majority_hit_top1 = False

    category_any_hit_top1 = None
    if preferred_categories:
        category_any_hit_top1 = any(c in categories_top1 for c in preferred_categories)

    signatures = [str(o.get("signature") or "") for o in top3]
    duplicate_signature_in_top3 = len(set(signatures)) != len(signatures) if signatures else False
    shared_item_ids_in_top3 = _shared_item_ids(top3) > 0

    wardrobe_hit_at_1 = None
    wardrobe_hit_at_3 = None
    wardrobe_items_per_outfit_top3_mean = None
    if user_id:
        wardrobe_hit_at_1 = _wardrobe_count(top1) > 0
        wardrobe_hit_at_3 = any(_wardrobe_count(o) > 0 for o in top3)
        if top3:
            wardrobe_items_per_outfit_top3_mean = statistics.mean([_wardrobe_count(o) for o in top3])

    return {
        "query": query,
        "user_id": user_id,
        "required_roles": required_roles,
        "missing_roles_top1": missing_roles_top1,
        "role_complete_top1": len(missing_roles_top1) == 0 if required_roles else None,
        "preferred_colors": preferred_colors,
        "color_any_hit_top1": color_any_hit_top1,
        "color_majority_hit_top1": color_majority_hit_top1,
        "preferred_categories": preferred_categories,
        "category_any_hit_top1": category_any_hit_top1,
        "top3_duplicate_signature": duplicate_signature_in_top3,
        "top3_shared_item_ids": shared_item_ids_in_top3,
        "wardrobe_hit_at_1": wardrobe_hit_at_1,
        "wardrobe_hit_at_3": wardrobe_hit_at_3,
        "wardrobe_items_per_outfit_top3_mean": wardrobe_items_per_outfit_top3_mean,
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate_bool(key: str) -> float | None:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return sum(1 for v in vals if bool(v)) / len(vals)

    missing_role_counts = [len(r.get("missing_roles_top1") or []) for r in rows]
    return {
        "count": len(rows),
        "role_complete_top1_rate": rate_bool("role_complete_top1"),
        "mean_missing_roles_top1": statistics.mean(missing_role_counts) if missing_role_counts else None,
        "color_any_hit_top1_rate": rate_bool("color_any_hit_top1"),
        "color_majority_hit_top1_rate": rate_bool("color_majority_hit_top1"),
        "category_any_hit_top1_rate": rate_bool("category_any_hit_top1"),
        "top3_duplicate_signature_rate": rate_bool("top3_duplicate_signature"),
        "top3_shared_item_ids_rate": rate_bool("top3_shared_item_ids"),
        "wardrobe_hit_at_1_rate": rate_bool("wardrobe_hit_at_1"),
        "wardrobe_hit_at_3_rate": rate_bool("wardrobe_hit_at_3"),
        "wardrobe_items_per_outfit_top3_mean": (
            statistics.mean(
                [r["wardrobe_items_per_outfit_top3_mean"] for r in rows if r.get("wardrobe_items_per_outfit_top3_mean") is not None]
            )
            if any(r.get("wardrobe_items_per_outfit_top3_mean") is not None for r in rows)
            else None
        ),
    }


@dataclass(frozen=True)
class Variant:
    name: str
    enable_openai_query_parser: bool
    enable_dense_retrieval_rerank: bool
    enable_openai_combo_composer: bool
    enable_openai_reranker: bool
    user_id: str | None


VARIANTS: dict[str, Variant] = {
    "baseline_det": Variant(
        name="baseline_det",
        enable_openai_query_parser=False,
        enable_dense_retrieval_rerank=False,
        enable_openai_combo_composer=False,
        enable_openai_reranker=False,
        user_id=None,
    ),
    "parser_only": Variant(
        name="parser_only",
        enable_openai_query_parser=True,
        enable_dense_retrieval_rerank=False,
        enable_openai_combo_composer=False,
        enable_openai_reranker=False,
        user_id=None,
    ),
    "parser_plus_dense": Variant(
        name="parser_plus_dense",
        enable_openai_query_parser=True,
        enable_dense_retrieval_rerank=True,
        enable_openai_combo_composer=False,
        enable_openai_reranker=False,
        user_id=None,
    ),
    "wardrobe_on": Variant(
        name="wardrobe_on",
        enable_openai_query_parser=True,
        enable_dense_retrieval_rerank=True,
        enable_openai_combo_composer=False,
        enable_openai_reranker=False,
        user_id="demo_user",
    ),
}


def _run_variant(variant: Variant, queries: list[dict[str, Any]]) -> dict[str, Any]:
    old_parser = settings.enable_openai_query_parser
    old_dense = settings.enable_dense_retrieval_rerank
    old_combo = settings.enable_openai_combo_composer
    old_reranker = settings.enable_openai_reranker

    settings.enable_openai_query_parser = variant.enable_openai_query_parser
    settings.enable_dense_retrieval_rerank = variant.enable_dense_retrieval_rerank
    settings.enable_openai_combo_composer = variant.enable_openai_combo_composer
    settings.enable_openai_reranker = variant.enable_openai_reranker

    try:
        items = []
        metrics_rows = []
        timings = []

        for q in queries:
            query_text = str(q["query"])
            t0 = time.time()
            api_result = build_outfits(query_text, user_id=variant.user_id)
            elapsed_ms = int((time.time() - t0) * 1000)
            timings.append(elapsed_ms)

            summary = _summarize_api_result(api_result)
            items.append(
                {
                    "id": q.get("id"),
                    "query": query_text,
                    "tags": q.get("tags") or [],
                    "elapsed_ms": elapsed_ms,
                    "result": summary,
                }
            )
            metrics_rows.append(_compute_metrics_for_query(query_text, summary, user_id=variant.user_id))

        return {
            "variant": variant.name,
            "config_snapshot": {
                "enable_openai_query_parser": settings.enable_openai_query_parser,
                "enable_dense_retrieval_rerank": settings.enable_dense_retrieval_rerank,
                "enable_openai_combo_composer": settings.enable_openai_combo_composer,
                "enable_openai_reranker": settings.enable_openai_reranker,
                "openai_model_query_parser": settings.openai_model_query_parser,
                "openai_model_judge": settings.openai_model_judge,
                "openai_embedding_model": settings.openai_embedding_model,
                "user_id": variant.user_id,
            },
            "timing_summary_ms": {
                "mean": statistics.mean(timings) if timings else None,
                "p95": (sorted(timings)[int(0.95 * (len(timings) - 1))] if timings else None),
                "max": max(timings) if timings else None,
            },
            "items": items,
            "metrics": {
                "rows": metrics_rows,
                "summary": _aggregate_metrics(metrics_rows),
            },
        }
    finally:
        settings.enable_openai_query_parser = old_parser
        settings.enable_dense_retrieval_rerank = old_dense
        settings.enable_openai_combo_composer = old_combo
        settings.enable_openai_reranker = old_reranker


def _judge_variant(run_payload: dict[str, Any]) -> dict[str, Any]:
    if not openai_is_configured():
        return {"ok": False, "error": "OpenAI is not configured (OPENAI_API_KEY missing)."}

    judged_items = []
    sum_overall = 0
    sum_relevance = 0
    sum_constraint_fit = 0
    sum_coherence = 0
    scored_count = 0

    for item in run_payload.get("items", []):
        query = item.get("query", "")
        result_obj = item.get("result", {}) or {}

        judge_in = {
            "parsed_constraints": result_obj.get("parsed_constraints", {}),
            "candidate_pools": result_obj.get("candidate_pools", {}),
            "top_outfits": result_obj.get("top_outfits", []),
            "llm_status": result_obj.get("llm_status"),
        }

        judge_result = llm_score_retrieval(user_query=query, retrieval_output=judge_in)

        derived = {"overall": None, "relevance": None, "constraint_fit": None, "coherence": None}
        if isinstance(judge_result, dict):
            scores = judge_result.get("scores") or {}
            if isinstance(scores, dict):
                try:
                    derived["relevance"] = int(scores.get("relevance")) if scores.get("relevance") is not None else None
                    derived["constraint_fit"] = (
                        int(scores.get("constraint_fit")) if scores.get("constraint_fit") is not None else None
                    )
                    derived["coherence"] = int(scores.get("coherence")) if scores.get("coherence") is not None else None
                    derived["overall"] = int(scores.get("overall")) if scores.get("overall") is not None else None
                except Exception:
                    derived = {"overall": None, "relevance": None, "constraint_fit": None, "coherence": None}

        if derived["overall"] is not None:
            scored_count += 1
            sum_overall += derived["overall"]
            sum_relevance += derived["relevance"] or 0
            sum_constraint_fit += derived["constraint_fit"] or 0
            sum_coherence += derived["coherence"] or 0

        judged_items.append(
            {
                "query": query,
                "judge_result": judge_result,
                "judge_status": "ok" if judge_result is not None else "null_response",
                "derived": derived,
            }
        )

    summary = {
        "count": len(run_payload.get("items", [])),
        "scored_count": scored_count,
        "mean_overall": (sum_overall / scored_count) if scored_count else None,
        "mean_subscores": {
            "relevance": (sum_relevance / scored_count) if scored_count else None,
            "constraint_fit": (sum_constraint_fit / scored_count) if scored_count else None,
            "coherence": (sum_coherence / scored_count) if scored_count else None,
        },
    }

    return {"ok": True, "summary": summary, "items": judged_items}


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value*100:.1f}%"


def _write_suite_report(suite_dir: Path, suite_payload: dict[str, Any]) -> None:
    lines = []
    lines.append(f"# Quantitative suite report ({suite_payload['suite_id']})")
    lines.append("")
    lines.append(f"- Query set: `{suite_payload['query_set']}`")
    lines.append(f"- Variants: {', '.join(suite_payload['variants'])}")
    lines.append("")

    lines.append("## Deterministic metrics (summary)")
    lines.append("")
    lines.append("| Variant | Role complete@1 | Color any@1 | Category any@1 | Dup signature@3 | Shared item@3 | Wardrobe hit@1 | Wardrobe hit@3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in suite_payload["variants"]:
        metrics = (suite_payload["results"].get(v, {}) or {}).get("metrics", {}) or {}
        s = metrics.get("summary", {}) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    v,
                    _format_pct(s.get("role_complete_top1_rate")),
                    _format_pct(s.get("color_any_hit_top1_rate")),
                    _format_pct(s.get("category_any_hit_top1_rate")),
                    _format_pct(s.get("top3_duplicate_signature_rate")),
                    _format_pct(s.get("top3_shared_item_ids_rate")),
                    _format_pct(s.get("wardrobe_hit_at_1_rate")),
                    _format_pct(s.get("wardrobe_hit_at_3_rate")),
                ]
            )
            + " |"
        )
    lines.append("")

    if suite_payload.get("judge_enabled"):
        lines.append("## LLM judge (mean overall)")
        lines.append("")
        lines.append("| Variant | Mean overall | Mean relevance | Mean constraint_fit | Mean coherence |")
        lines.append("|---|---:|---:|---:|---:|")
        for v in suite_payload["variants"]:
            judge = (suite_payload["results"].get(v, {}) or {}).get("judge", {}) or {}
            if not judge.get("ok"):
                lines.append(f"| {v} | n/a | n/a | n/a | n/a |")
                continue
            js = judge.get("summary", {}) or {}
            subs = js.get("mean_subscores", {}) or {}
            lines.append(
                f"| {v} | {js.get('mean_overall', 'n/a')} | {subs.get('relevance','n/a')} | {subs.get('constraint_fit','n/a')} | {subs.get('coherence','n/a')} |"
            )
        lines.append("")

    (suite_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-set",
        type=str,
        choices=["queries_eval_10_mens", "benchmark_queries"],
        default="queries_eval_10_mens",
        help="Which query set to run.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline_det,parser_only,parser_plus_dense,wardrobe_on",
        help="Comma-separated variant names.",
    )
    parser.add_argument("--judge", type=str, choices=["true", "false"], default="false", help="Run LLM judge.")
    args = parser.parse_args()

    if args.query_set == "queries_eval_10_mens":
        queries = [{"id": None, "query": q, "tags": []} for q in _load_queries_txt(EVAL_DIR / "queries_eval_10_mens.txt")]
        query_set_label = "eval/recommender_eval/queries_eval_10_mens.txt"
    else:
        queries = _load_queries_benchmark_json(EVAL_DIR / "benchmark_queries.json")
        query_set_label = "eval/recommender_eval/benchmark_queries.json"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_id = f"suite_{stamp}"
    suite_dir = EVAL_DIR / "artifacts" / "suites" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    variant_names = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    results: dict[str, Any] = {}

    for vname in variant_names:
        if vname not in VARIANTS:
            raise SystemExit(f"Unknown variant: {vname}. Known: {', '.join(sorted(VARIANTS))}")
        variant = VARIANTS[vname]

        vdir = suite_dir / vname
        vdir.mkdir(parents=True, exist_ok=True)

        run_payload = _run_variant(variant, queries)
        (vdir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        (vdir / "metrics.json").write_text(json.dumps(run_payload["metrics"]["rows"], indent=2), encoding="utf-8")
        (vdir / "metrics_summary.json").write_text(json.dumps(run_payload["metrics"]["summary"], indent=2), encoding="utf-8")

        judge_payload = None
        if args.judge == "true":
            judge_payload = _judge_variant(run_payload)
            (vdir / "judge.json").write_text(json.dumps(judge_payload, indent=2), encoding="utf-8")

        results[vname] = {
            "run_path": str((vdir / "run.json").as_posix()),
            "metrics": run_payload["metrics"],
            "judge": judge_payload,
        }

    suite_payload = {
        "suite_id": suite_id,
        "query_set": query_set_label,
        "variants": variant_names,
        "judge_enabled": args.judge == "true",
        "results": results,
    }
    (suite_dir / "suite.json").write_text(json.dumps(suite_payload, indent=2), encoding="utf-8")
    _write_suite_report(suite_dir, suite_payload)

    print(f"Wrote suite to {suite_dir}")
    print(f"- report: {suite_dir / 'report.md'}")


if __name__ == "__main__":
    main()

