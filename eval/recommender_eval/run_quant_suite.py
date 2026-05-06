"""Run a quantitative evaluation suite over multiple recommender variants.

Writes a clean folder under:
  eval/recommender_eval/artifacts/suites/<timestamp>/

Primary wardrobe benchmark variants:
  - catalog_sparse
  - catalog_dense
  - wardrobe_sparse
  - wardrobe_dense

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


def _load_queries_txt(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append({"id": f"txt_{index:03d}", "query": line, "tags": []})
    return queries


def _load_queries_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must be a list of query objects.")

    out: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        query_text = str(row.get("query") or "").strip()
        if not query_text:
            continue
        out.append(
            {
                "id": str(row.get("id") or ""),
                "query": query_text,
                "tags": row.get("tags") or [],
                "bucket": str(row.get("bucket") or "").strip() or None,
                "expected_wardrobe_use": str(row.get("expected_wardrobe_use") or "").strip() or None,
                "anchor_category": str(row.get("anchor_category") or "").strip() or None,
                "anchor_color": str(row.get("anchor_color") or "").strip() or None,
            }
        )
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
        role = str(item.get("role") or "").strip()
        if role:
            roles.add(role)
    return roles


def _colors_present(outfit: dict) -> set[str]:
    colors = set()
    for item in _safe_list(outfit.get("items")):
        color = item.get("normalized_color")
        if color:
            colors.add(str(color).lower())
    return colors


def _categories_present(outfit: dict) -> set[str]:
    categories = set()
    for item in _safe_list(outfit.get("items")):
        category = item.get("normalized_category")
        if category:
            categories.add(str(category).lower())
    return categories


def _wardrobe_count(outfit: dict) -> int:
    count = 0
    for item in _safe_list(outfit.get("items")):
        if str(item.get("source_type") or "").lower() == "wardrobe":
            count += 1
    return count


def _compute_metrics_for_query(query_row: dict[str, Any], summary: dict, *, user_id: str | None) -> dict[str, Any]:
    parsed = summary.get("parsed_constraints") or {}
    required_roles = [str(role) for role in _safe_list(parsed.get("required_roles"))]
    preferred_categories = [str(cat).lower() for cat in _safe_list(parsed.get("preferred_categories")) if str(cat).strip()]

    outfits = _safe_list(summary.get("top_outfits"))
    top1 = outfits[0] if outfits else {}

    roles_top1 = _roles_present(top1)
    categories_top1 = _categories_present(top1)
    colors_top1 = _colors_present(top1)

    missing_roles_top1 = [role for role in required_roles if role not in roles_top1]

    anchor_category = str(query_row.get("anchor_category") or "").strip().lower() or None
    anchor_color = str(query_row.get("anchor_color") or "").strip().lower() or None
    expected_wardrobe_use = str(query_row.get("expected_wardrobe_use") or "").strip().lower() or None
    bucket = str(query_row.get("bucket") or "").strip().lower() or None

    category_any_hit_top1 = None
    if anchor_category:
        category_any_hit_top1 = anchor_category in categories_top1
    elif preferred_categories:
        category_any_hit_top1 = any(category in categories_top1 for category in preferred_categories)

    wardrobe_hit_at_1 = None
    if user_id and expected_wardrobe_use == "yes":
        wardrobe_hit_at_1 = _wardrobe_count(top1) > 0

    wardrobe_constraint_override = None
    if user_id and expected_wardrobe_use == "yes" and _wardrobe_count(top1) > 0:
        category_miss = category_any_hit_top1 is False
        color_miss = bool(anchor_color) and (anchor_color not in colors_top1)
        wardrobe_constraint_override = category_miss or color_miss

    negative_control_wardrobe_intrusion = None
    if user_id and expected_wardrobe_use == "no":
        negative_control_wardrobe_intrusion = _wardrobe_count(top1) > 0

    return {
        "id": query_row.get("id"),
        "query": query_row.get("query"),
        "bucket": bucket,
        "expected_wardrobe_use": expected_wardrobe_use,
        "anchor_category": anchor_category,
        "anchor_color": anchor_color,
        "user_id": user_id,
        "required_roles": required_roles,
        "missing_roles_top1": missing_roles_top1,
        "role_complete_top1": len(missing_roles_top1) == 0 if required_roles else None,
        "category_any_hit_top1": category_any_hit_top1,
        "wardrobe_hit_at_1": wardrobe_hit_at_1,
        "wardrobe_constraint_override": wardrobe_constraint_override,
        "negative_control_wardrobe_intrusion": negative_control_wardrobe_intrusion,
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate_bool(key: str, *, bucket: str | None = None) -> float | None:
        values = []
        for row in rows:
            if bucket is not None and row.get("bucket") != bucket:
                continue
            value = row.get(key)
            if value is not None:
                values.append(bool(value))
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    missing_role_counts = [len(row.get("missing_roles_top1") or []) for row in rows]
    return {
        "count": len(rows),
        "role_complete_top1_rate": rate_bool("role_complete_top1"),
        "mean_missing_roles_top1": statistics.mean(missing_role_counts) if missing_role_counts else None,
        "category_any_hit_top1_rate": rate_bool("category_any_hit_top1"),
        "wardrobe_hit_at_1_rate": rate_bool("wardrobe_hit_at_1"),
        "wardrobe_constraint_override_rate": rate_bool("wardrobe_constraint_override"),
        "negative_control_wardrobe_intrusion_rate": rate_bool("negative_control_wardrobe_intrusion"),
        "bucket_counts": {
            "positive": sum(1 for row in rows if row.get("bucket") == "positive"),
            "mix": sum(1 for row in rows if row.get("bucket") == "mix"),
            "negative_control": sum(1 for row in rows if row.get("bucket") == "negative_control"),
        },
        "judge_reporting_scopes": {
            "wardrobe_hit_scope": "positive+mix only",
            "negative_control_scope": "negative_control only",
        },
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
    "catalog_sparse": Variant(
        name="catalog_sparse",
        enable_openai_query_parser=True,
        enable_dense_retrieval_rerank=False,
        enable_openai_combo_composer=False,
        enable_openai_reranker=False,
        user_id=None,
    ),
    "catalog_dense": Variant(
        name="catalog_dense",
        enable_openai_query_parser=True,
        enable_dense_retrieval_rerank=True,
        enable_openai_combo_composer=False,
        enable_openai_reranker=False,
        user_id=None,
    ),
    "wardrobe_sparse": Variant(
        name="wardrobe_sparse",
        enable_openai_query_parser=True,
        enable_dense_retrieval_rerank=False,
        enable_openai_combo_composer=False,
        enable_openai_reranker=False,
        user_id="demo_user",
    ),
    "wardrobe_dense": Variant(
        name="wardrobe_dense",
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

        for query_row in queries:
            query_text = str(query_row["query"])
            t0 = time.time()
            api_result = build_outfits(query_text, user_id=variant.user_id)
            elapsed_ms = int((time.time() - t0) * 1000)
            timings.append(elapsed_ms)

            summary = _summarize_api_result(api_result)
            items.append(
                {
                    "id": query_row.get("id"),
                    "query": query_text,
                    "tags": query_row.get("tags") or [],
                    "bucket": query_row.get("bucket"),
                    "expected_wardrobe_use": query_row.get("expected_wardrobe_use"),
                    "anchor_category": query_row.get("anchor_category"),
                    "anchor_color": query_row.get("anchor_color"),
                    "elapsed_ms": elapsed_ms,
                    "result": summary,
                }
            )
            metrics_rows.append(_compute_metrics_for_query(query_row, summary, user_id=variant.user_id))

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
                "id": item.get("id"),
                "query": query,
                "bucket": item.get("bucket"),
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
    return f"{value * 100:.1f}%"


def _format_delta(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "n/a"
    delta = current - previous
    if abs(delta) >= 1:
        return f"{delta:+.2f}"
    return f"{delta:+.1%}"


def _write_suite_report(suite_dir: Path, suite_payload: dict[str, Any]) -> None:
    lines = []
    lines.append(f"# Quantitative suite report ({suite_payload['suite_id']})")
    lines.append("")
    lines.append(f"- Query set: `{suite_payload['query_set']}`")
    lines.append(f"- Variants: {', '.join(suite_payload['variants'])}")
    lines.append("")

    lines.append("## Primary metrics (summary)")
    lines.append("")
    lines.append(
        "| Variant | Role complete@1 | Category hit@1 | Wardrobe hit@1 | Wardrobe override | Neg-control intrusion |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant_name in suite_payload["variants"]:
        metrics = (suite_payload["results"].get(variant_name, {}) or {}).get("metrics", {}) or {}
        summary = metrics.get("summary", {}) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    variant_name,
                    _format_pct(summary.get("role_complete_top1_rate")),
                    _format_pct(summary.get("category_any_hit_top1_rate")),
                    _format_pct(summary.get("wardrobe_hit_at_1_rate")),
                    _format_pct(summary.get("wardrobe_constraint_override_rate")),
                    _format_pct(summary.get("negative_control_wardrobe_intrusion_rate")),
                ]
            )
            + " |"
        )
    lines.append("")

    if suite_payload.get("judge_enabled"):
        lines.append("## Judge overall")
        lines.append("")
        lines.append("| Variant | Mean overall | Mean relevance | Mean constraint_fit | Mean coherence |")
        lines.append("|---|---:|---:|---:|---:|")
        for variant_name in suite_payload["variants"]:
            judge = (suite_payload["results"].get(variant_name, {}) or {}).get("judge", {}) or {}
            if not judge.get("ok"):
                lines.append(f"| {variant_name} | n/a | n/a | n/a | n/a |")
                continue
            judge_summary = judge.get("summary", {}) or {}
            sub = judge_summary.get("mean_subscores", {}) or {}
            lines.append(
                f"| {variant_name} | {judge_summary.get('mean_overall', 'n/a')} | {sub.get('relevance', 'n/a')} | {sub.get('constraint_fit', 'n/a')} | {sub.get('coherence', 'n/a')} |"
            )
        lines.append("")

    if "wardrobe_sparse" in suite_payload["variants"] and "wardrobe_dense" in suite_payload["variants"]:
        sparse_metrics = suite_payload["results"]["wardrobe_sparse"]["metrics"]["summary"]
        dense_metrics = suite_payload["results"]["wardrobe_dense"]["metrics"]["summary"]
        lines.append("## Dense lift (`wardrobe_dense - wardrobe_sparse`)")
        lines.append("")
        lines.append("| Metric | Dense lift |")
        lines.append("|---|---:|")
        lines.append(
            f"| Wardrobe hit@1 | {_format_delta(dense_metrics.get('wardrobe_hit_at_1_rate'), sparse_metrics.get('wardrobe_hit_at_1_rate'))} |"
        )
        lines.append(
            f"| Wardrobe override rate | {_format_delta(dense_metrics.get('wardrobe_constraint_override_rate'), sparse_metrics.get('wardrobe_constraint_override_rate'))} |"
        )
        lines.append(
            f"| Neg-control intrusion | {_format_delta(dense_metrics.get('negative_control_wardrobe_intrusion_rate'), sparse_metrics.get('negative_control_wardrobe_intrusion_rate'))} |"
        )
        if suite_payload.get("judge_enabled"):
            sparse_judge = (suite_payload["results"]["wardrobe_sparse"].get("judge") or {}).get("summary", {}) or {}
            dense_judge = (suite_payload["results"]["wardrobe_dense"].get("judge") or {}).get("summary", {}) or {}
            lines.append(
                f"| Judge overall | {_format_delta(dense_judge.get('mean_overall'), sparse_judge.get('mean_overall'))} |"
            )
        lines.append("")

    (suite_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-set",
        type=str,
        choices=["queries_eval_10_mens", "benchmark_queries", "wardrobe_eval_demo_user"],
        default="wardrobe_eval_demo_user",
        help="Which query set to run.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="catalog_sparse,catalog_dense,wardrobe_sparse,wardrobe_dense",
        help="Comma-separated variant names.",
    )
    parser.add_argument("--judge", type=str, choices=["true", "false"], default="false", help="Run LLM judge.")
    args = parser.parse_args()

    if args.query_set == "queries_eval_10_mens":
        queries = _load_queries_txt(EVAL_DIR / "queries_eval_10_mens.txt")
        query_set_label = "eval/recommender_eval/queries_eval_10_mens.txt"
    elif args.query_set == "benchmark_queries":
        queries = _load_queries_json(EVAL_DIR / "benchmark_queries.json")
        query_set_label = "eval/recommender_eval/benchmark_queries.json"
    else:
        queries = _load_queries_json(EVAL_DIR / "queries_wardrobe_eval_demo_user.json")
        query_set_label = "eval/recommender_eval/queries_wardrobe_eval_demo_user.json"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_id = f"suite_{stamp}"
    suite_dir = EVAL_DIR / "artifacts" / "suites" / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    variant_names = [name.strip() for name in str(args.variants).split(",") if name.strip()]
    results: dict[str, Any] = {}

    for variant_name in variant_names:
        if variant_name not in VARIANTS:
            raise SystemExit(f"Unknown variant: {variant_name}. Known: {', '.join(sorted(VARIANTS))}")
        variant = VARIANTS[variant_name]

        variant_dir = suite_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        run_payload = _run_variant(variant, queries)
        (variant_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        (variant_dir / "metrics.json").write_text(json.dumps(run_payload["metrics"]["rows"], indent=2), encoding="utf-8")
        (variant_dir / "metrics_summary.json").write_text(
            json.dumps(run_payload["metrics"]["summary"], indent=2), encoding="utf-8"
        )

        judge_payload = None
        if args.judge == "true":
            judge_payload = _judge_variant(run_payload)
            (variant_dir / "judge.json").write_text(json.dumps(judge_payload, indent=2), encoding="utf-8")

        results[variant_name] = {
            "run_path": str((variant_dir / "run.json").as_posix()),
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
