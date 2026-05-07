import json
from pathlib import Path
import sys

from openai import OpenAI

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qa.scripts import query_answer

top_k = 5
judge_model = "gpt-5.4-mini"
# query_answer.retrieve supports: "mix", "global"
retrieval_strategy = "global"  # options: mix, global

test_cases = [
    {
        "question": "What are the biggest seasonal fashion trends for spring and summer 2026?",
        "expected_scopes": ["seasonal_trends_2026"],
    },
    {
        "question": "Can you give me easy outfit formulas I can repeat for work and weekends?",
        "expected_scopes": ["outfit_formulas_and_styling_recipes"],
    },
    {
        "question": "How should I style wide-leg jeans in 2026 without looking sloppy?",
        "expected_scopes": ["denim_and_pants_trends", "garment_specific_styling"],
    },
    {
        "question": "Which shoe trends are replacing last year's styles in 2026?",
        "expected_scopes": ["footwear_and_shoe_trends"],
    },
    {
        "question": "What bag and jewelry trends are worth buying this year?",
        "expected_scopes": ["accessories_bags_and_jewelry"],
    },
    {
        "question": "Which colors are trending in fashion for summer 2026?",
        "expected_scopes": ["color_trends_and_color_theory"],
    },
    {
        "question": "What fabrics and textures are defining 2026 fashion?",
        "expected_scopes": ["materials_textures_and_fabric_trends"],
    },
    {
        "question": "What aesthetics are shaping culture and fashion right now?",
        "expected_scopes": ["aesthetic_and_cultural_trends"],
    },
    {
        "question": "What should I wear to look polished at the office in 2026?",
        "expected_scopes": ["workwear_and_office_style"],
    },
    {
        "question": "What are good outfit ideas for vacation and city walking in warm weather?",
        "expected_scopes": ["occasion_and_lifestyle_dressing"],
    },
    {
        "question": "What does sporty luxury look like in 2026 activewear?",
        "expected_scopes": ["activewear_athleisure_and_sporty_style"],
    },
    {
        "question": "What are celebrities and street style editors wearing this season?",
        "expected_scopes": ["celebrity_editor_and_street_style"],
    },
    {
        "question": "How do I build a capsule wardrobe for 2026?",
        "expected_scopes": ["wardrobe_staples_capsule_and_shopping_guides"],
    },
    {
        "question": "How is sustainability and resale changing fashion buying behavior?",
        "expected_scopes": ["sustainability_resale_and_industry_reports"],
    },
    {
        "question": "What is the 2026 fashion retail outlook and consumer trend?",
        "expected_scopes": ["fashion_business_market_and_consumer_behavior"],
    },
    {
        "question": "How can I style a white shirt and jeans in updated ways this year?",
        "expected_scopes": ["outfit_formulas_and_styling_recipes", "garment_specific_styling"],
    },
    {
        "question": "What are current blazer styling ideas for spring workwear?",
        "expected_scopes": ["workwear_and_office_style", "garment_specific_styling"],
    },
    {
        "question": "What are the most wearable shoe trends for commuting and office days?",
        "expected_scopes": ["footwear_and_shoe_trends", "workwear_and_office_style"],
    },
    {
        "question": "How are vintage and birth-year bags trending right now?",
        "expected_scopes": ["accessories_bags_and_jewelry", "sustainability_resale_and_industry_reports"],
    },
    {
        "question": "Is matcha green still trending and how should I style it?",
        "expected_scopes": ["color_trends_and_color_theory", "outfit_formulas_and_styling_recipes"],
    },
    {
        "question": "What denim silhouettes are strongest for spring 2026?",
        "expected_scopes": ["denim_and_pants_trends"],
    },
    {
        "question": "Which wardrobe staples are worth buying once and wearing repeatedly?",
        "expected_scopes": ["wardrobe_staples_capsule_and_shopping_guides"],
    },
    {
        "question": "What seasonal outfit trends overlap with office style this year?",
        "expected_scopes": ["seasonal_trends_2026", "workwear_and_office_style"],
    },
    {
        "question": "How are texture trends like suede and lace showing up in everyday outfits?",
        "expected_scopes": ["materials_textures_and_fabric_trends", "outfit_formulas_and_styling_recipes"],
    },
    {
        "question": "I need one look that works from office to dinner in rainy weather. What should I wear?",
        "expected_scopes": ["workwear_and_office_style", "occasion_and_lifestyle_dressing"],
    },
    {
        "question": "I avoid bright colors. How can I still look on-trend in 2026?",
        "expected_scopes": ["color_trends_and_color_theory", "outfit_formulas_and_styling_recipes"],
    },
    {
        "question": "I only wear black outfits. What 2026 updates can I adopt without changing my palette?",
        "expected_scopes": ["seasonal_trends_2026", "outfit_formulas_and_styling_recipes"],
    },
    {
        "question": "What should I wear courtside if I want stylish but not gym-like?",
        "expected_scopes": ["occasion_and_lifestyle_dressing", "aesthetic_and_cultural_trends"],
    },
    {
        "question": "What makes a strong activewear line launch in 2026?",
        "expected_scopes": [
            "activewear_athleisure_and_sporty_style",
            "fashion_business_market_and_consumer_behavior",
        ],
    },
    {
        "question": "Which 2026 trends are practical for real life, not just runway hype?",
        "expected_scopes": ["seasonal_trends_2026", "wardrobe_staples_capsule_and_shopping_guides"],
    },
]


def build_db_path() -> str:
    db_path = Path(query_answer.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(
            f"QA Chroma index not found at: {db_path}\n"
            "Build it first with: python scripts/qa_build_db.py"
        )
    return str(db_path)


def score_scope_recall_at_k(contexts: list[dict], expected_scopes: list[str]) -> float:
    expected_set = set(expected_scopes)
    retrieved_set = {item.get("scope", "") for item in contexts if item.get("scope", "")}
    hit_set = expected_set.intersection(retrieved_set)
    return len(hit_set) / len(expected_set)


def score_scope_precision_at_k(contexts: list[dict], expected_scopes: list[str]) -> float:
    expected_set = set(expected_scopes)
    retrieved_scopes = [item.get("scope", "") for item in contexts if item.get("scope", "")]
    if not retrieved_scopes:
        return 0.0
    hits = sum(1 for scope in retrieved_scopes if scope in expected_set)
    return hits / len(retrieved_scopes)


def score_scope_rr(contexts: list[dict], expected_scopes: list[str]) -> float:
    expected_set = set(expected_scopes)
    for rank, item in enumerate(contexts, start=1):
        if item.get("scope", "") in expected_set:
            return 1.0 / rank
    return 0.0


def judge_support_score(question: str, contexts: list[dict], answer: str, client: OpenAI) -> int:
    context_lines = []
    for idx, item in enumerate(contexts, start=1):
        context_lines.append(f"[source {idx}]")
        context_lines.append(f"title: {item.get('title', '')}")
        context_lines.append(f"url: {item.get('url', '')}")
        context_lines.append(f"content: {item.get('text', '')}")
    context_block = "\n".join(context_lines)

    judge_prompt = f"""
You are an evaluator for fashion question answering grounding.

Task:
Score whether the answer is supported by the retrieved context only.

Fashion-specific guidance:
- Focus on material claims: trend directions, brand examples, concrete items, dates, and numbers.
- Allow light paraphrasing and high-level generic phrasing if it does not add new specific facts.
- Do not penalize harmless style/wording choices that are not factual claims.
- Penalize invented specific facts (new brands, numbers, dates, events) not supported by context.

Scoring:
- 0: mostly unsupported, or contains major invented/contradictory claims.
- 1: partially supported; includes some unsupported specific claims.
- 2: strongly supported overall; core and most details are grounded, with no major hallucination.

Return json only:
{{"score": 0}}

Question:
{question}

Retrieved context:
{context_block}

Answer:
{answer}
""".strip()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": "You are a fair and evidence-focused judge."},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content.strip())
    return int(parsed["score"])


def evaluate_case(
    case_id: int,
    question: str,
    expected_scopes: list[str],
    db_path: str,
    client: OpenAI,
    retrieval_strategy: str,
) -> dict:
    contexts, scope_decision = query_answer.retrieve(
        question=question,
        top_k=top_k,
        db_path=db_path,
        return_scope_decision=True,
        retrieval_strategy=retrieval_strategy,
    )
    prompt = query_answer.llm_prompt(
        question=question,
        contexts=contexts,
        detected_scopes=scope_decision.get("top_scopes", []),
    )
    answer = query_answer.generate_answer(prompt, source_count=len(contexts))
    judge_score = judge_support_score(
        question=question,
        contexts=contexts,
        answer=answer,
        client=client,
    )
    recall_at_k = score_scope_recall_at_k(
        contexts=contexts,
        expected_scopes=expected_scopes,
    )
    precision_at_k = score_scope_precision_at_k(
        contexts=contexts,
        expected_scopes=expected_scopes,
    )
    mrr = score_scope_rr(
        contexts=contexts,
        expected_scopes=expected_scopes,
    )
    mode = scope_decision.get("retrieval_mode", "")

    return {
        "case_id": case_id,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "judge_score": judge_score,
        "mode": mode,
        "retrieved_count": len(contexts),
    }


def summarize_results(rows: list[dict]) -> dict:
    n = len(rows)
    avg_recall_at_k = sum(r["recall_at_k"] for r in rows) / n
    avg_precision_at_k = sum(r["precision_at_k"] for r in rows) / n
    avg_mrr = sum(r["mrr"] for r in rows) / n
    avg_judge_score = sum(r["judge_score"] for r in rows) / n
    return {
        "number_of_cases": n,
        "top_k": top_k,
        "retrieval_strategy": retrieval_strategy,
        "average_recall_at_k": avg_recall_at_k,
        "average_precision_at_k": avg_precision_at_k,
        "mrr": avg_mrr,
        "faithfulness": avg_judge_score,
    }


def print_summary_table(summary: dict) -> None:
    print("\nsummary")
    print(f"| number_of_cases      | {summary['number_of_cases']:<7} |")
    print(f"| top_k                | {summary['top_k']:<7} |")
    print(f"| retrieval_strategy   | {summary['retrieval_strategy']:<7} |")
    print(f"| average_recall_at_k  | {summary['average_recall_at_k']:<7.3f} |")
    print(f"| average_precision_at_k | {summary['average_precision_at_k']:<7.3f} |")
    print(f"| average_mrr          | {summary['mrr']:<7.3f} |")
    print(f"| average_judge_score  | {summary['faithfulness']:<7.3f} |")


def main() -> None:
    db_path = build_db_path()
    client = OpenAI()
    rows = []

    for case_id, case in enumerate(test_cases, start=1):
        row = evaluate_case(
            case_id=case_id,
            question=case["question"],
            expected_scopes=case["expected_scopes"],
            db_path=db_path,
            client=client,
            retrieval_strategy=retrieval_strategy,
        )
        rows.append(row)
        print(
            f"case {row['case_id']:02d} | recall@{top_k}: {row['recall_at_k']:.2f} | "
            f"precision@{top_k}: {row['precision_at_k']:.2f} | rr: {row['mrr']:.2f} | "
            f"judge: {row['judge_score']} | mode: {row['mode']} | k_returned: {row['retrieved_count']}"
        )

    summary = summarize_results(rows)
    print_summary_table(summary)


if __name__ == "__main__":
    main()
