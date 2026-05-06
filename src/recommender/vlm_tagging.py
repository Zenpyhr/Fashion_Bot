"""VLM (vision) tagging for wardrobe photos.

Turns a user-uploaded clothing image into a catalog-like metadata dict.

Important:
- This file must define ONE `tag_image` function.
- For wardrobe uploads, `source_type` must be `wardrobe` so downstream code can
  correctly attribute wardrobe usage in retrieval/eval.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from src.integrations.openai_client import create_openai_client, openai_is_configured
from src.shared.config import settings


REQUIRED_FIELDS: list[str] = [
    "item_id",
    "source_type",
    "image_path",
    "image_relative_path",
    "display_name",
    "description",
    "target_group",
    "recommendation_role",
    "normalized_category",
    "product_family",
    "normalized_color",
    "color_detail",
    "color_tone",
    "normalized_pattern",
    "section_theme",
    "product_type_name",
    "product_group_name",
    "index_name",
    "index_group_name",
    "section_name",
]


def tag_image(
    image_path: str,
    *,
    seed: dict[str, Any] | None = None,
    model: str | None = None,
    detail: str = "auto",
) -> dict[str, str]:
    """Tag one clothing image with OpenAI vision and return metadata in a fixed schema."""

    if not openai_is_configured():
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY to use VLM tagging.")

    path = Path(image_path)

    # Seed with stable defaults so the model doesn't omit fields.
    metadata: dict[str, str] = {k: "" for k in REQUIRED_FIELDS}
    metadata["item_id"] = path.stem
    metadata["source_type"] = "wardrobe"
    metadata["image_path"] = str(path)
    metadata["image_relative_path"] = path.name
    metadata["target_group"] = "men"
    metadata["index_name"] = "Wardrobe"
    metadata["index_group_name"] = "Wardrobe"
    metadata["section_name"] = "Wardrobe"

    if seed:
        for k, v in seed.items():
            if k in metadata and v is not None:
                metadata[k] = str(v)

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    image_base64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{image_base64}"

    schema = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in REQUIRED_FIELDS},
        "required": REQUIRED_FIELDS,
        "additionalProperties": False,
    }

    one_shot_example = {
        "item_id": "wardrobe_hoodie_001",
        "source_type": "wardrobe",
        "image_path": "data/recommender/user_wardrobe/demo_user/uploads/abcd1234.jpg",
        "image_relative_path": "abcd1234.jpg",
        "display_name": "Navy zip-up hoodie",
        "description": (
            "Dark navy men's zip-up hoodie with a clean solid base and a relaxed casual fit. "
            "Best for fall/winter or cool spring evenings; suitable for daily wear, travel, and laid-back weekends. "
            "Pairs well with jeans or joggers and casual sneakers; can be layered under a lightweight jacket."
        ),
        "target_group": "men",
        "recommendation_role": "top",
        "normalized_category": "hoodie",
        "product_family": "upper_body",
        "normalized_color": "blue",
        "color_detail": "dark navy",
        "color_tone": "dark",
        "normalized_pattern": "solid",
        "section_theme": "casual",
        "product_type_name": "Hoodie",
        "product_group_name": "Garment Upper body",
        "index_name": "Wardrobe",
        "index_group_name": "Wardrobe",
        "section_name": "Wardrobe",
    }

    prompt = (
        "You are a fashion wardrobe metadata tagger.\n"
        "You must return ONE JSON object only, with exactly the required keys (no extra keys, no markdown).\n"
        "Analyze the clothing item in the image carefully.\n"
        "Use seed metadata when provided unless the image clearly contradicts it.\n"
        "\n"
        "Hard rules (must follow):\n"
        "- source_type: always 'wardrobe'\n"
        "- target_group: always 'men'\n"
        "- index_name/index_group_name/section_name: keep Wardrobe context (use 'Wardrobe' if unsure)\n"
        "\n"
        "display_name rules:\n"
        "- 2–6 words; include color + garment type (e.g., 'Navy zip-up hoodie')\n"
        "- do NOT use filenames/ids; do NOT output 'unknown'\n"
        "\n"
        "description rules (2–4 natural sentences; not generic):\n"
        "1) visual details (color tone, pattern, fit/silhouette, notable design details)\n"
        "2) seasons/weather\n"
        "3) suitable occasions (daily, work, travel, date, weekend, sportswear)\n"
        "4) simple styling suggestions (what it pairs with)\n"
        "\n"
        "If the item is footwear, set recommendation_role='shoes' and choose an appropriate normalized_category (e.g., sneakers/boots).\n"
        "If it's outerwear (jacket/coat/blazer), set recommendation_role='outerwear' when appropriate.\n"
        "\n"
        "One-shot example output format (JSON):\n"
        f"{json.dumps(one_shot_example, indent=2)}\n"
    )

    client = create_openai_client()
    request_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_text", "text": f"Seed metadata:\n{json.dumps(metadata)}"},
                {"type": "input_image", "image_url": image_data_url, "detail": detail},
            ],
        }
    ]

    response = client.responses.create(
        model=model or settings.openai_model_reranker,
        input=request_input,
        text={
            "format": {
                "type": "json_schema",
                "name": "fashion_item_metadata",
                "schema": schema,
                "strict": True,
            }
        },
        temperature=0.1,
    )

    raw = json.loads(response.output_text)

    result: dict[str, str] = {}
    for k in REQUIRED_FIELDS:
        v = raw.get(k, "unknown")
        result[k] = str(v) if v is not None else "unknown"

    # Hard defaults for our system.
    result["source_type"] = "wardrobe"
    result["target_group"] = "men"
    if not result.get("index_name"):
        result["index_name"] = "Wardrobe"
    if not result.get("index_group_name"):
        result["index_group_name"] = "Wardrobe"
    if not result.get("section_name"):
        result["section_name"] = "Wardrobe"

    if not result["description"].strip() or result["description"].lower() == "unknown":
        result["description"] = (
            f"Men's {result.get('normalized_color', 'unknown')} {result.get('normalized_category', 'fashion item')} "
            f"with a {result.get('normalized_pattern', 'clean')} look and {result.get('section_theme', 'casual')} style. "
            "Suitable for everyday wear in mild to cool weather and easy to pair with jeans or casual pants."
        )

    if not result["display_name"].strip() or result["display_name"].lower() == "unknown":
        result["display_name"] = result.get("product_type_name", "").strip() or "unknown"

    return result

