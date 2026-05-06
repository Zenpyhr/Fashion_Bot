"""VLM (vision) tagging for wardrobe photos.

Turns a user-uploaded clothing image into a catalog-like metadata row.
"""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from src.integrations.openai_client import create_openai_client, openai_is_configured
from src.shared.config import settings


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

    required_fields = [
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

    path = Path(image_path)
    metadata: dict[str, str] = {k: "" for k in required_fields}
    metadata["source_type"] = "wardrobe"
    metadata["image_path"] = str(path)
    metadata["image_relative_path"] = path.name
    metadata["item_id"] = path.stem
    metadata["display_name"] = ""
    metadata["description"] = ""
    metadata["target_group"] = "men"

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
        "properties": {k: {"type": "string"} for k in required_fields},
        "required": required_fields,
        "additionalProperties": False,
    }

    one_shot_example = {
        "item_id": "wardrobe_hoodie_001",
        "source_type": "wardrobe",
        "image_path": "data/user_wardrobe/demo_user/uploads/abcd1234.jpg",
        "image_relative_path": "abcd1234.jpg",
        "display_name": "Navy hoodie",
        "description": "Dark navy men's hoodie with a solid pattern and relaxed casual vibe. Best for fall and winter or cool spring evenings, and suitable for daily wear, travel, and laid-back weekends. Pairs well with denim, joggers, or cargo trousers; works as a top layer or mid-layer under a lightweight jacket.",
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
        "Analyze the clothing item in the image carefully.\n"
        "Return one JSON object only, with exactly the required keys.\n"
        "Use seed metadata when provided, unless the image clearly contradicts it.\n"
        "Always set target_group to 'men'.\n"
        "display_name should be a human-friendly product name, not a filename.\n"
        "description is the most important field and must be detailed.\n"
        "For description, write 2-4 natural sentences that include:\n"
        "1) visual details (color tone, pattern, silhouette/fit, notable design details),\n"
        "2) recommended seasons/weather,\n"
        "3) suitable occasions,\n"
        "4) simple styling suggestions.\n"
        "Do not output a short generic description.\n"
        "One-shot example output format:\n"
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
    for k in required_fields:
        v = raw.get(k, "unknown")
        result[k] = str(v) if v is not None else "unknown"

    result["source_type"] = "wardrobe"
    result["target_group"] = "men"

    if not result["description"].strip() or result["description"].lower() == "unknown":
        result["description"] = (
            f"Men's {result.get('normalized_color', 'unknown')} {result.get('normalized_category', 'fashion item')} "
            f"with a {result.get('normalized_pattern', 'clean')} look and {result.get('section_theme', 'casual')} style. "
            "Suitable for everyday wear in mild to cool weather and easy to pair with jeans or casual pants."
        )
    if not result["display_name"].strip() or result["display_name"].lower() == "unknown":
        result["display_name"] = result.get("product_type_name", "").strip() or "unknown"

    return result

"""VLM (vision) tagging for wardrobe photos.

Turns a user-uploaded clothing image into a catalog-like metadata row.
"""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from src.integrations.openai_client import create_openai_client, openai_is_configured
from src.shared.config import settings


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

    required_fields = [
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

    path = Path(image_path)
    metadata: dict[str, str] = {k: "" for k in required_fields}
    metadata["source_type"] = "wardrobe"
    metadata["image_path"] = str(path)
    metadata["image_relative_path"] = path.name
    metadata["item_id"] = path.stem
    metadata["display_name"] = ""
    metadata["description"] = ""
    metadata["target_group"] = "men"

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
        "properties": {k: {"type": "string"} for k in required_fields},
        "required": required_fields,
        "additionalProperties": False,
    }

    one_shot_example = {
        "item_id": "wardrobe_hoodie_001",
        "source_type": "wardrobe",
        "image_path": "data/user_wardrobe/demo_user/uploads/abcd1234.jpg",
        "image_relative_path": "abcd1234.jpg",
        "display_name": "Navy hoodie",
        "description": "Dark navy men's hoodie with a solid pattern and relaxed casual vibe. Best for fall and winter or cool spring evenings, and suitable for daily wear, travel, and laid-back weekends. Pairs well with denim, joggers, or cargo trousers; works as a top layer or mid-layer under a lightweight jacket.",
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
        "Analyze the clothing item in the image carefully.\n"
        "Return one JSON object only, with exactly the required keys.\n"
        "Use seed metadata when provided, unless the image clearly contradicts it.\n"
        "Always set target_group to 'men'.\n"
        "display_name should be a human-friendly product name, not a filename.\n"
        "description is the most important field and must be detailed.\n"
        "For description, write 2-4 natural sentences that include:\n"
        "1) visual details (color tone, pattern, silhouette/fit, notable design details),\n"
        "2) recommended seasons/weather,\n"
        "3) suitable occasions,\n"
        "4) simple styling suggestions.\n"
        "Do not output a short generic description.\n"
        "One-shot example output format:\n"
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
    for k in required_fields:
        v = raw.get(k, "unknown")
        result[k] = str(v) if v is not None else "unknown"

    # Hard defaults for the current demo.
    result["source_type"] = "wardrobe"
    result["target_group"] = "men"

    if not result["description"].strip() or result["description"].lower() == "unknown":
        result["description"] = (
            f"Men's {result.get('normalized_color', 'unknown')} {result.get('normalized_category', 'fashion item')} "
            f"with a {result.get('normalized_pattern', 'clean')} look and {result.get('section_theme', 'casual')} style. "
            "Suitable for everyday wear in mild to cool weather and easy to pair with jeans or casual pants."
        )
    if not result["display_name"].strip() or result["display_name"].lower() == "unknown":
        result["display_name"] = result.get("product_type_name", "").strip() or "unknown"

    return result

"""VLM (vision) tagging for wardrobe photos and future enrichment.

This module turns a user-uploaded clothing photo into a catalog-like metadata row.
"""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from src.integrations.openai_client import create_openai_client, openai_is_configured
from src.shared.config import settings


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

    required_fields = [
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

    path = Path(image_path)
    metadata: dict[str, str] = {k: "" for k in required_fields}
    metadata["source_type"] = "wardrobe"
    metadata["image_path"] = str(path)
    metadata["image_relative_path"] = path.name
    metadata["item_id"] = path.stem
    metadata["display_name"] = ""
    metadata["description"] = ""
    metadata["target_group"] = "men"

    if seed:
        for k, v in seed.items():
            if k in metadata and v is not None:
                metadata[k] = str(v)

    if not metadata["display_name"].strip():
        metadata["display_name"] = metadata["product_type_name"].strip() or "unknown"

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    image_base64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{image_base64}"

    schema = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in required_fields},
        "required": required_fields,
        "additionalProperties": False,
    }

    one_shot_example = {
        "item_id": "wardrobe_hoodie_001",
        "source_type": "wardrobe",
        "image_path": "data/user_wardrobe/demo_user/uploads/abcd1234.jpg",
        "image_relative_path": "abcd1234.jpg",
        "display_name": "Navy hoodie",
        "description": "Dark navy men's hoodie with a solid pattern and relaxed casual vibe. Best for fall and winter or cool spring evenings, and suitable for daily wear, travel, and laid-back weekends. Pairs well with denim, joggers, or cargo trousers; works as a top layer or mid-layer under a lightweight jacket.",
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
        "Analyze the clothing item in the image carefully.\n"
        "Return one JSON object only, with exactly the required keys.\n"
        "Use seed metadata when provided, unless the image clearly contradicts it.\n"
        "Always set target_group to 'men'.\n"
        "display_name should be a human-friendly product name, not a filename.\n"
        "description is the most important field and must be detailed.\n"
        "For description, write 2-4 natural sentences that include:\n"
        "1) visual details (color tone, pattern, silhouette/fit, notable design details),\n"
        "2) recommended seasons/weather,\n"
        "3) suitable occasions,\n"
        "4) simple styling suggestions.\n"
        "Do not output a short generic description.\n"
        "One-shot example output format:\n"
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
    for k in required_fields:
        v = raw.get(k, "unknown")
        result[k] = str(v) if v is not None else "unknown"

    result["source_type"] = "wardrobe"
    result["target_group"] = "men"

    if not result["description"].strip() or result["description"].lower() == "unknown":
        result["description"] = (
            f"Men's {result.get('normalized_color', 'unknown')} {result.get('normalized_category', 'fashion item')} "
            f"with a {result.get('normalized_pattern', 'clean')} look and {result.get('section_theme', 'casual')} style. "
            "Suitable for everyday wear in mild to cool weather and easy to pair with jeans or casual pants."
        )
    if not result["display_name"].strip() or result["display_name"].lower() == "unknown":
        result["display_name"] = result.get("product_type_name", "").strip() or "unknown"

    return result


def enrich_catalog_items() -> None:
    """Placeholder for future catalog-wide VLM enrichment."""

    raise NotImplementedError

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from openai import OpenAI


def tag_image(
    image_path: str,
    seed: dict[str, Any] | None = None,
    model: str = "gpt-5.4-mini",
    detail: str = "auto",
) -> dict[str, str]:
    """
    Tag one clothing image with OpenAI vision and return metadata in your target schema.
    """
    required_fields = [
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
        "section_name"
    ]

    path = Path(image_path)
    metadata: dict[str, str] = {k: "" for k in required_fields}
    metadata["source_type"] = "personal cloth"
    metadata["image_path"] = str(path)
    metadata["image_relative_path"] = path.name
    metadata["item_id"] = path.stem
    metadata["display_name"] = ""
    metadata["description"] = ""
    metadata["target_group"] = "men"

    if seed:
        for k, v in seed.items():
            if k in metadata and v is not None:
                metadata[k] = str(v)
    metadata["source_type"] = "personal clothes"
    metadata["target_group"] = "men"
    if not metadata["display_name"].strip():
        metadata["display_name"] = metadata["product_type_name"].strip() or "unknown"

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    image_base64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    image_data_url = f"data:{mime_type};base64,{image_base64}"

    schema = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in required_fields},
        "required": required_fields,
        "additionalProperties": False,
    }

    one_shot_example = {
        "item_id": "0504658004",
        "source_type": "catalog",
        "image_path": "data/processed/demo_images/050/0504658004.jpg",
        "image_relative_path": "050/0504658004.jpg",
        "display_name": "Birkir canvas (1)",
        "description": "Dark blue men's hoodie with a clean solid pattern and casual streetwear vibe. Best for fall and winter or cool spring evenings, and suitable for daily wear, campus, travel, and relaxed weekend outings. Works well with denim, joggers, or cargo pants; ideal as a top layer or mid-layer under a lightweight jacket.",
        "target_group": "men",
        "recommendation_role": "top",
        "normalized_category": "hoodie",
        "product_family": "upper_body",
        "normalized_color": "blue",
        "color_detail": "dark blue",
        "color_tone": "dark",
        "normalized_pattern": "solid",
        "section_theme": "casual",
        "product_type_name": "Hoodie",
        "product_group_name": "Garment Upper body",
        "index_name": "Menswear",
        "index_group_name": "Menswear",
        "section_name": "Men Casual"
    }

    prompt = (
        "You are a fashion cloth metadata tagger.\n"
        "You should analyze the components and elements of the image carefully.\n"
        "Return one JSON object only, with exactly the required keys.\n"
        "Use seed metadata when provided, unless image clearly contradicts it.\n"
        "For source_type, use 'personal cloth' for user-uploaded images by default.\n"
        "Always set target_group to 'men'.\n"
        "display_name should be the product name, not a filename.\n"
        "description is the most important field and must be detailed.\n"
        "For description, write 2-4 natural sentences that include:\n"
        "1) visual details (color tone, pattern, silhouette/fit, notable design details),\n"
        "2) recommended seasons/weather,\n"
        "3) suitable occasions (e.g., daily, work, travel, date, party, sports/casual),\n"
        "4) simple styling suggestions (what it pairs with).\n"
        "Do not output a short generic description.\n"
        "Do not leave description empty or 'unknown'.\n"
        "One-shot example output format:\n"
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
    for k in required_fields:
        v = raw.get(k, "unknown")
        result[k] = str(v) if v is not None else "unknown"
    result["target_group"] = "men"
    if not result["description"].strip() or result["description"].lower() == "unknown":
        result["description"] = (
            f"Men's {result.get('normalized_color', 'unknown')} {result.get('normalized_category', 'fashion item')} "
            f"with a {result.get('normalized_pattern', 'clean')} look and {result.get('section_theme', 'casual')} style. "
            "Suitable for everyday wear in mild to cool weather and easy to pair with jeans or casual pants."
        )
    if not result["display_name"].strip() or result["display_name"].lower() == "unknown":
        result["display_name"] = result.get("product_type_name", "").strip() or "unknown"
    return result

