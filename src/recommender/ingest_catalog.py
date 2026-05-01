"""Load and parse H&M catalog metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_CATALOG_COLUMNS = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]

RAW_CATALOG_DTYPES = {
    "article_id": "string",
    "product_code": "string",
    "prod_name": "string",
    "product_type_name": "string",
    "product_group_name": "string",
    "graphical_appearance_name": "string",
    "colour_group_name": "string",
    "perceived_colour_value_name": "string",
    "perceived_colour_master_name": "string",
    "department_name": "string",
    "index_name": "string",
    "index_group_name": "string",
    "section_name": "string",
    "garment_group_name": "string",
    "detail_desc": "string",
}

DEFAULT_SOURCE_PATH = Path("data/raw/hm/articles.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/catalog_items/catalog_raw_subset.csv")


def ingest_catalog(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Read the raw H&M catalog and keep only the columns needed downstream."""

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Catalog source not found: {source}")

    df = pd.read_csv(
        source,
        usecols=RAW_CATALOG_COLUMNS,
        dtype=RAW_CATALOG_DTYPES,
        keep_default_na=True,
    )

    df = df.replace({r"^\s*$": pd.NA}, regex=True)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

    return df
