import pandas as pd

from src.recommender.normalize_catalog import (
    _map_recommendation_role,
    _map_target_group,
    normalize_catalog_dataframe,
)


def test_recommendation_role_mapping() -> None:
    assert _map_recommendation_role("Blouse") == "top"
    assert _map_recommendation_role("Boots") == "shoes"
    assert _map_recommendation_role("Dress") is None
    assert _map_recommendation_role("Bra") is None


def test_target_group_mapping() -> None:
    assert _map_target_group("Ladieswear", "Ladieswear", "Womens Everyday Collection") == "women"
    assert _map_target_group("Menswear", "Menswear", "Men H&M Sport") == "men"
    assert _map_target_group("Baby/Children", "Baby Sizes 50-98", "Baby Essentials & Complements") == "baby"
    assert _map_target_group("Baby/Children", "Children Sizes 92-140", "Kids Girl") == "kids"


def test_normalize_catalog_dataframe_filters_to_mvp_roles() -> None:
    raw = pd.DataFrame(
        [
            {
                "article_id": "1",
                "product_code": "0001",
                "prod_name": "Classic Blouse",
                "product_type_name": "Blouse",
                "product_group_name": "Garment Upper body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "White",
                "perceived_colour_value_name": "Light",
                "perceived_colour_master_name": "White",
                "department_name": "Blouse",
                "index_name": "Ladieswear",
                "index_group_name": "Ladieswear",
                "section_name": "Womens Everyday Collection",
                "garment_group_name": "Blouses",
                "detail_desc": "A lightweight woven blouse.",
            },
            {
                "article_id": "2",
                "product_code": "0002",
                "prod_name": "Soft Bra",
                "product_type_name": "Bra",
                "product_group_name": "Underwear",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "Black",
                "perceived_colour_value_name": "Dark",
                "perceived_colour_master_name": "Black",
                "department_name": "Expressive Lingerie",
                "index_name": "Lingeries/Tights",
                "index_group_name": "Ladieswear",
                "section_name": "Womens Lingerie",
                "garment_group_name": "Under-, Nightwear",
                "detail_desc": "A bra we should filter out.",
            },
            {
                "article_id": "3",
                "product_code": "0003",
                "prod_name": "Running Sneakers",
                "product_type_name": "Sneakers",
                "product_group_name": "Shoes",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "Black",
                "perceived_colour_value_name": "Dark",
                "perceived_colour_master_name": "Black",
                "department_name": "Men Sport Tops",
                "index_name": "Sport",
                "index_group_name": "Sport",
                "section_name": "Men H&M Sport",
                "garment_group_name": "Shoes",
                "detail_desc": "Performance sneakers.",
            },
            {
                "article_id": "4",
                "product_code": "0004",
                "prod_name": "Soft Slippers",
                "product_type_name": "Slippers",
                "product_group_name": "Shoes",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "Black",
                "perceived_colour_value_name": "Dark",
                "perceived_colour_master_name": "Black",
                "department_name": "Men Shoes",
                "index_name": "Menswear",
                "index_group_name": "Menswear",
                "section_name": "Mens Outerwear",
                "garment_group_name": "Shoes",
                "detail_desc": "House slippers we should filter out.",
            },
        ]
    )

    normalized = normalize_catalog_dataframe(raw)

    assert list(normalized["item_id"]) == ["3", "1"]
    assert set(normalized["recommendation_role"]) == {"shoes", "top"}
