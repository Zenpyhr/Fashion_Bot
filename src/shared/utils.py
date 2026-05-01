"""Small shared helpers."""


def slugify_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")
