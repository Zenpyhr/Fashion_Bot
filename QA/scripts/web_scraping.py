#!/usr/bin/env python3

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

url_list_path = Path("data/url_list.json")
output_dir = Path("data/raw_articles")
min_words = 30

request_headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

boilerplate_patterns = [
    r"sign up for vogue",
    r"cond[ée]\s+nast",
    r"all rights reserved",
    r"affiliate partnership",
    r"ad choices",
    r"who what wear is part of future us",
    r"future us,\s*inc\.",
    r"full 7th floor,\s*130 west 42nd street",
    r"visit our corporate site",
    r"when you purchase through links on our site",
    r"here.?s how it works",
]

paywall_patterns = [
    r"already a subscriber",
    r"sign in",
    r"subscribe",
    r"unlimited access",
    r"create an account",
]


def _is_boilerplate(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in boilerplate_patterns)


def _looks_like_paywall(text: str) -> bool:
    normalized = text.strip().lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in paywall_patterns)


def _extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return "untitled_article"


def _select_content_root(soup: BeautifulSoup):
    selectors = [
        "article",
        "main article",
        "[itemprop='articleBody']",
        "main",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node and node.get_text(" ", strip=True):
            return node
    return soup.body or soup


def _extract_clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    root = _select_content_root(soup)
    paragraphs = []
    seen = set()

    for p in root.find_all("p"):
        text = p.get_text(" ", strip=True)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 45:
            continue
        if _is_boilerplate(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        paragraphs.append(text)

    text = "\n\n".join(paragraphs).strip()
    if text:
        return text

    # Fallback for pages that don't use standard paragraph tags.
    lines = []
    for line in root.get_text("\n", strip=True).splitlines():
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < 45:
            continue
        if _is_boilerplate(line):
            continue
        lines.append(line)
    return "\n\n".join(lines).strip()


def _validate_extracted_text(text: str, min_words: int) -> None:
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < min_words:
        raise RuntimeError(
            f"Extracted only {word_count} words (< {min_words}). "
            "Likely teaser/paywall/signup page or non-article content."
        )
    if _looks_like_paywall(text):
        raise RuntimeError("Detected paywall/signup content in extracted text.")


def scrape_article_to_txt(
    url: str,
    scope: str,
    output_dir: Path = output_dir,
    min_words: int = min_words,
) -> Path:
    response = requests.get(url, timeout=20, headers=request_headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title = _extract_title(soup)
    text = _extract_clean_text(soup)
    _validate_extracted_text(text, min_words=min_words)

    safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_")[:80] or "article"
    domain = urlparse(url).netloc.replace(".", "_")

    safe_scope = re.sub(r"[^a-zA-Z0-9_-]+", "_", scope).strip("_")[:60] or "unknown_scope"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{safe_scope}__{domain}_{safe_title}.txt"

    with file_path.open("w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Scope: {scope}\n")
        f.write(f"Title: {title}\n\n")
        f.write(text)

    return file_path


def load_scoped_urls(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(data).__name__}.")

    scoped_urls = {}
    for scope, urls in data.items():
        if not isinstance(scope, str):
            raise ValueError("All top-level keys in url_list.json must be strings.")
        if not isinstance(urls, list):
            raise ValueError(f"Scope '{scope}' must map to a list of URLs.")
        scoped_urls[scope] = [str(url).strip() for url in urls if str(url).strip()]
    return scoped_urls


if __name__ == "__main__":
    scoped_urls = load_scoped_urls(url_list_path)
    total_urls = sum(len(urls) for urls in scoped_urls.values())
    print(f"Loaded {total_urls} URLs across {len(scoped_urls)} scopes from {url_list_path}")

    for scope, urls in scoped_urls.items():
        print(f"\nScope: {scope} ({len(urls)} URLs)")
        for url in urls:
            try:
                saved_file = scrape_article_to_txt(
                    url=url,
                    scope=scope,
                    output_dir=output_dir,
                    min_words=min_words,
                )
                print(f"Saved: {saved_file}")
            except Exception as e:
                print(f"Failed: [{scope}] {url} -> {e}")
