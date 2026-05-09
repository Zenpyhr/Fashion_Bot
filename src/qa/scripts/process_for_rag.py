#!/usr/bin/env python3
"""Clean scraped fashion articles and split them into RAG-ready chunks."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QA_DATA_DIR = PROJECT_ROOT / "data" / "qa"
default_input_dir = QA_DATA_DIR / "raw_articles"
default_output_dir = QA_DATA_DIR / "processed_articles"
chunk_words = 150
overlap_words = 40
min_chunk_words = 50

sentence_split_regex = re.compile(r'(?<=[.!?])\s+(?=(?:["\']?[A-Z0-9]))')
word_regex = re.compile(r"\b\w+\b")


@dataclass
class RawArticle:
    """Structured version of a scraped article text file."""

    source_file: Path
    url: str
    scope: str
    title: str
    text: str


def word_count(text: str) -> int:
    """Count words with the same regex used by the chunking logic."""

    return len(word_regex.findall(text))


def normalize_space(text: str) -> str:
    """Normalize line endings and repeated horizontal whitespace."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text)


def slugify(text: str, max_length: int = 80) -> str:
    """Create a stable, filesystem-safe article ID from the title."""

    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        return "article"
    return slug[:max_length].strip("_") or "article"


def parse_raw_article(path: Path) -> RawArticle:
    """Read one raw article file and separate metadata headers from body text."""

    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = normalize_space(raw).strip()
    lines = raw.splitlines()

    url = ""
    scope = ""
    title = ""
    header_end = -1

    # Header metadata is expected near the top of each scraped text file.
    for idx, line in enumerate(lines[:40]):
        line_stripped = line.strip()
        lower = line_stripped.lower()
        if lower.startswith("url:"):
            url = line_stripped.split(":", 1)[1].strip()
            header_end = idx
        elif lower.startswith("scope:"):
            scope = line_stripped.split(":", 1)[1].strip()
            header_end = idx
        elif lower.startswith("title:"):
            title = line_stripped.split(":", 1)[1].strip()
            header_end = idx

    body_start = header_end + 1 if header_end >= 0 else 0

    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1

    body = "\n".join(lines[body_start:]).strip()
    if not title:
        title = path.stem.replace("_", " ").strip()
    if not scope and "__" in path.stem:
        scope = path.stem.split("__", 1)[0]

    return RawArticle(source_file=path, url=url, scope=scope, title=title, text=body)


def clean_article_text(text: str) -> str:
    """Remove duplicate paragraphs and collapse noisy whitespace."""

    text = normalize_space(text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

    cleaned = []
    seen = set()
    for paragraph in paragraphs:
        # Normalize each paragraph before comparing so small spacing differences do not create duplicates.
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        fingerprint = paragraph.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        cleaned.append(paragraph)

    if cleaned:
        return "\n\n".join(cleaned).strip()
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences while keeping abbreviations reasonably intact."""

    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []
    return [s.strip() for s in sentence_split_regex.split(collapsed) if s.strip()]


def fallback_word_chunks(
    text: str, target_words: int, overlap_words: int, min_chunk_words: int
) -> list[str]:
    """Chunk by word count when sentence splitting cannot produce useful chunks."""

    words = text.split()
    if not words:
        return []

    chunks = []
    # The step size is smaller than the chunk size, which creates overlap between chunks.
    step = max(1, target_words - overlap_words)
    for start in range(0, len(words), step):
        piece = words[start : start + target_words]
        if not piece:
            continue
        # Very short trailing chunks usually add weak retrieval context, so skip them.
        if start > 0 and len(piece) < min_chunk_words:
            break
        chunks.append(" ".join(piece).strip())
        if start + target_words >= len(words):
            break

    return chunks


def suffix_prefix_overlap_words(previous: str, current: str) -> int:
    """Measure repeated words at the boundary between adjacent chunks."""

    previous_words = previous.split()
    current_words = current.split()
    max_overlap = min(len(previous_words), len(current_words))

    # Search from the largest possible overlap down to find the exact repeated boundary.
    for size in range(max_overlap, 0, -1):
        if previous_words[-size:] == current_words[:size]:
            return size
    return 0


def dedupe_adjacent_chunks(chunks: list[str], min_new_words: int) -> list[str]:
    """Drop or merge chunks that add too little new information."""

    if not chunks:
        return []

    cleaned_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        previous = cleaned_chunks[-1]
        previous_normalized = re.sub(r"\s+", " ", previous).strip().lower()
        chunk_normalized = re.sub(r"\s+", " ", chunk).strip().lower()

        # Skip exact duplicate chunks produced by overlap or repeated article text.
        if chunk_normalized == previous_normalized:
            continue

        overlap = suffix_prefix_overlap_words(previous, chunk)
        chunk_words = chunk.split()
        new_words = len(chunk_words) - overlap

        # If the chunk mostly repeats the previous one, append only the new tail text.
        if new_words < min_new_words:
            if new_words > 0:
                novelty = " ".join(chunk_words[overlap:])
                cleaned_chunks[-1] = f"{previous} {novelty}".strip()
            continue

        cleaned_chunks.append(chunk)

    return cleaned_chunks


def sentence_chunks(
    text: str, target_words: int, overlap_words: int, min_chunk_words: int
) -> list[str]:
    """Build overlapping sentence-based chunks for retrieval."""

    sentences = split_sentences(text)
    if not sentences:
        return fallback_word_chunks(text, target_words, overlap_words, min_chunk_words)

    # Pre-compute sentence lengths so the chunking loop can make fast size decisions.
    sentence_word_counts = [word_count(sentence) for sentence in sentences]
    chunks = []
    start = 0

    while start < len(sentences):
        end = start
        total_words = 0

        # Grow each chunk until it reaches the target size without cutting a sentence.
        while end < len(sentences):
            next_words = sentence_word_counts[end]
            # Stop before adding a sentence that would push this chunk past the target size.
            if end > start and total_words + next_words > target_words:
                break

            total_words += next_words
            end += 1

            if total_words >= target_words:
                break

        chunk_text = " ".join(sentences[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(sentences):
            break

        # Count sentences backward from the chunk end until we reach the desired overlap.
        overlap_total = 0
        next_start = end
        # Move the next chunk start backward to preserve context overlap.
        while next_start > start and overlap_total < overlap_words:
            next_start -= 1
            overlap_total += sentence_word_counts[next_start]

        if next_start <= start:
            # Always move forward at least one sentence to prevent an infinite loop.
            next_start = start + 1
        start = next_start

    # Remove chunks where the overlap leaves too little genuinely new text.
    chunks = dedupe_adjacent_chunks(
        chunks=chunks,
        min_new_words=max(8, min_chunk_words // 4),
    )

    # Merge a tiny final chunk into the previous chunk so retrieval gets enough context.
    if len(chunks) >= 2 and word_count(chunks[-1]) < min_chunk_words:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks


def build_chunks_for_article(
    article: RawArticle,
    article_id: str,
    target_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[dict]:
    """Convert one cleaned article into JSONL records with retrieval metadata."""

    cleaned_text = clean_article_text(article.text)
    chunks = sentence_chunks(cleaned_text, target_words, overlap_words, min_chunk_words)

    parsed = urlparse(article.url) if article.url else None
    domain = parsed.netloc.lower() if parsed and parsed.netloc else "unknown"

    records = []
    total = len(chunks)

    for idx, chunk_text in enumerate(chunks):
        # Each chunk keeps article metadata so the answer can cite its source later.
        records.append(
            {
                "article_id": article_id,
                "chunk_index": idx,
                "chunk_count": total,
                "text": chunk_text,
                "title": article.title,
                "url": article.url,
                "scope": article.scope,
                "domain": domain,
            }
        )

    return records


def main() -> None:
    """Process all scraped article files into clean article and chunk JSONL files."""

    input_dir = Path(default_input_dir)
    output_dir = Path(default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {input_dir}")

    all_chunks = []
    all_articles = []
    seen_article_ids = set()

    for path in files:
        # Parse one scraped file, clean it, then turn it into multiple retrieval chunks.
        article = parse_raw_article(path)
        base_article_id = slugify(article.title, max_length=80)
        article_id = base_article_id
        suffix = 2
        # Ensure every article has a unique ID even when titles repeat.
        while article_id in seen_article_ids:
            article_id = f"{base_article_id}_{suffix}"
            suffix += 1
        seen_article_ids.add(article_id)

        cleaned = clean_article_text(article.text)
        chunks = build_chunks_for_article(
            article=article,
            article_id=article_id,
            target_words=chunk_words,
            overlap_words=overlap_words,
            min_chunk_words=min_chunk_words,
        )

        all_articles.append(
            {
                "article_id": article_id,
                "title": article.title,
                "url": article.url,
                "scope": article.scope,
                "clean_text": cleaned,
                "chunk_count": len(chunks),
            }
        )
        all_chunks.extend(chunks)

    chunks_path = output_dir / "fashion_qa_chunks.jsonl"
    # Chunks are the records later embedded into the vector database.
    with chunks_path.open("w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    articles_path = output_dir / "fashion_qa_articles_clean.jsonl"
    # The article file keeps the full cleaned text for inspection/debugging.
    with articles_path.open("w", encoding="utf-8") as f:
        for record in all_articles:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
