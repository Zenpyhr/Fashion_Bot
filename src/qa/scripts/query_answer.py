"""Retrieve fashion QA context from Chroma and generate grounded LLM answers."""

import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Ensure .env is loaded for local script runs and that LangChain can see the key.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from src.shared.config import settings

    if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
except Exception:
    # QA scripts should still run retrieval-only even if shared settings aren't available.
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QA_DATA_DIR = PROJECT_ROOT / "data" / "qa"
db = str(QA_DATA_DIR / "index" / "fashion_chroma_db")
collection_name = "fashion_articles"
default_top_k = 5
embed_model_name = "all-MiniLM-L6-v2"
llm_use = "gpt-5.4-mini"
scope_top_n = 3
mix_scoped_ratio = 0.7
url_list_file = QA_DATA_DIR / "url_list.json"

_embed_model = None
_allowed_scopes = None
_vector_store = None
_vector_store_path = None


def get_embed_model() -> HuggingFaceEmbeddings:
    """Create the embedding model once and reuse it across queries."""

    global _embed_model
    if _embed_model is None:
        # Normalized embeddings make similarity search more stable for cosine-style distance.
        _embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embed_model


def get_vector_store(db_path: str = db) -> Chroma:
    """Create or reuse the Chroma vector store for the selected database path."""

    global _vector_store, _vector_store_path
    if _vector_store is None or _vector_store_path != db_path:
        # Recreate the store only when the caller switches to a different database path.
        _vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=get_embed_model(),
        )
        _vector_store_path = db_path
    return _vector_store


def load_allowed_scopes(path: Path = url_list_file) -> list[str]:
    """Load the valid fashion scopes from the URL configuration file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data.keys())


def get_allowed_scopes() -> list[str]:
    """Cache allowed scopes so repeated QA requests avoid re-reading JSON."""

    global _allowed_scopes
    if _allowed_scopes is None:
        _allowed_scopes = load_allowed_scopes()
    return _allowed_scopes


def map_question_to_scopes(question: str, allowed_scopes: list[str]) -> dict:
    """Ask the LLM to map a user question to the closest configured scopes."""

    client = ChatOpenAI(model=llm_use, temperature=0)
    # Scope classification narrows retrieval to the most relevant fashion categories.
    prompt = (
        "Classify the user question into fashion scopes.\n"
        f"Allowed scopes: {allowed_scopes}\n"
        "Return JSON only.\n"
        f"Return at most {scope_top_n} scopes from the allowed list with confidence in the interval of 0 to 1.\n"
        "If no scope fits, return unknown.\n"
        "JSON format:\n"
        '{"scopes":[{"scope":"allowed_scope_name","confidence":0.0}],"unknown":false}\n'
        "or\n"
        '{"scopes":[],"unknown":true}\n'
        f"Question: {question}"
    )

    response = client.invoke(
        [
            SystemMessage(content="You are a precise scope classifier."),
            HumanMessage(content=prompt),
        ]
    )
    content = response.content.strip()
    # The classifier is instructed to return JSON only, so parse it into a dict.
    parsed = json.loads(content)

    unknown = parsed.get("unknown", False)
    raw_scopes = parsed.get("scopes", [])

    # Keep only valid scopes and the highest confidence for each scope.
    best_by_scope = {}
    for item in raw_scopes:
        scope = item.get("scope")
        confidence = float(item.get("confidence", 0.0))
        if scope not in allowed_scopes:
            continue
        if scope not in best_by_scope or confidence > best_by_scope[scope]:
            best_by_scope[scope] = confidence

    ranked = sorted(best_by_scope.items(), key=lambda x: x[1], reverse=True)
    # Keep only the strongest few scopes so retrieval stays focused.
    top_scopes = [scope for scope, _ in ranked[:scope_top_n]]
    top_confidence = ranked[0][1] if ranked else 0.0
    mode = "unknown" if unknown or not top_scopes else "scoped"
    return {
        "top_scopes": top_scopes,
        "top_confidence": top_confidence,
        "mode": mode,
    }


def query_candidates(
    collection, query_emb: list[float], n_results: int, where: dict | None = None
) -> list[dict]:
    """Query Chroma and normalize raw results into candidate context records."""

    query_kwargs = {
        "query_embeddings": [query_emb],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        # The where filter lets mixed retrieval search only inside predicted scopes.
        query_kwargs["where"] = where

    # Query Chroma directly with the already-computed question embedding.
    results = collection._collection.query(**query_kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []
    for idx, (text, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        # Source keys help avoid returning many near-duplicate chunks from one article.
        source_key = (
            meta.get("article_id")
            or meta.get("url")
            or meta.get("title")
            or f"source_{idx}"
        )
        candidates.append(
            {
                "text": text,
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "article_id": meta.get("article_id", ""),
                "scope": meta.get("scope", ""),
                "source_key": source_key,
                "distance": distance,
            }
        )
    return candidates


def select_diverse(
    candidates: list[dict], limit: int, seen_sources: set | None = None
) -> tuple[list[dict], set]:
    """Select top candidates while preferring coverage across different articles."""

    if seen_sources is None:
        seen_sources = set()

    unique_first_pass = []
    fallback_chunks = []

    # Prefer one chunk per article first to improve coverage/diversity.
    for item in candidates:
        source_key = item.get("source_key")
        if source_key not in seen_sources:
            # First pass favors new articles so the final context is not one repeated source.
            unique_first_pass.append(item)
            seen_sources.add(source_key)
        else:
            # Repeated articles are kept as fallback in case we still need more chunks.
            fallback_chunks.append(item)

    selected = unique_first_pass[:limit]
    if len(selected) < limit:
        # If there are not enough unique articles, fill from duplicate-source chunks.
        selected.extend(fallback_chunks[: limit - len(selected)])
    return selected, seen_sources


def retrieve(
    question: str,
    top_k: int,
    db_path: str = db,
    return_scope_decision: bool = False,
    retrieval_strategy: str = "mix",
) -> list[dict] | tuple[list[dict], dict]:
    """Retrieve relevant chunks using global search or scope-aware mixed search."""

    model = get_embed_model()
    collection = get_vector_store(db_path)

    # Overfetch so diversity filtering still has enough candidates to choose from.
    overfetch_k = max(top_k * 4, top_k)
    query_emb = model.embed_query(question)

    selected = []
    seen_sources = set()

    if retrieval_strategy not in {"mix", "global"}:
        raise ValueError("retrieval_strategy must be one of: mix, global")

    scope_decision = {
        "top_scopes": [],
        "top_confidence": 0.0,
        "mode": "global",
        "retrieval_mode": "global_only",
    }

    if retrieval_strategy == "global":
        # Global retrieval ignores scope labels and searches the whole vector database.
        global_candidates = query_candidates(collection, query_emb, n_results=overfetch_k)
        selected, seen_sources = select_diverse(global_candidates, top_k, seen_sources)
    else:
        allowed_scopes = get_allowed_scopes()
        # Mixed retrieval first predicts likely scopes for the question.
        scope_decision = map_question_to_scopes(question, allowed_scopes)
        top_scopes = scope_decision["top_scopes"]

        # Mixed retrieval: prioritize mapper scopes first, then fill remaining with global search.
        if top_scopes:
            scope_decision["retrieval_mode"] = "mixed_scoped_priority"

            # Reserve about 70 percent of results for scoped matches, then backfill globally.
            scoped_target = max(1, int(round(top_k * mix_scoped_ratio)))
            # Global candidates are fetched early so they can fill gaps after scoped selection.
            global_candidates = query_candidates(collection, query_emb, n_results=overfetch_k)

            # Scoped candidates search only chunks whose metadata scope matches the classifier.
            scoped_candidates = query_candidates(
                collection,
                query_emb,
                n_results=overfetch_k,
                where={"scope": {"$in": top_scopes}},
            )

            scoped_selected, seen_sources = select_diverse(
                scoped_candidates, min(scoped_target, top_k), seen_sources
            )
            selected.extend(scoped_selected)

            if len(selected) < top_k:
                # Backfill from global search to avoid missing useful context outside predicted scopes.
                remaining = top_k - len(selected)
                global_selected, seen_sources = select_diverse(
                    global_candidates, remaining, seen_sources
                )
                selected.extend(global_selected)
        else:
            scope_decision["retrieval_mode"] = "global_only_no_scope"
            # If the classifier cannot find a useful scope, fall back to normal semantic search.
            global_candidates = query_candidates(collection, query_emb, n_results=overfetch_k)
            selected, seen_sources = select_diverse(global_candidates, top_k, seen_sources)

    if return_scope_decision:
        return selected, scope_decision
    return selected


def llm_prompt(question: str, contexts: list[dict], detected_scopes: list[str] | None = None) -> str:
    """Build the answer prompt with source context and strict citation rules."""

    context = ""
    for i, item in enumerate(contexts, start=1):
        # Number each retrieved chunk so the LLM can cite it as [Source i].
        context += f"\n[Source {i}]\n"
        context += f"Title: {item['title']}\n"
        context += f"URL: {item['url']}\n"
        context += f"Content: {item['text']}\n"
    source_count = len(contexts)
    # The registry gives the model an exact source list to reproduce in the final answer.
    source_registry = "\n".join(
        f"{i}. [Source {i}] {item.get('title', '')} - {item.get('url', '')}"
        for i, item in enumerate(contexts, start=1)
    )

    scope_context = ""
    if detected_scopes:
        # Include detected scopes as extra context, but the article chunks remain the main evidence.
        scope_context = f"\nDetected scopes: {detected_scopes}\n"

    # The prompt forces an evidence check before answering to reduce hallucinated trend advice.
    prompt = f"""
You are an experienced fashion assistant specializing in trends and outfit analysis.

Answer the question using the provided context as your primary source.
Do NOT fabricate or hallucinate information.

If the question is ambiguous (e.g., "sports outfit"), briefly clarify the possible meanings before answering.

If the context is incomplete or missing key information:
- clearly state what is missing
- provide a cautious, general answer if helpful
- explicitly distinguish between information from the context and general knowledge

---

First, run an evidence sufficiency gate before writing:
- "Sufficient evidence" means the retrieved sources contain direct, relevant support for the core user question.
- If the retrieved sources are definitely irrelevant to the question, or do not contain enough direct support for the core claims, you MUST use Insufficient Evidence Mode.
- In Insufficient Evidence Mode, do not give trend advice from general knowledge.

Sufficient Evidence Mode output structure:

### Answer
Provide a clear and detailed answer in 2 to 3 short paragraphs.
- Focus on practical and concrete insights (e.g., specific clothing items, styling elements, trends)
- Use natural, conversational language.
- Include at least 4 concrete details from the context (items, silhouettes, fabrics, colors, or styling directions)
- If seasonal evidence exists, briefly compare spring/summer vs fall/winter
- End with one practical takeaway for how someone can apply the trends
- If needed, include a short clarification of the question meaning

### Key Trends
List distinct trends supported by the context:
- Each bullet = ONE clear trend
- Start each bullet with a short bold trend label like: **Trend label**: detail...
- Include specific examples (e.g., sneakers, tracksuits, tennis skirts, jerseys)
- Add inline citations like [Source 1]
- Combine multiple sources for a trend when appropriate
- If fewer than 3 supported trends exist, say:
  "The available evidence is limited and supports only X trends"

### Evidence
Explain how the sources support the trends:
- Do NOT repeat the trend statements
- Summarize what each source contributes
- Highlight overlaps or differences between sources when relevant
- Keep explanations concise but informative
- You must reference every source at least once in this section using [Source i].

### Sources
List ALL provided sources exactly once, in numeric order from [Source 1] to [Source {source_count}], using this format:
1. [Source i] Title - URL

---

Insufficient Evidence Mode output structure (use exactly this structure):

### Answer
I do not have enough reliable evidence in the retrieved sources to answer this question directly.

### Key Trends
- No supported trend can be concluded from the retrieved sources.

### Evidence
- The retrieved context does not contain enough direct, relevant evidence for the user question.
- Briefly state what is missing (for example: missing topic coverage, missing timeframe, or missing item/category evidence).

---

Rules:
- Prioritize the provided context
- Do NOT invent unsupported facts
- Do NOT copy text verbatim from the context
- Prefer specific details over vague generalizations
- Be transparent about uncertainty or gaps
- Citation validity is strict: only cite sources in the range [Source 1] to [Source {source_count}]
- Never output out-of-range citations such as [Source {source_count + 1}] or higher
- If and only if you are in Sufficient Evidence Mode:
  - Use every source ID from [Source 1] to [Source {source_count}] at least once in Key Trends or Evidence.
  - In ### Sources, include all and only the provided sources, each exactly once, and keep source IDs consistent.
- If you are in Insufficient Evidence Mode:
  - Do not provide unsupported fashion advice.
  - Do not claim trends, brands, numbers, or dates as facts.
  - Follow the Insufficient Evidence Mode structure exactly.
- Before final answer, self-check citation integrity and evidence sufficiency. Do not output extra commentary.

---

Provided source registry (must be preserved in the Sources section):
{source_registry}

---

Context:
{scope_context}
{context}

Question:
{question}

Answer:
""".strip()
    return prompt 

def generate_answer(prompt: str, source_count: int | None = None) -> str:
    """Send the grounded QA prompt to the chat model and return its answer."""

    client = ChatOpenAI(model=llm_use, temperature=0)

    # Temperature 0 makes the answer more deterministic and citation-following.
    response = client.invoke(
        [
            SystemMessage(
                content=(
                    "You are an experienced fashion assistant. "
                    "Answer only with supported evidence from retrieved context and cite sources."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    return response.content.strip()


def qa_answer(question, retrieval_strategy: str = "mix"):
    """End-to-end QA helper used by the app and command-line script."""

    question = question 
    top_k = default_top_k
    db_path = db

    # Retrieve relevant article chunks and keep the scope decision for prompt context.
    contexts, scope_decision = retrieve(
        question,
        top_k=top_k,
        db_path=db_path,
        return_scope_decision=True,
        retrieval_strategy=retrieval_strategy,
    )
    # Build a grounded prompt from retrieved chunks, then ask the LLM to answer with citations.
    prompt = llm_prompt(question, contexts, detected_scopes=scope_decision.get("top_scopes", []))
    ans = generate_answer(prompt, source_count=len(contexts))
    return ans

if __name__ == "__main__":
    question = "How should I style wide-leg jeans in 2026 without looking sloppy?"
    output = qa_answer(question)
    print(output)
