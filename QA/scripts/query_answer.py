import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

db = "index/fashion_chroma_db"
collection_name = "fashion_articles"
default_top_k = 5
embed_model_name = "all-MiniLM-L6-v2"
llm_use = "gpt-5.4-mini"
scope_confidence_high = 0.85
scope_confidence_medium = 0.6
scope_top_n = 3
base_dir = Path(__file__).resolve().parents[1]
url_list_file = base_dir / "data" / "url_list.json"

_embed_model = None
_allowed_scopes = None
_vector_store = None
_vector_store_path = None


def get_embed_model() -> HuggingFaceEmbeddings:
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embed_model


def get_vector_store(db_path: str = db) -> Chroma:
    global _vector_store, _vector_store_path
    if _vector_store is None or _vector_store_path != db_path:
        _vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=get_embed_model(),
        )
        _vector_store_path = db_path
    return _vector_store


def load_allowed_scopes(path: Path = url_list_file) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data.keys())


def get_allowed_scopes() -> list[str]:
    global _allowed_scopes
    if _allowed_scopes is None:
        _allowed_scopes = load_allowed_scopes()
    return _allowed_scopes


def map_question_to_scopes(question: str, allowed_scopes: list[str]) -> dict:
    client = ChatOpenAI(model=llm_use, temperature=0)
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
    parsed = json.loads(content)

    unknown = parsed.get("unknown", False)
    raw_scopes = parsed.get("scopes", [])

    best_by_scope = {}
    for item in raw_scopes:
        scope = item.get("scope")
        confidence = float(item.get("confidence", 0.0))
        if scope not in allowed_scopes:
            continue
        if scope not in best_by_scope or confidence > best_by_scope[scope]:
            best_by_scope[scope] = confidence

    ranked = sorted(best_by_scope.items(), key=lambda x: x[1], reverse=True)
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
    query_kwargs = {
        "query_embeddings": [query_emb],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection._collection.query(**query_kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []
    for idx, (text, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
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
    if seen_sources is None:
        seen_sources = set()

    unique_first_pass = []
    fallback_chunks = []

    # Prefer one chunk per article first to improve coverage/diversity.
    for item in candidates:
        source_key = item.get("source_key")
        if source_key not in seen_sources:
            unique_first_pass.append(item)
            seen_sources.add(source_key)
        else:
            fallback_chunks.append(item)

    selected = unique_first_pass[:limit]
    if len(selected) < limit:
        selected.extend(fallback_chunks[: limit - len(selected)])
    return selected, seen_sources


def retrieve(
    question: str,
    top_k: int,
    db_path: str = db,
    return_scope_decision: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    model = get_embed_model()
    collection = get_vector_store(db_path)

    allowed_scopes = get_allowed_scopes()
    scope_decision = map_question_to_scopes(question, allowed_scopes)
    top_scopes = scope_decision["top_scopes"]
    top_confidence = scope_decision["top_confidence"]

    overfetch_k = max(top_k * 4, top_k)
    query_emb = model.embed_query(question)

    selected = []
    seen_sources = set()

    if top_scopes and top_confidence >= scope_confidence_high:
        scope_decision["retrieval_mode"] = "scoped_high_confidence"
        scoped_candidates = query_candidates(
            collection,
            query_emb,
            n_results=overfetch_k,
            where={"scope": {"$in": top_scopes}},
        )
        scoped_selected, seen_sources = select_diverse(scoped_candidates, top_k, seen_sources)
        selected.extend(scoped_selected)

        if len(selected) < top_k:
            global_candidates = query_candidates(collection, query_emb, n_results=overfetch_k)
            backfill, seen_sources = select_diverse(
                global_candidates, top_k - len(selected), seen_sources
            )
            selected.extend(backfill)

    elif top_scopes and scope_confidence_medium <= top_confidence < scope_confidence_high:
        scope_decision["retrieval_mode"] = "mixed_medium_confidence"
        scoped_target = max(1, top_k // 2)
        global_target = max(1, top_k - scoped_target)

        scoped_candidates = query_candidates(
            collection,
            query_emb,
            n_results=overfetch_k,
            where={"scope": {"$in": top_scopes}},
        )
        global_candidates = query_candidates(collection, query_emb, n_results=overfetch_k)

        scoped_selected, seen_sources = select_diverse(
            scoped_candidates, min(scoped_target, top_k), seen_sources
        )
        selected.extend(scoped_selected)

        if len(selected) < top_k:
            global_selected, seen_sources = select_diverse(
                global_candidates, min(global_target, top_k - len(selected)), seen_sources
            )
            selected.extend(global_selected)

        if len(selected) < top_k:
            scoped_backfill, seen_sources = select_diverse(
                scoped_candidates, top_k - len(selected), seen_sources
            )
            selected.extend(scoped_backfill)

        if len(selected) < top_k:
            global_backfill, seen_sources = select_diverse(
                global_candidates, top_k - len(selected), seen_sources
            )
            selected.extend(global_backfill)

    else:
        scope_decision["retrieval_mode"] = "global_only"
        global_candidates = query_candidates(collection, query_emb, n_results=overfetch_k)
        selected, seen_sources = select_diverse(global_candidates, top_k, seen_sources)

    if return_scope_decision:
        return selected, scope_decision
    return selected


def llm_prompt(question: str, contexts: list[dict], detected_scopes: list[str] | None = None) -> str:
    context = ""
    for i, item in enumerate(contexts, start=1):
        context += f"\n[Source {i}]\n"
        context += f"Title: {item['title']}\n"
        context += f"URL: {item['url']}\n"
        context += f"Content: {item['text']}\n"

    scope_context = ""
    if detected_scopes:
        scope_context = f"\nDetected scopes: {detected_scopes}\n"

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

Your response must follow this structure:

### Answer
Provide a clear and concise answer.
- Focus on practical and concrete insights (e.g., specific clothing items, styling elements, trends)
- Use natural, conversational language
- Ground your answer using phrases like "Based on the provided sources..."
- If needed, include a short clarification of the question meaning

### Key Trends
List distinct trends supported by the context:
- Each bullet = ONE clear trend
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

### Sources
List all cited sources:
1. Title - URL

---

Rules:
- Prioritize the provided context
- Do NOT invent unsupported facts
- Do NOT copy text verbatim from the context
- Prefer specific details over vague generalizations
- Be transparent about uncertainty or gaps

---

Context:
{scope_context}
{context}

Question:
{question}

Answer:
""".strip()
    return prompt 

def generate_answer(prompt: str) -> str:
    client = ChatOpenAI(model=llm_use, temperature=0.1)

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


def qa_answer(question):
    question = question 
    top_k = default_top_k
    db_path = db

    contexts, scope_decision = retrieve(
        question, top_k=top_k, db_path=db_path, return_scope_decision=True
    )
    prompt = llm_prompt(question, contexts, detected_scopes=scope_decision.get("top_scopes", []))
    ans = generate_answer(prompt)
    return ans

if __name__ == "__main__":
    question = "What is the overall seasonal fashion trend for year 2026?"
    output = qa_answer(question)
    print(output)
