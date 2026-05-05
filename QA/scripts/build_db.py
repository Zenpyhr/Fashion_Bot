import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

base_dir = Path(__file__).resolve().parents[1]
chunk_file = base_dir / "data" / "processed_articles" / "fashion_qa_chunks.jsonl"
db_dir = base_dir / "index" / "fashion_chroma_db"
name = "fashion_articles"
batch_size = 32
embed_model_name = "all-MiniLM-L6-v2"

model = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name=name,
    persist_directory=str(db_dir),
    embedding_function=model,
)
collection = vector_store._collection

documents, metadatas, ids = [], [], []

with open(chunk_file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        item = json.loads(line)
        doc_id = f"{item['article_id']}::{item['chunk_index']}"
        documents.append(item["text"])
        metadata = {
            "article_id": item["article_id"],
            "chunk_index": item["chunk_index"],
            "chunk_count": item["chunk_count"],
            "title": item["title"],
            "url": item["url"],
            "domain": item["domain"],
            "scope": item["scope"],
        }
        metadatas.append(metadata)
        ids.append(doc_id)

for start in range(0, len(documents), batch_size):
    end = start + batch_size
    batch_docs = documents[start:end]
    batch_meta = metadatas[start:end]
    batch_ids = ids[start:end]

    embeddings = model.embed_documents(batch_docs)

    collection.upsert(
        documents=batch_docs,
        embeddings=embeddings,
        metadatas=batch_meta,
        ids=batch_ids
    )
print(f"Saved {len(documents)} chunks to ChromaDB at {db_dir}.")
