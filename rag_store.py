from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from pathlib import Path
import hashlib, re

# ------------------ Setup ------------------
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
client = QdrantClient("http://localhost:6333")


def _make_id(path: str) -> str:
    """Stable unique id per resume file."""
    return hashlib.md5(Path(path).name.encode()).hexdigest()[:8]


def _normalize(txt: str) -> str:
    """Normalize whitespace and line breaks."""
    return re.sub(r"\s+", " ", txt.strip())


# ------------------ Index Resume ------------------
def index_resume(path: str, text: str) -> str:
    """
    Create or replace a Qdrant collection for each resume.
    Deletes any old copy before indexing.
    """
    cid = _make_id(path)
    cname = f"resume_{cid}"

    # Delete old collection if it exists
    try:
        existing = [c.name for c in client.get_collections().collections]
        if cname in existing:
            client.delete_collection(collection_name=cname)
            print(f"[QDRANT] Old collection '{cname}' cleared.")
    except Exception as e:
        print(f"[QDRANT] Warning while clearing old collection: {e}")

    # Create fresh collection
    client.recreate_collection(
        collection_name=cname,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # Chunk text and embed
    chunks = [text[i:i + 800] for i in range(0, len(text), 800)]
    embeddings = encoder.encode(chunks).tolist()

    # Insert all chunks
    client.upsert(
        collection_name=cname,
        points=[
            {"id": i, "vector": embeddings[i], "payload": {"chunk": chunks[i]}}
            for i in range(len(chunks))
        ]
    )

    print(f"[QDRANT] Indexed {len(chunks)} chunks for '{Path(path).name}'.")
    return cname


# ------------------ Retrieve Context ------------------
def retrieve_context(cname: str, query: str, limit: int = 3) -> str:
    """Retrieve top relevant chunks for a query."""
    qv = encoder.encode([_normalize(query)])[0]
    hits = client.search(collection_name=cname, query_vector=qv, limit=limit)
    return " ".join([h.payload["chunk"] for h in hits])

# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# import hashlib, re

# # Load once
# encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# client = QdrantClient("http://localhost:6333")

# def _make_id(path: str):
#     """Stable unique id per resume."""
#     return hashlib.md5(Path(path).name.encode()).hexdigest()[:8]

# def _normalize(txt: str):
#     return re.sub(r"\s+", " ", txt.strip())

# def index_resume(path: str, text: str):
#     """Create isolated collection and index text chunks."""
#     cid = _make_id(path)
#     cname = f"resume_{cid}"

#     # reset collection (per resume isolation)
#     client.recreate_collection(
#         collection_name=cname,
#         vectors_config=VectorParams(size=768, distance=Distance.COSINE),
#     )

#     chunks = [text[i:i+800] for i in range(0, len(text), 800)]
#     embeddings = encoder.encode(chunks).tolist()
#     client.upsert(
#         collection_name=cname,
#         points=[{"id": i, "vector": embeddings[i], "payload": {"chunk": chunks[i]}}
#                 for i in range(len(chunks))]
#     )
#     return cname

# def retrieve_context(cname: str, query: str, limit=3):
#     """Retrieve top chunks for query."""
#     qv = encoder.encode([_normalize(query)])[0]
#     hits = client.search(collection_name=cname, query_vector=qv, limit=limit)
#     return " ".join([h.payload["chunk"] for h in hits])

# def certification_context(collection_name: str):
#     """Retrieve only certifications and publications context."""
#     query = "certification OR certificate OR course OR training OR publication"
#     return retrieve_context(collection_name, query)

