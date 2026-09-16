import chromadb

from app.config import CHROMA_PATH

client = chromadb.PersistentClient(path=CHROMA_PATH)

print("Collections:", [c.name for c in client.list_collections()])

collection = client.get_collection("click")
data = collection.get(include=["documents", "metadatas"])

ids = data["ids"]
metadatas = data["metadatas"]
documents = data["documents"]

# We asked for these in `include`, so they're guaranteed to be present.
# The asserts tell the type checker what we already know.
assert metadatas is not None
assert documents is not None

print(f"\nTotal chunks: {len(ids)}")
print(f"First 5 chunk IDs: {ids[:5]}")

print(f"No duplicated IDs: {len(data['ids']) == len(set(data['ids']))}")

collection = client.get_collection("click")
data2 = collection.get()
print(f"After re-ingest: {len(data2['ids'])} chunks")
assert len(data2["ids"]) == len(data["ids"]), "Duplicate chunks — delete didn't work"

for i in range(min(5, len(ids))):
    meta = metadatas[i]
    doc = documents[i]
    print(
        f"\n[{i}] {meta['file_path']}:{meta['start_line']}-{meta['end_line']}  ({meta['chunk_type']} {meta['name']})"
    )
    print(f"    Preview: {doc[:120]!r}")
