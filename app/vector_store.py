import chromadb

from app.config import CHROMA_PATH

_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(name: str):
    return _client.get_or_create_collection(name)


def get_collection(name: str):
    return _client.get_collection(name)


def list_collections() -> list[str]:
    return [c.name for c in _client.list_collections()]
