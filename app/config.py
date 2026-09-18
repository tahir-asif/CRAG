import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBEDDING_BATCH_SIZE = 64  # CPU-friendly batch size; tune down if memory-constrained

DEFAULT_INGEST_DEPTH = 1

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
REPO_CACHE_DIR = str(Path(__file__).parent.parent / "repo_cache")

SKIP_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    "target",
    ".idea",
    ".vscode",
    "site-packages",
    "vendor",
}

MAX_FILE_SIZE_KB = 500
